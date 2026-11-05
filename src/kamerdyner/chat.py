"""One user's chat with the assistant: the incoming messages, their memory and the model put
together, and the archive of everything said."""

import dataclasses
import json
import logging
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import assert_never

from kamerdyner.common.llm import (
    LLM,
    GenerateResult,
    Incomplete,
    LLMError,
    Prompt,
    Reply,
    Role,
    ToolCall,
    ToolResult,
    ToolRound,
    ToolSpec,
    Unavailable,
)
from kamerdyner.common.messages import Message, format_time_with_offset, without_ephemeral_text
from kamerdyner.common.store import MessageFile, message_record
from kamerdyner.memory.memory import CompactionOutcome, Memory
from kamerdyner.prompt import build_prompt
from kamerdyner.users import User, find_timezone, save_timezone

logger = logging.getLogger(__name__)

type Clock = Callable[[], datetime]
"""Returns the current time as an aware datetime."""


def utc_now() -> datetime:
    return datetime.now(UTC)


class ConversationLog:
    """The archive of every message, one JSON object per line in `log.jsonl`, in the format of
    `kamerdyner.common.store`; appended to durably, never read on the way to the model. An
    ephemeral message is archived without its text (`without_ephemeral_text`)."""

    def __init__(self, directory: Path) -> None:
        self._path = directory / "log.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        _drop_torn_tail(self._path)

    def append(self, message: Message) -> None:
        """Once this returns, the message survives a crash."""
        archived = without_ephemeral_text(message)
        record = message_record(archived)
        line = json.dumps(record, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            descriptor = file.fileno()
            os.fsync(descriptor)


class Chat:
    """Answers a user's messages; each answered exchange goes to memory.

    A client stores each incoming message with `receive` as soon as it arrives, asks for an
    answer with `reply`, and calls `compact` when it is a good moment to tidy up memory: right
    after showing the reply, after a pause in the conversation, or on the user's request. The
    messages still waiting for a reply are kept in `pending`, so a crash loses none of them; the
    model sees memory's conversation followed by them.

    Messages may keep arriving while a reply or a compaction is under way: they wait in
    `pending` and are answered by the next `reply`. `reply` and `compact` themselves must not
    overlap; the client runs them one at a time.

    The model may call tools while answering: one round of them, then it answers. What they did
    stays in the conversation as text at the head of the reply, and the client never sees that.
    """

    def __init__(
        self,
        user: User,
        directory: Path,
        log: ConversationLog,
        pending: MessageFile,
        memory: Memory,
        llm: LLM,
        clock: Clock,
    ) -> None:
        self._user = user
        self._directory = directory
        self._log = log
        self._pending = pending
        self._memory = memory
        self._llm = llm
        self._clock = clock
        self._drop_answered()

    @property
    def awaiting_reply(self) -> bool:
        """Whether some of the user's messages have no reply yet."""
        return bool(self._pending.messages)

    @property
    def conversation_tokens(self) -> int:
        """The size of memory's recent conversation in (approximate) model tokens."""
        return self._memory.conversation_tokens()

    def receive(self, text: str) -> None:
        """Stores an incoming message; once this returns, it survives a crash."""
        received_at = self._now()
        incoming = Message(role=Role.USER, text=text, at=received_at)
        self._log.append(incoming)
        self._pending.append(incoming)

    async def reply(self) -> GenerateResult:
        """Answers the messages waiting for a reply; a full reply goes to memory together with the
        messages the model saw, and only those leave the waiting ones. On any other outcome the
        messages keep waiting, to be answered together with the next ones."""
        seen = self._pending.messages
        assert seen, f"Internal error. Nothing to reply to for user {self._user.id}"
        memory = self._memory.content()
        conversation = (*self._memory.conversation, *seen)
        prompt = build_prompt(self._user, memory, conversation)
        first = await self._llm.generate(prompt, _TOOLS)
        result, tools_record = await self._finish_tools(prompt, first)
        match result:
            case Reply(text=reply_text, grounded=grounded):
                answered_at = self._now()
                # Search results may be shown and kept for the conversation, not archived or learned
                # from: a grounded reply is ephemeral.
                reply = Message(
                    role=Role.MODEL,
                    text=tools_record + reply_text,
                    at=answered_at,
                    ephemeral=grounded,
                )
                self._log.append(reply)
                exchange = (*seen, reply)
                # Memory first: a crash before the next line leaves the exchange in both, and
                # `_drop_answered` settles that on the next start.
                self._memory.add(exchange)
                arrived = self._messages_after(seen)
                self._pending.replace(arrived)
            case Incomplete(reason=reason):
                logger.warning("Incomplete reply for user %s: %s", self._user.id, reason)
            case Unavailable(detail=detail):
                logger.warning("Model unavailable for user %s: %s", self._user.id, detail)
            case _:
                assert_never(result)
        return result

    async def compact(self, keep_tokens: int) -> CompactionOutcome:
        """Moves the oldest part of memory's recent conversation into its long-term memory,
        keeping at least `keep_tokens` of the newest."""
        return await self._memory.compact(keep_tokens)

    async def _finish_tools(
        self, prompt: Prompt, first: GenerateResult
    ) -> tuple[GenerateResult, str]:
        """Takes the model's first response to the end: runs the tools it asked for and gets its
        answer, with no tools allowed any more. Returns that final result and the record of the
        tools to keep in front of the answer, empty when none ran. A tool that ran stays done
        even if the answer then fails, and may run again when the messages are answered anew."""
        if isinstance(first, LLMError) or not first.tool_calls:
            return first, ""
        results = self._run_tools(first.tool_calls)
        tool_round = ToolRound(first.provider_turn, results)
        with_results = dataclasses.replace(prompt, tool_round=tool_round)
        final = await self._llm.generate(with_results)
        if isinstance(final, LLMError):
            return final, ""
        assert not final.tool_calls, (
            f"Internal error. Tool calls {final.tool_calls} with no tools allowed, "
            f"user {self._user.id}"
        )
        # What the first turn found on the web went into the answer too.
        whole = dataclasses.replace(
            final,
            grounded=first.grounded or final.grounded,
            searches=(*first.searches, *final.searches),
        )
        return whole, _tools_record(results)

    def _run_tools(self, calls: Sequence[ToolCall]) -> tuple[ToolResult, ...]:
        results: list[ToolResult] = []
        for call in calls:
            text = self._run_tool(call)
            logger.info("Tool %s of user %s: %s", call, self._user.id, text)
            result = ToolResult(call=call, text=text)
            results.append(result)
        return tuple(results)

    def _run_tool(self, call: ToolCall) -> str:
        """Runs a call of the model; a call out of contract gets its error told to the model."""
        match call:
            case ToolCall(name=_SET_TIMEZONE.name, arguments={"zone": str() as zone}):
                return self._set_timezone(zone)
            case _:
                logger.warning("Rejected a tool call of user %s: %s", self._user.id, call)
                return f"Error: no tool {call.name} taking arguments {dict(call.arguments)}."

    def _set_timezone(self, zone: str) -> str:
        timezone = find_timezone(zone)
        if timezone is None:
            return f'Error: unknown time zone "{zone}". Use an IANA name such as Europe/Kyiv.'
        save_timezone(self._directory, timezone)
        self._user = dataclasses.replace(self._user, timezone=timezone)
        now = self._now()
        stamp = format_time_with_offset(now)
        return f"Time zone set to {zone}. Local time is now {stamp}."

    def _now(self) -> datetime:
        """The current time on the user's clock."""
        now = self._clock()
        return now.astimezone(self._user.timezone)

    def _messages_after(self, prefix: Sequence[Message]) -> tuple[Message, ...]:
        """The waiting messages that follow `prefix`, which must still be their start."""
        messages = self._pending.messages
        count = len(prefix)
        assert messages[:count] == tuple(prefix), (
            f"Internal error. The waiting messages of user {self._user.id} no longer start with "
            f"the {count} they had; there are {len(messages)}"
        )
        return messages[count:]

    def _drop_answered(self) -> None:
        """Drops the waiting messages that memory already holds: those the process stored in
        memory with their reply but died before removing from the waiting ones."""
        held = set(self._memory.conversation)
        waiting = [message for message in self._pending.messages if message not in held]
        dropped = len(self._pending.messages) - len(waiting)
        if dropped:
            logger.warning(
                "Dropping %d waiting messages of user %s already answered in memory",
                dropped,
                self._user.id,
            )
            self._pending.replace(waiting)


_SET_TIMEZONE = ToolSpec(
    name="set_timezone",
    description=(
        "Sets the user's time zone, which the timestamps of their messages follow from now on. "
        "Call it only when the user plainly says they moved, are travelling or want it changed; "
        "never on a guess."
    ),
    parameters={
        "type": "object",
        "properties": {
            "zone": {"type": "string", "description": "IANA time zone name, e.g. Europe/Kyiv"},
        },
        "required": ["zone"],
    },
)

_TOOLS = (_SET_TIMEZONE,)


def _tools_record(results: Sequence[ToolResult]) -> str:
    """What the tools did, as it stands before the answer in the conversation."""
    lines: list[str] = []
    for result in results:
        call = _render_call(result.call)
        lines.append(f"Model requested tool {call}.")
        lines.append(f"The system ran it, result: {result.text}")
    lines.append("In the next request the model replied:\n")
    return "\n".join(lines)


def _render_call(call: ToolCall) -> str:
    arguments: list[str] = []
    for name, value in call.arguments.items():
        rendered = json.dumps(value, ensure_ascii=False)
        arguments.append(f"{name}={rendered}")
    joined = ", ".join(arguments)
    return f"{call.name}({joined})"


def _drop_torn_tail(path: Path) -> None:
    """Cuts off a last line left unfinished by a crash mid-write. Such a message was never
    acknowledged as stored, and appending after it would glue two records into one line."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    end = data.rfind(b"\n") + 1
    if end == len(data):
        return
    logger.warning("Dropping a torn last line of %s: %r", path, data[end:])
    with path.open("r+b") as file:
        file.truncate(end)
        file.flush()
        descriptor = file.fileno()
        os.fsync(descriptor)
