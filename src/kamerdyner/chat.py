"""One user's chat with the assistant: the incoming messages, their memory and the model put
together, and the archive of everything said."""

import json
import logging
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import assert_never

from kamerdyner.common.llm import LLM, GenerateResult, Incomplete, Reply, Role, Unavailable
from kamerdyner.common.messages import Message
from kamerdyner.common.store import MessageFile
from kamerdyner.memory import CompactionOutcome, Memory
from kamerdyner.prompt import build_prompt
from kamerdyner.users import User

logger = logging.getLogger(__name__)

type Clock = Callable[[], datetime]
"""Returns the current time as an aware datetime."""


def utc_now() -> datetime:
    return datetime.now(UTC)


class ConversationLog:
    """The archive of every message, one JSON object per line in `log.jsonl`, in the format of
    `kamerdyner.common.store`; appended to durably, never read on the way to the model."""

    def __init__(self, directory: Path) -> None:
        self._path = directory / "log.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        _drop_torn_tail(self._path)

    def append(self, message: Message) -> None:
        """Once this returns, the message survives a crash."""
        record = _record(message)
        line = json.dumps(record, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            descriptor = file.fileno()
            os.fsync(descriptor)


class Chat:
    """Answers a user's messages; each answered exchange goes to memory.

    A client stores each incoming message with `receive` as soon as it arrives and asks for an
    answer with `reply` — `send` does both — and calls `compact` when it is a good moment to tidy
    up memory: right after showing the reply, after a pause in the conversation, or on the
    user's request. The messages still waiting for a reply are kept in `pending`, so a crash
    loses none of them; the model sees memory's conversation followed by them.

    Messages may keep arriving while a reply or a compaction is under way: they wait in
    `pending` and are answered by the next `reply`. `reply` and `compact` themselves must not
    overlap; the client runs them one at a time.
    """

    def __init__(
        self,
        user: User,
        log: ConversationLog,
        pending: MessageFile,
        memory: Memory,
        llm: LLM,
        clock: Clock,
    ) -> None:
        self._user = user
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
        result = await self._llm.generate(prompt)
        match result:
            case Reply(text=reply_text):
                answered_at = self._now()
                reply = Message(role=Role.MODEL, text=reply_text, at=answered_at)
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

    async def send(self, text: str) -> GenerateResult:
        """Stores the message, then answers it."""
        self.receive(text)
        return await self.reply()

    async def compact(self, keep_tokens: int) -> CompactionOutcome:
        """Moves the oldest part of memory's recent conversation into its long-term memory,
        keeping at least `keep_tokens` of the newest."""
        return await self._memory.compact(keep_tokens)

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


def _record(message: Message) -> dict[str, str]:
    return {"role": message.role.value, "text": message.text, "at": message.at.isoformat()}


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
