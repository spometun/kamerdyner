"""Memory as the recent conversation plus the assistant's notes on each older stretch of it.

The memory directory holds:

    latest.json  the recent conversation, the messages the model sees (see `kamerdyner.store`).
    notes.md     long-term memory, what goes into the system instruction: one entry per absorbed
                 stretch, oldest first, each opening with the heading of its time span; rewritten
                 whole on every change.
    memory.log   what memory did, for a person to read: every pass with the conversation it took
                 and the notes it wrote, or why it failed. Append-only, never read back.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import assert_never

from kamerdyner.llm import LLM, Incomplete, Reply, Unavailable
from kamerdyner.memory import CompactionOutcome, NotAbsorbed
from kamerdyner.memory.compaction import Compact, NoAction, conversation_tokens, split_newest
from kamerdyner.memory.summary import span_heading, summarize, transcript
from kamerdyner.messages import Message, Role, format_time
from kamerdyner.store import MessageFile, write_atomically
from kamerdyner.tokens import ApproximateTokenCounter

logger = logging.getLogger(__name__)


class NotesMemory:
    """Keeps the recent conversation; compacting turns its older stretch into notes written by
    the model (`summary`) and appends them to `notes.md`. The content is all the notes so far.

    `name` is how the user is called in the notes; `llm` writes them.
    """

    def __init__(self, directory: Path, name: str, llm: LLM) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._conversation = MessageFile(directory / "latest.json")
        self._notes_path = directory / "notes.md"
        self._journal_path = directory / "memory.log"
        self._name = name
        self._llm = llm
        self._counter = ApproximateTokenCounter()
        self._notes = _load_notes(self._notes_path)

    @property
    def messages(self) -> tuple[Message, ...]:
        return self._conversation.messages

    def content(self) -> str:
        return self._notes.rstrip("\n")

    def conversation_tokens(self) -> int:
        return conversation_tokens(self._conversation.messages, self._counter)

    def add(self, exchange: Sequence[Message]) -> None:
        roles = [message.role for message in exchange]
        well_formed = len(roles) >= 2 and roles[-1] is Role.MODEL and Role.MODEL not in roles[:-1]
        if not well_formed:
            raise ValueError(
                f"Invalid input. An exchange is the user's messages and then one reply of the "
                f"model, got roles {[role.value for role in roles]}"
            )
        messages = (*self._conversation.messages, *exchange)
        self._conversation.replace(messages)

    async def compact(self, keep_tokens: int) -> CompactionOutcome:
        """The conversation is cut only once the notes on the older stretch are stored; when the
        model fails to write them, memory stays as it was."""
        plan = split_newest(self._conversation.messages, self._counter, keep_tokens)
        match plan:
            case NoAction():
                return plan
            case Compact(to_compress=to_compress):
                absorbed = await self._absorb(to_compress)
                match absorbed:
                    case _Absorbed():
                        remaining = self._messages_after(to_compress)
                        self._conversation.replace(remaining)
                        logger.info(
                            "Moved %d messages of %s into notes, %d remain",
                            len(to_compress),
                            self._name,
                            len(remaining),
                        )
                        return plan
                    case NotAbsorbed():
                        return absorbed
                    case _:
                        assert_never(absorbed)
            case _:
                assert_never(plan)

    async def _absorb(self, messages: Sequence[Message]) -> "_Absorbed | NotAbsorbed":
        conversation = transcript(self._name, messages)
        result = await summarize(self._llm, self._name, messages)
        match result:
            case Reply(text=text):
                entry = text.strip()
                self._store(entry)
                sections = [("conversation", conversation), ("notes", entry)]
                heading = span_heading(messages)
                first_line = entry.split("\n", 1)[0]
                if first_line != heading:
                    warning = f"the notes open with {first_line!r} instead of {heading!r}"
                    logger.warning("Notes of %s: %s", self._name, warning)
                    sections.append(("warning", warning))
                self._journal(f"absorbed {len(messages)} messages", sections)
                return _Absorbed()
            case Incomplete(reason=reason, partial_text=partial_text):
                outcome = NotAbsorbed(reason=f"the notes came back incomplete: {reason}")
                sections = [("conversation", conversation), ("partial notes", partial_text)]
            case Unavailable(detail=detail):
                outcome = NotAbsorbed(reason=f"the model is unavailable: {detail}")
                sections = [("conversation", conversation)]
            case _:
                assert_never(result)
        logger.warning("Notes of %s not updated: %s", self._name, outcome.reason)
        self._journal(f"could not absorb {len(messages)} messages: {outcome.reason}", sections)
        return outcome

    def _messages_after(self, prefix: Sequence[Message]) -> tuple[Message, ...]:
        """The messages of the conversation that follow `prefix`, which must still be its start."""
        messages = self._conversation.messages
        count = len(prefix)
        assert messages[:count] == tuple(prefix), (
            f"Internal error. The conversation of {self._name} no longer starts with the {count} "
            f"messages it had; it holds {len(messages)}"
        )
        return messages[count:]

    def _store(self, entry: str) -> None:
        notes = self._notes + entry + "\n\n"
        write_atomically(self._notes_path, notes)
        self._notes = notes

    def _journal(self, title: str, sections: Sequence[tuple[str, str]]) -> None:
        # The machine's local time: the journal is read by the person running the program.
        now = datetime.now().astimezone()
        stamp = format_time(now)
        parts = [f"==== [{stamp}] {title} ====\n"]
        for name, body in sections:
            parts.append(f"--- {name} ---\n{body}\n")
        record = "\n".join(parts) + "\n"
        with self._journal_path.open("a", encoding="utf-8") as file:
            file.write(record)


@dataclass(frozen=True)
class _Absorbed:
    """The messages are now in the notes and may leave the recent conversation."""


def _load_notes(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
