"""Memory as the assistant's notes on each stretch of conversation, oldest first.

The memory directory holds:

    notes.md     the memory itself, what goes into the system instruction: one entry per absorbed
                 stretch, each opening with the heading of its time span; rewritten whole on
                 every change.
    latest.json  the current conversation, the messages the model sees (see `kamerdyner.store`).
    memory.log   what memory did, for a person to read: every pass with the conversation it took
                 and the notes it wrote, or why it failed. Append-only, never read back.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import assert_never

from kamerdyner.llm import LLM, Incomplete, Reply, Unavailable
from kamerdyner.memory import Absorbed, AbsorbResult, NotAbsorbed
from kamerdyner.memory.summary import span_heading, summarize, transcript
from kamerdyner.messages import Clock, Message
from kamerdyner.prompt import format_time
from kamerdyner.store import write_atomically
from kamerdyner.users import User

logger = logging.getLogger(__name__)


class NotesMemory:
    """Absorbs each stretch of conversation as notes written by the model (`summary`) and
    appends them to `notes.md`; the content is all the notes so far."""

    def __init__(self, directory: Path, user: User, llm: LLM, clock: Clock) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._notes_path = directory / "notes.md"
        self._journal_path = directory / "memory.log"
        self._user = user
        self._llm = llm
        self._clock = clock
        self._notes = _load_notes(self._notes_path)

    def content(self) -> str:
        return self._notes.rstrip("\n")

    async def absorb(self, messages: Sequence[Message]) -> AbsorbResult:
        """Keeps the notes unchanged when the model fails to write them."""
        if not messages:
            # Nothing to remember; not worth a call to the model.
            return Absorbed()
        conversation = transcript(self._user, messages)
        result = await summarize(self._llm, self._user, messages)
        match result:
            case Reply(text=text):
                entry = text.strip()
                self._store(entry)
                sections = [("conversation", conversation), ("notes", entry)]
                heading = span_heading(self._user, messages)
                first_line = entry.split("\n", 1)[0]
                if first_line != heading:
                    warning = f"the notes open with {first_line!r} instead of {heading!r}"
                    logger.warning("Notes of user %s: %s", self._user.id, warning)
                    sections.append(("warning", warning))
                self._journal(f"absorbed {len(messages)} messages", sections)
                logger.info("Absorbed %d messages of user %s", len(messages), self._user.id)
                return Absorbed()
            case Incomplete(reason=reason, partial_text=partial_text):
                outcome = NotAbsorbed(reason=f"the notes came back incomplete: {reason}")
                sections = [("conversation", conversation), ("partial notes", partial_text)]
            case Unavailable(detail=detail):
                outcome = NotAbsorbed(reason=f"the model is unavailable: {detail}")
                sections = [("conversation", conversation)]
            case _:
                assert_never(result)
        logger.warning("Memory of user %s not updated: %s", self._user.id, outcome.reason)
        self._journal(f"could not absorb {len(messages)} messages: {outcome.reason}", sections)
        return outcome

    def _store(self, entry: str) -> None:
        notes = self._notes + entry + "\n\n"
        write_atomically(self._notes_path, notes)
        self._notes = notes

    def _journal(self, title: str, sections: Sequence[tuple[str, str]]) -> None:
        now = self._clock()
        stamp = format_time(now, self._user.timezone)
        parts = [f"==== [{stamp}] {title} ====\n"]
        for name, body in sections:
            parts.append(f"--- {name} ---\n{body}\n")
        record = "\n".join(parts) + "\n"
        with self._journal_path.open("a", encoding="utf-8") as file:
            file.write(record)


def _load_notes(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
