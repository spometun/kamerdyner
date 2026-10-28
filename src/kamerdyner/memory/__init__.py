"""Memory of a user: the recent conversation, and what the assistant keeps of the older one.

`Memory` is the interface; each implementation lives in its own module of this package
(`notes`), and the ways of compressing messages into memory text next to them (`summary`). An
implementation keeps everything of its own in one directory given to it, loads it from there when
created, and makes the helpers it needs itself. Whatever the way, long-term memory keeps both
sides of the conversation: the user first of all, and the assistant itself — what it said,
advised and promised."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from kamerdyner.memory.compaction import Compact, NoAction
from kamerdyner.messages import Message


@dataclass(frozen=True)
class NotAbsorbed:
    """Long-term memory could not take the messages and is unchanged; they stay in the recent
    conversation."""

    reason: str


type CompactionOutcome = NoAction | Compact | NotAbsorbed
"""What `Memory.compact` did: nothing, the plan carried out, or nothing because long-term memory
could not take the messages."""


class Memory(Protocol):
    @property
    def messages(self) -> tuple[Message, ...]:
        """The recent conversation, oldest first: whole exchanges, as `add` took them."""
        ...

    def content(self) -> str:
        """Long-term memory as text for the model's system instruction; empty when nothing is
        known yet."""
        ...

    def conversation_tokens(self) -> int:
        """The size of `messages` in (approximate) model tokens."""
        ...

    def add(self, exchange: Sequence[Message]) -> None:
        """Appends an exchange to the recent conversation: one or more messages of the user and
        then the model's reply to them. Once this returns, it survives a crash."""
        ...

    async def compact(self, keep_tokens: int) -> CompactionOutcome:
        """Moves the oldest part of the recent conversation into long-term memory, keeping at
        least `keep_tokens` of the newest; 0 moves everything."""
        ...
