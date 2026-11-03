"""The interface of memory: `Memory`, and what compacting it can come to."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from kamerdyner.common.messages import Message


@dataclass(frozen=True)
class NothingToCompact:
    """No cut leaves enough of the newest conversation; nothing moved."""


@dataclass(frozen=True)
class Compacted:
    """`moved` messages went from the recent conversation into long-term memory; `kept`
    remain."""

    moved: int
    kept: int


@dataclass(frozen=True)
class NotAbsorbed:
    """Long-term memory could not take the messages and is unchanged; they stay in the recent
    conversation."""

    reason: str


type CompactionOutcome = NothingToCompact | Compacted | NotAbsorbed
"""What `Memory.compact` did."""


class Memory(Protocol):
    @property
    def conversation(self) -> tuple[Message, ...]:
        """The recent conversation, oldest first: whole exchanges, as `add` took them."""
        ...

    def content(self) -> str:
        """Long-term memory as text for the model's system instruction; empty when nothing is
        known yet."""
        ...

    def conversation_tokens(self) -> int:
        """The size of `conversation` in (approximate) model tokens."""
        ...

    def add(self, exchange: Sequence[Message]) -> None:
        """Appends an exchange to the recent conversation: one or more messages of the user and
        then the model's reply to them. Once this returns, it survives a crash."""
        ...

    async def compact(self, keep_tokens: int) -> CompactionOutcome:
        """Moves the oldest part of the recent conversation into long-term memory, keeping at
        least `keep_tokens` of the newest; 0 moves everything. Nothing is added while it runs."""
        ...
