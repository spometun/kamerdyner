"""Long-term memory of a user: what the assistant knows beyond the recent conversation.

Each implementation lives in its own module of this package (`notes`); the ways of compressing
messages into memory text live next to them (`summary`). Whatever the way, memory keeps both
sides of the conversation: the user first of all, and the assistant itself — what it said,
advised and promised."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from kamerdyner.messages import Message


@dataclass(frozen=True)
class Absorbed:
    """The messages are now part of memory and may leave the recent conversation."""


@dataclass(frozen=True)
class NotAbsorbed:
    """Memory could not take the messages and is unchanged; they must stay in the conversation."""

    reason: str


type AbsorbResult = Absorbed | NotAbsorbed


class Memory(Protocol):
    def content(self) -> str:
        """The memory as text for the model's system instruction; empty when nothing is known."""
        ...

    async def absorb(self, messages: Sequence[Message]) -> AbsorbResult:
        """Folds messages leaving the recent conversation into memory, oldest first."""
        ...


class EmptyMemory:
    """Remembers nothing: absorbed messages are dropped and the content is always empty."""

    def content(self) -> str:
        return ""

    async def absorb(self, messages: Sequence[Message]) -> AbsorbResult:
        return Absorbed()
