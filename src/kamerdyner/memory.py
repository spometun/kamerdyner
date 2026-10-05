"""Long-term memory of a user: what the assistant knows beyond the recent conversation."""

from collections.abc import Sequence
from typing import Protocol

from kamerdyner.messages import Message


class Memory(Protocol):
    def content(self) -> str:
        """The memory as text for the model's system instruction; empty when nothing is known."""
        ...

    async def absorb(self, messages: Sequence[Message]) -> None:
        """Folds messages leaving the recent conversation into memory, oldest first."""
        ...


class EmptyMemory:
    """Remembers nothing: absorbed messages are dropped and the content is always empty."""

    def content(self) -> str:
        return ""

    async def absorb(self, messages: Sequence[Message]) -> None:
        pass
