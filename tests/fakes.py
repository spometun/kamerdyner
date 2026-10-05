"""Test doubles and builders shared by the tests."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from kamerdyner.llm import GenerateResult, Prompt
from kamerdyner.messages import Message, Role
from kamerdyner.users import User

START = datetime(2026, 10, 4, 18, 32, tzinfo=UTC)
"""Sunday 2026-10-04, 14:32 in Toronto."""


def make_user() -> User:
    timezone = ZoneInfo("America/Toronto")
    return User(id="marta", name="Марта", timezone=timezone)


def make_message(role: Role, text: str, minutes: int = 0) -> Message:
    """A message written `minutes` after START."""
    offset = timedelta(minutes=minutes)
    return Message(role=role, text=text, at=START + offset)


class FakeClock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, minutes: int) -> None:
        self.now += timedelta(minutes=minutes)


class ScriptedLLM:
    """Returns the given results in order and records every prompt it receives."""

    def __init__(self, results: Sequence[GenerateResult]) -> None:
        self._results = list(results)
        self.prompts: list[Prompt] = []

    async def generate(self, prompt: Prompt) -> GenerateResult:
        self.prompts.append(prompt)
        return self._results.pop(0)


class RecordingMemory:
    """Keeps absorbed batches for inspection; its content is a fixed text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.absorbed: list[tuple[Message, ...]] = []

    def content(self) -> str:
        return self.text

    async def absorb(self, messages: Sequence[Message]) -> None:
        batch = tuple(messages)
        self.absorbed.append(batch)
