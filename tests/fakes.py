"""Test doubles and builders shared by the tests."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from kamerdyner.llm import GenerateResult, Prompt
from kamerdyner.memory import CompactionOutcome
from kamerdyner.memory.compaction import NoAction, conversation_tokens
from kamerdyner.messages import Message, Role
from kamerdyner.tokens import ApproximateTokenCounter
from kamerdyner.users import User

TORONTO = ZoneInfo("America/Toronto")

START = datetime(2026, 10, 4, 14, 32, tzinfo=TORONTO)
"""Sunday 2026-10-04, 14:32 in Toronto."""


def make_user() -> User:
    return User(id="marta", name="Марта", timezone=TORONTO)


def make_message(role: Role, text: str, minutes: int = 0) -> Message:
    """A message written `minutes` after START, in Toronto time."""
    offset = timedelta(minutes=minutes)
    return Message(role=role, text=text, at=START + offset)


class FakeClock:
    """Tells UTC time, as the system clock does, starting at START."""

    def __init__(self) -> None:
        self.now = START.astimezone(UTC)

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


class FakeMemory:
    """Keeps the recent conversation in a list, starting with `messages`; its content is a fixed
    text. `compact` records the tokens asked to keep, moves nothing and answers `outcome`,
    `NoAction` unless set."""

    def __init__(self, text: str, messages: Sequence[Message] = ()) -> None:
        self.text = text
        self.conversation = list(messages)
        self.compactions: list[int] = []
        self.outcome: CompactionOutcome = NoAction()

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self.conversation)

    def content(self) -> str:
        return self.text

    def conversation_tokens(self) -> int:
        counter = ApproximateTokenCounter()
        return conversation_tokens(self.conversation, counter)

    def add(self, exchange: Sequence[Message]) -> None:
        self.conversation.extend(exchange)

    async def compact(self, keep_tokens: int) -> CompactionOutcome:
        self.compactions.append(keep_tokens)
        return self.outcome
