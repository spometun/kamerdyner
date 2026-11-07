"""Test doubles and builders shared by the tests."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from kamerdyner.common.llm import (
    LLMError,
    ModelReply,
    ModelTurn,
    Prompt,
    Role,
    ToolSpec,
    ToolTurn,
    UserTurn,
)
from kamerdyner.common.messages import Message
from kamerdyner.memory.impl.compaction import conversation_tokens
from kamerdyner.memory.memory import CompactionOutcome, NothingToCompact
from kamerdyner.users import User

TORONTO = ZoneInfo("America/Toronto")

START = datetime(2026, 10, 4, 14, 32, tzinfo=TORONTO)
"""Sunday 2026-10-04, 14:32 in Toronto."""


def make_user() -> User:
    return User(id="marta", name="Марта", timezone=TORONTO)


def plain_turn(turn: UserTurn | ModelTurn | ToolTurn) -> UserTurn | ModelTurn:
    """`turn`, which the test expects not to be a `ToolTurn`."""
    assert not isinstance(turn, ToolTurn), f"Expected a plain turn, got {turn!r}"
    return turn


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
    """Returns the given results in order and records every prompt it receives and the tools it
    was given with it."""

    def __init__(self, results: Sequence[ModelReply | LLMError]) -> None:
        self._results = list(results)
        self.prompts: list[Prompt] = []
        self.tools: list[Sequence[ToolSpec]] = []

    async def generate(
        self, prompt: Prompt, tools: Sequence[ToolSpec] = ()
    ) -> ModelReply | LLMError:
        self.prompts.append(prompt)
        self.tools.append(tools)
        return self._results.pop(0)


class FakeMemory:
    """Keeps the recent conversation in a list, starting with `messages`; its content is a fixed
    text. `compact` records the tokens asked to keep, moves nothing and answers `outcome`,
    `NothingToCompact` unless set."""

    def __init__(self, text: str, messages: Sequence[Message] = ()) -> None:
        self.text = text
        self._messages = list(messages)
        self.compactions: list[int] = []
        self.outcome: CompactionOutcome = NothingToCompact()

    @property
    def conversation(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    def content(self) -> str:
        return self.text

    def conversation_tokens(self) -> int:
        return conversation_tokens(self._messages)

    def add(self, exchange: Sequence[Message]) -> None:
        self._messages.extend(exchange)

    async def compact(self, keep_tokens: int) -> CompactionOutcome:
        self.compactions.append(keep_tokens)
        return self.outcome
