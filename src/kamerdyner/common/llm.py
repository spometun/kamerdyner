"""The boundary with the language model: what a request to it is and what can come back.

Each provider implements `LLM` in its own module of `kamerdyner.llm` (`gemini`); everything
that is specific to a provider stays inside its module."""

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


class Role(enum.Enum):
    """Who speaks in a dialogue with the model: the user, or the model itself."""

    USER = "user"
    MODEL = "model"


@dataclass(frozen=True)
class Turn:
    """One turn of the dialogue as the model sees it: the text already rendered for the model."""

    role: Role
    text: str


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the model is told about it: `description` is what decides when the model calls
    it, `parameters` the JSON Schema of its arguments."""

    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ToolCall:
    """The model asks to run a tool. The arguments come from the model: untrusted."""

    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ToolResult:
    """What running `call` came to, in words for the model; a failure is told the same way."""

    call: ToolCall
    text: str


@dataclass(frozen=True)
class ToolRound:
    """The tools the model asked for, run: `provider_turn` is the `Reply.provider_turn` that
    asked, `results` hold one result per call of that reply, in its order."""

    provider_turn: object
    results: tuple[ToolResult, ...]


@dataclass(frozen=True)
class Prompt:
    """A request to the model: the system instruction and the dialogue so far, and, when the
    model asked for tools in answer to these turns, what running them came to.

    Turns alternate between the user and the model, starting and ending with the user.
    """

    system_instruction: str
    turns: tuple[Turn, ...]
    tool_round: ToolRound | None = None

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("Invalid input. A prompt needs at least one turn")
        if self.turns[0].role is not Role.USER or self.turns[-1].role is not Role.USER:
            raise ValueError(
                f"Invalid input. Prompt turns must start and end with the user, got "
                f"{self.turns[0].role} ... {self.turns[-1].role}"
            )
        count = len(self.turns)
        for index in range(1, count):
            if self.turns[index].role is self.turns[index - 1].role:
                raise ValueError(
                    f"Invalid input. Prompt turns must alternate, turns {index - 1} and {index} "
                    f"are both {self.turns[index].role}"
                )


@dataclass(frozen=True)
class WebSearch:
    """A web search the model ran to answer: its query, and the page of its results."""

    query: str
    url: str


@dataclass(frozen=True)
class Reply:
    """The model answered in full. `grounded`: the answer rests on what the model found on the
    web, by searching or by reading a page it was given; `searches`: the searches it ran, which
    the provider's terms may require showing with the answer.

    With `tool_calls`, the model is not done: it waits for their results, and `text` (possibly
    empty) is not the answer. `provider_turn` is then the provider's own record of this turn, to
    hand back unread in `ToolRound`; without calls it is None."""

    text: str
    grounded: bool = False
    searches: tuple[WebSearch, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    provider_turn: object = None


@dataclass(frozen=True)
class LLMError:
    """The model gave no answer to use. Match it to treat all failures alike, or its subclasses
    to tell them apart."""


@dataclass(frozen=True)
class Incomplete(LLMError):
    """The model stopped early or answered nothing usable: cut off by the output limit, blocked
    by a safety filter, or an empty answer. `partial_text` is whatever text did arrive."""

    reason: str
    partial_text: str


@dataclass(frozen=True)
class Unavailable(LLMError):
    """The model could not be reached or refused the request: network, rate limit, server error."""

    detail: str


type GenerateResult = Reply | Incomplete | Unavailable


class LLM(Protocol):
    async def generate(self, prompt: Prompt, tools: Sequence[ToolSpec] = ()) -> GenerateResult:
        """The model's response to `prompt`. It can call only `tools`: without any, a `Reply`
        never has `tool_calls`, and a model that tries anyway comes back as an `LLMError`."""
        ...
