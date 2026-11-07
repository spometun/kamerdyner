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
class ToolSpec:
    """A tool as the model is told about it: `description` is what decides when the model calls
    it, `parameters` the JSON Schema of its arguments."""

    name: str
    description: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class ToolCall:
    """The model asks to run a tool; `id` ties the result to this call. The arguments come from
    the model: untrusted."""

    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class ToolResult:
    """What running `call` came to, in words for the model; a failure is told the same way."""

    call: ToolCall
    text: str


@dataclass(frozen=True)
class UserTurn:
    """What the user said, already rendered for the model."""

    text: str


@dataclass(frozen=True)
class ModelTurn:
    """What the model said. `provider_data` is the `ModelReply.provider_data` of this turn, empty
    when there is none. The tools it asked for here are in the `ToolTurn` that follows."""

    text: str
    provider_data: str = ""


@dataclass(frozen=True)
class ToolTurn:
    """What running the calls of the model's turn before it came to: one result per call, in
    their order."""

    results: tuple[ToolResult, ...]


@dataclass(frozen=True)
class Prompt:
    """A request to the model: the system instruction and the dialogue so far.

    Model turns alternate with those answering them, of the user or of tools; the dialogue starts
    and ends with the latter.
    """

    system_instruction: str
    turns: tuple[UserTurn | ModelTurn | ToolTurn, ...]

    def __post_init__(self) -> None:
        if not self.turns:
            raise ValueError("Invalid input. A prompt needs at least one turn")
        by_model = [isinstance(turn, ModelTurn) for turn in self.turns]
        if by_model[0] or by_model[-1]:
            raise ValueError(
                f"Invalid input. Prompt turns must not start or end with the model, got "
                f"{type(self.turns[0]).__name__} ... {type(self.turns[-1]).__name__}"
            )
        count = len(by_model)
        for index in range(1, count):
            if by_model[index] == by_model[index - 1]:
                raise ValueError(
                    f"Invalid input. Prompt turns must alternate, turns {index - 1} and {index} "
                    f"are {type(self.turns[index - 1]).__name__} and "
                    f"{type(self.turns[index]).__name__}"
                )


@dataclass(frozen=True)
class WebSearch:
    """A web search the model ran to answer: its query, and the page of its results."""

    query: str
    url: str


@dataclass(frozen=True)
class ModelReply:
    """The model answered in full. `grounded`: the answer rests on what the model found on the
    web, by searching or by reading a page it was given; `searches`: the searches it ran, which
    the provider's terms may require showing with the answer.

    With `tool_calls`, the model is not done: it waits for their results, and `text` (possibly
    empty) is not the answer.

    `provider_data` is the provider's own addition to this turn, unread outside its module: given
    back in `ModelTurn.provider_data` next to the text, it lets the model continue from what it
    thought here. It belongs to the provider that wrote it; empty means there is none."""

    text: str
    grounded: bool = False
    searches: tuple[WebSearch, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    provider_data: str = ""


@dataclass(frozen=True)
class Incomplete:
    """The model stopped early or answered nothing usable: cut off by the output limit, blocked
    by a safety filter, or an empty answer. `partial_text` is whatever text did arrive."""

    reason: str
    partial_text: str


@dataclass(frozen=True)
class Unavailable:
    """The model could not be reached or refused the request: network, rate limit, server error."""

    detail: str


LLMError = Incomplete | Unavailable
"""The model gave no answer to use."""


class LLM(Protocol):
    async def generate(
        self, prompt: Prompt, tools: Sequence[ToolSpec] = ()
    ) -> ModelReply | LLMError:
        """The model's response to `prompt`. It can call only `tools`: without any, a `ModelReply`
        never has `tool_calls`, and a model that tries anyway comes back as an `LLMError`."""
        ...
