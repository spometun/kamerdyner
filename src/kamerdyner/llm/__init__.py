"""The boundary with the language model: what a request to it is and what can come back.

Each provider implements `LLM` in its own module of this package (`gemini`); everything that
is specific to a provider stays inside its module."""

from dataclasses import dataclass
from typing import Protocol

from kamerdyner.messages import Role


@dataclass(frozen=True)
class Turn:
    """One turn of the dialogue as the model sees it: the text already rendered for the model."""

    role: Role
    text: str


@dataclass(frozen=True)
class Prompt:
    """A request to the model: the system instruction and the dialogue so far.

    Turns alternate between the user and the model, starting and ending with the user.
    """

    system_instruction: str
    turns: tuple[Turn, ...]

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
class Reply:
    """The model answered in full."""

    text: str


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


type GenerateResult = Reply | Incomplete | Unavailable


class LLM(Protocol):
    async def generate(self, prompt: Prompt) -> GenerateResult: ...
