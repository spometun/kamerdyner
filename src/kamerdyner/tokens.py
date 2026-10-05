"""Measuring text size in model tokens."""

import math
from dataclasses import dataclass
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class ApproximateTokenCounter:
    """Estimates tokens from the UTF-8 size, without calling the model's tokenizer.

    Counts in bytes rather than characters, so Cyrillic (two bytes a character) weighs more
    than Latin, as it does for real tokenizers. A rough estimate, erring towards more tokens.
    """

    bytes_per_token: int = 4

    def __post_init__(self) -> None:
        if self.bytes_per_token <= 0:
            raise ValueError(
                f"Invalid input. bytes_per_token must be positive, got {self.bytes_per_token}"
            )

    def count(self, text: str) -> int:
        encoded = text.encode("utf-8")
        size = len(encoded)
        return math.ceil(size / self.bytes_per_token)
