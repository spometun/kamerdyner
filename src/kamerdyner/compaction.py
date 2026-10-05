"""Deciding when the recent conversation has grown too long and which part of it goes to memory."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from kamerdyner.messages import Message, Role
from kamerdyner.tokens import TokenCounter


@dataclass(frozen=True)
class NoAction:
    """The conversation stays as it is."""


@dataclass(frozen=True)
class Compact:
    """The older messages `to_compress` go to memory; `to_keep` stay in the conversation.

    Together they are the whole conversation, in order. `to_keep` is empty or starts with a user
    message, so the dialogue left behind begins with a whole exchange.
    """

    to_compress: tuple[Message, ...]
    to_keep: tuple[Message, ...]


type CompactionPlan = NoAction | Compact


class CompactionPolicy(Protocol):
    def plan(self, messages: Sequence[Message], now: datetime) -> CompactionPlan: ...


@dataclass(frozen=True)
class SizeThresholdPolicy:
    """Compacts once the conversation reaches `limit_tokens`, moving at least
    `min_compress_tokens` of the oldest messages to memory.

    Moving a large share at once rather than just the overflow keeps compaction rare: memory
    gets fewer, fuller entries, and the start of the prompt stays stable between compactions.
    """

    counter: TokenCounter
    limit_tokens: int = 16_000
    min_compress_tokens: int = 8_000

    def __post_init__(self) -> None:
        if not 0 < self.min_compress_tokens <= self.limit_tokens:
            raise ValueError(
                f"Invalid input. Need 0 < min_compress_tokens <= limit_tokens, got "
                f"min_compress_tokens={self.min_compress_tokens}, "
                f"limit_tokens={self.limit_tokens}"
            )

    def plan(self, messages: Sequence[Message], now: datetime) -> CompactionPlan:
        sizes = [self.counter.count(message.text) for message in messages]
        total = sum(sizes)
        if total < self.limit_tokens:
            return NoAction()
        count = len(messages)
        compressed = 0
        for cut in range(1, count + 1):
            compressed += sizes[cut - 1]
            at_turn_start = cut == count or messages[cut].role is Role.USER
            if compressed >= self.min_compress_tokens and at_turn_start:
                to_compress = tuple(messages[:cut])
                to_keep = tuple(messages[cut:])
                return Compact(to_compress=to_compress, to_keep=to_keep)
        raise AssertionError(
            f"Internal error. No cut found in {len(messages)} messages of {total} tokens, "
            f"limit {self.limit_tokens}, min_compress {self.min_compress_tokens}"
        )
