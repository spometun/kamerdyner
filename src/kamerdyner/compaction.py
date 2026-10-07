"""Deciding when the current conversation has grown too long and which part of it goes to memory."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from kamerdyner.messages import Message, Role
from kamerdyner.tokens import TokenCounter


@dataclass(frozen=True)
class NoAction:
    """The conversation stays as it is."""


@dataclass(frozen=True)
class Compact:
    """The older messages `to_compress` go to memory; `to_keep` stay in the conversation.

    Together they are the whole conversation, in order. The cut falls right after a reply of the
    model, so no exchange — the user's message(s) and the reply to them — is split.
    """

    to_compress: tuple[Message, ...]
    to_keep: tuple[Message, ...]


type CompactionPlan = NoAction | Compact


class CompactionPolicy(Protocol):
    def plan(self, messages: Sequence[Message]) -> CompactionPlan: ...


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

    def plan(self, messages: Sequence[Message]) -> CompactionPlan:
        total = conversation_tokens(messages, self.counter)
        if total < self.limit_tokens:
            return NoAction()
        return split_oldest(messages, self.counter, self.min_compress_tokens)


@dataclass(frozen=True)
class ShareCompactionPolicy:
    """Always compacts, moving at least `share` of the conversation's tokens, oldest first."""

    counter: TokenCounter
    share: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.share <= 1:
            raise ValueError(f"Invalid input. share must be in (0, 1], got {self.share}")

    def plan(self, messages: Sequence[Message]) -> CompactionPlan:
        total = conversation_tokens(messages, self.counter)
        min_tokens = math.ceil(total * self.share)
        return split_oldest(messages, self.counter, min_tokens)


def conversation_tokens(messages: Sequence[Message], counter: TokenCounter) -> int:
    sizes = [counter.count(message.text) for message in messages]
    return sum(sizes)


def split_oldest(messages: Sequence[Message], counter: TokenCounter, min_tokens: int) -> Compact:
    """Splits off the oldest messages worth at least `min_tokens` — or all of them, if the
    conversation holds less — cutting only right after a reply of the model."""
    if min_tokens < 0:
        raise ValueError(f"Invalid input. min_tokens must not be negative, got {min_tokens}")
    count = len(messages)
    compressed = 0
    for cut in range(1, count):
        compressed += counter.count(messages[cut - 1].text)
        after_reply = messages[cut - 1].role is Role.MODEL
        if compressed >= min_tokens and after_reply:
            to_compress = tuple(messages[:cut])
            to_keep = tuple(messages[cut:])
            return Compact(to_compress=to_compress, to_keep=to_keep)
    everything = tuple(messages)
    return Compact(to_compress=everything, to_keep=())
