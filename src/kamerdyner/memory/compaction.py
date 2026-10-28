"""Choosing which part of the recent conversation goes to long-term memory."""

from collections.abc import Sequence
from dataclasses import dataclass

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


def conversation_tokens(messages: Sequence[Message], counter: TokenCounter) -> int:
    sizes = [counter.count(message.text) for message in messages]
    return sum(sizes)


def split_newest(
    messages: Sequence[Message], counter: TokenCounter, keep_tokens: int
) -> CompactionPlan:
    """Splits off the oldest messages, as many as leave at least `keep_tokens` of the newest,
    cutting only right after a reply of the model; `NoAction` when no such cut exists.

    Moving everything older at once rather than just an overflow keeps compaction rare: memory
    gets fewer, fuller entries, and the start of the prompt stays stable between compactions. An
    exchange too large to leave `keep_tokens` behind it stays until enough newer messages follow.
    """
    if keep_tokens < 0:
        raise ValueError(f"Invalid input. keep_tokens must not be negative, got {keep_tokens}")
    kept = 0
    for cut in range(len(messages), 0, -1):
        after_reply = messages[cut - 1].role is Role.MODEL
        if kept >= keep_tokens and after_reply:
            to_compress = tuple(messages[:cut])
            to_keep = tuple(messages[cut:])
            return Compact(to_compress=to_compress, to_keep=to_keep)
        kept += counter.count(messages[cut - 1].text)
    return NoAction()
