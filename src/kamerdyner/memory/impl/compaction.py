"""Choosing which part of the recent conversation goes to long-term memory."""

import math
from collections.abc import Sequence

from kamerdyner.common.llm import Role
from kamerdyner.common.messages import Message


def conversation_tokens(messages: Sequence[Message]) -> int:
    """The size of the messages' texts in model tokens, estimated without the model's tokenizer
    from the UTF-8 size, so Cyrillic (two bytes a character) weighs more than Latin, as it does
    for real tokenizers. A rough estimate, erring towards more tokens."""
    sizes = [_tokens(message.text) for message in messages]
    return sum(sizes)


def split_newest(
    messages: Sequence[Message], keep_tokens: int
) -> tuple[tuple[Message, ...], tuple[Message, ...]]:
    """Splits the messages into the oldest, to compress, and the newest, to keep — together the
    whole conversation, in order. The cut falls as late as leaves at least `keep_tokens` of the
    newest, and only right after a reply of the model, so no exchange — the user's message(s) and
    the reply to them — is split. Without such a cut there is nothing to compress.

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
            return tuple(messages[:cut]), tuple(messages[cut:])
        kept += _tokens(messages[cut - 1].text)
    return (), tuple(messages)


_BYTES_PER_TOKEN = 4


def _tokens(text: str) -> int:
    encoded = text.encode("utf-8")
    return math.ceil(len(encoded) / _BYTES_PER_TOKEN)
