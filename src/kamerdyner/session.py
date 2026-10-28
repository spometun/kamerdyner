"""A user's chat driven by a client whose messages arrive at any moment, not one after a reply."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import assert_never

from kamerdyner.chat import Chat
from kamerdyner.llm import GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.memory import CompactionOutcome

type Deliver = Callable[[GenerateResult], Awaitable[None]]
"""Shows the user the outcome of a reply."""

type Replying = Callable[[], AbstractAsyncContextManager[None]]
"""A scope that lasts while the model is writing a reply, e.g. to show that it is typing."""


class Session:
    """Runs replies and compactions of one chat one at a time.

    Every incoming message is stored at once with `receive`; `catch_up` then answers. Messages
    that arrive while a reply or a compaction is running wait for it and are answered together
    by the next reply. After a full reply that brings the conversation to `limit_tokens`, it is
    compacted down to at least `keep_tokens` of the newest, silently: the client shows nothing
    for it.
    """

    def __init__(self, chat: Chat, limit_tokens: int, keep_tokens: int) -> None:
        if not 0 < keep_tokens < limit_tokens:
            raise ValueError(
                f"Invalid input. Need 0 < keep_tokens < limit_tokens, got "
                f"keep_tokens={keep_tokens}, limit_tokens={limit_tokens}"
            )
        self._chat = chat
        self._limit_tokens = limit_tokens
        self._keep_tokens = keep_tokens
        self._lock = asyncio.Lock()

    @property
    def conversation_tokens(self) -> int:
        return self._chat.conversation_tokens

    def receive(self, text: str) -> None:
        self._chat.receive(text)

    async def catch_up(self, deliver: Deliver, replying: Replying) -> None:
        """Answers until no message is left without a reply. When the model fails, the messages
        stay unanswered until the next `catch_up`."""
        async with self._lock:
            while self._chat.awaiting_reply:
                async with replying():
                    result = await self._chat.reply()
                await deliver(result)
                match result:
                    case Reply():
                        if self._chat.conversation_tokens >= self._limit_tokens:
                            await self._chat.compact(self._keep_tokens)
                    case Incomplete() | Unavailable():
                        return
                    case _:
                        assert_never(result)

    async def compact(self, keep_tokens: int) -> CompactionOutcome:
        """Compacts now, keeping at least `keep_tokens` of the newest, once the running reply or
        compaction is over."""
        async with self._lock:
            return await self._chat.compact(keep_tokens)
