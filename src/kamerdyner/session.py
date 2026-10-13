"""A user's chat driven by a client whose messages arrive at any moment, not one after a reply."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from typing import assert_never

from kamerdyner.chat import Chat, CompactionOutcome
from kamerdyner.compaction import CompactionPolicy
from kamerdyner.llm import GenerateResult, Incomplete, Reply, Unavailable

type Deliver = Callable[[GenerateResult], Awaitable[None]]
"""Shows the user the outcome of a reply."""

type Replying = Callable[[], AbstractAsyncContextManager[None]]
"""A scope that lasts while the model is writing a reply, e.g. to show that it is typing."""


class Session:
    """Runs replies and compactions of one chat one at a time.

    Every incoming message is stored at once with `receive`; `catch_up` then answers. Messages
    that arrive while a reply or a compaction is running wait for it and are answered together
    by the next reply. After each full reply the conversation is compacted with `after_reply`,
    silently: the client shows nothing for it.
    """

    def __init__(self, chat: Chat, after_reply: CompactionPolicy) -> None:
        self._chat = chat
        self._after_reply = after_reply
        self._lock = asyncio.Lock()

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
                        await self._chat.compact(self._after_reply)
                    case Incomplete() | Unavailable():
                        return
                    case _:
                        assert_never(result)

    async def compact(self, policy: CompactionPolicy) -> CompactionOutcome:
        """Compacts now with `policy`, once the running reply or compaction is over."""
        async with self._lock:
            return await self._chat.compact(policy)
