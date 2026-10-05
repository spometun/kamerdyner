"""One user's chat with the assistant: the conversation, its memory and the model put together."""

import logging
from typing import assert_never

from kamerdyner.compaction import Compact, CompactionPolicy, NoAction
from kamerdyner.llm import LLM, GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.memory import Memory
from kamerdyner.messages import Clock, Message, Role
from kamerdyner.prompt import build_prompt
from kamerdyner.store import ConversationLog
from kamerdyner.users import User

logger = logging.getLogger(__name__)


class Chat:
    """Answers a user's messages and hands the overflow of the conversation to memory.

    A client calls `send` for each incoming message and `compact` when it is a good moment to
    tidy up: right after showing the reply, or after a pause in the conversation.
    """

    def __init__(
        self,
        user: User,
        log: ConversationLog,
        memory: Memory,
        policy: CompactionPolicy,
        llm: LLM,
        clock: Clock,
    ) -> None:
        self._user = user
        self._log = log
        self._memory = memory
        self._policy = policy
        self._llm = llm
        self._clock = clock

    async def send(self, text: str) -> GenerateResult:
        """Stores the message before calling the model, then stores a full reply. On any other
        outcome the message stays in the conversation without a reply."""
        received_at = self._clock()
        incoming = Message(role=Role.USER, text=text, at=received_at)
        self._log.append(incoming)
        recent = self._log.recent()
        memory = self._memory.content()
        prompt = build_prompt(self._user, memory, recent)
        result = await self._llm.generate(prompt)
        match result:
            case Reply(text=reply_text):
                answered_at = self._clock()
                reply = Message(role=Role.MODEL, text=reply_text, at=answered_at)
                self._log.append(reply)
            case Incomplete(reason=reason):
                logger.warning("Incomplete reply for user %s: %s", self._user.id, reason)
            case Unavailable(detail=detail):
                logger.warning("Model unavailable for user %s: %s", self._user.id, detail)
            case _:
                assert_never(result)
        return result

    async def compact(self) -> None:
        """Moves the oldest part of the conversation into memory if the policy says so."""
        now = self._clock()
        recent = self._log.recent()
        plan = self._policy.plan(recent, now)
        match plan:
            case NoAction():
                pass
            case Compact(to_compress=to_compress):
                count = len(to_compress)
                await self._memory.absorb(to_compress)
                self._log.fold(count)
                logger.info("Folded %d messages of user %s into memory", count, self._user.id)
            case _:
                assert_never(plan)
