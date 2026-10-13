"""One user's chat with the assistant: the conversation, its memory and the model put together."""

import logging
from collections.abc import Sequence
from typing import assert_never

from kamerdyner.compaction import Compact, CompactionPlan, CompactionPolicy, NoAction
from kamerdyner.llm import LLM, GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.memory import Absorbed, Memory, NotAbsorbed
from kamerdyner.messages import Clock, Message, Role
from kamerdyner.prompt import build_prompt
from kamerdyner.store import ConversationLog, CurrentConversation
from kamerdyner.users import User

logger = logging.getLogger(__name__)

type CompactionOutcome = CompactionPlan | NotAbsorbed
"""What `Chat.compact` did: nothing, the plan carried out, or nothing because memory could not
take the messages."""


class Chat:
    """Answers a user's messages and hands the overflow of the conversation to memory.

    A client stores each incoming message with `receive` as soon as it arrives and asks for an
    answer with `reply` — `send` does both — and calls `compact` when it is a good moment to tidy
    up: right after showing the reply, after a pause in the conversation, or on the user's
    request, with the policy that fits the moment.

    Messages may keep arriving while a reply or a compaction is under way: they stay in the
    conversation and are answered by the next `reply`. `reply` and `compact` themselves must not
    overlap; the client runs them one at a time.
    """

    def __init__(
        self,
        user: User,
        log: ConversationLog,
        current: CurrentConversation,
        memory: Memory,
        llm: LLM,
        clock: Clock,
    ) -> None:
        self._user = user
        self._log = log
        self._current = current
        self._memory = memory
        self._llm = llm
        self._clock = clock

    @property
    def awaiting_reply(self) -> bool:
        """Whether the conversation ends with the user's messages that have no reply yet."""
        messages = self._current.messages
        return bool(messages) and messages[-1].role is Role.USER

    def receive(self, text: str) -> None:
        """Stores an incoming message; once this returns, it survives a crash."""
        received_at = self._clock()
        incoming = Message(role=Role.USER, text=text, at=received_at)
        self._log.append(incoming)
        self._current.append(incoming)

    async def reply(self) -> GenerateResult:
        """Answers the conversation so far and stores a full reply right after the messages the
        model saw, before any that arrived meanwhile. On any other outcome the user's messages
        stay without a reply, to be answered together with the next ones."""
        seen = self._current.messages
        assert self.awaiting_reply, (
            f"Internal error. Nothing to reply to for user {self._user.id}: the conversation "
            f"of {len(seen)} messages does not end with the user's"
        )
        memory = self._memory.content()
        prompt = build_prompt(self._user, memory, seen)
        result = await self._llm.generate(prompt)
        match result:
            case Reply(text=reply_text):
                answered_at = self._clock()
                reply = Message(role=Role.MODEL, text=reply_text, at=answered_at)
                self._log.append(reply)
                arrived = self._messages_after(seen)
                messages = (*seen, reply, *arrived)
                self._current.replace(messages)
            case Incomplete(reason=reason):
                logger.warning("Incomplete reply for user %s: %s", self._user.id, reason)
            case Unavailable(detail=detail):
                logger.warning("Model unavailable for user %s: %s", self._user.id, detail)
            case _:
                assert_never(result)
        return result

    async def send(self, text: str) -> GenerateResult:
        """Stores the message, then answers it."""
        self.receive(text)
        return await self.reply()

    async def compact(self, policy: CompactionPolicy) -> CompactionOutcome:
        """Moves the oldest part of the conversation into memory if the policy says so. The
        conversation is cut only once memory has taken the messages."""
        plan = policy.plan(self._current.messages)
        match plan:
            case NoAction():
                return plan
            case Compact(to_compress=to_compress):
                absorbed = await self._memory.absorb(to_compress)
                match absorbed:
                    case Absorbed():
                        # Keeps what arrived while memory was absorbing, besides `to_keep`.
                        remaining = self._messages_after(to_compress)
                        self._current.replace(remaining)
                        logger.info(
                            "Moved %d messages of user %s into memory, %d remain",
                            len(to_compress),
                            self._user.id,
                            len(remaining),
                        )
                        return plan
                    case NotAbsorbed(reason=reason):
                        logger.warning(
                            "Kept %d messages of user %s in the conversation: %s",
                            len(to_compress),
                            self._user.id,
                            reason,
                        )
                        return absorbed
                    case _:
                        assert_never(absorbed)
            case _:
                assert_never(plan)

    def _messages_after(self, prefix: Sequence[Message]) -> tuple[Message, ...]:
        """The messages of the conversation that follow `prefix`, which must still be its start."""
        messages = self._current.messages
        count = len(prefix)
        assert messages[:count] == tuple(prefix), (
            f"Internal error. The conversation of user {self._user.id} no longer starts with the "
            f"{count} messages it had; it holds {len(messages)}"
        )
        return messages[count:]
