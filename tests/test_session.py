import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

from fakes import FakeClock, RecordingMemory, make_user

from kamerdyner.chat import Chat
from kamerdyner.compaction import ShareCompactionPolicy, SizeThresholdPolicy
from kamerdyner.llm import GenerateResult, Prompt, Reply, Unavailable
from kamerdyner.session import Session
from kamerdyner.store import ConversationLog, CurrentConversation
from kamerdyner.tokens import ApproximateTokenCounter


class _GatedLLM:
    """Answers with the given results, each only once the test opens the gate."""

    def __init__(self, results: list[GenerateResult]) -> None:
        self._results = results
        self.gate = asyncio.Event()
        self.prompts: list[Prompt] = []

    async def generate(self, prompt: Prompt) -> GenerateResult:
        self.prompts.append(prompt)
        await self.gate.wait()
        return self._results.pop(0)


class _Client:
    """Records what the user is shown: each delivered result and each typing scope."""

    def __init__(self) -> None:
        self.delivered: list[GenerateResult] = []
        self.typing_scopes = 0

    async def deliver(self, result: GenerateResult) -> None:
        self.delivered.append(result)

    @contextlib.asynccontextmanager
    async def replying(self) -> AsyncGenerator[None]:
        self.typing_scopes += 1
        yield


def _session(directory: Path, llm: _GatedLLM) -> tuple[Session, CurrentConversation]:
    current = CurrentConversation(directory)
    log = ConversationLog(directory)
    chat = Chat(make_user(), log, current, RecordingMemory(""), llm, FakeClock())
    counter = ApproximateTokenCounter()
    policy = SizeThresholdPolicy(counter, limit_tokens=10_000, keep_tokens=5_000)
    return Session(chat, policy), current


async def test_messages_arriving_during_a_reply_get_one_reply_together(tmp_path: Path) -> None:
    llm = _GatedLLM([Reply("Відповідь 1"), Reply("Відповідь 2")])
    session, current = _session(tmp_path, llm)
    client = _Client()

    session.receive("Перше")
    first = asyncio.create_task(session.catch_up(client.deliver, client.replying))
    await asyncio.sleep(0)
    session.receive("Друге")
    second = asyncio.create_task(session.catch_up(client.deliver, client.replying))
    session.receive("Третє")
    third = asyncio.create_task(session.catch_up(client.deliver, client.replying))
    llm.gate.set()
    await asyncio.gather(first, second, third)

    texts = [message.text for message in current.messages]
    assert texts == ["Перше", "Відповідь 1", "Друге", "Третє", "Відповідь 2"]
    assert client.delivered == [Reply("Відповідь 1"), Reply("Відповідь 2")]
    assert client.typing_scopes == 2
    assert len(llm.prompts) == 2


async def test_a_failed_reply_leaves_the_messages_for_the_next_catch_up(tmp_path: Path) -> None:
    llm = _GatedLLM([Unavailable("503"), Reply("Нарешті")])
    llm.gate.set()
    session, current = _session(tmp_path, llm)
    client = _Client()

    session.receive("Перше")
    await session.catch_up(client.deliver, client.replying)
    session.receive("Друге")
    await session.catch_up(client.deliver, client.replying)

    assert client.delivered == [Unavailable("503"), Reply("Нарешті")]
    texts = [message.text for message in current.messages]
    assert texts == ["Перше", "Друге", "Нарешті"]


async def test_compaction_on_request_waits_for_the_running_reply(tmp_path: Path) -> None:
    llm = _GatedLLM([Reply("Відповідь")])
    session, current = _session(tmp_path, llm)
    client = _Client()
    counter = ApproximateTokenCounter()
    on_request = ShareCompactionPolicy(counter, share=1.0)

    session.receive("Перше")
    replying = asyncio.create_task(session.catch_up(client.deliver, client.replying))
    await asyncio.sleep(0)
    compacting = asyncio.create_task(session.compact(on_request))
    await asyncio.sleep(0)
    assert not compacting.done()
    llm.gate.set()
    await asyncio.gather(replying, compacting)

    assert current.messages == ()
