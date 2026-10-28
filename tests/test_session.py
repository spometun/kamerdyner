import asyncio
import contextlib
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
from fakes import FakeClock, FakeMemory, make_user

from kamerdyner.chat import Chat
from kamerdyner.llm import GenerateResult, Prompt, Reply, Unavailable
from kamerdyner.session import Session
from kamerdyner.store import ConversationLog, MessageFile


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


def _session(
    directory: Path, llm: _GatedLLM, limit_tokens: int = 10_000, keep_tokens: int = 5_000
) -> tuple[Session, FakeMemory]:
    memory = FakeMemory("")
    pending = MessageFile(directory / "pending.json")
    log = ConversationLog(directory)
    chat = Chat(make_user(), log, pending, memory, llm, FakeClock())
    session = Session(chat, limit_tokens, keep_tokens)
    return session, memory


async def test_messages_arriving_during_a_reply_get_one_reply_together(tmp_path: Path) -> None:
    llm = _GatedLLM([Reply("Відповідь 1"), Reply("Відповідь 2")])
    session, memory = _session(tmp_path, llm)
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

    texts = [message.text for message in memory.messages]
    assert texts == ["Перше", "Відповідь 1", "Друге", "Третє", "Відповідь 2"]
    assert client.delivered == [Reply("Відповідь 1"), Reply("Відповідь 2")]
    assert client.typing_scopes == 2
    assert len(llm.prompts) == 2


async def test_a_failed_reply_leaves_the_messages_for_the_next_catch_up(tmp_path: Path) -> None:
    llm = _GatedLLM([Unavailable("503"), Reply("Нарешті")])
    llm.gate.set()
    session, memory = _session(tmp_path, llm)
    client = _Client()

    session.receive("Перше")
    await session.catch_up(client.deliver, client.replying)
    session.receive("Друге")
    await session.catch_up(client.deliver, client.replying)

    assert client.delivered == [Unavailable("503"), Reply("Нарешті")]
    texts = [message.text for message in memory.messages]
    assert texts == ["Перше", "Друге", "Нарешті"]


async def test_a_reply_reaching_the_limit_compacts_down_to_keep(tmp_path: Path) -> None:
    # 400 ASCII characters are 100 approximate tokens.
    llm = _GatedLLM([Reply("коротко"), Reply("x" * 400)])
    llm.gate.set()
    session, memory = _session(tmp_path, llm, limit_tokens=100, keep_tokens=40)
    client = _Client()

    session.receive("Перше")
    await session.catch_up(client.deliver, client.replying)
    assert memory.compactions == []
    session.receive("Друге")
    await session.catch_up(client.deliver, client.replying)

    assert memory.compactions == [40]


async def test_compaction_on_request_waits_for_the_running_reply(tmp_path: Path) -> None:
    llm = _GatedLLM([Reply("Відповідь")])
    session, memory = _session(tmp_path, llm)
    client = _Client()

    session.receive("Перше")
    replying = asyncio.create_task(session.catch_up(client.deliver, client.replying))
    await asyncio.sleep(0)
    compacting = asyncio.create_task(session.compact(0))
    await asyncio.sleep(0)
    assert not compacting.done()
    llm.gate.set()
    await asyncio.gather(replying, compacting)

    assert memory.compactions == [0]
    assert len(memory.messages) == 2


@pytest.mark.parametrize(("limit", "keep"), [(1000, 0), (1000, 1000), (0, 0)])
def test_rejects_inconsistent_sizes(tmp_path: Path, limit: int, keep: int) -> None:
    llm = _GatedLLM([])

    with pytest.raises(ValueError, match="Invalid input"):
        _session(tmp_path, llm, limit_tokens=limit, keep_tokens=keep)
