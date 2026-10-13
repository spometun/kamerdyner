import json
from collections.abc import Callable, Sequence
from pathlib import Path

from fakes import FakeClock, RecordingMemory, ScriptedLLM, make_user

from kamerdyner.chat import Chat
from kamerdyner.compaction import Compact, ShareCompactionPolicy, SizeThresholdPolicy
from kamerdyner.llm import GenerateResult, Incomplete, Prompt, Reply, Unavailable
from kamerdyner.memory import AbsorbResult, NotAbsorbed
from kamerdyner.messages import Message, Role
from kamerdyner.store import ConversationLog, CurrentConversation
from kamerdyner.tokens import ApproximateTokenCounter


class _Setup:
    def __init__(self, directory: Path, results: list[GenerateResult]) -> None:
        self.directory = directory
        self.clock = FakeClock()
        self.llm = ScriptedLLM(results)
        self.memory = RecordingMemory("Любить каву без цукру.")
        self.current = CurrentConversation(directory)
        log = ConversationLog(directory)
        # One token per byte; compact at 150 tokens, keeping at least 80.
        self.counter = ApproximateTokenCounter(bytes_per_token=1)
        self.policy = SizeThresholdPolicy(self.counter, limit_tokens=150, keep_tokens=80)
        user = make_user()
        self.chat = Chat(user, log, self.current, self.memory, self.llm, self.clock)

    def logged_texts(self) -> list[str]:
        lines = (self.directory / "log.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["text"] for line in lines]

    def current_texts(self) -> list[str]:
        return [message.text for message in self.current.messages]


async def test_reply_is_stored_and_returned(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Reply("Привіт, Марто!")])

    result = await setup.chat.send("Привіт")

    assert result == Reply("Привіт, Марто!")
    stored = [(message.role, message.text) for message in setup.current.messages]
    assert stored == [(Role.USER, "Привіт"), (Role.MODEL, "Привіт, Марто!")]
    assert setup.logged_texts() == ["Привіт", "Привіт, Марто!"]
    prompt = setup.llm.prompts[0]
    assert prompt.turns[-1].text == "[2026-10-04 Sun 14:32] Привіт"
    assert "Любить каву без цукру." in prompt.system_instruction


async def test_message_is_on_disk_before_the_model_is_called(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Unavailable("503")])

    await setup.chat.send("Перше")

    reopened = CurrentConversation(tmp_path)
    assert [message.text for message in reopened.messages] == ["Перше"]
    assert setup.logged_texts() == ["Перше"]


async def test_failed_reply_keeps_the_message_and_the_next_one_joins_it(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Unavailable("503"), Incomplete("MAX_TOKENS", "Пів"), Reply("Ось")])

    unavailable = await setup.chat.send("Перше")
    setup.clock.advance(1)
    incomplete = await setup.chat.send("Друге")
    setup.clock.advance(1)
    await setup.chat.send("Третє")

    assert unavailable == Unavailable("503")
    assert incomplete == Incomplete("MAX_TOKENS", "Пів")
    assert setup.current_texts() == ["Перше", "Друге", "Третє", "Ось"]
    last_prompt = setup.llm.prompts[-1]
    assert len(last_prompt.turns) == 1
    assert "Перше" in last_prompt.turns[0].text
    assert "Третє" in last_prompt.turns[0].text


async def test_compact_hands_the_oldest_messages_to_memory(tmp_path: Path) -> None:
    replies: list[GenerateResult] = [Reply("x" * 20) for _ in range(5)]
    setup = _Setup(tmp_path, replies)
    for index in range(3):
        await setup.chat.send(f"question {index} " + "y" * 10)
        await setup.chat.compact(setup.policy)
        assert setup.memory.absorbed == [], f"compacted too early, after message {index}"

    # 4 exchanges of 21 + 20 tokens = 164 >= 150. Keeping the last exchange alone (41 tokens) is
    # less than 80, so the last two stay (82 tokens) and the first two go.
    await setup.chat.send("question 3 " + "y" * 10)
    await setup.chat.compact(setup.policy)

    [absorbed] = setup.memory.absorbed
    assert [message.text[:10] for message in absorbed] == [
        "question 0",
        "x" * 10,
        "question 1",
        "x" * 10,
    ]
    remaining = [text[:10] for text in setup.current_texts()]
    assert remaining == ["question 2", "x" * 10, "question 3", "x" * 10]
    # The archive keeps everything; the model no longer sees what went to memory.
    assert len(setup.logged_texts()) == 8
    await setup.chat.send("question 4")
    first_turn = setup.llm.prompts[-1].turns[0]
    assert "question 2" in first_turn.text


async def test_compact_on_request_moves_at_least_half_and_reports_it(tmp_path: Path) -> None:
    replies: list[GenerateResult] = [Reply("x" * 20) for _ in range(2)]
    setup = _Setup(tmp_path, replies)
    await setup.chat.send("question 0 " + "y" * 10)
    await setup.chat.send("question 1 " + "y" * 10)
    on_request = ShareCompactionPolicy(setup.counter, share=0.5)

    plan = await setup.chat.compact(on_request)

    assert isinstance(plan, Compact)
    assert [message.text[:10] for message in plan.to_compress] == ["question 0", "x" * 10]
    assert setup.memory.absorbed == [plan.to_compress]
    assert [text[:10] for text in setup.current_texts()] == ["question 1", "x" * 10]


async def test_compact_keeps_the_conversation_when_memory_fails(tmp_path: Path) -> None:
    replies: list[GenerateResult] = [Reply("x" * 20) for _ in range(2)]
    setup = _Setup(tmp_path, replies)
    setup.memory.result = NotAbsorbed("the model is unavailable: 503")
    await setup.chat.send("question 0 " + "y" * 10)
    await setup.chat.send("question 1 " + "y" * 10)
    on_request = ShareCompactionPolicy(setup.counter, share=0.5)

    outcome = await setup.chat.compact(on_request)

    assert outcome == NotAbsorbed("the model is unavailable: 503")
    assert len(setup.memory.absorbed) == 1
    assert len(setup.current_texts()) == 4
    reopened = CurrentConversation(tmp_path)
    assert len(reopened.messages) == 4


class _MidwayLLM:
    """Replies with fixed texts; before the first reply, runs `meanwhile` — a message arriving
    while the model is thinking."""

    def __init__(self, texts: list[str], meanwhile: Callable[[], None]) -> None:
        self._texts = texts
        self._meanwhile = meanwhile
        self.prompts: list[Prompt] = []

    async def generate(self, prompt: Prompt) -> GenerateResult:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            self._meanwhile()
        text = self._texts.pop(0)
        return Reply(text)


async def test_a_message_arriving_during_a_reply_is_answered_next(tmp_path: Path) -> None:
    clock = FakeClock()
    current = CurrentConversation(tmp_path)
    log = ConversationLog(tmp_path)
    memory = RecordingMemory("")
    holder: list[Chat] = []
    llm = _MidwayLLM(["Відповідь 1", "Відповідь 2"], lambda: holder[0].receive("Друге"))
    chat = Chat(make_user(), log, current, memory, llm, clock)
    holder.append(chat)

    chat.receive("Перше")
    await chat.reply()

    assert [message.text for message in current.messages] == ["Перше", "Відповідь 1", "Друге"]
    assert chat.awaiting_reply
    await chat.reply()
    assert not chat.awaiting_reply
    assert "Друге" in llm.prompts[1].turns[-1].text


async def test_a_message_arriving_during_compaction_stays(tmp_path: Path) -> None:
    replies: list[GenerateResult] = [Reply("x" * 20) for _ in range(2)]
    setup = _Setup(tmp_path, replies)
    await setup.chat.send("question 0 " + "y" * 10)
    await setup.chat.send("question 1 " + "y" * 10)
    original_absorb = setup.memory.absorb

    async def absorb_while_one_arrives(messages: Sequence[Message]) -> AbsorbResult:
        setup.chat.receive("arrived meanwhile")
        return await original_absorb(messages)

    setup.memory.absorb = absorb_while_one_arrives
    on_request = ShareCompactionPolicy(setup.counter, share=0.5)

    await setup.chat.compact(on_request)

    remaining = [text[:10] for text in setup.current_texts()]
    assert remaining == ["question 1", "x" * 10, "arrived me"]
    assert setup.chat.awaiting_reply
