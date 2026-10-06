import json
from pathlib import Path

from fakes import FakeClock, RecordingMemory, ScriptedLLM, make_user

from kamerdyner.chat import Chat
from kamerdyner.compaction import SizeThresholdPolicy
from kamerdyner.llm import GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.messages import Role
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
        # One token per byte; compact at 150 tokens, moving at least 50.
        counter = ApproximateTokenCounter(bytes_per_token=1)
        policy = SizeThresholdPolicy(counter, limit_tokens=150, min_compress_tokens=50)
        user = make_user()
        self.chat = Chat(user, log, self.current, self.memory, policy, self.llm, self.clock)

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
        await setup.chat.compact()
        assert setup.memory.absorbed == [], f"compacted too early, after message {index}"

    # 4 exchanges of 21 + 20 tokens = 164 >= 150. The minimum of 50 is passed inside the second
    # exchange, so the cut moves to its end: the first two exchanges (82 tokens) go.
    await setup.chat.send("question 3 " + "y" * 10)
    await setup.chat.compact()

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
