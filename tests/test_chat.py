import json
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path

from fakes import START, FakeClock, FakeMemory, ScriptedLLM, make_message, make_user

from kamerdyner.chat import Chat
from kamerdyner.common.llm import GenerateResult, Incomplete, Prompt, Reply, Role, Turn, Unavailable
from kamerdyner.common.messages import Message
from kamerdyner.common.store import ConversationLog, MessageFile
from kamerdyner.memory import NotAbsorbed


class _Setup:
    def __init__(
        self,
        directory: Path,
        results: list[GenerateResult],
        remembered: Sequence[Message] = (),
    ) -> None:
        self.directory = directory
        self.clock = FakeClock()
        self.llm = ScriptedLLM(results)
        self.memory = FakeMemory("Любить каву без цукру.", remembered)
        self.pending = MessageFile(directory / "pending.json")
        self.log = ConversationLog(directory)
        self.chat = self.reopen()

    def reopen(self) -> Chat:
        user = make_user()
        return Chat(user, self.log, self.pending, self.memory, self.llm, self.clock)

    def logged_texts(self) -> list[str]:
        lines = (self.directory / "log.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["text"] for line in lines]

    def memory_texts(self) -> list[str]:
        return [message.text for message in self.memory.conversation]

    def pending_texts(self) -> list[str]:
        return [message.text for message in self.pending.messages]


async def test_reply_goes_to_memory_with_the_message(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Reply("Привіт, Марто!")])

    result = await setup.chat.send("Привіт")

    assert result == Reply("Привіт, Марто!")
    stored = [(message.role, message.text) for message in setup.memory.conversation]
    assert stored == [(Role.USER, "Привіт"), (Role.MODEL, "Привіт, Марто!")]
    assert setup.pending_texts() == []
    assert not setup.chat.awaiting_reply
    assert setup.logged_texts() == ["Привіт", "Привіт, Марто!"]
    prompt = setup.llm.prompts[0]
    assert prompt.turns[-1].text == "[2026-10-04 Sun 14:32] Привіт"
    assert "Любить каву без цукру." in prompt.system_instruction


async def test_messages_carry_the_users_local_time(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Reply("Ось")])

    await setup.chat.send("Котра година?")

    [question, answer] = setup.memory.conversation
    assert question.at == START
    assert question.at.utcoffset() == timedelta(hours=-4)
    assert answer.at.utcoffset() == timedelta(hours=-4)
    record = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["at"] == "2026-10-04T14:32:00-04:00"


async def test_message_is_on_disk_before_the_model_is_called(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Unavailable("503")])

    await setup.chat.send("Перше")

    reopened = MessageFile(tmp_path / "pending.json")
    assert [message.text for message in reopened.messages] == ["Перше"]
    assert setup.memory_texts() == []
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
    assert setup.memory_texts() == ["Перше", "Друге", "Третє", "Ось"]
    assert setup.pending_texts() == []
    last_prompt = setup.llm.prompts[-1]
    assert len(last_prompt.turns) == 1
    assert "Перше" in last_prompt.turns[0].text
    assert "Третє" in last_prompt.turns[0].text


async def test_model_sees_memorys_conversation_then_the_waiting_messages(tmp_path: Path) -> None:
    earlier = [
        make_message(Role.USER, "Як справи?", minutes=-60),
        make_message(Role.MODEL, "Добре.", minutes=-59),
    ]
    setup = _Setup(tmp_path, [Reply("Ось")], remembered=earlier)

    await setup.chat.send("А в тебе?")

    assert setup.llm.prompts[0].turns == (
        Turn(Role.USER, "[2026-10-04 Sun 13:32] Як справи?"),
        Turn(Role.MODEL, "Добре."),
        Turn(Role.USER, "[2026-10-04 Sun 14:32] А в тебе?"),
    )


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
    memory = FakeMemory("")
    pending = MessageFile(tmp_path / "pending.json")
    log = ConversationLog(tmp_path)
    holder: list[Chat] = []
    llm = _MidwayLLM(["Відповідь 1", "Відповідь 2"], lambda: holder[0].receive("Друге"))
    chat = Chat(make_user(), log, pending, memory, llm, FakeClock())
    holder.append(chat)

    chat.receive("Перше")
    await chat.reply()

    assert [message.text for message in memory.conversation] == ["Перше", "Відповідь 1"]
    assert [message.text for message in pending.messages] == ["Друге"]
    assert chat.awaiting_reply
    await chat.reply()
    assert not chat.awaiting_reply
    texts = [message.text for message in memory.conversation]
    assert texts == ["Перше", "Відповідь 1", "Друге", "Відповідь 2"]
    assert "Друге" in llm.prompts[1].turns[-1].text


async def test_waiting_messages_already_in_memory_are_dropped_on_start(tmp_path: Path) -> None:
    # A crash after memory took the exchange, before the waiting messages were updated.
    answered = make_message(Role.USER, "Перше", minutes=0)
    reply = make_message(Role.MODEL, "Відповідь", minutes=1)
    unanswered = make_message(Role.USER, "Друге", minutes=2)
    setup = _Setup(tmp_path, [], remembered=[answered, reply])
    setup.pending.replace([answered, unanswered])

    setup.reopen()

    assert setup.pending_texts() == ["Друге"]
    reopened = MessageFile(tmp_path / "pending.json")
    assert [message.text for message in reopened.messages] == ["Друге"]


async def test_compact_asks_memory_to_keep_the_given_tokens(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [])
    setup.memory.outcome = NotAbsorbed("the model is unavailable: 503")

    outcome = await setup.chat.compact(80)

    assert outcome == NotAbsorbed("the model is unavailable: 503")
    assert setup.memory.compactions == [80]
