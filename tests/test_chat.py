import json
from collections.abc import Callable, Sequence
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from fakes import START, FakeClock, FakeMemory, ScriptedLLM, make_message, make_user

from kamerdyner.chat import Chat, ConversationLog
from kamerdyner.common.llm import (
    GenerateResult,
    Incomplete,
    Prompt,
    Reply,
    Role,
    ToolCall,
    ToolSpec,
    Turn,
    Unavailable,
    WebSearch,
)
from kamerdyner.common.messages import Message
from kamerdyner.common.store import MessageFile
from kamerdyner.memory.memory import NotAbsorbed
from kamerdyner.users import load_user


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
        return Chat(user, self.directory, self.log, self.pending, self.memory, self.llm, self.clock)

    async def send(self, text: str) -> GenerateResult:
        self.chat.receive(text)
        return await self.chat.reply()

    def logged_texts(self) -> list[str]:
        lines = (self.directory / "log.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line)["text"] for line in lines]

    def memory_texts(self) -> list[str]:
        return [message.text for message in self.memory.conversation]

    def pending_texts(self) -> list[str]:
        return [message.text for message in self.pending.messages]


async def test_reply_goes_to_memory_with_the_message(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Reply("Привіт, Марто!")])

    result = await setup.send("Привіт")

    assert result == Reply("Привіт, Марто!")
    stored = [(message.role, message.text) for message in setup.memory.conversation]
    assert stored == [(Role.USER, "Привіт"), (Role.MODEL, "Привіт, Марто!")]
    assert setup.pending_texts() == []
    assert not setup.chat.awaiting_reply
    assert setup.logged_texts() == ["Привіт", "Привіт, Марто!"]
    prompt = setup.llm.prompts[0]
    assert prompt.turns[-1].text == "[2026-10-04 Sun 14:32 UTC-04:00] Привіт"
    assert "Любить каву без цукру." in prompt.system_instruction


async def test_messages_carry_the_users_local_time(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Reply("Ось")])

    await setup.send("Котра година?")

    [question, answer] = setup.memory.conversation
    assert question.at == START
    assert question.at.utcoffset() == timedelta(hours=-4)
    assert answer.at.utcoffset() == timedelta(hours=-4)
    record = json.loads((tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["at"] == "2026-10-04T14:32:00-04:00"


async def test_message_is_on_disk_before_the_model_is_called(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Unavailable("503")])

    await setup.send("Перше")

    reopened = MessageFile(tmp_path / "pending.json")
    assert [message.text for message in reopened.messages] == ["Перше"]
    assert setup.memory_texts() == []
    assert setup.logged_texts() == ["Перше"]


async def test_failed_reply_keeps_the_message_and_the_next_one_joins_it(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Unavailable("503"), Incomplete("MAX_TOKENS", "Пів"), Reply("Ось")])

    unavailable = await setup.send("Перше")
    setup.clock.advance(1)
    incomplete = await setup.send("Друге")
    setup.clock.advance(1)
    await setup.send("Третє")

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

    await setup.send("А в тебе?")

    assert setup.llm.prompts[0].turns == (
        Turn(Role.USER, "[2026-10-04 Sun 13:32 UTC-04:00] Як справи?"),
        Turn(Role.MODEL, "Добре."),
        Turn(Role.USER, "[2026-10-04 Sun 14:32 UTC-04:00] А в тебе?"),
    )


class _MidwayLLM:
    """Replies with fixed texts; before the first reply, runs `meanwhile` — a message arriving
    while the model is thinking."""

    def __init__(self, texts: list[str], meanwhile: Callable[[], None]) -> None:
        self._texts = texts
        self._meanwhile = meanwhile
        self.prompts: list[Prompt] = []

    async def generate(self, prompt: Prompt, tools: Sequence[ToolSpec] = ()) -> GenerateResult:
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
    chat = Chat(make_user(), tmp_path, log, pending, memory, llm, FakeClock())
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


def _logged_texts(directory: Path) -> list[str]:
    lines = (directory / "log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["text"] for line in lines]


def test_log_keeps_every_message_in_order(tmp_path: Path) -> None:
    log = ConversationLog(tmp_path)
    for index in range(3):
        message = make_message(Role.USER, f"Повідомлення {index}", minutes=index)
        log.append(message)

    reopened = ConversationLog(tmp_path)
    last = make_message(Role.MODEL, "Відповідь", minutes=3)
    reopened.append(last)

    assert _logged_texts(tmp_path) == [
        "Повідомлення 0",
        "Повідомлення 1",
        "Повідомлення 2",
        "Відповідь",
    ]


def test_log_drops_a_torn_last_line(tmp_path: Path) -> None:
    log = ConversationLog(tmp_path)
    complete = make_message(Role.USER, "complete")
    log.append(complete)
    with (tmp_path / "log.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"role": "model", "te')

    reopened = ConversationLog(tmp_path)
    after_crash = make_message(Role.USER, "after the crash", minutes=1)
    reopened.append(after_crash)

    assert _logged_texts(tmp_path) == ["complete", "after the crash"]


async def test_a_grounded_reply_is_remembered_whole_but_archived_without_its_text(
    tmp_path: Path,
) -> None:
    setup = _Setup(tmp_path, [Reply("Python 3.15.2", grounded=True)])

    await setup.send("Яка остання версія Python?")

    assert setup.memory_texts() == ["Яка остання версія Python?", "Python 3.15.2"]
    assert setup.memory.conversation[-1].ephemeral
    logged = setup.logged_texts()
    assert logged[0] == "Яка остання версія Python?"
    assert logged[1].startswith("[Kamerdyner answered from a web search")


def _set_timezone_call(zone: object) -> ToolCall:
    return ToolCall("set_timezone", {"zone": zone})


def _asks_for(*calls: ToolCall, grounded: bool = False) -> Reply:
    return Reply("", grounded=grounded, tool_calls=calls)


def _with_user_file(setup: _Setup) -> None:
    settings = 'name = "Марта"\ntimezone = "America/Toronto"\n'
    (setup.directory / "user.toml").write_text(settings, encoding="utf-8")


async def test_a_reply_without_tool_calls_takes_one_request(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [Reply("Привіт")])

    await setup.send("Привіт")

    assert len(setup.llm.prompts) == 1
    assert [spec.name for spec in setup.llm.tools[0]] == ["set_timezone"]
    assert setup.memory_texts() == ["Привіт", "Привіт"]


async def test_a_tool_call_runs_and_the_model_answers_with_no_tools_allowed(
    tmp_path: Path,
) -> None:
    call = _set_timezone_call("Europe/Kyiv")
    setup = _Setup(tmp_path, [_asks_for(call), Reply("Записала, тепер Київ")])
    _with_user_file(setup)

    result = await setup.send("Я тепер у Києві")

    assert result == Reply("Записала, тепер Київ")
    assert setup.llm.tools[1] == ()
    stamp = "2026-10-04 Sun 21:32 UTC+03:00"
    text = f"Time zone set to Europe/Kyiv. Local time is now {stamp}."
    asked = Turn(Role.MODEL, 'Model requested tool set_timezone(zone="Europe/Kyiv").')
    told = Turn(Role.USER, f"The system ran it, result: {text}")
    assert setup.llm.prompts[1].turns == (*setup.llm.prompts[0].turns, asked, told)


async def test_the_new_zone_is_saved_and_applies_from_the_reply_on(tmp_path: Path) -> None:
    call = _set_timezone_call("Europe/Kyiv")
    setup = _Setup(tmp_path, [_asks_for(call), Reply("Ось"), Reply("Ще")])
    _with_user_file(setup)

    await setup.send("Я тепер у Києві")
    await setup.send("А зараз?")

    assert load_user(tmp_path).timezone == ZoneInfo("Europe/Kyiv")
    [asked, answer, next_message, _] = setup.memory.conversation
    assert asked.at.isoformat() == "2026-10-04T14:32:00-04:00"
    assert answer.at.isoformat() == "2026-10-04T21:32:00+03:00"
    assert next_message.at.isoformat() == "2026-10-04T21:32:00+03:00"


async def test_tools_are_recorded_before_the_answer_but_the_client_gets_the_answer_alone(
    tmp_path: Path,
) -> None:
    good = _set_timezone_call("Europe/Kyiv")
    bad = _set_timezone_call("Mars/Base")
    setup = _Setup(tmp_path, [_asks_for(good, bad), Reply("Готово")])
    _with_user_file(setup)

    result = await setup.send("Постав")

    assert result == Reply("Готово")
    assert setup.memory_texts()[-1] == (
        'Model requested tool set_timezone(zone="Europe/Kyiv").\n'
        "The system ran it, result: Time zone set to Europe/Kyiv. "
        "Local time is now 2026-10-04 Sun 21:32 UTC+03:00.\n"
        'Model requested tool set_timezone(zone="Mars/Base").\n'
        'The system ran it, result: Error: unknown time zone "Mars/Base". '
        "Use an IANA name such as Europe/Kyiv.\n"
        "In the next request the model replied:\n"
        "Готово"
    )
    assert setup.logged_texts()[-1] == setup.memory_texts()[-1]


async def test_an_unknown_zone_is_told_to_the_model_and_changes_nothing(tmp_path: Path) -> None:
    setup = _Setup(tmp_path, [_asks_for(_set_timezone_call("Mars/Base")), Reply("Не знаю такого")])
    _with_user_file(setup)

    await setup.send("Я на Марсі")

    assert 'Error: unknown time zone "Mars/Base"' in setup.llm.prompts[1].turns[-1].text
    assert load_user(tmp_path).timezone == ZoneInfo("America/Toronto")
    assert setup.memory.conversation[-1].at.isoformat() == "2026-10-04T14:32:00-04:00"


async def test_calls_out_of_contract_are_told_to_the_model(tmp_path: Path) -> None:
    unknown = ToolCall("delete_everything", {})
    not_a_string = _set_timezone_call(5)
    missing = ToolCall("set_timezone", {})
    setup = _Setup(tmp_path, [_asks_for(unknown, not_a_string, missing), Reply("Ось")])

    await setup.send("Постав")

    assert setup.llm.prompts[1].turns[-1].text == (
        "The system ran it, result: Error: no tool delete_everything taking arguments {}.\n"
        "The system ran it, result: Error: no tool set_timezone taking arguments {'zone': 5}.\n"
        "The system ran it, result: Error: no tool set_timezone taking arguments {}."
    )


async def test_when_the_answer_after_tools_fails_the_message_keeps_waiting(tmp_path: Path) -> None:
    call = _set_timezone_call("Europe/Kyiv")
    setup = _Setup(
        tmp_path,
        [_asks_for(call), Unavailable("503"), _asks_for(call), Reply("Записала")],
    )
    _with_user_file(setup)

    failed = await setup.send("Я тепер у Києві")

    assert failed == Unavailable("503")
    assert setup.pending_texts() == ["Я тепер у Києві"]
    assert setup.memory_texts() == []
    assert load_user(tmp_path).timezone == ZoneInfo("Europe/Kyiv")
    result = await setup.chat.reply()
    assert result == Reply("Записала")
    assert setup.pending_texts() == []


async def test_what_the_first_turn_found_on_the_web_counts_for_the_answer(tmp_path: Path) -> None:
    search = WebSearch("погода Київ", "https://example.com/search")
    first = Reply(
        "",
        grounded=True,
        searches=(search,),
        tool_calls=(_set_timezone_call("Europe/Kyiv"),),
    )
    setup = _Setup(tmp_path, [first, Reply("У Києві +5")])
    _with_user_file(setup)

    result = await setup.send("Постав пояс і знайди погоду")

    assert result == Reply("У Києві +5", grounded=True, searches=(search,))
    assert setup.memory.conversation[-1].ephemeral
