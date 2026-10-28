"""Telegram bot: each user talks to the assistant in their own private chat with the bot.

    python -m kamerdyner.apps.telegram <data directory>

Every user directory in the data directory whose `user.toml` has `telegram_id = <id>` is a user
of the bot; anyone else who writes is told their Telegram id, to be added that way, and nothing
of theirs is kept. A user's conversation and memory are the same files the console uses.
TELEGRAM_BOT_TOKEN, GEMINI_API_KEY and GEMINI_MODEL come from the environment or `.env`. The log
of this program is `telegram.log` in the data directory. Ctrl+C stops the bot.

Replies go out as plain text. Messages that arrive while the assistant is replying get one reply
together, next. On start, messages left without a reply by an earlier run are answered.
"""

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import tomllib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.types import ErrorEvent
from aiogram.types import Message as TelegramMessage
from dotenv import load_dotenv
from google.genai import types

from kamerdyner.chat import Chat
from kamerdyner.llm import LLM, GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.llm.gemini import GeminiLLM, make_client
from kamerdyner.memory import CompactionOutcome, NotAbsorbed
from kamerdyner.memory.compaction import Compact, NoAction
from kamerdyner.memory.notes import NotesMemory
from kamerdyner.messages import utc_now
from kamerdyner.session import Session
from kamerdyner.store import ConversationLog, MessageFile
from kamerdyner.users import User, load_user

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kamerdyner as a Telegram bot.")
    parser.add_argument("data", type=Path, help="the directory of user directories")
    args = parser.parse_args()
    data: Path = args.data

    load_dotenv()
    token = _required_env(parser, "TELEGRAM_BOT_TOKEN")
    api_key = _required_env(parser, "GEMINI_API_KEY")
    model = _required_env(parser, "GEMINI_MODEL")
    try:
        directories = load_telegram_users(data)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(
        filename=data / "telegram.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = make_client(api_key, timeout_seconds=120, attempts=3)
    # A fixed seed makes replies repeatable: a change in them comes from our prompt or code.
    llm = GeminiLLM(client, model, types.ThinkingLevel.LOW, seed=0)
    memory_llm = GeminiLLM(client, model, types.ThinkingLevel.HIGH, seed=0)
    members: dict[int, _Member] = {}
    for telegram_id, directory in directories.items():
        member = _member(directory, llm, memory_llm)
        members[telegram_id] = member

    asyncio.run(_serve(token, members))


def load_telegram_users(data: Path) -> dict[int, Path]:
    """The user directories in `data` whose `user.toml` names a `telegram_id`, by that id.
    Raises ValueError naming the file on an id that is not an integer or is taken twice."""
    users: dict[int, Path] = {}
    for path in sorted(data.glob("*/user.toml")):
        with path.open("rb") as file:
            settings = tomllib.load(file)
        match settings:
            case {"telegram_id": int() as telegram_id}:
                pass
            case {"telegram_id": other}:
                raise ValueError(f"User file {path} has a non-integer telegram_id {other!r}")
            case _:
                continue
        if telegram_id in users:
            raise ValueError(
                f"User file {path} has telegram_id {telegram_id}, already taken by "
                f"{users[telegram_id]}"
            )
        users[telegram_id] = path.parent
    return users


def split_message(text: str, limit: int) -> list[str]:
    """`text` in pieces of at most `limit` characters, each cut at the last line break before
    the limit, else at the last space, else right at the limit."""
    if limit <= 0:
        raise ValueError(f"Invalid input. limit must be positive, got {limit}")
    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(rest[:cut])
        rest = rest[cut:].lstrip()
    pieces.append(rest)
    return pieces


# Temporarily 4K instead of 16K, to see compaction happen in testing.
_LIMIT_TOKENS = 4_000
_KEEP_TOKENS = 2_000

# Telegram allows 4096 characters; it counts some in UTF-16 units, hence the margin.
_MESSAGE_LIMIT = 4000
# Telegram shows a chat action for about five seconds.
_TYPING_RENEW_SECONDS = 4

_HELP = """\
Пиши мені про що завгодно — я відповім і запам'ятаю важливе.

/compact — перенести старшу половину розмови в пам'ять
/help — ця довідка"""

_TEXT_ONLY = """\
Поки що я розумію лише текст. Якщо зручніше голосом — на клавіатурі телефона, найімовірніше, \
є кнопка мікрофона: надиктуй повідомлення, і мені прийде вже текст."""


@dataclass(frozen=True)
class _Member:
    user: User
    session: Session


def _member(directory: Path, llm: LLM, memory_llm: LLM) -> _Member:
    user = load_user(directory)
    log_directory = directory / "log"
    log = ConversationLog(log_directory)
    pending = MessageFile(directory / "pending.json")
    memory_directory = directory / "memory"
    memory = NotesMemory(memory_directory, user.name, memory_llm)
    chat = Chat(user, log, pending, memory, llm, utc_now)
    session = Session(chat, _LIMIT_TOKENS, _KEEP_TOKENS)
    return _Member(user=user, session=session)


async def _serve(token: str, members: dict[int, _Member]) -> None:
    bot = Bot(token=token)
    try:
        me = await bot.get_me()
    except TelegramAPIError as error:
        await bot.session.close()
        sys.exit(f"Cannot start the bot: {error}")
    names = [member.user.name for member in members.values()]
    users = ", ".join(names) if names else "nobody yet (add telegram_id to a user.toml)"
    print(f"@{me.username} is running for {users}. Ctrl+C stops it.")
    logger.info("Bot @%s started for %d users", me.username, len(members))
    dispatcher = _dispatcher(bot, members)
    async with asyncio.TaskGroup() as group:
        for telegram_id, member in members.items():
            group.create_task(_answer_waiting(bot, telegram_id, member))
        # aiogram types the default of `allowed_updates` with an unannotated sentinel.
        await dispatcher.start_polling(bot)  # pyright: ignore[reportUnknownMemberType]


def _dispatcher(bot: Bot, members: dict[int, _Member]) -> Dispatcher:
    router = Router()
    router.message.filter(F.chat.type == ChatType.PRIVATE)

    async def member_of(message: TelegramMessage) -> _Member | None:
        """The member who wrote `message`; anyone else is told their Telegram id."""
        sender = message.from_user
        if sender is None:
            logger.warning("Ignoring message %s without a sender", message.message_id)
            return None
        member = members.get(sender.id)
        if member is None:
            logger.info("Message from unknown Telegram user %s (@%s)", sender.id, sender.username)
            reply = f"Це приватний бот від Sergiy Pometun. Твій Telegram id: {sender.id}"
            await _send(bot, message.chat.id, reply)
        return member

    @router.message(CommandStart())
    async def on_start(message: TelegramMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        member = await member_of(message)
        if member is None:
            return
        greeting = f"Привіт, {member.user.name}! Я Камердинер.\n\n{_HELP}"
        await _send(bot, message.chat.id, greeting)

    @router.message(Command("help"))
    async def on_help(message: TelegramMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        member = await member_of(message)
        if member is None:
            return
        await _send(bot, message.chat.id, _HELP)

    @router.message(Command("compact"))
    async def on_compact(message: TelegramMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        member = await member_of(message)
        if member is None:
            return
        # Up to the older half: at least half of the newest stays.
        half = member.session.conversation_tokens // 2
        outcome = await member.session.compact(half)
        report = _compaction_report(outcome)
        await _send(bot, message.chat.id, report)
        # Messages may have arrived while compacting.
        await _catch_up(bot, message.chat.id, member)

    @router.message(F.text)
    async def on_text(message: TelegramMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        member = await member_of(message)
        if member is None:
            return
        text = message.text
        assert text is not None, f"Internal error. Message {message.message_id} passed F.text"
        if text.startswith("/"):
            await _send(bot, message.chat.id, "Не знаю такої команди. /help — що я вмію.")
            return
        member.session.receive(text)
        # Shown at once, even when the reply has to wait for a running one.
        await _send_typing(bot, message.chat.id)
        await _catch_up(bot, message.chat.id, member)

    @router.message()
    async def on_other(message: TelegramMessage) -> None:  # pyright: ignore[reportUnusedFunction]
        member = await member_of(message)
        if member is None:
            return
        await _send(bot, message.chat.id, _TEXT_ONLY)

    dispatcher = Dispatcher()
    dispatcher.include_router(router)

    @dispatcher.errors()
    async def on_error(event: ErrorEvent) -> None:  # pyright: ignore[reportUnusedFunction]
        """The boundary of every update: a failure is logged and the user told, the bot runs on."""
        update_id = event.update.update_id
        logger.error("Failed to handle update %s", update_id, exc_info=event.exception)
        message = event.update.message
        if message is not None:
            await _send(bot, message.chat.id, "Щось пішло не так. Я записав це в лог.")

    return dispatcher


async def _answer_waiting(bot: Bot, telegram_id: int, member: _Member) -> None:
    """Answers messages an earlier run left without a reply. The boundary of this background
    task: a failure is logged and the bot runs on."""
    try:
        # In a private chat, the chat id is the user's id.
        await _catch_up(bot, telegram_id, member)
    except Exception:
        logger.exception("Failed to answer the waiting messages of user %s", member.user.id)


async def _catch_up(bot: Bot, chat_id: int, member: _Member) -> None:
    async def deliver(result: GenerateResult) -> None:
        text = _reply_text(result)
        await _send(bot, chat_id, text)

    await member.session.catch_up(deliver, lambda: _typing(bot, chat_id))


def _reply_text(result: GenerateResult) -> str:
    match result:
        case Reply(text=text):
            return text
        case Incomplete(partial_text=partial_text):
            notice = "[відповідь обірвалася — напиши ще щось, і я відповім знову]"
            return f"{partial_text}\n\n{notice}".strip()
        case Unavailable():
            return "Модель зараз недоступна. Напиши ще раз трохи згодом — я відповім на все разом."
        case _:
            assert_never(result)


def _compaction_report(outcome: CompactionOutcome) -> str:
    match outcome:
        case NoAction():
            return "Немає чого переносити в пам'ять."
        case Compact(to_compress=to_compress):
            return f"Переніс у пам'ять повідомлень: {len(to_compress)}."
        case NotAbsorbed(reason=reason):
            return f"Не вдалося перенести в пам'ять, розмова лишилась як була: {reason}"
        case _:
            assert_never(outcome)


async def _send(bot: Bot, chat_id: int, text: str) -> None:
    """Sends `text` as plain text, in pieces if long; a failure is logged, not raised."""
    pieces = split_message(text, _MESSAGE_LIMIT)
    for piece in pieces:
        try:
            await bot.send_message(chat_id=chat_id, text=piece)
        except TelegramAPIError as error:
            logger.warning("Could not send to chat %s: %r", chat_id, error)
            return


@contextlib.asynccontextmanager
async def _typing(bot: Bot, chat_id: int) -> AsyncGenerator[None]:
    """Shows that the bot is typing for as long as the scope lasts."""
    async with asyncio.TaskGroup() as group:
        keeping = group.create_task(_keep_typing(bot, chat_id))
        try:
            yield
        finally:
            keeping.cancel()


async def _keep_typing(bot: Bot, chat_id: int) -> None:
    while True:
        await _send_typing(bot, chat_id)
        await asyncio.sleep(_TYPING_RENEW_SECONDS)


async def _send_typing(bot: Bot, chat_id: int) -> None:
    try:
        await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    except TelegramAPIError as error:
        logger.warning("Could not show typing in chat %s: %r", chat_id, error)


def _required_env(parser: argparse.ArgumentParser, name: str) -> str:
    value = os.environ.get(name)
    if not value:
        parser.error(f"{name} is not set; put it in .env (see .env.example)")
    return value


if __name__ == "__main__":
    main()
