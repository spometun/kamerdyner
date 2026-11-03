"""The bot itself: one member per user, an update dispatcher, replies sent to Telegram."""

import asyncio
import contextlib
import logging
import sys
import tomllib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatAction, ChatType, ParseMode
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import ErrorEvent
from aiogram.types import Message as TelegramMessage

from kamerdyner.apps.telegram.impl.formatting import split_message, to_telegram_html
from kamerdyner.chat import Chat, ConversationLog, utc_now
from kamerdyner.common.llm import LLM, GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.common.store import MessageFile
from kamerdyner.memory.memory import Compacted, CompactionOutcome, NotAbsorbed, NothingToCompact
from kamerdyner.memory.notes import NotesMemory
from kamerdyner.session import Session
from kamerdyner.users import User, load_user

logger = logging.getLogger(__name__)


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


async def serve(token: str, directories: dict[int, Path], llm: LLM, memory_llm: LLM) -> None:
    """Runs the bot for the users in `directories`, by Telegram id, until cancelled. `llm`
    answers, `memory_llm` compacts memory."""
    members: dict[int, _Member] = {}
    for telegram_id, directory in directories.items():
        member = _member(directory, llm, memory_llm)
        members[telegram_id] = member
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
        await _send_markdown(bot, chat_id, text)

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
        case NothingToCompact():
            return "Немає чого переносити в пам'ять."
        case Compacted(moved=moved):
            return f"Переніс у пам'ять повідомлень: {moved}."
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


async def _send_markdown(bot: Bot, chat_id: int, text: str) -> None:
    """Sends the model's Markdown `text` formatted, in pieces if long; a piece Telegram cannot
    parse goes out as plain text instead. A failure to send is logged, not raised."""
    pieces = split_message(text, _MESSAGE_LIMIT)
    for piece in pieces:
        formatted = to_telegram_html(piece)
        try:
            await bot.send_message(chat_id=chat_id, text=formatted, parse_mode=ParseMode.HTML)
        except TelegramBadRequest as error:
            if "can't parse entities" not in error.message:
                logger.warning("Could not send to chat %s: %r", chat_id, error)
                return
            # Our HTML is at fault, so the error is loud; the user still gets the text.
            logger.error("Telegram rejected the HTML %r: %s", formatted, error.message)
            await _send(bot, chat_id, piece)
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
