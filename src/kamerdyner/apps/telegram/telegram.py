"""Telegram bot: each user talks to the assistant in their own private chat with the bot.

    python -m kamerdyner.apps.telegram <data directory>

Every user directory in the data directory whose `user.toml` has `telegram_id = <id>` is a user
of the bot; anyone else who writes is told their Telegram id, to be added that way, and nothing
of theirs is kept. A user's conversation and memory are the same files the console uses.
TELEGRAM_BOT_TOKEN, GEMINI_API_KEY and GEMINI_MODEL come from the environment or `.env`. The log
of this program is `telegram.log` in the data directory. Ctrl+C stops the bot.

Replies go out formatted: the model's Markdown as Telegram HTML. Messages that arrive while the
assistant is replying get one reply together, next. On start, messages left without a reply by an
earlier run are answered.
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types

from kamerdyner.apps.telegram.impl.bot import load_telegram_users, serve
from kamerdyner.llm.gemini import GeminiLLM, make_client


def main() -> None:
    """Runs the bot as a program: arguments, `.env`, the log, then the bot until Ctrl+C."""
    parser = argparse.ArgumentParser(
        prog="python -m kamerdyner.apps.telegram", description="Run Kamerdyner as a Telegram bot."
    )
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
    llm = GeminiLLM(client, model, types.ThinkingLevel.LOW, seed=0, web=True)
    memory_llm = GeminiLLM(client, model, types.ThinkingLevel.HIGH, seed=0, web=False)
    serving = serve(token, directories, llm, memory_llm)
    asyncio.run(serving)


def _required_env(parser: argparse.ArgumentParser, name: str) -> str:
    value = os.environ.get(name)
    if not value:
        parser.error(f"{name} is not set; put it in .env (see .env.example)")
    return value
