"""Console chat: one user talks to the assistant in the terminal.

    python -m kamerdyner.apps.console <user directory>

The user directory holds everything about the user: `user.toml` with their settings, the
conversation, and `console.log`, the log of this program. GEMINI_API_KEY and GEMINI_MODEL come
from the environment or `.env`. Ctrl+D or Ctrl+C ends the chat.
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import assert_never

from dotenv import load_dotenv
from google.genai import types

from kamerdyner.chat import Chat
from kamerdyner.compaction import SizeThresholdPolicy
from kamerdyner.gemini import GeminiLLM, make_client
from kamerdyner.llm import GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.memory import EmptyMemory
from kamerdyner.messages import utc_now
from kamerdyner.store import ConversationLog
from kamerdyner.tokens import ApproximateTokenCounter
from kamerdyner.users import load_user


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Kamerdyner in the terminal.")
    parser.add_argument("directory", type=Path, help="the user's directory, with user.toml")
    args = parser.parse_args()
    directory: Path = args.directory

    try:
        user = load_user(directory)
    except FileNotFoundError as error:
        parser.error(f"no {error.filename}, copy user.example.toml there")

    load_dotenv()
    api_key = _required_env(parser, "GEMINI_API_KEY")
    model = _required_env(parser, "GEMINI_MODEL")
    log_path = directory / "console.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = make_client(api_key, timeout_seconds=120, attempts=3)
    llm = GeminiLLM(client, model, types.ThinkingLevel.LOW)
    log = ConversationLog(directory)
    memory = EmptyMemory()
    counter = ApproximateTokenCounter()
    policy = SizeThresholdPolicy(counter, limit_tokens=16_000, min_compress_tokens=8_000)
    chat = Chat(user, log, memory, policy, llm, utc_now)

    print(f"Kamerdyner — {user.name}. Ctrl+D to quit.")
    with asyncio.Runner() as runner:
        while True:
            try:
                text = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text.strip():
                continue
            sending = chat.send(text)
            result = runner.run(sending)
            _show(result)
            compacting = chat.compact()
            runner.run(compacting)


def _required_env(parser: argparse.ArgumentParser, name: str) -> str:
    value = os.environ.get(name)
    if not value:
        parser.error(f"{name} is not set; put it in .env (see .env.example)")
    return value


def _show(result: GenerateResult) -> None:
    match result:
        case Reply(text=text):
            print(f"\n{text}")
        case Incomplete(reason=reason, partial_text=partial_text):
            print(f"\n{partial_text}\n[the reply was cut short: {reason}]")
        case Unavailable(detail=detail):
            print(f"\n[the model is unavailable, try again: {detail}]")
        case _:
            assert_never(result)


if __name__ == "__main__":
    main()
