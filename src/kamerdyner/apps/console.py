"""Console chat: one user talks to the assistant in the terminal.

    python -m kamerdyner.apps.console <user_id> [--data DIR]

Reads GEMINI_API_KEY and GEMINI_MODEL from the environment or `.env`, users from
`<data>/users.toml`, and keeps each user's conversation in `<data>/<user_id>/`. The log of the
program goes to `<data>/console.log`. Ctrl+D or Ctrl+C ends the chat.
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
from kamerdyner.users import load_users


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Kamerdyner in the terminal.")
    parser.add_argument("user_id", help="a user from <data>/users.toml")
    parser.add_argument("--data", type=Path, default=Path("data"), help="data directory")
    args = parser.parse_args()
    data: Path = args.data
    user_id: str = args.user_id

    users_path = data / "users.toml"
    try:
        users = load_users(users_path)
    except FileNotFoundError:
        parser.error(f"no users file {users_path}, copy users.example.toml there")
    if user_id not in users:
        parser.error(f"unknown user {user_id!r}, {users_path} has {sorted(users)}")
    user = users[user_id]

    load_dotenv()
    api_key = _required_env(parser, "GEMINI_API_KEY")
    model = _required_env(parser, "GEMINI_MODEL")
    log_path = data / "console.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = make_client(api_key, timeout_seconds=120, attempts=3)
    llm = GeminiLLM(client, model, types.ThinkingLevel.LOW)
    user_directory = data / user.id
    log = ConversationLog(user_directory)
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
