"""Console chat: one user talks to the assistant in the terminal.

    python -m kamerdyner.apps.console <user directory>

The user directory holds everything about the user: `user.toml` with their settings, and in
`log/` the conversation and `console.log`, the log of this program. GEMINI_API_KEY and
GEMINI_MODEL come from the environment or `.env`. /exit, Ctrl+D or Ctrl+C ends the chat.

A line starting with "/" is a command, not a message to the assistant; see `_COMMANDS`.
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
from kamerdyner.compaction import (
    Compact,
    CompactionPlan,
    CompactionPolicy,
    NoAction,
    ShareCompactionPolicy,
    SizeThresholdPolicy,
    conversation_tokens,
)
from kamerdyner.llm import GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.llm.gemini import GeminiLLM, make_client
from kamerdyner.memory import EmptyMemory
from kamerdyner.messages import utc_now
from kamerdyner.store import ConversationLog, CurrentConversation
from kamerdyner.tokens import ApproximateTokenCounter, TokenCounter
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
    log_directory = directory / "log"
    log_directory.mkdir(exist_ok=True)
    log_path = log_directory / "console.log"
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    client = make_client(api_key, timeout_seconds=120, attempts=3)
    llm = GeminiLLM(client, model, types.ThinkingLevel.LOW)
    log = ConversationLog(log_directory)
    current = CurrentConversation(log_directory)
    memory = EmptyMemory()
    counter = ApproximateTokenCounter()
    after_reply = SizeThresholdPolicy(counter, limit_tokens=16_000, min_compress_tokens=8_000)
    on_request = ShareCompactionPolicy(counter, share=0.5)
    chat = Chat(user, log, current, memory, llm, utc_now)

    print(f"Kamerdyner — {user.name}. /exit or Ctrl+D to quit, /help for commands.")
    with asyncio.Runner() as runner:
        while True:
            try:
                text = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text.strip():
                continue
            if text.startswith("/"):
                keep_going = _run_command(text, runner, chat, on_request, counter)
                if not keep_going:
                    break
                continue
            sending = chat.send(text)
            result = runner.run(sending)
            _show(result)
            compacting = chat.compact(after_reply)
            plan = runner.run(compacting)
            match plan:
                case NoAction():
                    pass
                case Compact():
                    _show_compaction(plan, counter)
                case _:
                    assert_never(plan)


_COMMANDS = {
    "/compact": "move at least half of the current conversation into memory, oldest first",
    "/help": "list the commands",
    "/exit": "end the chat",
}


def _run_command(
    command: str,
    runner: asyncio.Runner,
    chat: Chat,
    policy: CompactionPolicy,
    counter: TokenCounter,
) -> bool:
    """Runs a console command; returns False when the chat should end."""
    match command.strip():
        case "/compact":
            compacting = chat.compact(policy)
            plan = runner.run(compacting)
            _show_compaction(plan, counter)
        case "/help":
            print()
            for name, description in _COMMANDS.items():
                print(f"{name:10} {description}")
        case "/exit":
            return False
        case unknown:
            print(f"\n[unknown command {unknown!r}, /help lists the commands]")
    return True


def _show_compaction(plan: CompactionPlan, counter: TokenCounter) -> None:
    match plan:
        case NoAction():
            print("\n[nothing to compact]")
        case Compact(to_compress=to_compress, to_keep=to_keep):
            moved = conversation_tokens(to_compress, counter)
            kept = conversation_tokens(to_keep, counter)
            print(
                f"\n[moved {len(to_compress)} messages (~{moved} tokens) into memory, "
                f"{len(to_keep)} messages (~{kept} tokens) remain]"
            )
        case _:
            assert_never(plan)


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
