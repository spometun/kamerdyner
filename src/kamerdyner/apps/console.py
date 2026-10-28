"""Console chat: one user talks to the assistant in the terminal.

    python -m kamerdyner.apps.console <user directory>

The user directory holds everything about the user: `user.toml` with their settings,
`pending.json` with their messages still waiting for a reply, in `log/` the archive of the
conversation and `console.log`, the log of this program, and in `memory/` the memory: the recent
conversation, the notes and `memory.log`, the record of what memory did. GEMINI_API_KEY
and GEMINI_MODEL come from the environment or `.env`. /exit, Ctrl+D or Ctrl+C ends the chat.

A line starting with "/" is a command, not a message to the assistant; see `_COMMANDS`.
What is typed while the assistant is busy is not echoed; it appears in the next prompt. Each
reply opens with the size of the recent conversation in memory after it, in (approximate)
tokens: `[122]`.
"""

import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import assert_never

from dotenv import load_dotenv
from google.genai import types

from kamerdyner.apps.terminal import TypeAhead
from kamerdyner.chat import Chat, utc_now
from kamerdyner.common.llm import GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.common.store import ConversationLog, MessageFile
from kamerdyner.llm.gemini import GeminiLLM, make_client
from kamerdyner.memory import Compacted, CompactionOutcome, NotAbsorbed, NothingToCompact
from kamerdyner.memory.notes import NotesMemory
from kamerdyner.users import load_user


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Kamerdyner in the terminal.")
    parser.add_argument("directory", type=Path, help="the user's directory, with user.toml")
    args = parser.parse_args()
    directory: Path = args.directory

    try:
        user = load_user(directory)
    except FileNotFoundError as error:
        parser.error(f"no {error.filename}; cp -r user.example {directory}, then fill in the name")

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
    # A fixed seed makes replies repeatable: a change in them comes from our prompt or code.
    llm = GeminiLLM(client, model, types.ThinkingLevel.LOW, seed=0)
    memory_llm = GeminiLLM(client, model, types.ThinkingLevel.HIGH, seed=0)
    log = ConversationLog(log_directory)
    pending = MessageFile(directory / "pending.json")
    memory_directory = directory / "memory"
    memory = NotesMemory(memory_directory, user.name, memory_llm)
    chat = Chat(user, log, pending, memory, llm, utc_now)

    type_ahead = TypeAhead()
    print(f"Kamerdyner — {user.name}. /exit or Ctrl+D to quit, /help for commands.")
    with asyncio.Runner() as runner:
        while True:
            try:
                text = type_ahead.ask("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text.strip():
                continue
            if text.startswith("/"):
                keep_going = _run_command(text, runner, chat, type_ahead)
                if not keep_going:
                    break
                continue
            sending = chat.send(text)
            with type_ahead.collecting():
                result = runner.run(sending)
            _show(result, chat.conversation_tokens)
            if chat.conversation_tokens < _LIMIT_TOKENS:
                continue
            compacting = chat.compact(_KEEP_TOKENS)
            with type_ahead.collecting():
                outcome = runner.run(compacting)
            match outcome:
                case NothingToCompact():
                    pass
                case Compacted() | NotAbsorbed():
                    _show_compaction(outcome, chat)
                case _:
                    assert_never(outcome)


# Temporarily 4K instead of 16K, to see compaction happen in testing.
_LIMIT_TOKENS = 4_000
_KEEP_TOKENS = 2_000

_COMMANDS = {
    "/compact": "move up to the older half of the recent conversation into memory",
    "/help": "list the commands",
    "/exit": "end the chat",
}


def _run_command(
    command: str,
    runner: asyncio.Runner,
    chat: Chat,
    type_ahead: TypeAhead,
) -> bool:
    """Runs a console command; returns False when the chat should end."""
    match command.strip():
        case "/compact":
            half = chat.conversation_tokens // 2
            compacting = chat.compact(half)
            with type_ahead.collecting():
                outcome = runner.run(compacting)
            _show_compaction(outcome, chat)
        case "/help":
            print()
            for name, description in _COMMANDS.items():
                print(f"{name:10} {description}")
        case "/exit":
            return False
        case unknown:
            print(f"\n[unknown command {unknown!r}, /help lists the commands]")
    return True


def _show_compaction(outcome: CompactionOutcome, chat: Chat) -> None:
    match outcome:
        case NothingToCompact():
            print("\n[nothing to compact]")
        case Compacted(moved=moved, kept=kept):
            print(
                f"\n[moved {moved} messages into memory, {kept} messages "
                f"(~{chat.conversation_tokens} tokens) remain]"
            )
        case NotAbsorbed(reason=reason):
            print(f"\n[memory could not take the old messages, they stay: {reason}]")
        case _:
            assert_never(outcome)


def _required_env(parser: argparse.ArgumentParser, name: str) -> str:
    value = os.environ.get(name)
    if not value:
        parser.error(f"{name} is not set; put it in .env (see .env.example)")
    return value


def _show(result: GenerateResult, tokens: int) -> None:
    """Prints the reply, opening with the size of the recent conversation in memory."""
    match result:
        case Reply(text=text):
            print(f"\n[{tokens}] {text}")
        case Incomplete(reason=reason, partial_text=partial_text):
            print(f"\n[{tokens}] {partial_text}\n[the reply was cut short: {reason}]")
        case Unavailable(detail=detail):
            print(f"\n[{tokens}] [the model is unavailable, try again: {detail}]")
        case _:
            assert_never(result)


if __name__ == "__main__":
    main()
