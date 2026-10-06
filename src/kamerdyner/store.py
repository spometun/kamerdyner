"""A user's conversation on disk.

A user's directory holds:

    log.jsonl     every message ever written, one per line, append-only: the archive. It is never
                  read on the way to the model.
    current.json  the current conversation, the messages the model sees: {"messages": [...]},
                  rewritten whole on every change.

A message is stored as {"role": "user" | "model", "text": "...", "at": "<ISO 8601, UTC>"}.
"""

import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from kamerdyner.messages import Message, Role

logger = logging.getLogger(__name__)


class ConversationLog:
    """The archive of every message, appended to durably."""

    def __init__(self, directory: Path) -> None:
        self._path = directory / "log.jsonl"
        directory.mkdir(parents=True, exist_ok=True)
        _drop_torn_tail(self._path)

    def append(self, message: Message) -> None:
        """Once this returns, the message survives a crash."""
        record = _record(message)
        line = json.dumps(record, ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            descriptor = file.fileno()
            os.fsync(descriptor)


class CurrentConversation:
    """The messages the model sees, oldest first. Read from disk once, then kept in memory and
    written through on every change, so a crash loses nothing that was acknowledged."""

    def __init__(self, directory: Path) -> None:
        self._path = directory / "current.json"
        directory.mkdir(parents=True, exist_ok=True)
        self._messages = _load_current(self._path)

    @property
    def messages(self) -> tuple[Message, ...]:
        return self._messages

    def append(self, message: Message) -> None:
        messages = (*self._messages, message)
        self.replace(messages)

    def replace(self, messages: Sequence[Message]) -> None:
        """Makes `messages` the whole current conversation."""
        records = [_record(message) for message in messages]
        text = json.dumps({"messages": records}, ensure_ascii=False, indent=1)
        _write_atomically(self._path, text)
        self._messages = tuple(messages)


def _record(message: Message) -> dict[str, str]:
    return {"role": message.role.value, "text": message.text, "at": message.at.isoformat()}


def _load_current(path: Path) -> tuple[Message, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    try:
        content = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Corrupted current conversation {path}: {error}") from error
    match content:
        case {"messages": [*records]}:
            pass
        case _:
            raise ValueError(f"Corrupted current conversation {path}: no 'messages' list")
    messages: list[Message] = []
    for index, record in enumerate(records):
        message = _message(record, path, index)
        messages.append(message)
    return tuple(messages)


def _message(record: Any, path: Path, index: int) -> Message:
    match record:
        case {"role": str() as role, "text": str() as text, "at": str() as at}:
            pass
        case _:
            raise ValueError(
                f"Corrupted current conversation {path}: message {index} is {record!r}"
            )
    try:
        parsed_role = Role(role)
        parsed_at = datetime.fromisoformat(at)
        return Message(role=parsed_role, text=text, at=parsed_at)
    except ValueError as error:
        raise ValueError(
            f"Corrupted current conversation {path}: message {index}: {error}"
        ) from error


def _drop_torn_tail(path: Path) -> None:
    """Cuts off a last line left unfinished by a crash mid-write. Such a message was never
    acknowledged as stored, and appending after it would glue two records into one line."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    end = data.rfind(b"\n") + 1
    if end == len(data):
        return
    logger.warning("Dropping a torn last line of %s: %r", path, data[end:])
    with path.open("r+b") as file:
        file.truncate(end)
        file.flush()
        descriptor = file.fileno()
        os.fsync(descriptor)


def _write_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        descriptor = file.fileno()
        os.fsync(descriptor)
    os.replace(temporary, path)
