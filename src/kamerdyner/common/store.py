"""Messages on disk: a list of messages that changes as a whole, kept by `MessageFile` as
{"messages": [...]} and rewritten whole on every change.

A message is stored as {"role": "user" | "model", "text": "...", "at": "<ISO 8601>"}, the time
with the user's local offset: "2026-10-04T14:32:00-04:00".
"""

import json
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from kamerdyner.common.llm import Role
from kamerdyner.common.messages import Message


class MessageFile:
    """A list of messages, oldest first, kept in a file; a missing file is an empty list. Read
    from disk once, then kept in memory and written through on every change, so a crash loses
    nothing that was acknowledged."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._messages = _load_messages(path)

    @property
    def messages(self) -> tuple[Message, ...]:
        return self._messages

    def append(self, message: Message) -> None:
        messages = (*self._messages, message)
        self.replace(messages)

    def replace(self, messages: Sequence[Message]) -> None:
        """Makes `messages` the whole list."""
        records = [_record(message) for message in messages]
        text = json.dumps({"messages": records}, ensure_ascii=False, indent=1)
        write_atomically(self._path, text)
        self._messages = tuple(messages)


def write_atomically(path: Path, text: str) -> None:
    """Replaces the file with `text` so that a crash leaves either the old file or the new one."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        descriptor = file.fileno()
        os.fsync(descriptor)
    os.replace(temporary, path)


def _record(message: Message) -> dict[str, str]:
    return {"role": message.role.value, "text": message.text, "at": message.at.isoformat()}


def _load_messages(path: Path) -> tuple[Message, ...]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    try:
        content = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Corrupted message file {path}: {error}") from error
    match content:
        case {"messages": [*records]}:
            pass
        case _:
            raise ValueError(f"Corrupted message file {path}: no 'messages' list")
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
            raise ValueError(f"Corrupted message file {path}: message {index} is {record!r}")
    try:
        parsed_role = Role(role)
        parsed_at = datetime.fromisoformat(at)
        return Message(role=parsed_role, text=text, at=parsed_at)
    except ValueError as error:
        raise ValueError(f"Corrupted message file {path}: message {index}: {error}") from error
