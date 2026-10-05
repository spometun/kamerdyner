"""A user's conversation on disk.

A user's directory holds:

    log.jsonl    every message ever written, one JSON object per line, append-only:
                 {"role": "user" | "model", "text": "...", "at": "<ISO 8601, UTC>"}
    state.json   {"watermark": N}: the first N messages of the log are folded into memory;
                 the messages after them are the recent conversation
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from kamerdyner.messages import Message, Role

logger = logging.getLogger(__name__)


class ConversationLog:
    def __init__(self, directory: Path) -> None:
        self._log_path = directory / "log.jsonl"
        self._state_path = directory / "state.json"
        directory.mkdir(parents=True, exist_ok=True)
        _drop_torn_tail(self._log_path)

    def append(self, message: Message) -> None:
        """Appends to the log durably: once this returns, the message survives a crash."""
        record = {"role": message.role.value, "text": message.text, "at": message.at.isoformat()}
        line = json.dumps(record, ensure_ascii=False)
        with self._log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
            descriptor = file.fileno()
            os.fsync(descriptor)

    def recent(self) -> list[Message]:
        """The messages not yet folded into memory, oldest first."""
        messages = self._read_all()
        watermark = self._read_watermark()
        assert watermark <= len(messages), (
            f"Internal error. Watermark {watermark} is past the last message "
            f"{len(messages)} of {self._log_path}"
        )
        return messages[watermark:]

    def fold(self, count: int) -> None:
        """Marks the `count` oldest recent messages as folded into memory."""
        if count < 0:
            raise ValueError(f"Invalid input. count must not be negative, got {count}")
        messages = self._read_all()
        watermark = self._read_watermark() + count
        assert watermark <= len(messages), (
            f"Internal error. Folding {count} messages moves the watermark to {watermark}, "
            f"past the last message {len(messages)} of {self._log_path}"
        )
        state = json.dumps({"watermark": watermark})
        _write_atomically(self._state_path, state)

    def _read_all(self) -> list[Message]:
        try:
            text = self._log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        messages: list[Message] = []
        lines = text.splitlines()
        for number, line in enumerate(lines, start=1):
            message = _parse_message(line, self._log_path, number)
            messages.append(message)
        return messages

    def _read_watermark(self) -> int:
        try:
            text = self._state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return 0
        match json.loads(text):
            case {"watermark": int() as watermark} if watermark >= 0:
                return watermark
            case state:
                raise ValueError(f"Corrupted state file {self._state_path}: {state!r}")


def _parse_message(line: str, path: Path, number: int) -> Message:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"Corrupted log {path} line {number}: {error}: {line!r}") from error
    match record:
        case {"role": str() as role, "text": str() as text, "at": str() as at}:
            pass
        case _:
            raise ValueError(f"Corrupted log {path} line {number}: unexpected record {line!r}")
    try:
        parsed_role = Role(role)
        parsed_at = datetime.fromisoformat(at)
        return Message(role=parsed_role, text=text, at=parsed_at)
    except ValueError as error:
        raise ValueError(f"Corrupted log {path} line {number}: {error}: {line!r}") from error


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
