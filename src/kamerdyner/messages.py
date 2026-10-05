"""Messages of a conversation between a user and the assistant."""

import enum
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

type Clock = Callable[[], datetime]
"""Returns the current time as an aware UTC datetime."""


class Role(enum.Enum):
    USER = "user"
    MODEL = "model"


@dataclass(frozen=True)
class Message:
    """One message of a conversation; `at` is when it was written, an aware UTC datetime."""

    role: Role
    text: str
    at: datetime

    def __post_init__(self) -> None:
        if self.at.utcoffset() != timedelta(0):
            raise ValueError(f"Invalid input. Message time must be aware UTC, got {self.at!r}")


def utc_now() -> datetime:
    return datetime.now(UTC)
