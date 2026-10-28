"""Messages of a conversation between a user and the assistant."""

import enum
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

type Clock = Callable[[], datetime]
"""Returns the current time as an aware datetime."""


class Role(enum.Enum):
    USER = "user"
    MODEL = "model"


@dataclass(frozen=True)
class Message:
    """One message of a conversation; `at` is when it was written, an aware datetime in the
    user's local time — the time on their clock when they wrote or read it."""

    role: Role
    text: str
    at: datetime

    def __post_init__(self) -> None:
        if self.at.utcoffset() is None:
            raise ValueError(f"Invalid input. Message time must be aware, got {self.at!r}")


def format_time(at: datetime) -> str:
    """`2026-10-04 Sun 14:32`, in the time zone `at` carries."""
    weekday = _WEEKDAYS[at.weekday()]
    return f"{at:%Y-%m-%d} {weekday} {at:%H:%M}"


def utc_now() -> datetime:
    return datetime.now(UTC)


_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
