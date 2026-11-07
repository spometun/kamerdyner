"""Messages of a conversation between a user and the assistant."""

import dataclasses
from dataclasses import dataclass
from datetime import datetime

from kamerdyner.common.llm import Role


@dataclass(frozen=True)
class Message:
    """One message of a conversation; `at` is when it was written, an aware datetime in the
    user's local time — the time on their clock when they wrote or read it. `ephemeral`: the text
    may be shown and kept in the recent conversation, but neither archived nor compressed into
    long-term memory (`without_ephemeral_text`) — a reply resting on a web search, whose results
    may not be kept or learned from. `provider_data` is the `Reply.provider_data` of a reply, kept
    for the model's next requests only: it is not archived."""

    role: Role
    text: str
    at: datetime
    ephemeral: bool = False
    provider_data: str = ""

    def __post_init__(self) -> None:
        if self.at.utcoffset() is None:
            raise ValueError(f"Invalid input. Message time must be aware, got {self.at!r}")


def without_ephemeral_text(message: Message) -> Message:
    """`message` as it may be archived or compressed into memory: an ephemeral message's text is
    replaced by a note that there was one; any other message is returned as it is."""
    if not message.ephemeral:
        return message
    return dataclasses.replace(message, text=_EPHEMERAL_STAND_IN)


def format_time(at: datetime) -> str:
    """`2026-10-04 Sun 14:32`, in the time zone `at` carries."""
    weekday = _WEEKDAYS[at.weekday()]
    return f"{at:%Y-%m-%d} {weekday} {at:%H:%M}"


def format_time_with_offset(at: datetime) -> str:
    """`2026-11-03 Tue 13:46 UTC-05:00`: `format_time` and the offset from UTC that `at` carries,
    so the moment is unambiguous to someone who does not know the zone."""
    offset = at.strftime("%z")
    if not offset:
        raise ValueError(f"Invalid input. The time must be aware, got {at!r}")
    return f"{format_time(at)} UTC{offset[:3]}:{offset[3:5]}"


_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

_EPHEMERAL_STAND_IN = (
    "[Kamerdyner answered from a web search or a web page; the answer is not kept]"
)
