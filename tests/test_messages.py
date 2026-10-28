from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from kamerdyner.messages import Message, Role, format_time


def test_format_time_shows_the_time_zone_the_moment_carries() -> None:
    moment = datetime(2026, 10, 30, 1, 30, tzinfo=UTC)
    toronto = moment.astimezone(ZoneInfo("America/Toronto"))
    kyiv = moment.astimezone(ZoneInfo("Europe/Kyiv"))

    assert format_time(toronto) == "2026-10-29 Thu 21:30"
    assert format_time(kyiv) == "2026-10-30 Fri 03:30"


def test_message_rejects_a_time_without_a_zone() -> None:
    naive = datetime(2026, 10, 4, 14, 32)

    with pytest.raises(ValueError, match="Invalid input"):
        Message(role=Role.USER, text="x", at=naive)
