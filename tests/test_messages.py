import dataclasses
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from fakes import make_message

from kamerdyner.common.llm import Role
from kamerdyner.common.messages import Message, format_time, without_ephemeral_text


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


def test_an_ephemeral_message_loses_only_its_text() -> None:
    plain = make_message(Role.MODEL, "391")
    ephemeral = dataclasses.replace(plain, text="Python 3.15.2", ephemeral=True)

    kept_out = without_ephemeral_text(ephemeral)

    assert kept_out.text.startswith("[Kamerdyner answered from a web search")
    assert "3.15.2" not in kept_out.text
    assert (kept_out.role, kept_out.at, kept_out.ephemeral) == (Role.MODEL, plain.at, True)
    assert without_ephemeral_text(plain) is plain
