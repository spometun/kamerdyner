"""Users of the assistant.

A user is a directory holding everything about them: their settings in `user.toml`, and next to
it their messages waiting for a reply (`pending.json`) and the archive of their conversation
(`log/`), both kept by `kamerdyner.chat`, and their memory (`memory/`, see
`kamerdyner.memory`).
`user.toml`:

    name = "Марта"
    timezone = "America/Toronto"
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from kamerdyner.common.store import write_atomically


@dataclass(frozen=True)
class User:
    """A user: `id` is the name of their directory, `name` is how the assistant calls them, and
    `timezone` is where they live — times shown to the model are local to it."""

    id: str
    name: str
    timezone: ZoneInfo

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(f"Invalid input. User {self.id!r} has an empty name")


def load_user(directory: Path) -> User:
    """Reads the user's settings; raises ValueError naming the file on a bad one."""
    path = directory / "user.toml"
    with path.open("rb") as file:
        settings = tomllib.load(file)
    match settings:
        case {"name": str() as name, "timezone": str() as timezone}:
            pass
        case _:
            raise ValueError(
                f"User file {path} needs string 'name' and 'timezone', got {settings!r}"
            )
    zone = find_timezone(timezone)
    if zone is None:
        raise ValueError(f"User file {path} has unknown timezone {timezone!r}")
    return User(id=directory.name, name=name, timezone=zone)


def find_timezone(name: str) -> ZoneInfo | None:
    """The time zone with this IANA name, which may come from outside; None if there is none."""
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return None


def save_timezone(directory: Path, timezone: ZoneInfo) -> None:
    """Rewrites the time zone in the user's `user.toml`, leaving the rest of the file as it is;
    raises ValueError naming the file when it has no such setting on a line of its own."""
    path = directory / "user.toml"
    settings = path.read_text(encoding="utf-8")
    updated, count = _TIMEZONE_SETTING.subn(rf'\g<1>"{timezone.key}"', settings)
    if count != 1:
        raise ValueError(f"User file {path} has {count} timezone settings instead of one")
    write_atomically(path, updated)


_TIMEZONE_SETTING = re.compile(r"""^(timezone\s*=\s*)("[^"\n]*"|'[^'\n]*')""", re.MULTILINE)
