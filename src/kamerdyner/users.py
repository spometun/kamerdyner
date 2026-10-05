"""Users of the assistant.

A user is a directory holding everything about them: their settings in `user.toml`, and next to
it their conversation (see `kamerdyner.store`). `user.toml`:

    name = "Марта"
    timezone = "America/Toronto"
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


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
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f"User file {path} has unknown timezone {timezone!r}") from error
    return User(id=directory.name, name=name, timezone=zone)
