"""Users of the assistant and the file that lists them.

The users file is TOML, one table per user, keyed by the user id:

    [serhiy]
    name = "Сергій"
    timezone = "America/Toronto"
"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


@dataclass(frozen=True)
class User:
    """A user: `id` names their data directory, `name` is how the assistant calls them, and
    `timezone` is where they live — times shown to the model are local to it."""

    id: str
    name: str
    timezone: ZoneInfo

    def __post_init__(self) -> None:
        if _USER_ID.fullmatch(self.id) is None:
            raise ValueError(
                f"Invalid input. User id must match {_USER_ID.pattern}, got {self.id!r}"
            )
        if not self.name.strip():
            raise ValueError(f"Invalid input. User {self.id!r} has an empty name")


def load_users(path: Path) -> dict[str, User]:
    """Reads the users file; raises ValueError naming the file and the user on a bad entry."""
    with path.open("rb") as file:
        tables = tomllib.load(file)
    users: dict[str, User] = {}
    for user_id, table in tables.items():
        user = _user_from_table(path, user_id, table)
        users[user_id] = user
    return users


_USER_ID = re.compile(r"[a-z0-9_-]+")


def _user_from_table(path: Path, user_id: str, table: Any) -> User:
    match table:
        case {"name": str() as name, "timezone": str() as timezone}:
            pass
        case _:
            raise ValueError(
                f"Users file {path}: user {user_id!r} needs string 'name' and 'timezone', "
                f"got {table!r}"
            )
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"Users file {path}: user {user_id!r} has unknown timezone {timezone!r}"
        ) from error
    return User(id=user_id, name=name, timezone=zone)
