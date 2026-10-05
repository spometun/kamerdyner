from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kamerdyner.users import User, load_users


def test_loads_users_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "users.toml"
    path.write_text(
        '[andriy]\nname = "Андрій"\ntimezone = "America/Toronto"\n\n'
        '[marta]\nname = "Марта"\ntimezone = "Europe/Kyiv"\n',
        encoding="utf-8",
    )

    users = load_users(path)

    toronto = ZoneInfo("America/Toronto")
    kyiv = ZoneInfo("Europe/Kyiv")
    assert users == {
        "andriy": User(id="andriy", name="Андрій", timezone=toronto),
        "marta": User(id="marta", name="Марта", timezone=kyiv),
    }


@pytest.mark.parametrize(
    ("entry", "complaint"),
    [
        ('[marta]\ntimezone = "Europe/Kyiv"\n', "needs string 'name'"),
        ('[marta]\nname = "Марта"\ntimezone = "Mars/Olympus"\n', "unknown timezone"),
        ('[Marta]\nname = "Марта"\ntimezone = "Europe/Kyiv"\n', "User id must match"),
    ],
)
def test_rejects_bad_entries(tmp_path: Path, entry: str, complaint: str) -> None:
    path = tmp_path / "users.toml"
    path.write_text(entry, encoding="utf-8")

    with pytest.raises(ValueError, match=complaint):
        load_users(path)
