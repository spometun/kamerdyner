from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kamerdyner.users import User, load_user


def test_user_is_read_from_their_directory(tmp_path: Path) -> None:
    directory = tmp_path / "marta"
    directory.mkdir()
    (directory / "user.toml").write_text(
        'name = "Марта"\ntimezone = "Europe/Kyiv"\n', encoding="utf-8"
    )

    user = load_user(directory)

    kyiv = ZoneInfo("Europe/Kyiv")
    assert user == User(id="marta", name="Марта", timezone=kyiv)


@pytest.mark.parametrize(
    ("settings", "complaint"),
    [
        ('timezone = "Europe/Kyiv"\n', "needs string 'name'"),
        ('name = "Марта"\ntimezone = "Mars/Olympus"\n', "unknown timezone"),
        ('name = " "\ntimezone = "Europe/Kyiv"\n', "empty name"),
    ],
)
def test_rejects_bad_settings(tmp_path: Path, settings: str, complaint: str) -> None:
    (tmp_path / "user.toml").write_text(settings, encoding="utf-8")

    with pytest.raises(ValueError, match=complaint):
        load_user(tmp_path)
