from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kamerdyner.users import User, find_timezone, load_user, save_timezone


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


def test_saving_a_timezone_rewrites_only_its_value(tmp_path: Path) -> None:
    settings = (
        '# timezone = "Comment/Stays"\n'
        'name = "Марта"\n'
        'timezone = "America/Toronto"  # where she lives\n'
        "telegram_id = 42\n"
    )
    (tmp_path / "user.toml").write_text(settings, encoding="utf-8")
    kyiv = ZoneInfo("Europe/Kyiv")

    save_timezone(tmp_path, kyiv)

    saved = (tmp_path / "user.toml").read_text(encoding="utf-8")
    assert saved == settings.replace('"America/Toronto"', '"Europe/Kyiv"')
    assert load_user(tmp_path).timezone == kyiv


@pytest.mark.parametrize("settings", ['name = "Марта"\n', 'timezone = "A/B"\ntimezone = "C/D"\n'])
def test_saving_needs_exactly_one_timezone_setting(tmp_path: Path, settings: str) -> None:
    (tmp_path / "user.toml").write_text(settings, encoding="utf-8")
    kyiv = ZoneInfo("Europe/Kyiv")

    with pytest.raises(ValueError, match="timezone settings instead of one"):
        save_timezone(tmp_path, kyiv)


def test_a_known_timezone_is_found() -> None:
    assert find_timezone("Europe/Kyiv") == ZoneInfo("Europe/Kyiv")


@pytest.mark.parametrize("name", ["Mars/Base", "Europe", "../etc/passwd", "/etc/passwd", ""])
def test_anything_else_is_no_timezone(name: str) -> None:
    assert find_timezone(name) is None
