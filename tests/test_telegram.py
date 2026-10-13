from pathlib import Path

import pytest

from kamerdyner.apps.telegram import load_telegram_users, split_message


def _user_file(data: Path, name: str, extra: str) -> None:
    directory = data / name
    directory.mkdir()
    settings = f'name = "{name}"\ntimezone = "America/Toronto"\n{extra}'
    (directory / "user.toml").write_text(settings, encoding="utf-8")


def test_users_are_found_by_telegram_id_and_others_skipped(tmp_path: Path) -> None:
    _user_file(tmp_path, "sergiy", "telegram_id = 111\n")
    _user_file(tmp_path, "olenka", "telegram_id = 222\n")
    _user_file(tmp_path, "marta", "")

    users = load_telegram_users(tmp_path)

    assert users == {111: tmp_path / "sergiy", 222: tmp_path / "olenka"}


@pytest.mark.parametrize(
    ("second", "message"),
    [('telegram_id = "111"\n', "non-integer"), ("telegram_id = 111\n", "already taken")],
)
def test_a_bad_or_repeated_telegram_id_is_refused(
    tmp_path: Path, second: str, message: str
) -> None:
    _user_file(tmp_path, "a", "telegram_id = 111\n")
    _user_file(tmp_path, "b", second)

    with pytest.raises(ValueError, match=message):
        load_telegram_users(tmp_path)


@pytest.mark.parametrize(
    ("text", "limit", "pieces"),
    [
        ("коротко", 10, ["коротко"]),
        ("рядок один\nрядок два", 12, ["рядок один", "рядок два"]),
        ("слово слово слово", 12, ["слово слово", "слово"]),
        ("абвгґдеєжзи", 4, ["абвг", "ґдеє", "жзи"]),
    ],
)
def test_split_message(text: str, limit: int, pieces: list[str]) -> None:
    assert split_message(text, limit) == pieces
