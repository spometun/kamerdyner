import pytest

from kamerdyner.apps.terminal import apply_keystrokes


# Ukrainian text: Cyrillic letters that look Latin are meant (RUF001).
@pytest.mark.parametrize(
    ("line", "keys", "expected"),
    [
        ("", "привіт", "привіт"),
        ("при", "віт", "привіт"),
        ("", "привіт\x7f\x7fе", "приве"),  # noqa: RUF001
        ("", "\x7f\x7fок", "ок"),  # noqa: RUF001
        ("", "так\rі ні", "так і ні"),  # noqa: RUF001
        ("", "a\x1b[Db\x1bOAc", "abc"),
        ("", "x\x04\x01y", "xy"),
    ],
)
def test_apply_keystrokes(line: str, keys: str, expected: str) -> None:
    assert apply_keystrokes(line, keys) == expected
