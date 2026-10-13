import pytest

from kamerdyner.apps.terminal import apply_keystrokes


@pytest.mark.parametrize(
    ("line", "keys", "expected"),
    [
        ("", "привіт", "привіт"),
        ("при", "віт", "привіт"),
        ("", "привіт\x7f\x7fе", "приве"),
        ("", "\x7f\x7fок", "ок"),
        ("", "так\rі ні", "так і ні"),
        ("", "a\x1b[Db\x1bOAc", "abc"),
        ("", "x\x04\x01y", "xy"),
    ],
)
def test_apply_keystrokes(line: str, keys: str, expected: str) -> None:
    assert apply_keystrokes(line, keys) == expected
