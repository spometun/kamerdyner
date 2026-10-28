import json
from pathlib import Path

import pytest
from fakes import make_message

from kamerdyner.common.llm import Role
from kamerdyner.common.store import MessageFile


def test_a_missing_message_file_is_an_empty_list(tmp_path: Path) -> None:
    messages = MessageFile(tmp_path / "marta" / "latest.json")

    assert messages.messages == ()


def test_message_file_survives_reopening(tmp_path: Path) -> None:
    question = make_message(Role.USER, "Привіт! Як справи?", minutes=0)
    answer = make_message(Role.MODEL, "Добре, дякую.", minutes=1)
    current = MessageFile(tmp_path / "latest.json")
    current.append(question)
    current.append(answer)

    reopened = MessageFile(tmp_path / "latest.json")

    assert reopened.messages == (question, answer)


def test_replace_makes_the_given_messages_the_whole_list(tmp_path: Path) -> None:
    messages = [make_message(Role.USER, f"message {index}", minutes=index) for index in range(4)]
    current = MessageFile(tmp_path / "latest.json")
    for message in messages:
        current.append(message)

    current.replace(messages[2:])
    reopened = MessageFile(tmp_path / "latest.json")

    assert current.messages == tuple(messages[2:])
    assert reopened.messages == tuple(messages[2:])


def test_corrupted_message_file_names_the_message(tmp_path: Path) -> None:
    good = {"role": "user", "text": "fine", "at": "2026-10-04T18:32:00+00:00"}
    bad = {"role": "nobody", "text": "x", "at": "2026-10-04T18:33:00+00:00"}
    content = json.dumps({"messages": [good, bad]})
    (tmp_path / "latest.json").write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="message 1"):
        MessageFile(tmp_path / "latest.json")
