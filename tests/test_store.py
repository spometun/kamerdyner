import dataclasses
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


def test_only_an_ephemeral_message_stores_the_flag_and_it_survives_reopening(
    tmp_path: Path,
) -> None:
    question = make_message(Role.USER, "Яка версія Python?", minutes=0)
    plain = make_message(Role.MODEL, "Перевірю.", minutes=1)
    ephemeral = make_message(Role.MODEL, "3.15.2", minutes=2)
    ephemeral = dataclasses.replace(ephemeral, ephemeral=True)
    path = tmp_path / "latest.json"
    MessageFile(path).replace([question, plain, ephemeral])

    records = json.loads(path.read_text(encoding="utf-8"))["messages"]
    reopened = MessageFile(path)

    assert ["ephemeral" in record for record in records] == [False, False, True]
    assert reopened.messages == (question, plain, ephemeral)


def test_a_non_boolean_ephemeral_flag_is_corruption(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    record = {"role": "model", "text": "x", "at": "2026-10-04T14:32:00-04:00", "ephemeral": "yes"}
    path.write_text(json.dumps({"messages": [record]}), encoding="utf-8")

    with pytest.raises(ValueError, match="message 0"):
        MessageFile(path)


def test_provider_data_is_stored_only_on_the_messages_that_have_it(tmp_path: Path) -> None:
    plain = make_message(Role.USER, "Привіт")
    with_data = make_message(Role.MODEL, "Вітаю", minutes=1)
    with_data = dataclasses.replace(with_data, provider_data="data")
    path = tmp_path / "latest.json"
    MessageFile(path).replace([plain, with_data])

    records = json.loads(path.read_text(encoding="utf-8"))["messages"]
    reopened = MessageFile(path)

    assert ["provider_data" in record for record in records] == [False, True]
    assert reopened.messages == (plain, with_data)


def test_non_text_provider_data_is_corruption(tmp_path: Path) -> None:
    path = tmp_path / "latest.json"
    record = {"role": "model", "text": "x", "at": "2026-10-04T14:32:00-04:00", "provider_data": 5}
    path.write_text(json.dumps({"messages": [record]}), encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted message file"):
        MessageFile(path)
