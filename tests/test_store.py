from pathlib import Path

import pytest
from fakes import make_message

from kamerdyner.messages import Role
from kamerdyner.store import ConversationLog


def test_new_user_has_an_empty_conversation(tmp_path: Path) -> None:
    log = ConversationLog(tmp_path / "marta")

    assert log.recent() == []


def test_messages_survive_reopening(tmp_path: Path) -> None:
    question = make_message(Role.USER, "Привіт! Як справи?", minutes=0)
    answer = make_message(Role.MODEL, "Добре, дякую.", minutes=1)
    log = ConversationLog(tmp_path)
    log.append(question)
    log.append(answer)

    reopened = ConversationLog(tmp_path)

    assert reopened.recent() == [question, answer]


def test_fold_hides_the_oldest_messages_and_survives_reopening(tmp_path: Path) -> None:
    messages = [make_message(Role.USER, f"message {index}", minutes=index) for index in range(5)]
    log = ConversationLog(tmp_path)
    for message in messages:
        log.append(message)

    log.fold(2)
    log.fold(1)
    reopened = ConversationLog(tmp_path)

    assert reopened.recent() == messages[3:]


def test_fold_past_the_end_is_a_bug(tmp_path: Path) -> None:
    log = ConversationLog(tmp_path)
    message = make_message(Role.USER, "one")
    log.append(message)

    with pytest.raises(AssertionError, match="Internal error"):
        log.fold(2)


def test_torn_last_line_is_dropped(tmp_path: Path) -> None:
    complete = make_message(Role.USER, "complete", minutes=0)
    log = ConversationLog(tmp_path)
    log.append(complete)
    with (tmp_path / "log.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"role": "model", "te')
    after_crash = make_message(Role.USER, "after the crash", minutes=1)

    reopened = ConversationLog(tmp_path)
    reopened.append(after_crash)

    assert reopened.recent() == [complete, after_crash]


def test_corrupted_line_names_its_number(tmp_path: Path) -> None:
    log = ConversationLog(tmp_path)
    message = make_message(Role.USER, "fine")
    log.append(message)
    with (tmp_path / "log.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"role": "nobody", "text": "x", "at": "2026-10-04T18:32:00+00:00"}\n')

    with pytest.raises(ValueError, match="line 2"):
        log.recent()
