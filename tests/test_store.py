import json
from pathlib import Path

import pytest
from fakes import make_message

from kamerdyner.messages import Role
from kamerdyner.store import ConversationLog, CurrentConversation


def _logged_texts(directory: Path) -> list[str]:
    lines = (directory / "log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["text"] for line in lines]


def test_log_keeps_every_message_in_order(tmp_path: Path) -> None:
    log = ConversationLog(tmp_path)
    for index in range(3):
        message = make_message(Role.USER, f"Повідомлення {index}", minutes=index)
        log.append(message)

    reopened = ConversationLog(tmp_path)
    last = make_message(Role.MODEL, "Відповідь", minutes=3)
    reopened.append(last)

    assert _logged_texts(tmp_path) == [
        "Повідомлення 0",
        "Повідомлення 1",
        "Повідомлення 2",
        "Відповідь",
    ]


def test_log_drops_a_torn_last_line(tmp_path: Path) -> None:
    log = ConversationLog(tmp_path)
    complete = make_message(Role.USER, "complete")
    log.append(complete)
    with (tmp_path / "log.jsonl").open("a", encoding="utf-8") as file:
        file.write('{"role": "model", "te')

    reopened = ConversationLog(tmp_path)
    after_crash = make_message(Role.USER, "after the crash", minutes=1)
    reopened.append(after_crash)

    assert _logged_texts(tmp_path) == ["complete", "after the crash"]


def test_new_user_has_an_empty_current_conversation(tmp_path: Path) -> None:
    current = CurrentConversation(tmp_path / "marta")

    assert current.messages == ()


def test_current_conversation_survives_reopening(tmp_path: Path) -> None:
    question = make_message(Role.USER, "Привіт! Як справи?", minutes=0)
    answer = make_message(Role.MODEL, "Добре, дякую.", minutes=1)
    current = CurrentConversation(tmp_path)
    current.append(question)
    current.append(answer)

    reopened = CurrentConversation(tmp_path)

    assert reopened.messages == (question, answer)


def test_replace_makes_the_given_messages_the_whole_conversation(tmp_path: Path) -> None:
    messages = [make_message(Role.USER, f"message {index}", minutes=index) for index in range(4)]
    current = CurrentConversation(tmp_path)
    for message in messages:
        current.append(message)

    current.replace(messages[2:])
    reopened = CurrentConversation(tmp_path)

    assert current.messages == tuple(messages[2:])
    assert reopened.messages == tuple(messages[2:])


def test_corrupted_current_conversation_names_the_message(tmp_path: Path) -> None:
    good = {"role": "user", "text": "fine", "at": "2026-10-04T18:32:00+00:00"}
    bad = {"role": "nobody", "text": "x", "at": "2026-10-04T18:33:00+00:00"}
    content = json.dumps({"messages": [good, bad]})
    (tmp_path / "current.json").write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="message 1"):
        CurrentConversation(tmp_path)
