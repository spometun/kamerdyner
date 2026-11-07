import dataclasses

import pytest
from fakes import make_message, make_user

from kamerdyner.common.llm import ModelTurn, Prompt, Role, UserTurn
from kamerdyner.prompt import build_prompt


def test_user_messages_carry_local_time_with_weekday_and_offset() -> None:
    user = make_user()
    question = make_message(Role.USER, "Котра година?")
    answer = make_message(Role.MODEL, "Пів на третю.", minutes=1)
    follow_up = make_message(Role.USER, "Дякую", minutes=90)

    prompt = build_prompt(user, "", [question, answer, follow_up])

    assert prompt.turns == (
        UserTurn("[2026-10-04 Sun 14:32 UTC-04:00] Котра година?"),
        ModelTurn("Пів на третю."),
        UserTurn("[2026-10-04 Sun 16:02 UTC-04:00] Дякую"),
    )


def test_consecutive_user_messages_merge_into_one_turn() -> None:
    user = make_user()
    first = make_message(Role.USER, "Перше", minutes=0)
    second = make_message(Role.USER, "Друге", minutes=3)

    prompt = build_prompt(user, "", [first, second])

    assert prompt.turns == (
        UserTurn(
            "[2026-10-04 Sun 14:32 UTC-04:00] Перше\n\n[2026-10-04 Sun 14:35 UTC-04:00] Друге",
        ),
    )


def test_memory_goes_into_the_system_instruction() -> None:
    user = make_user()
    question = make_message(Role.USER, "Що я люблю?")

    prompt = build_prompt(user, "Любить каву без цукру.", [question])

    assert "Марта" in prompt.system_instruction
    assert prompt.system_instruction.endswith("Любить каву без цукру.")


@pytest.mark.parametrize(
    "turns",
    [
        (),
        (ModelTurn("a"), UserTurn("b")),
        (UserTurn("a"), ModelTurn("b")),
        (UserTurn("a"), UserTurn("b")),
    ],
)
def test_prompt_rejects_malformed_dialogue(turns: tuple[UserTurn | ModelTurn, ...]) -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        Prompt(system_instruction="", turns=turns)


def test_a_turn_carries_the_provider_data_of_its_message() -> None:
    question = make_message(Role.USER, "Як справи?")
    answer = make_message(Role.MODEL, "Добре", minutes=1)
    answer = dataclasses.replace(answer, provider_data="data")
    follow_up = make_message(Role.USER, "Чудово", minutes=2)

    prompt = build_prompt(make_user(), "", [question, answer, follow_up])

    assert prompt.turns[1] == ModelTurn("Добре", provider_data="data")


def test_a_merged_turn_keeps_the_provider_data_of_its_last_message() -> None:
    first = make_message(Role.MODEL, "Перше")
    first = dataclasses.replace(first, provider_data="older")
    second = make_message(Role.MODEL, "Друге", minutes=1)
    second = dataclasses.replace(second, provider_data="newer")
    question = make_message(Role.USER, "Ну?", minutes=2)
    opening = make_message(Role.USER, "Привіт", minutes=-1)

    prompt = build_prompt(make_user(), "", [opening, first, second, question])

    assert prompt.turns[1] == ModelTurn("Перше\n\nДруге", provider_data="newer")
