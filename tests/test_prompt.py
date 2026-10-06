import pytest
from fakes import make_message, make_user

from kamerdyner.llm import Prompt, Turn
from kamerdyner.messages import Role
from kamerdyner.prompt import build_prompt


def test_user_messages_carry_local_time_with_weekday() -> None:
    user = make_user()
    question = make_message(Role.USER, "Котра година?")
    answer = make_message(Role.MODEL, "Пів на третю.", minutes=1)
    follow_up = make_message(Role.USER, "Дякую", minutes=90)

    prompt = build_prompt(user, "", [question, answer, follow_up])

    assert prompt.turns == (
        Turn(Role.USER, "[2026-10-04 Sun 14:32] Котра година?"),
        Turn(Role.MODEL, "Пів на третю."),
        Turn(Role.USER, "[2026-10-04 Sun 16:02] Дякую"),
    )


def test_consecutive_user_messages_merge_into_one_turn() -> None:
    user = make_user()
    first = make_message(Role.USER, "Перше", minutes=0)
    second = make_message(Role.USER, "Друге", minutes=3)

    prompt = build_prompt(user, "", [first, second])

    assert prompt.turns == (
        Turn(Role.USER, "[2026-10-04 Sun 14:32] Перше\n\n[2026-10-04 Sun 14:35] Друге"),
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
        (Turn(Role.MODEL, "a"), Turn(Role.USER, "b")),
        (Turn(Role.USER, "a"), Turn(Role.MODEL, "b")),
        (Turn(Role.USER, "a"), Turn(Role.USER, "b")),
    ],
)
def test_prompt_rejects_malformed_dialogue(turns: tuple[Turn, ...]) -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        Prompt(system_instruction="", turns=turns)
