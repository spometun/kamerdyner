import pytest
from fakes import ScriptedLLM, make_message

from kamerdyner.common.llm import Reply, Role, Turn
from kamerdyner.memory.summary import span_heading, summarize, transcript


def test_transcript_names_both_sides_with_local_times() -> None:
    messages = [
        make_message(Role.USER, "Завтра до лікаря", minutes=0),
        make_message(Role.MODEL, "Нагадати зранку?", minutes=1),
    ]

    text = transcript("Марта", messages)

    assert text == (
        "[2026-10-04 Sun 14:32] Марта: Завтра до лікаря\n\n"
        "[2026-10-04 Sun 14:33] Kamerdyner: Нагадати зранку?"
    )


def test_span_heading_holds_the_first_and_the_last_time() -> None:
    two_days_later = 2 * 24 * 60 + 5
    messages = [
        make_message(Role.USER, "a", minutes=0),
        make_message(Role.MODEL, "b", minutes=1),
        make_message(Role.USER, "c", minutes=two_days_later),
    ]

    heading = span_heading(messages)

    assert heading == "## [2026-10-04 Sun 14:32] — [2026-10-06 Tue 14:37]"


async def test_summarize_sends_the_transcript_and_returns_the_reply() -> None:
    messages = [
        make_message(Role.USER, "Привіт", minutes=0),
        make_message(Role.MODEL, "Привіт, Марто!", minutes=1),
    ]
    llm = ScriptedLLM([Reply("- Марта привіталася.")])

    result = await summarize(llm, "Марта", messages)

    assert result == Reply("- Марта привіталася.")
    [prompt] = llm.prompts
    conversation = transcript("Марта", messages)
    assert prompt.turns == (Turn(Role.USER, conversation),)
    assert "Марта" in prompt.system_instruction


async def test_summarize_refuses_an_empty_stretch() -> None:
    llm = ScriptedLLM([])

    with pytest.raises(ValueError, match="Nothing to summarize"):
        await summarize(llm, "Марта", [])
