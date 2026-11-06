from kamerdyner.common.llm import Prompt, Reply, Role, ToolCall, Turn
from kamerdyner.tools.tools import ToolResult, tools_record, with_tool_results

_QUESTION = Turn(Role.USER, "Постав Київ і Варшаву")
_PROMPT = Prompt(system_instruction="Будь ввічливою", turns=(_QUESTION,))
_KYIV = ToolResult(ToolCall("set_timezone", {"zone": "Europe/Kyiv"}), "Done.")
_WARSAW = ToolResult(ToolCall("set_timezone", {"zone": "Europe/Warsaw"}), "Також.")


def test_the_next_request_holds_what_the_model_said_asked_for_and_got() -> None:
    reply = Reply("Зараз поставлю", tool_calls=(_KYIV.call, _WARSAW.call))

    prompt = with_tool_results(_PROMPT, reply, (_KYIV, _WARSAW))

    assert prompt.system_instruction == "Будь ввічливою"
    assert prompt.turns == (
        _QUESTION,
        Turn(
            Role.MODEL,
            "Зараз поставлю\n"
            'Model requested tool set_timezone(zone="Europe/Kyiv").\n'
            'Model requested tool set_timezone(zone="Europe/Warsaw").',
        ),
        Turn(
            Role.USER,
            "The system ran it, result: Done.\nThe system ran it, result: Також.",
        ),
    )


def test_a_model_that_said_nothing_gets_a_turn_of_its_requests_alone() -> None:
    reply = Reply("", tool_calls=(_KYIV.call,))

    prompt = with_tool_results(_PROMPT, reply, (_KYIV,))

    assert prompt.turns[1].text == 'Model requested tool set_timezone(zone="Europe/Kyiv").'


def test_the_record_pairs_each_request_with_its_result() -> None:
    assert tools_record((_KYIV,)) == (
        'Model requested tool set_timezone(zone="Europe/Kyiv").\n'
        "The system ran it, result: Done.\n"
        "In the next request the model replied:\n"
    )
