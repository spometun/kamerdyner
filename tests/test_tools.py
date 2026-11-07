from kamerdyner.common.llm import (
    ModelReply,
    ModelTurn,
    Prompt,
    ToolCall,
    ToolResult,
    ToolTurn,
    UserTurn,
)
from kamerdyner.tools.tools import tools_record, with_tool_results

_QUESTION = UserTurn("Постав Київ і Варшаву")
_PROMPT = Prompt(system_instruction="Будь ввічливою", turns=(_QUESTION,))
_KYIV = ToolResult(ToolCall("id-1", "set_timezone", {"zone": "Europe/Kyiv"}), "Done.")
_WARSAW = ToolResult(ToolCall("id-2", "set_timezone", {"zone": "Europe/Warsaw"}), "Також.")


def test_the_next_request_holds_what_the_model_said_asked_for_and_got() -> None:
    calls = (_KYIV.call, _WARSAW.call)
    reply = ModelReply("Зараз поставлю", tool_calls=calls, provider_data="data")

    prompt = with_tool_results(_PROMPT, reply, (_KYIV, _WARSAW))

    assert prompt.system_instruction == "Будь ввічливою"
    assert prompt.turns == (
        _QUESTION,
        ModelTurn("Зараз поставлю", provider_data="data"),
        ToolTurn((_KYIV, _WARSAW)),
    )


def test_the_record_pairs_each_request_with_its_result() -> None:
    assert tools_record((_KYIV,)) == (
        'Model requested tool set_timezone(zone="Europe/Kyiv").\n'
        "The system ran it, result: Done.\n"
        "In the next request the model replied:\n"
    )
