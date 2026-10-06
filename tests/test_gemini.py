"""The mapping from SDK responses to results, tested directly, without a network call."""

from google.genai import types

from kamerdyner.llm import Incomplete, Reply
from kamerdyner.llm.gemini import _result  # pyright: ignore[reportPrivateUsage]


def _response(text: str, finish_reason: types.FinishReason) -> types.GenerateContentResponse:
    part = types.Part(text=text)
    content = types.Content(role="model", parts=[part])
    candidate = types.Candidate(content=content, finish_reason=finish_reason)
    return types.GenerateContentResponse(candidates=[candidate])


def test_full_answer_is_a_reply() -> None:
    response = _response("Привіт", types.FinishReason.STOP)

    assert _result(response) == Reply("Привіт")


def test_answer_cut_by_the_output_limit_is_incomplete() -> None:
    response = _response("Пів", types.FinishReason.MAX_TOKENS)

    result = _result(response)

    assert isinstance(result, Incomplete)
    assert result.partial_text == "Пів"
    assert "MAX_TOKENS" in result.reason


def test_empty_answer_is_incomplete() -> None:
    response = _response("", types.FinishReason.STOP)

    result = _result(response)

    assert isinstance(result, Incomplete)


def test_blocked_prompt_is_incomplete() -> None:
    feedback = types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY)
    response = types.GenerateContentResponse(candidates=[], prompt_feedback=feedback)

    result = _result(response)

    assert isinstance(result, Incomplete)
    assert "SAFETY" in result.reason
