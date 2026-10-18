"""The mapping from SDK responses to results, tested directly, without a network call."""

import logging

import pytest
from google.genai import types

from kamerdyner.llm import Incomplete, Reply
from kamerdyner.llm.gemini import _result  # pyright: ignore[reportPrivateUsage]


def _signed(text: str) -> types.Part:
    return types.Part(text=text, thought_signature=b"signature")


def _unsigned(text: str) -> types.Part:
    return types.Part(text=text)


def _response(
    parts: list[types.Part], finish_reason: types.FinishReason
) -> types.GenerateContentResponse:
    content = types.Content(role="model", parts=parts)
    candidate = types.Candidate(content=content, finish_reason=finish_reason)
    return types.GenerateContentResponse(candidates=[candidate])


def test_full_answer_is_a_reply() -> None:
    part = _signed("Привіт")
    response = _response([part], types.FinishReason.STOP)

    assert _result(response) == Reply("Привіт")


def test_unsigned_reasoning_before_the_answer_is_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    reasoning = _unsigned("The user asks to be called Оля. Wait, ...")
    answer = _signed("Зрозумів, Олю!")
    response = _response([reasoning, answer], types.FinishReason.STOP)

    with caplog.at_level(logging.WARNING):
        result = _result(response)

    assert result == Reply("Зрозумів, Олю!")
    assert "The user asks to be called" in caplog.text


def test_signed_parts_join_into_one_answer() -> None:
    first = _signed("Зрозумів, ")
    second = _signed("Олю!")
    response = _response([first, second], types.FinishReason.STOP)

    assert _result(response) == Reply("Зрозумів, Олю!")


def test_answer_without_any_signed_part_is_incomplete() -> None:
    part = _unsigned("The user asks...")
    response = _response([part], types.FinishReason.STOP)

    result = _result(response)

    assert isinstance(result, Incomplete)


def test_answer_cut_by_the_output_limit_is_incomplete() -> None:
    part = _signed("Пів")
    response = _response([part], types.FinishReason.MAX_TOKENS)

    result = _result(response)

    assert isinstance(result, Incomplete)
    assert result.partial_text == "Пів"
    assert "MAX_TOKENS" in result.reason


def test_empty_answer_is_incomplete() -> None:
    part = _signed("")
    response = _response([part], types.FinishReason.STOP)

    result = _result(response)

    assert isinstance(result, Incomplete)


def test_candidate_without_content_is_incomplete() -> None:
    candidate = types.Candidate(finish_reason=types.FinishReason.SAFETY)
    response = types.GenerateContentResponse(candidates=[candidate])

    result = _result(response)

    assert isinstance(result, Incomplete)
    assert "SAFETY" in result.reason


def test_blocked_prompt_is_incomplete() -> None:
    feedback = types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY)
    response = types.GenerateContentResponse(candidates=[], prompt_feedback=feedback)

    result = _result(response)

    assert isinstance(result, Incomplete)
    assert "SAFETY" in result.reason
