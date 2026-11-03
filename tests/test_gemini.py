"""The mapping from SDK responses to results, tested directly, without a network call."""

import logging

import pytest
from google.genai import types

from kamerdyner.common.llm import Incomplete, Reply, WebSearch
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


def _grounded_response(metadata: types.GroundingMetadata) -> types.GenerateContentResponse:
    part = _signed("Python 3.15.2")
    content = types.Content(role="model", parts=[part])
    candidate = types.Candidate(
        content=content, finish_reason=types.FinishReason.STOP, grounding_metadata=metadata
    )
    return types.GenerateContentResponse(candidates=[candidate])


_CHIP_URL = "https://www.google.com/search?q=latest+python+release&client=app-vertex-grounding"


def test_an_answer_after_a_search_is_grounded_with_googles_links_to_its_searches() -> None:
    source = types.GroundingChunkWeb(uri="https://example.com/r", title="python.org")
    chunk = types.GroundingChunk(web=source)
    widget = (
        '<style>.chip {}</style><div class="container"><svg class="logo"></svg>'
        '<div class="carousel"><a class="chip" '
        f'href="{_CHIP_URL.replace("&", "&amp;")}">latest\n  python release</a></div></div>'
    )
    entry = types.SearchEntryPoint(rendered_content=widget)
    metadata = types.GroundingMetadata(
        web_search_queries=["latest python release"],
        grounding_chunks=[chunk],
        search_entry_point=entry,
    )
    response = _grounded_response(metadata)

    search = WebSearch(query="latest python release", url=_CHIP_URL)
    assert _result(response) == Reply("Python 3.15.2", grounded=True, searches=(search,))


def test_searches_without_googles_widget_are_linked_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    metadata = types.GroundingMetadata(web_search_queries=["toronto weather"])
    response = _grounded_response(metadata)

    with caplog.at_level(logging.WARNING):
        result = _result(response)

    search = WebSearch(
        query="toronto weather", url="https://www.google.com/search?q=toronto+weather"
    )
    assert result == Reply("Python 3.15.2", grounded=True, searches=(search,))
    assert "without Google's suggestion links" in caplog.text


def test_an_answer_from_a_page_it_read_is_grounded() -> None:
    page = types.GroundingChunkWeb(uri="https://docs.python.org/3/whatsnew/", title="What's New")
    chunk = types.GroundingChunk(web=page)
    metadata = types.GroundingMetadata(grounding_chunks=[chunk])
    response = _grounded_response(metadata)

    assert _result(response) == Reply("Python 3.15.2", grounded=True)


def test_empty_grounding_metadata_is_not_grounded() -> None:
    metadata = types.GroundingMetadata()
    response = _grounded_response(metadata)

    assert _result(response) == Reply("Python 3.15.2", grounded=False)
