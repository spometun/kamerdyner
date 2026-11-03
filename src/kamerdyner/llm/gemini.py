"""The model on the Gemini API. All use of the google-genai SDK lives in this module."""

import logging
from collections.abc import Sequence

import aiohttp
import httpx
from google import genai
from google.genai import errors, types

from kamerdyner.common.llm import GenerateResult, Incomplete, Prompt, Reply, Turn, Unavailable

logger = logging.getLogger(__name__)


class GeminiLLM:
    """Generates with one Gemini model at a fixed thinking level and a fixed sampling seed, so
    the same prompt gets the same reply (best effort on Google's side, and only for the same
    model version). With `web`, the model may search Google and read the pages it is given, when
    it decides to; such a reply is `grounded`. Retries of transient failures (429, 5xx, timeouts)
    are configured on the client; what still fails comes back as a value."""

    def __init__(
        self,
        client: genai.Client,
        model: str,
        thinking_level: types.ThinkingLevel,
        seed: int,
        web: bool,
    ) -> None:
        self._client = client
        self._model = model
        self._thinking_level = thinking_level
        self._seed = seed
        self._tools = _web_tools() if web else None

    async def generate(self, prompt: Prompt) -> GenerateResult:
        contents = _contents(prompt.turns)
        thinking = types.ThinkingConfig(thinking_level=self._thinking_level)
        # Tool calls, once there are tools, are run by us, not by the SDK.
        manual_calls = types.AutomaticFunctionCallingConfig(disable=True)
        config = types.GenerateContentConfig(
            system_instruction=prompt.system_instruction,
            thinking_config=thinking,
            automatic_function_calling=manual_calls,
            seed=self._seed,
            tools=self._tools,
        )
        try:
            # The SDK's signature mentions PIL images, unknown without Pillow installed.
            response = await self._client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
                model=self._model, contents=contents, config=config
            )
        except errors.APIError as error:
            return Unavailable(detail=f"Gemini API error {error.code}: {error.message}")
        except (aiohttp.ClientError, httpx.HTTPError, TimeoutError) as error:
            return Unavailable(detail=f"Cannot reach Gemini: {error!r}")
        _log_usage(response)
        return _result(response)


def make_client(api_key: str, timeout_seconds: int, attempts: int) -> genai.Client:
    """A Gemini client that retries transient failures (408, 429, 5xx, network) up to
    `attempts` calls in total, with exponential backoff, each call capped at `timeout_seconds`."""
    retry = types.HttpRetryOptions(attempts=attempts)
    options = types.HttpOptions(timeout=timeout_seconds * 1000, retry_options=retry)
    return genai.Client(api_key=api_key, http_options=options)


def _contents(turns: Sequence[Turn]) -> list[types.Content]:
    contents: list[types.Content] = []
    for turn in turns:
        part = types.Part(text=turn.text)
        content = types.Content(role=turn.role.value, parts=[part])
        contents.append(content)
    return contents


def _result(response: types.GenerateContentResponse) -> GenerateResult:
    if not response.candidates:
        feedback = response.prompt_feedback
        block_reason = None if feedback is None else feedback.block_reason
        return Incomplete(
            reason=f"no candidates, prompt block reason {block_reason}", partial_text=""
        )
    candidate = response.candidates[0]
    text = _answer_text(candidate)
    finish_reason = candidate.finish_reason
    if finish_reason is types.FinishReason.STOP and text:
        grounded = _grounded(candidate)
        return Reply(text=text, grounded=grounded)
    return Incomplete(reason=f"finish reason {finish_reason}, {len(text)} chars", partial_text=text)


def _answer_text(candidate: types.Candidate) -> str:
    """The joined text of the parts that carry a thought signature. Gemini 3 signs every answer;
    a part without a signature is the model's reasoning come back unmarked, as if it were the answer
    (gemini-3.8-flash at a low thinking level, a few percent of replies). Such parts are dropped
    and logged."""
    content = candidate.content
    # A candidate stopped before any output (e.g. by safety filters) comes without content.
    parts = [] if content is None or content.parts is None else content.parts
    texts: list[str] = []
    for part in parts:
        if part.thought_signature is None:
            logger.warning("Dropped a Gemini response part without a thought signature: %r", part)
        elif part.text is not None:
            texts.append(part.text)
    return "".join(texts)


def _web_tools() -> list[types.Tool]:
    search = types.Tool(google_search=types.GoogleSearch())
    pages = types.Tool(url_context=types.UrlContext())
    return [search, pages]


def _grounded(candidate: types.Candidate) -> bool:
    """Whether the answer rests on a Google search or on pages it read; what was searched and
    read is logged."""
    metadata = candidate.grounding_metadata
    # Without web tools, or when the model used none, the metadata is absent or empty.
    if metadata is None:
        return False
    queries = [] if metadata.web_search_queries is None else metadata.web_search_queries
    chunks = [] if metadata.grounding_chunks is None else metadata.grounding_chunks
    pages = [chunk.web.uri for chunk in chunks if chunk.web is not None]
    if not queries and not pages:
        return False
    logger.info("Grounded on searches %s, pages %s", queries, pages)
    return True


def _log_usage(response: types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    assert usage is not None, f"Unexpected Gemini response without usage metadata: {response!r}"
    logger.info(
        "Gemini tokens: prompt %s (cached %s), thoughts %s, output %s",
        usage.prompt_token_count,
        usage.cached_content_token_count,
        usage.thoughts_token_count,
        usage.candidates_token_count,
    )
