"""The model on the Gemini API. All use of the google-genai SDK lives in this module."""

import base64
import logging
import urllib.parse
from collections.abc import Sequence
from html.parser import HTMLParser

import aiohttp
import httpx
from google import genai
from google.genai import errors, types

from kamerdyner.common.llm import (
    GenerateResult,
    Incomplete,
    LLMError,
    Prompt,
    Reply,
    ToolCall,
    ToolSpec,
    Turn,
    Unavailable,
    WebSearch,
)

logger = logging.getLogger(__name__)


class GeminiLLM:
    """Generates with one Gemini model at a fixed thinking level and a fixed sampling seed, so
    the same prompt gets the same reply (best effort on Google's side, and only for the same
    model version). With `web`, the model may search Google and read the pages it is given, when
    it decides to; such a reply is `grounded`. Retries of transient failures (429, 5xx, timeouts)
    are configured on the client; what still fails comes back as a value.

    Provider data is the thought signature of the reply's text in base64: sent back on the text,
    it brings what the model thought in that turn into the request."""

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
        self._web_tools = _web_tools() if web else []

    async def generate(self, prompt: Prompt, tools: Sequence[ToolSpec] = ()) -> GenerateResult:
        contents = _contents(prompt.turns)
        thinking = types.ThinkingConfig(thinking_level=self._thinking_level)
        # Tool calls are run by us, not by the SDK.
        manual_calls = types.AutomaticFunctionCallingConfig(disable=True)
        functions = _function_tools(tools)
        # The API refuses our functions next to its built-in tools without this.
        mixed = bool(functions) and bool(self._web_tools)
        tool_config = types.ToolConfig(include_server_side_tool_invocations=mixed)
        config = types.GenerateContentConfig(
            system_instruction=prompt.system_instruction,
            thinking_config=thinking,
            automatic_function_calling=manual_calls,
            seed=self._seed,
            tools=[*functions, *self._web_tools],
            tool_config=tool_config,
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
        result = _result(response)
        return result if tools else _without_tool_calls(result)


def make_client(api_key: str, timeout_seconds: int, attempts: int) -> genai.Client:
    """A Gemini client that retries transient failures (408, 429, 5xx, network) up to
    `attempts` calls in total, with exponential backoff, each call capped at `timeout_seconds`."""
    retry = types.HttpRetryOptions(attempts=attempts)
    options = types.HttpOptions(timeout=timeout_seconds * 1000, retry_options=retry)
    return genai.Client(api_key=api_key, http_options=options)


def _contents(turns: Sequence[Turn]) -> list[types.Content]:
    contents: list[types.Content] = []
    for turn in turns:
        signature = _decoded_signature(turn.provider_data)
        part = types.Part(text=turn.text, thought_signature=signature)
        content = types.Content(role=turn.role.value, parts=[part])
        contents.append(content)
    return contents


def _decoded_signature(provider_data: str) -> bytes | None:
    return base64.b64decode(provider_data, validate=True) if provider_data else None


def _encoded_signature(candidate: types.Candidate) -> str:
    """The signature of the last signed text part, the one that closes the answer; empty when
    there is none. The text of a turn goes back as one part, so one signature goes with it."""
    signature = b""
    for part in _content_parts(candidate.content):
        if part.text is not None and part.thought_signature is not None:
            signature = part.thought_signature
    encoded = base64.b64encode(signature)
    return encoded.decode("ascii")


def _function_tools(specs: Sequence[ToolSpec]) -> list[types.Tool]:
    declarations: list[types.FunctionDeclaration] = []
    for spec in specs:
        declaration = types.FunctionDeclaration(
            name=spec.name,
            description=spec.description,
            parameters_json_schema=spec.parameters,
        )
        declarations.append(declaration)
    return [types.Tool(function_declarations=declarations)] if declarations else []


def _result(response: types.GenerateContentResponse) -> GenerateResult:
    if not response.candidates:
        feedback = response.prompt_feedback
        block_reason = None if feedback is None else feedback.block_reason
        return Incomplete(
            reason=f"no candidates, prompt block reason {block_reason}", partial_text=""
        )
    candidate = response.candidates[0]
    text = _answer_text(candidate)
    calls = _tool_calls(candidate)
    finish_reason = candidate.finish_reason
    if finish_reason is types.FinishReason.STOP and (text or calls):
        grounded = _grounded(candidate)
        searches = _searches(candidate)
        provider_data = _encoded_signature(candidate)
        return Reply(
            text=text,
            grounded=grounded,
            searches=searches,
            tool_calls=calls,
            provider_data=provider_data,
        )
    return Incomplete(reason=f"finish reason {finish_reason}, {len(text)} chars", partial_text=text)


def _without_tool_calls(result: GenerateResult) -> GenerateResult:
    """`result` of a request that allowed no tools; a model calling one anyway is an error."""
    if isinstance(result, LLMError) or not result.tool_calls:
        return result
    names = [call.name for call in result.tool_calls]
    logger.warning("The model called %s when no tools were allowed", names)
    return Incomplete(reason=f"tool calls {names} when none allowed", partial_text=result.text)


def _content_parts(content: types.Content | None) -> list[types.Part]:
    # A candidate stopped before any output (e.g. by safety filters) comes without content.
    return [] if content is None or content.parts is None else content.parts


def _tool_calls(candidate: types.Candidate) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for part in _content_parts(candidate.content):
        function_call = part.function_call
        if function_call is None:
            continue
        name = function_call.name
        assert name is not None, f"Unexpected Gemini function call without a name: {part!r}"
        # A call without arguments may come without them altogether.
        arguments = {} if function_call.args is None else function_call.args
        call = ToolCall(name=name, arguments=arguments)
        calls.append(call)
    return tuple(calls)


def _answer_text(candidate: types.Candidate) -> str:
    """The joined text of the text parts that carry a thought signature. Gemini 3 signs every
    answer; a text part without a signature is the model's reasoning come back unmarked, as if it
    were the answer (gemini-3.8-flash at a low thinking level, a few percent of replies). Such
    parts are dropped and logged."""
    texts: list[str] = []
    for part in _content_parts(candidate.content):
        if part.text is None:
            continue
        if part.thought_signature is None:
            logger.warning("Dropped a Gemini response part without a thought signature: %r", part)
        else:
            texts.append(part.text)
    return "".join(texts)


def _web_tools() -> list[types.Tool]:
    search = types.Tool(google_search=types.GoogleSearch())
    pages = types.Tool(url_context=types.UrlContext())
    return [search, pages]


def _grounded(candidate: types.Candidate) -> bool:
    """Whether the answer rests on a Google search or on pages it read; what was searched and
    read is logged. A turn that also calls our tools shows its searches only as `tool_call` parts,
    without metadata."""
    used_web = any(part.tool_call is not None for part in _content_parts(candidate.content))
    metadata = candidate.grounding_metadata
    # Without web tools, or when the model used none, the metadata is absent or empty.
    if metadata is None:
        return used_web
    queries = [] if metadata.web_search_queries is None else metadata.web_search_queries
    chunks = [] if metadata.grounding_chunks is None else metadata.grounding_chunks
    pages = [chunk.web.uri for chunk in chunks if chunk.web is not None]
    if not queries and not pages:
        return used_web
    logger.info("Grounded on searches %s, pages %s", queries, pages)
    return True


def _searches(candidate: types.Candidate) -> tuple[WebSearch, ...]:
    """The Google searches behind the answer, each with the link Google's own suggestion widget
    gives it (the "chips" of `search_entry_point`)."""
    metadata = candidate.grounding_metadata
    if metadata is None:
        return ()
    queries = [] if metadata.web_search_queries is None else metadata.web_search_queries
    entry = metadata.search_entry_point
    rendered = "" if entry is None or entry.rendered_content is None else entry.rendered_content
    parser = _ChipParser()
    parser.feed(rendered)
    parser.close()
    if queries and not parser.chips:
        logger.warning("Searches %s came without Google's suggestion links; linking them", queries)
        linked = [WebSearch(query=query, url=_search_url(query)) for query in queries]
        return tuple(linked)
    return tuple(parser.chips)


def _search_url(query: str) -> str:
    encoded = urllib.parse.quote_plus(query)
    return f"https://www.google.com/search?q={encoded}"


class _ChipParser(HTMLParser):
    """Collects the links of Google's suggestion widget: `<a class="chip" href="...">query</a>`."""

    def __init__(self) -> None:
        super().__init__()
        self.chips: list[WebSearch] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        href = values.get("href")
        if tag == "a" and values.get("class") == "chip" and href is not None:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._href is None:
            return
        words = "".join(self._text).split()
        query = " ".join(words)
        chip = WebSearch(query=query, url=self._href)
        self.chips.append(chip)
        self._href = None


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
