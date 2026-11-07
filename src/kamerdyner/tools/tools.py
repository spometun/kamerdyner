"""The model's tools: which there are, running what it asked for, and telling it how that went.

Nothing here depends on the provider: a tool round is a `ModelTurn` and a `ToolTurn` with the calls
and their results, which each provider puts into its own form."""

import dataclasses
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from kamerdyner.common.llm import ModelReply, ModelTurn, Prompt, ToolCall, ToolResult, ToolTurn
from kamerdyner.tools.impl.outcome import ToolOutcome
from kamerdyner.tools.impl.set_timezone import SET_TIMEZONE, set_timezone
from kamerdyner.users import User

logger = logging.getLogger(__name__)

TOOLS = (SET_TIMEZONE,)


@dataclass(frozen=True)
class ToolsRun:
    """`results` hold one result per call, in the order of the calls; `user` is the user as the
    tools left them."""

    results: tuple[ToolResult, ...]
    user: User


def run_tools(calls: Sequence[ToolCall], user: User, directory: Path, now: datetime) -> ToolsRun:
    """Runs the model's calls in order for the user living in `directory`; `now` is the current
    time. A call out of contract gets its error told to the model."""
    results: list[ToolResult] = []
    for call in calls:
        outcome = _run_tool(call, user, directory, now)
        logger.info("Tool %s of user %s: %s", call, user.id, outcome.text)
        user = outcome.user
        result = ToolResult(call=call, text=outcome.text)
        results.append(result)
    return ToolsRun(results=tuple(results), user=user)


def with_tool_results(prompt: Prompt, reply: ModelReply, results: Sequence[ToolResult]) -> Prompt:
    """The request that follows `prompt` once the calls of `reply` have run: the same dialogue,
    then the model's turn with what it said and asked for, then a turn with `results`."""
    asked = ModelTurn(text=reply.text, provider_data=reply.provider_data)
    answered = ToolTurn(results=tuple(results))
    turns = (*prompt.turns, asked, answered)
    return dataclasses.replace(prompt, turns=turns)


def tools_record(results: Sequence[ToolResult]) -> str:
    """What the tools did, as it stands before the answer in the conversation."""
    lines: list[str] = []
    for result in results:
        request = _request_line(result.call)
        outcome = _result_line(result)
        lines.append(request)
        lines.append(outcome)
    lines.append("In the next request the model replied:\n")
    return "\n".join(lines)


def _run_tool(call: ToolCall, user: User, directory: Path, now: datetime) -> ToolOutcome:
    match call:
        case ToolCall(name=SET_TIMEZONE.name, arguments={"zone": str() as zone}):
            return set_timezone(zone, user, directory, now)
        case _:
            logger.warning("Rejected a tool call of user %s: %s", user.id, call)
            text = f"Error: no tool {call.name} taking arguments {dict(call.arguments)}."
            return ToolOutcome(text=text, user=user)


def _request_line(call: ToolCall) -> str:
    rendered = _render_call(call)
    return f"Model requested tool {rendered}."


def _result_line(result: ToolResult) -> str:
    return f"The system ran it, result: {result.text}"


def _render_call(call: ToolCall) -> str:
    arguments: list[str] = []
    for name, value in call.arguments.items():
        rendered = json.dumps(value, ensure_ascii=False)
        arguments.append(f"{name}={rendered}")
    joined = ", ".join(arguments)
    return f"{call.name}({joined})"
