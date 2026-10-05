"""Assembling the request to the model from the user, their memory and the recent conversation."""

from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from kamerdyner.llm import Prompt, Turn
from kamerdyner.messages import Message, Role
from kamerdyner.users import User


def build_prompt(user: User, memory: str, messages: Sequence[Message]) -> Prompt:
    """The system instruction carries the stable part (who the user is, what is remembered about
    them) and comes first; the recent messages follow. Each user message is prefixed with its
    local time, and consecutive messages of one role are merged into one turn.

    `messages` must end with a user message.
    """
    system_instruction = _system_instruction(user, memory)
    turns = _turns(messages, user.timezone)
    return Prompt(system_instruction=system_instruction, turns=turns)


def format_time(at: datetime, timezone: ZoneInfo) -> str:
    """`2026-10-04 Sun 14:32` in the given timezone."""
    local = at.astimezone(timezone)
    weekday = _WEEKDAYS[local.weekday()]
    return f"{local:%Y-%m-%d} {weekday} {local:%H:%M}"


_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

_INSTRUCTION = """\
You are Kamerdyner, the personal assistant of {name}. Help with anything they ask, from \
everyday questions to personal matters: be warm, direct and concise, and reply in the \
language they write in.

Every message from {name} starts with a timestamp in square brackets, [YYYY-MM-DD Day HH:MM], \
in their local time. The system adds it; {name} does not type it. The timestamp of the latest \
message is the current time. Use the timestamps to understand time ("yesterday", "on Monday", \
how long ago something was said), but never write timestamps in your replies."""

_MEMORY_SECTION = """

# What you remember about {name}

{memory}"""


def _system_instruction(user: User, memory: str) -> str:
    instruction = _INSTRUCTION.format(name=user.name)
    if memory:
        instruction += _MEMORY_SECTION.format(name=user.name, memory=memory)
    return instruction


def _turns(messages: Sequence[Message], timezone: ZoneInfo) -> tuple[Turn, ...]:
    turns: list[Turn] = []
    for message in messages:
        text = _render(message, timezone)
        if turns and turns[-1].role is message.role:
            merged = Turn(role=message.role, text=f"{turns[-1].text}\n\n{text}")
            turns[-1] = merged
        else:
            turn = Turn(role=message.role, text=text)
            turns.append(turn)
    return tuple(turns)


def _render(message: Message, timezone: ZoneInfo) -> str:
    match message.role:
        case Role.USER:
            stamp = format_time(message.at, timezone)
            return f"[{stamp}] {message.text}"
        case Role.MODEL:
            return message.text
