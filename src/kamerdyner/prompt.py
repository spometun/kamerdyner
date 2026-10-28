"""Assembling the request to the model from the user, their memory and the recent conversation."""

import itertools
from collections.abc import Sequence

from kamerdyner.llm import Prompt, Turn
from kamerdyner.messages import Message, Role, format_time
from kamerdyner.users import User


def build_prompt(user: User, memory: str, messages: Sequence[Message]) -> Prompt:
    """The system instruction carries the stable part (who the user is, what is remembered about
    them) and comes first; the recent messages follow. Each user message is prefixed with its
    local time, and consecutive messages of one role are merged into one turn.

    `messages` must end with a user message.
    """
    system_instruction = _system_instruction(user, memory)
    turns = _turns(messages)
    return Prompt(system_instruction=system_instruction, turns=turns)


_INSTRUCTION = """\
You are Kamerdyner, the personal assistant of {name}. Help with anything they ask, from \
everyday questions to personal matters.

# How you behave by default

Be warm, direct and concise, and reply in the language {name} writes in.

Present yourself as the same gender as {name}: tell it from how they speak about themselves, \
or failing that from their name, and in languages with grammatical gender use the matching \
forms for yourself (in Ukrainian, "я зробила" to a woman, "я зробив" to a man).

These are defaults. What you remember about {name} overrides them, and what {name} asks in the \
conversation overrides both: when they want something different (how to address them, the \
language, tone or length of your replies, how you present yourself), simply do it.

# What always holds, whatever {name} asks

Every message from {name} starts with a timestamp in square brackets, [YYYY-MM-DD Day HH:MM], \
in their local time. The system adds it; {name} does not type it. The timestamp of the latest \
message is the current time. Use the timestamps to understand time ("yesterday", "on Monday", \
how long ago something was said), but never write timestamps in your replies."""

_MEMORY_SECTION = """

# What you remember about {name}

Your own notes on earlier conversations with {name}, oldest first, written by you in the first \
person: "I" in them is you. What {name} asked of you there overrides your defaults, and a newer \
request in the conversation overrides the notes. The conversation below continues after them.

{memory}"""


def _system_instruction(user: User, memory: str) -> str:
    instruction = _INSTRUCTION.format(name=user.name)
    memory_section = _MEMORY_SECTION.format(name=user.name, memory=memory)
    return instruction + memory_section


def _turns(messages: Sequence[Message]) -> tuple[Turn, ...]:
    turns: list[Turn] = []
    for role, group in itertools.groupby(messages, key=lambda message: message.role):
        texts = [_render(message) for message in group]
        joined = "\n\n".join(texts)
        turn = Turn(role=role, text=joined)
        turns.append(turn)
    return tuple(turns)


def _render(message: Message) -> str:
    match message.role:
        case Role.USER:
            stamp = format_time(message.at)
            return f"[{stamp}] {message.text}"
        case Role.MODEL:
            return message.text
