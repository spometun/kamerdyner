"""The simplest compression: a stretch of conversation in, the assistant's notes on it out.

The notes are written by the assistant in the first person, in the language of the
conversation, and keep both sides: what the user told and how (the main focus) and what the
assistant itself said, advised and promised. They open with `span_heading`, written by the
model itself. Their length follows the substance of the conversation; nothing caps it.
"""

from collections.abc import Sequence

from kamerdyner.llm import LLM, GenerateResult, Prompt, Turn
from kamerdyner.messages import Message, Role
from kamerdyner.prompt import format_time
from kamerdyner.users import User


async def summarize(llm: LLM, user: User, messages: Sequence[Message]) -> GenerateResult:
    """Asks the model for notes on `messages`; a full reply carries them as its text.

    `messages` must not be empty.
    """
    if not messages:
        raise ValueError(f"Invalid input. Nothing to summarize for user {user.id!r}")
    instruction = _INSTRUCTION.format(name=user.name)
    conversation = transcript(user, messages)
    turn = Turn(role=Role.USER, text=conversation)
    prompt = Prompt(system_instruction=instruction, turns=(turn,))
    return await llm.generate(prompt)


def transcript(user: User, messages: Sequence[Message]) -> str:
    """The messages as the summarizing model reads them: one per paragraph, each opening with
    its local time and its author, `[2026-10-04 Sun 14:32] Марта: ...`."""
    lines: list[str] = []
    for message in messages:
        stamp = format_time(message.at, user.timezone)
        author = _author(message.role, user)
        lines.append(f"[{stamp}] {author}: {message.text}")
    return "\n\n".join(lines)


_ASSISTANT = "Kamerdyner"

_INSTRUCTION = """\
You are Kamerdyner, the personal assistant of {name}. The user message holds a stretch of \
your conversation with {name}. Each message opens with its local time, \
[YYYY-MM-DD Day HH:MM], and its author: {name}, or Kamerdyner — that is you.

This stretch is leaving your view. The notes you write now are all you will keep of it: \
later you will read them before talking to {name} again, without the conversation itself. \
Write them for yourself, in the first person — you are "I", {name} is called by name.

The notes are above all about {name}: what you learned about them and their life that will \
help you know them and help them later. Most of the notes should be about {name}. Keep what \
you yourself said brief — what you need to stay consistent with yourself and keep your \
word.

Open the notes with a heading line holding the time of the first and of the last message of \
the stretch, copied exactly as they stand in the conversation:
## [2026-10-04 Sun 14:32] — [2026-10-06 Tue 09:15]

Rules:
- Turn relative times into dates: "tomorrow" said on 2026-10-04 is 2026-10-05. Keep the time \
of day when it matters.
- Note plans and promises with their dates, {name}'s and your own.
- Quote a short phrase of {name}'s when how they said it matters.
- Keep what is worth remembering later. Skip greetings and empty small talk. For general \
answers that can be given again (a recipe, how something works), note that you gave it and \
what for, not its content.
- Never add anything the conversation does not say.
- Write in the language of the conversation.
- Let the length follow the substance: a short exchange gets a line or two, a rich one more.
- Below the heading, short bullet points grouped as you see fit, with no further headings.
- Reply with the notes only."""


def span_heading(user: User, messages: Sequence[Message]) -> str:
    """The heading the notes on `messages` must open with: the local times of the first and of
    the last message, `## [2026-10-04 Sun 14:32] — [2026-10-06 Tue 09:15]`."""
    first = format_time(messages[0].at, user.timezone)
    last = format_time(messages[-1].at, user.timezone)
    return f"## [{first}] — [{last}]"


def _author(role: Role, user: User) -> str:
    match role:
        case Role.USER:
            return user.name
        case Role.MODEL:
            return _ASSISTANT
