"""The tool that changes the user's time zone."""

import dataclasses
from datetime import datetime
from pathlib import Path

from kamerdyner.common.llm import ToolSpec
from kamerdyner.common.messages import format_time_with_offset
from kamerdyner.tools.impl.outcome import ToolOutcome
from kamerdyner.users import User, find_timezone, save_timezone

SET_TIMEZONE = ToolSpec(
    name="set_timezone",
    description=(
        "Sets the user's time zone, which the timestamps of their messages follow from now on. "
        "Call it only when the user plainly says they moved, are travelling or want it changed; "
        "never on a guess."
    ),
    parameters={
        "type": "object",
        "properties": {
            "zone": {"type": "string", "description": "IANA time zone name, e.g. Europe/Kyiv"},
        },
        "required": ["zone"],
    },
)


def set_timezone(zone: str, user: User, directory: Path, now: datetime) -> ToolOutcome:
    """Moves the user in `directory` to `zone`, a name from the model; an unknown one changes
    nothing."""
    timezone = find_timezone(zone)
    if timezone is None:
        text = f'Error: unknown time zone "{zone}". Use an IANA name such as Europe/Kyiv.'
        return ToolOutcome(text=text, user=user)
    save_timezone(directory, timezone)
    moved = dataclasses.replace(user, timezone=timezone)
    local_now = now.astimezone(timezone)
    stamp = format_time_with_offset(local_now)
    text = f"Time zone set to {zone}. Local time is now {stamp}."
    return ToolOutcome(text=text, user=moved)
