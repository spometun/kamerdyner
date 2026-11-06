"""What running one tool comes to."""

from dataclasses import dataclass

from kamerdyner.users import User


@dataclass(frozen=True)
class ToolOutcome:
    """`text` tells the model what happened, a failure included; `user` is the user as the tool
    left them."""

    text: str
    user: User
