import pytest
from fakes import START, make_message

from kamerdyner.compaction import Compact, NoAction, SizeThresholdPolicy
from kamerdyner.messages import Message, Role
from kamerdyner.tokens import ApproximateTokenCounter

# One token per byte, so a message of n ASCII characters weighs n tokens.
_COUNTER = ApproximateTokenCounter(bytes_per_token=1)


def _dialogue(*sizes: int) -> list[Message]:
    """Alternating user and model messages, starting with the user, of the given token sizes."""
    messages: list[Message] = []
    for index, size in enumerate(sizes):
        role = Role.USER if index % 2 == 0 else Role.MODEL
        message = make_message(role, "x" * size, minutes=index)
        messages.append(message)
    return messages


def test_below_limit_keeps_everything() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=1000, min_compress_tokens=500)
    messages = _dialogue(300, 300, 300)

    assert policy.plan(messages, START) == NoAction()


def test_empty_conversation_keeps_everything() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=1000, min_compress_tokens=500)

    assert policy.plan([], START) == NoAction()


def test_at_limit_compresses_at_least_the_minimum_and_keeps_the_rest() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=16_000, min_compress_tokens=8_000)
    messages = _dialogue(*[1000] * 16)

    plan = policy.plan(messages, START)

    assert plan == Compact(to_compress=tuple(messages[:8]), to_keep=tuple(messages[8:]))


def test_cut_extends_to_the_start_of_the_next_exchange() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=6000, min_compress_tokens=3000)
    messages = _dialogue(*[1000] * 6)

    plan = policy.plan(messages, START)

    # The minimum is reached after the third message, a user's; their reply goes along.
    assert plan == Compact(to_compress=tuple(messages[:4]), to_keep=tuple(messages[4:]))


def test_compresses_everything_when_no_exchange_starts_after_the_minimum() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=6000, min_compress_tokens=3000)
    messages = _dialogue(1000, 20_000)

    plan = policy.plan(messages, START)

    assert plan == Compact(to_compress=tuple(messages), to_keep=())


@pytest.mark.parametrize(("limit", "minimum"), [(1000, 0), (1000, 1001), (0, 0)])
def test_rejects_inconsistent_sizes(limit: int, minimum: int) -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        SizeThresholdPolicy(_COUNTER, limit_tokens=limit, min_compress_tokens=minimum)
