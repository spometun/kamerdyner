import pytest
from fakes import make_message

from kamerdyner.compaction import (
    Compact,
    NoAction,
    ShareCompactionPolicy,
    SizeThresholdPolicy,
    split_oldest,
)
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

    assert policy.plan(messages) == NoAction()


def test_empty_conversation_keeps_everything() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=1000, min_compress_tokens=500)

    assert policy.plan([]) == NoAction()


def test_at_limit_compresses_at_least_the_minimum_and_keeps_the_rest() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=16_000, min_compress_tokens=8_000)
    messages = _dialogue(*[1000] * 16)

    plan = policy.plan(messages)

    assert plan == Compact(to_compress=tuple(messages[:8]), to_keep=tuple(messages[8:]))


def test_cut_extends_to_the_start_of_the_next_exchange() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=6000, min_compress_tokens=3000)
    messages = _dialogue(*[1000] * 6)

    plan = policy.plan(messages)

    # The minimum is reached after the third message, a user's; their reply goes along.
    assert plan == Compact(to_compress=tuple(messages[:4]), to_keep=tuple(messages[4:]))


def test_compresses_everything_when_no_exchange_starts_after_the_minimum() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=6000, min_compress_tokens=3000)
    messages = _dialogue(1000, 20_000)

    plan = policy.plan(messages)

    assert plan == Compact(to_compress=tuple(messages), to_keep=())


@pytest.mark.parametrize(("limit", "minimum"), [(1000, 0), (1000, 1001), (0, 0)])
def test_rejects_inconsistent_sizes(limit: int, minimum: int) -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        SizeThresholdPolicy(_COUNTER, limit_tokens=limit, min_compress_tokens=minimum)


def test_cut_never_separates_an_unanswered_message_from_the_next_one() -> None:
    # The second user message went unanswered, so the third one shares its reply.
    roles = [Role.USER, Role.MODEL, Role.USER, Role.USER, Role.MODEL, Role.USER, Role.MODEL]
    messages = [make_message(role, "x" * 1000, minutes=index) for index, role in enumerate(roles)]

    plan = split_oldest(messages, _COUNTER, min_tokens=3000)

    assert plan == Compact(to_compress=tuple(messages[:5]), to_keep=tuple(messages[5:]))


def test_split_rejects_a_negative_minimum() -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        split_oldest([], _COUNTER, min_tokens=-1)


def test_share_policy_moves_at_least_the_share_whatever_the_size() -> None:
    policy = ShareCompactionPolicy(_COUNTER, share=0.5)
    messages = _dialogue(100, 100, 100, 100, 100, 100)

    plan = policy.plan(messages)

    # Half is 300 tokens, reached on a user message; the cut moves past its reply.
    assert plan == Compact(to_compress=tuple(messages[:4]), to_keep=tuple(messages[4:]))


def test_share_policy_leaves_an_empty_conversation_alone() -> None:
    policy = ShareCompactionPolicy(_COUNTER, share=0.5)

    assert policy.plan([]) == NoAction()


@pytest.mark.parametrize("share", [0, -0.5, 1.5])
def test_share_policy_rejects_a_share_outside_zero_to_one(share: float) -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        ShareCompactionPolicy(_COUNTER, share=share)
