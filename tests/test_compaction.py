import pytest
from fakes import make_message

from kamerdyner.compaction import (
    Compact,
    NoAction,
    ShareCompactionPolicy,
    SizeThresholdPolicy,
    split_newest,
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
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=1000, keep_tokens=500)
    messages = _dialogue(300, 300, 300)

    assert policy.plan(messages) == NoAction()


def test_empty_conversation_keeps_everything() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=1000, keep_tokens=500)

    assert policy.plan([]) == NoAction()


def test_at_limit_compresses_all_but_the_newest_kept_tokens() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=4000, keep_tokens=2000)
    messages = _dialogue(*[500] * 8)

    plan = policy.plan(messages)

    assert plan == Compact(to_compress=tuple(messages[:4]), to_keep=tuple(messages[4:]))


def test_keeps_more_than_the_minimum_to_cut_right_after_a_reply() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=4000, keep_tokens=2000)
    messages = _dialogue(*[600] * 7)

    plan = policy.plan(messages)

    # Keeping 2000 needs the last four messages (2400), which start with a model reply; the
    # cut moves back past the user message before it, keeping five (3000).
    assert plan == Compact(to_compress=tuple(messages[:2]), to_keep=tuple(messages[2:]))


def test_a_huge_exchange_waits_until_enough_newer_messages_follow_it() -> None:
    policy = SizeThresholdPolicy(_COUNTER, limit_tokens=4000, keep_tokens=2000)
    messages = _dialogue(3500, 300, 100, 200)

    # Over the limit, but cutting after either reply would keep less than 2000.
    assert policy.plan(messages) == NoAction()

    messages += _dialogue(900, 900)
    plan = policy.plan(messages)

    # 100 + 200 + 900 + 900 = 2100 now follow the huge exchange, so it can go.
    assert plan == Compact(to_compress=tuple(messages[:2]), to_keep=tuple(messages[2:]))


@pytest.mark.parametrize(("limit", "keep"), [(1000, 0), (1000, 1000), (0, 0)])
def test_rejects_inconsistent_sizes(limit: int, keep: int) -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        SizeThresholdPolicy(_COUNTER, limit_tokens=limit, keep_tokens=keep)


def test_split_newest_rejects_a_negative_keep() -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        split_newest([], _COUNTER, keep_tokens=-1)


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


def test_share_policy_on_an_empty_conversation_moves_nothing() -> None:
    policy = ShareCompactionPolicy(_COUNTER, share=0.5)

    assert policy.plan([]) == Compact(to_compress=(), to_keep=())


@pytest.mark.parametrize("share", [0, -0.5, 1.5])
def test_share_policy_rejects_a_share_outside_zero_to_one(share: float) -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        ShareCompactionPolicy(_COUNTER, share=share)
