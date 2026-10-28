import pytest
from fakes import make_message

from kamerdyner.common.llm import Role
from kamerdyner.common.messages import Message
from kamerdyner.common.tokens import ApproximateTokenCounter
from kamerdyner.memory.compaction import Compact, NoAction, split_newest

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


def test_empty_conversation_has_nothing_to_compact() -> None:
    assert split_newest([], _COUNTER, keep_tokens=500) == NoAction()


def test_compresses_all_but_the_newest_kept_tokens() -> None:
    messages = _dialogue(*[500] * 8)

    plan = split_newest(messages, _COUNTER, keep_tokens=2000)

    assert plan == Compact(to_compress=tuple(messages[:4]), to_keep=tuple(messages[4:]))


def test_keeping_nothing_compresses_everything() -> None:
    messages = _dialogue(100, 100, 100, 100)

    plan = split_newest(messages, _COUNTER, keep_tokens=0)

    assert plan == Compact(to_compress=tuple(messages), to_keep=())


def test_keeps_more_than_the_minimum_to_cut_right_after_a_reply() -> None:
    messages = _dialogue(*[600] * 8)

    plan = split_newest(messages, _COUNTER, keep_tokens=2000)

    # Keeping 2000 needs the last four messages (2400), which start with a user message right
    # after a reply, so the cut falls there.
    assert plan == Compact(to_compress=tuple(messages[:4]), to_keep=tuple(messages[4:]))

    plan = split_newest(messages, _COUNTER, keep_tokens=2500)

    # Five messages would do (3000), but they start with a reply; the cut moves back past the
    # user message before it, keeping six (3600).
    assert plan == Compact(to_compress=tuple(messages[:2]), to_keep=tuple(messages[2:]))


def test_a_huge_exchange_waits_until_enough_newer_messages_follow_it() -> None:
    messages = _dialogue(3500, 300, 100, 200)

    # Cutting after either reply would keep less than 2000.
    assert split_newest(messages, _COUNTER, keep_tokens=2000) == NoAction()

    messages += _dialogue(900, 900)
    plan = split_newest(messages, _COUNTER, keep_tokens=2000)

    # 100 + 200 + 900 + 900 = 2100 now follow the huge exchange, so it can go.
    assert plan == Compact(to_compress=tuple(messages[:2]), to_keep=tuple(messages[2:]))


def test_cut_never_separates_messages_sharing_one_reply() -> None:
    # The second user message went unanswered, so the third one shares its reply.
    roles = [Role.USER, Role.MODEL, Role.USER, Role.USER, Role.MODEL, Role.USER, Role.MODEL]
    messages = [make_message(role, "x" * 1000, minutes=index) for index, role in enumerate(roles)]

    plan = split_newest(messages, _COUNTER, keep_tokens=2500)

    assert plan == Compact(to_compress=tuple(messages[:2]), to_keep=tuple(messages[2:]))


def test_rejects_a_negative_keep() -> None:
    with pytest.raises(ValueError, match="Invalid input"):
        split_newest([], _COUNTER, keep_tokens=-1)
