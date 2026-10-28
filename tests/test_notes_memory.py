from pathlib import Path

import pytest
from fakes import ScriptedLLM, make_message

from kamerdyner.common.llm import GenerateResult, Incomplete, Prompt, Reply, Role, Unavailable
from kamerdyner.common.messages import Message
from kamerdyner.memory import Compacted, NotAbsorbed, NothingToCompact
from kamerdyner.memory.notes import NotesMemory

_HEADING = "## [2026-10-04 Sun 14:32] — [2026-10-04 Sun 14:33]"


def _exchange(minutes: int = 0, size: int = 0) -> list[Message]:
    """A question and its answer; `size` pads each to that many characters of ASCII."""
    question = make_message(Role.USER, "Завтра до лікаря".ljust(size, "."), minutes=minutes)
    answer = make_message(Role.MODEL, "Нагадаю зранку.".ljust(size, "."), minutes=minutes + 1)
    return [question, answer]


def _memory(directory: Path, results: list[GenerateResult]) -> tuple[NotesMemory, ScriptedLLM]:
    llm = ScriptedLLM(results)
    memory = NotesMemory(directory, "Марта", llm)
    return memory, llm


def _journal(directory: Path) -> str:
    return (directory / "memory.log").read_text(encoding="utf-8")


def test_new_memory_is_empty(tmp_path: Path) -> None:
    memory, _ = _memory(tmp_path, [])

    assert memory.conversation == ()
    assert memory.content() == ""
    assert memory.conversation_tokens() == 0


def test_conversation_survives_a_restart(tmp_path: Path) -> None:
    memory, _ = _memory(tmp_path, [])
    exchange = _exchange()

    memory.add(exchange)
    reopened, _ = _memory(tmp_path, [])

    assert reopened.conversation == tuple(exchange)


@pytest.mark.parametrize(
    "roles",
    [
        [],
        [Role.USER],
        [Role.MODEL],
        [Role.MODEL, Role.USER],
        [Role.USER, Role.MODEL, Role.MODEL],
    ],
)
def test_add_rejects_what_is_not_an_exchange(tmp_path: Path, roles: list[Role]) -> None:
    memory, _ = _memory(tmp_path, [])
    messages = [make_message(role, "x", minutes=index) for index, role in enumerate(roles)]

    with pytest.raises(ValueError, match="Invalid input"):
        memory.add(messages)


async def test_compacted_notes_are_the_content_and_survive_a_restart(tmp_path: Path) -> None:
    notes = f"{_HEADING}\n- Марта 2026-10-05 йде до лікаря; я пообіцяв нагадати зранку."
    memory, _ = _memory(tmp_path, [Reply(notes + "\n")])
    exchange = _exchange()
    memory.add(exchange)

    outcome = await memory.compact(0)

    assert outcome == Compacted(moved=2, kept=0)
    assert memory.content() == notes
    assert memory.conversation == ()
    reopened, _ = _memory(tmp_path, [])
    assert reopened.content() == notes
    assert reopened.conversation == ()
    journal = _journal(tmp_path)
    assert "absorbed 2 messages" in journal
    assert "[2026-10-04 Sun 14:32] Марта: Завтра до лікаря" in journal
    assert notes in journal
    assert "warning" not in journal


async def test_compact_keeps_the_newest_tokens(tmp_path: Path) -> None:
    memory, llm = _memory(tmp_path, [Reply(f"{_HEADING}\n- перше")])
    # 400 characters of ASCII padding are about 100 tokens: each exchange weighs about 200.
    first = _exchange(minutes=0, size=400)
    second = _exchange(minutes=60, size=400)
    third = _exchange(minutes=120, size=400)
    for exchange in (first, second, third):
        memory.add(exchange)

    outcome = await memory.compact(300)

    assert outcome == Compacted(moved=2, kept=4)
    assert memory.conversation == (*second, *third)
    [prompt] = llm.prompts
    assert "14:32" in prompt.turns[0].text
    assert "15:32" not in prompt.turns[0].text


async def test_each_compaction_adds_an_entry_after_the_previous_ones(tmp_path: Path) -> None:
    second_heading = "## [2026-10-04 Sun 15:32] — [2026-10-04 Sun 15:33]"
    first = f"{_HEADING}\n- перше"
    second = f"{second_heading}\n- друге"
    memory, _ = _memory(tmp_path, [Reply(first), Reply(second)])

    memory.add(_exchange())
    await memory.compact(0)
    memory.add(_exchange(minutes=60))
    await memory.compact(0)

    assert memory.content() == f"{first}\n\n{second}"


async def test_nothing_to_compact_calls_no_model(tmp_path: Path) -> None:
    memory, llm = _memory(tmp_path, [])

    empty = await memory.compact(0)
    memory.add(_exchange())
    too_little = await memory.compact(10_000)

    assert empty == NothingToCompact()
    assert too_little == NothingToCompact()
    assert llm.prompts == []
    assert len(memory.conversation) == 2


async def test_notes_with_a_wrong_heading_are_kept_with_a_warning(tmp_path: Path) -> None:
    memory, _ = _memory(tmp_path, [Reply("- без заголовка")])
    memory.add(_exchange())

    outcome = await memory.compact(0)

    assert outcome == Compacted(moved=2, kept=0)
    assert memory.content() == "- без заголовка"
    journal = _journal(tmp_path)
    assert "--- warning ---" in journal
    assert _HEADING in journal


async def test_unavailable_model_leaves_memory_unchanged(tmp_path: Path) -> None:
    memory, _ = _memory(tmp_path, [Unavailable("503")])
    exchange = _exchange()
    memory.add(exchange)

    outcome = await memory.compact(0)

    assert outcome == NotAbsorbed("the model is unavailable: 503")
    assert memory.content() == ""
    assert memory.conversation == tuple(exchange)
    assert not (tmp_path / "notes.md").exists()
    journal = _journal(tmp_path)
    assert "could not absorb 2 messages: the model is unavailable: 503" in journal
    assert "Марта: Завтра до лікаря" in journal


async def test_incomplete_notes_are_not_kept_but_journaled(tmp_path: Path) -> None:
    memory, _ = _memory(tmp_path, [Incomplete("finish reason MAX_TOKENS", f"{_HEADING}\n- Лік")])
    memory.add(_exchange())

    outcome = await memory.compact(0)

    assert outcome == NotAbsorbed("the notes came back incomplete: finish reason MAX_TOKENS")
    assert memory.content() == ""
    assert len(memory.conversation) == 2
    assert f"--- partial notes ---\n{_HEADING}\n- Лік" in _journal(tmp_path)


class _AddingLLM:
    """Writes the notes, but first runs `meanwhile` — an exchange added while it writes."""

    def __init__(self, notes: str, meanwhile: list[Message], memory: list[NotesMemory]) -> None:
        self._notes = notes
        self._meanwhile = meanwhile
        self._memory = memory

    async def generate(self, prompt: Prompt) -> GenerateResult:
        self._memory[0].add(self._meanwhile)
        return Reply(self._notes)


async def test_an_exchange_added_during_compaction_stays(tmp_path: Path) -> None:
    later = _exchange(minutes=60)
    holder: list[NotesMemory] = []
    llm = _AddingLLM(f"{_HEADING}\n- перше", later, holder)
    memory = NotesMemory(tmp_path, "Марта", llm)
    holder.append(memory)
    memory.add(_exchange())

    await memory.compact(0)

    assert memory.conversation == tuple(later)
