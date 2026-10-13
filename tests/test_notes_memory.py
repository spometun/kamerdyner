from pathlib import Path

from fakes import FakeClock, ScriptedLLM, make_message, make_user

from kamerdyner.llm import GenerateResult, Incomplete, Reply, Unavailable
from kamerdyner.memory import Absorbed, NotAbsorbed
from kamerdyner.memory.notes import NotesMemory
from kamerdyner.messages import Message, Role

_HEADING = "## [2026-10-04 Sun 14:32] — [2026-10-04 Sun 14:33]"


def _exchange(minutes: int = 0) -> list[Message]:
    question = make_message(Role.USER, "Завтра до лікаря", minutes=minutes)
    answer = make_message(Role.MODEL, "Нагадаю зранку.", minutes=minutes + 1)
    return [question, answer]


def _memory(directory: Path, results: list[GenerateResult]) -> tuple[NotesMemory, ScriptedLLM]:
    llm = ScriptedLLM(results)
    user = make_user()
    clock = FakeClock()
    memory = NotesMemory(directory, user, llm, clock)
    return memory, llm


def _journal(directory: Path) -> str:
    return (directory / "memory.log").read_text(encoding="utf-8")


async def test_absorbed_notes_are_the_content_and_survive_a_restart(tmp_path: Path) -> None:
    notes = f"{_HEADING}\n- Марта 2026-10-05 йде до лікаря; я пообіцяв нагадати зранку."
    memory, _ = _memory(tmp_path, [Reply(notes + "\n")])

    result = await memory.absorb(_exchange())

    assert result == Absorbed()
    assert memory.content() == notes
    reopened, _ = _memory(tmp_path, [])
    assert reopened.content() == notes
    journal = _journal(tmp_path)
    assert "absorbed 2 messages" in journal
    assert "[2026-10-04 Sun 14:32] Марта: Завтра до лікаря" in journal
    assert notes in journal
    assert "warning" not in journal


async def test_each_stretch_adds_an_entry_after_the_previous_ones(tmp_path: Path) -> None:
    second_heading = "## [2026-10-04 Sun 15:32] — [2026-10-04 Sun 15:33]"
    first = f"{_HEADING}\n- перше"
    second = f"{second_heading}\n- друге"
    memory, _ = _memory(tmp_path, [Reply(first), Reply(second)])

    await memory.absorb(_exchange())
    await memory.absorb(_exchange(minutes=60))

    assert memory.content() == f"{first}\n\n{second}"


async def test_notes_with_a_wrong_heading_are_kept_with_a_warning(tmp_path: Path) -> None:
    memory, _ = _memory(tmp_path, [Reply("- без заголовка")])

    result = await memory.absorb(_exchange())

    assert result == Absorbed()
    assert memory.content() == "- без заголовка"
    journal = _journal(tmp_path)
    assert "--- warning ---" in journal
    assert _HEADING in journal


async def test_unavailable_model_leaves_memory_unchanged(tmp_path: Path) -> None:
    memory, _ = _memory(tmp_path, [Unavailable("503")])

    result = await memory.absorb(_exchange())

    assert result == NotAbsorbed("the model is unavailable: 503")
    assert memory.content() == ""
    assert not (tmp_path / "notes.md").exists()
    journal = _journal(tmp_path)
    assert "could not absorb 2 messages: the model is unavailable: 503" in journal
    assert "Марта: Завтра до лікаря" in journal


async def test_incomplete_notes_are_not_kept_but_journaled(tmp_path: Path) -> None:
    memory, _ = _memory(tmp_path, [Incomplete("finish reason MAX_TOKENS", f"{_HEADING}\n- Лік")])

    result = await memory.absorb(_exchange())

    assert result == NotAbsorbed("the notes came back incomplete: finish reason MAX_TOKENS")
    assert memory.content() == ""
    assert f"--- partial notes ---\n{_HEADING}\n- Лік" in _journal(tmp_path)


async def test_nothing_to_absorb_calls_no_model(tmp_path: Path) -> None:
    memory, llm = _memory(tmp_path, [])

    result = await memory.absorb([])

    assert result == Absorbed()
    assert llm.prompts == []
    assert memory.content() == ""
