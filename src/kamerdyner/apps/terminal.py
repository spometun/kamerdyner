"""Terminal input for the console: what the user types while the assistant is busy is kept and
offered, already typed, in the next prompt."""

import contextlib
import os
import re
import readline
import select
import sys
import termios
import tty
from collections.abc import Generator


class TypeAhead:
    """Collects keystrokes typed while the program is busy and hands them to the next `ask`.

    While `collecting`, the terminal does not echo, so typing does not break into the output;
    what was typed shows up in the next prompt instead. Enter pressed meanwhile becomes a space:
    nothing is sent before the user has seen it. On a stdin that is not a terminal, nothing is
    collected.
    """

    def __init__(self) -> None:
        self._pending = ""

    @contextlib.contextmanager
    def collecting(self) -> Generator[None]:
        if not sys.stdin.isatty():
            yield
            return
        descriptor = sys.stdin.fileno()
        saved = termios.tcgetattr(descriptor)
        tty.setcbreak(descriptor)
        try:
            yield
            raw = _read_available(descriptor)
        finally:
            # TCSADRAIN keeps unread input, unlike TCSAFLUSH.
            termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
        text = raw.decode("utf-8", errors="replace")
        self._pending = apply_keystrokes(self._pending, text)

    def ask(self, prompt: str) -> str:
        """`input(prompt)`, with what was collected already typed in."""
        prefill = self._pending
        self._pending = ""
        readline.set_startup_hook(lambda: readline.insert_text(prefill))
        try:
            return input(prompt)
        finally:
            readline.set_startup_hook()


def apply_keystrokes(line: str, keys: str) -> str:
    """The line after typing `keys` at its end: Backspace erases, Enter becomes a space, escape
    sequences (arrows and the like) and other control characters are dropped."""
    plain = _ESCAPE_SEQUENCE.sub("", keys)
    characters = list(line)
    for key in plain:
        if key in _ERASE:
            if characters:
                characters.pop()
        elif key in _ENTER:
            characters.append(" ")
        elif key.isprintable():
            characters.append(key)
    return "".join(characters)


_ESCAPE_SEQUENCE = re.compile(r"\x1b(\[[0-9;?]*[ -/]*[@-~]|O.|.)?")
_ERASE = "\x7f\x08"
_ENTER = "\r\n"


def _read_available(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        ready, _, _ = select.select([descriptor], [], [], 0)
        if not ready:
            break
        chunk = os.read(descriptor, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)
