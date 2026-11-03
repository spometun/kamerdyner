"""Text shaped for Telegram: cut to its message size, and the model's Markdown as the HTML subset
Telegram renders.

Telegram knows bold, italic, strikethrough, inline code, code blocks, links and quotes. What it
lacks is rewritten into plain lines: a heading becomes a bold line, a list item a line opening
with "•" or its number, a table row its cells joined by "|", a rule a line of dashes. Everything
else is text, escaped: raw HTML in the Markdown is shown as typed, never interpreted.
"""

import html
from collections.abc import Sequence

from markdown_it import MarkdownIt
from markdown_it.token import Token


def split_message(text: str, limit: int) -> list[str]:
    """`text` in pieces of at most `limit` characters, each cut at the last line break before
    the limit, else at the last space, else right at the limit."""
    if limit <= 0:
        raise ValueError(f"Invalid input. limit must be positive, got {limit}")
    pieces: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(rest[:cut])
        rest = rest[cut:].lstrip()
    pieces.append(rest)
    return pieces


def to_telegram_html(markdown: str) -> str:
    """`markdown` (CommonMark with strikethrough and tables) as Telegram HTML."""
    tokens = _PARSER.parse(markdown)
    renderer = _Renderer()
    renderer.blocks(tokens)
    return renderer.result()


_PARSER = MarkdownIt("commonmark", {"html": False}).enable(["strikethrough", "table"])

# Link targets Telegram accepts; a link to anything else is shown as its text alone.
_LINK_SCHEMES = ("http://", "https://", "tg://", "mailto:")

_INLINE_TAGS = {
    "strong_open": "<b>",
    "strong_close": "</b>",
    "em_open": "<i>",
    "em_close": "</i>",
    "s_open": "<s>",
    "s_close": "</s>",
}

_RULE = "——————"
_INDENT = "    "


class _Renderer:
    """Writes blocks separated by a blank line, list items and table rows by a line break.

    A separation is held back until the next content arrives, so nothing trails the text and
    nothing is put between a container's opening (a list item's marker, a quote's tag) and its
    first block.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._gap = 0
        self._just_opened = False
        # One entry per open list: whether it is ordered.
        self._lists: list[bool] = []
        self._first_cell = False
        self._link_open: list[bool] = []

    def result(self) -> str:
        return "".join(self._parts)

    def blocks(self, tokens: Sequence[Token]) -> None:
        for token in tokens:
            self._block(token)

    def _block(self, token: Token) -> None:
        match token.type:
            case "paragraph_open":
                if not token.hidden:
                    self._separate(2)
            case "paragraph_close":
                pass
            case "heading_open":
                self._separate(2)
                self._emit("<b>")
            case "heading_close":
                self._close("</b>")
            case "bullet_list_open" | "ordered_list_open":
                if not self._lists:
                    self._separate(2)
                ordered = token.type == "ordered_list_open"
                self._lists.append(ordered)
            case "bullet_list_close" | "ordered_list_close":
                self._lists.pop()
            case "list_item_open":
                self._separate(1)
                depth = len(self._lists)
                ordered = self._lists[-1]
                marker = f"{token.info}." if ordered else "•"
                indent = _INDENT * (depth - 1)
                self._emit(f"{indent}{marker} ")
                self._just_opened = True
            case "list_item_close":
                pass
            case "blockquote_open":
                self._separate(2)
                self._emit("<blockquote>")
                self._just_opened = True
            case "blockquote_close":
                self._close("</blockquote>")
            case "fence" | "code_block":
                self._separate(2)
                self._code_block(token)
            case "hr":
                self._separate(2)
                self._emit(_RULE)
            case "table_open":
                self._separate(2)
            case "tr_open":
                self._separate(1)
                self._first_cell = True
            case "th_open" | "td_open":
                if not self._first_cell:
                    self._emit(" | ")
                self._first_cell = False
                if token.type == "th_open":
                    self._emit("<b>")
            case "th_close":
                self._close("</b>")
            case "table_close" | "thead_open" | "thead_close" | "tbody_open" | "tbody_close":
                pass
            case "tr_close" | "td_close":
                pass
            case "inline":
                assert token.children is not None, (
                    f"Internal error. An inline token without children: {token!r}"
                )
                for child in token.children:
                    self._inline(child)
            case "html_block":
                escaped = html.escape(token.content.rstrip("\n"), quote=False)
                self._separate(2)
                self._emit(escaped)
            case _:
                raise AssertionError(f"Internal error. Unexpected Markdown block {token.type!r}")

    def _inline(self, token: Token) -> None:
        tag = _INLINE_TAGS.get(token.type)
        if tag is not None:
            self._emit(tag)
            return
        match token.type:
            case "text" | "html_inline":
                escaped = html.escape(token.content, quote=False)
                self._emit(escaped)
            case "softbreak" | "hardbreak":
                self._emit("\n")
            case "code_inline":
                escaped = html.escape(token.content, quote=False)
                self._emit(f"<code>{escaped}</code>")
            case "link_open":
                href = token.attrGet("href")
                assert isinstance(href, str), f"Internal error. A link without a target: {token!r}"
                usable = href.startswith(_LINK_SCHEMES)
                self._link_open.append(usable)
                if usable:
                    escaped = html.escape(href, quote=True)
                    self._emit(f'<a href="{escaped}">')
            case "link_close":
                usable = self._link_open.pop()
                if usable:
                    self._emit("</a>")
            case "image":
                # The alternative text stands for the picture Telegram cannot show inline.
                assert token.children is not None, (
                    f"Internal error. An image token without children: {token!r}"
                )
                for child in token.children:
                    self._inline(child)
            case _:
                raise AssertionError(f"Internal error. Unexpected Markdown inline {token.type!r}")

    def _code_block(self, token: Token) -> None:
        code = token.content.rstrip("\n")
        escaped = html.escape(code, quote=False)
        language = token.info.strip().split(" ", 1)[0]
        if language:
            attribute = html.escape(language, quote=True)
            self._emit(f'<pre><code class="language-{attribute}">{escaped}</code></pre>')
        else:
            self._emit(f"<pre>{escaped}</pre>")

    def _separate(self, newlines: int) -> None:
        self._gap = max(self._gap, newlines)

    def _emit(self, text: str) -> None:
        if self._parts and not self._just_opened:
            self._parts.append("\n" * self._gap)
        self._gap = 0
        self._just_opened = False
        self._parts.append(text)

    def _close(self, tag: str) -> None:
        """Closes a tag right after its content, before any separation that is held back."""
        self._parts.append(tag)
