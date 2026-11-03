import pytest

from kamerdyner.apps.telegram.impl.formatting import to_telegram_html


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        ("Просто текст.", "Просто текст."),
        (
            "**жирний**, *курсив*, _теж курсив_, ~~закреслений~~",
            "<b>жирний</b>, <i>курсив</i>, <i>теж курсив</i>, <s>закреслений</s>",
        ),
        ("виклик `f(x)`", "виклик <code>f(x)</code>"),
        (
            "[сайт](https://example.com/a?b=1&c=2)",
            '<a href="https://example.com/a?b=1&amp;c=2">сайт</a>',
        ),
        ("рядок\nдругий", "рядок\nдругий"),
    ],
)
def test_inline_markup(markdown: str, expected: str) -> None:
    assert to_telegram_html(markdown) == expected


def test_text_that_looks_like_html_is_escaped_not_interpreted() -> None:
    markdown = "a < b && <b>не тег</b>"

    assert to_telegram_html(markdown) == "a &lt; b &amp;&amp; &lt;b&gt;не тег&lt;/b&gt;"


def test_a_link_telegram_cannot_open_keeps_only_its_text() -> None:
    assert to_telegram_html("[тут](/relative/path)") == "тут"


def test_blocks_are_separated_by_a_blank_line_and_headings_are_bold() -> None:
    markdown = "# Заголовок\n\nПерший абзац.\n\nДругий абзац.\n\n---\n\nПісля риски."

    assert to_telegram_html(markdown) == (
        "<b>Заголовок</b>\n\nПерший абзац.\n\nДругий абзац.\n\n——————\n\nПісля риски."
    )


def test_lists_become_lines_with_markers_nested_ones_indented() -> None:
    markdown = "Список:\n\n- один\n- два\n  1. перше\n  2. друге\n- три\n\nКінець."

    assert to_telegram_html(markdown) == (
        "Список:\n\n• один\n• два\n    1. перше\n    2. друге\n• три\n\nКінець."
    )


def test_an_ordered_list_keeps_its_numbers() -> None:
    assert to_telegram_html("3. три\n4. **чотири**") == "3. три\n4. <b>чотири</b>"


def test_a_loose_list_is_as_compact_as_a_tight_one() -> None:
    assert to_telegram_html("- один\n\n- два") == "• один\n• два"


def test_code_blocks_keep_their_language_and_escape_their_content() -> None:
    markdown = "```python\nif a < b:\n    pass\n```\n\n    відступ"

    assert to_telegram_html(markdown) == (
        '<pre><code class="language-python">if a &lt; b:\n    pass</code></pre>\n\n'
        "<pre>відступ</pre>"
    )


def test_a_quote_is_a_blockquote() -> None:
    assert to_telegram_html("> цитата\n> далі\n\nпісля") == (
        "<blockquote>цитата\nдалі</blockquote>\n\nпісля"
    )


def test_a_table_becomes_rows_of_cells_with_a_bold_header() -> None:
    markdown = "| Хто | Скільки |\n|---|---|\n| Оля | 2 |\n| Сергій | 3 |"

    assert to_telegram_html(markdown) == "<b>Хто</b> | <b>Скільки</b>\nОля | 2\nСергій | 3"


def test_an_image_shows_its_alternative_text() -> None:
    assert to_telegram_html("![кіт на дивані](https://example.com/cat.png)") == "кіт на дивані"


def test_empty_text_stays_empty() -> None:
    assert to_telegram_html("") == ""
