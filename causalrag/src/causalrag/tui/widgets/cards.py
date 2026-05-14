"""Log lines, command echoes, and the embedded Card widget.

Mirrors the design's `L`, `CmdEcho`, and `tui-card` components. Each is a
self-contained Textual widget so the LogView can append them as the pipeline
streams output.
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.containers import Vertical
from textual.widgets import Static


LOG_KIND_STYLES: dict[str, tuple[str, str]] = {
    # name -> (gutter_color, body_color)
    "":      ("#4d5773", "#cfd6e4"),
    "muted": ("#4d5773", "#9aa3b5"),
    "dim":   ("#4d5773", "#6b7691"),
    "acc":   ("#5fa8ff", "#9ec2ff"),
    "ok":    ("#7ed2e6", "#7ed2e6"),
    "warn":  ("#a3b6da", "#a3b6da"),
    "err":   ("#e08877", "#e08877"),
    "you":   ("#5fa8ff", "#eef2f9"),
}


class LogLine(Static):
    """One streamed log line — small gutter glyph + body text."""

    DEFAULT_CSS = ""

    def __init__(
        self,
        body: RenderableType | str,
        kind: str = "",
        gutter: str = "·",
    ) -> None:
        gcolor, bcolor = LOG_KIND_STYLES.get(kind, LOG_KIND_STYLES[""])
        if isinstance(body, str):
            body_text = Text(body, style=bcolor)
        else:
            body_text = body
        renderable = Text()
        renderable.append(f" {gutter} ", style=gcolor)
        if isinstance(body_text, Text):
            renderable.append_text(body_text)
        else:
            renderable.append_text(Text.from_markup(str(body_text)))
        super().__init__(renderable, markup=False)


class CmdEcho(Static):
    """Echo of a user-submitted command line, with the slash highlighted."""

    DEFAULT_CSS = ""

    def __init__(self, cmd: str) -> None:
        text = Text()
        text.append("›  ", style="#5fa8ff")
        if cmd.startswith("/"):
            head, _, rest = cmd.partition(" ")
            text.append(head, style="#9ec2ff bold")
            if rest:
                text.append(" " + rest, style="#cfd6e4")
        else:
            text.append(cmd, style="#cfd6e4")
        super().__init__(text, markup=False)


class CardWidget(Vertical):
    """A bordered card with a dashed-rule header (step · title · meta) and an
    arbitrary renderable body. Used by every command that wants to show a
    table or formatted result."""

    DEFAULT_CSS = ""

    def __init__(
        self,
        title: str,
        step: str | None = None,
        meta: str | None = None,
        body: RenderableType | None = None,
    ) -> None:
        super().__init__()
        self._title = title
        self._step = step
        self._meta = meta
        self._body = body

    def compose(self):
        header = Text()
        if self._step:
            header.append(f"{self._step} ", style="#5fa8ff bold")
        header.append(self._title, style="#cfd6e4")
        if self._meta:
            header.append("    ")
            header.append(self._meta, style="#4d5773")
        yield Static(header, classes="card-header", markup=False)
        body = self._body if self._body is not None else Text("")
        yield Static(body, classes="card-body", markup=False)


def kv_table(rows: list[tuple[str, RenderableType | str]]) -> Table:
    """Two-column key/value table (Field | Value) styled like the design."""
    t = Table(show_header=False, show_edge=False, box=None, pad_edge=False)
    t.add_column("k", style="#9aa3b5", no_wrap=False)
    t.add_column("v", style="#cfd6e4")
    for k, v in rows:
        t.add_row(k, v)
    return t


def column_table(
    columns: list[tuple[str, str]],
    rows: list[tuple],
) -> Table:
    """Multi-column tabular display matching the design's `tbl` look —
    no row dividers, dashed header underline, monospace numerics.

    ``columns`` is a list of ``(label, justify)`` pairs where justify is one
    of ``"left" | "right" | "center"``.
    """
    t = Table(
        show_header=True,
        show_edge=False,
        box=None,
        pad_edge=False,
        header_style="#4d5773",
        show_lines=False,
        expand=False,
    )
    for label, justify in columns:
        t.add_column(label, justify=justify, style="#cfd6e4")
    for row in rows:
        t.add_row(*row)
    return t


__all__ = ["LogLine", "CmdEcho", "CardWidget", "kv_table", "column_table"]
