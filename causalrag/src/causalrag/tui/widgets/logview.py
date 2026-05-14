"""LogView — scrollable container that hosts the streaming log + cards.

Provides a small append-API so command runners can post to it from any task:

    log.line("Loaded 12,480 × 12", kind="ok", gutter="✓")
    log.card(title="Profile", body=table)
    log.echo("/discover data/cohort.parquet")
"""

from __future__ import annotations

from rich.console import RenderableType
from textual.containers import VerticalScroll
from textual.widget import Widget

from causalrag.tui.widgets.cards import CardWidget, CmdEcho, LogLine


class LogView(VerticalScroll):
    """A scroll container that auto-pins to the bottom when new widgets land."""

    DEFAULT_CSS = ""

    def __init__(self) -> None:
        super().__init__()
        self.can_focus = False

    # --- Public append API ------------------------------------------------

    def line(
        self,
        body: RenderableType | str,
        kind: str = "",
        gutter: str = "·",
    ) -> None:
        self._mount_and_scroll(LogLine(body, kind=kind, gutter=gutter))

    def echo(self, cmd: str) -> None:
        self._mount_and_scroll(CmdEcho(cmd))

    def card(
        self,
        title: str,
        body: RenderableType,
        step: str | None = None,
        meta: str | None = None,
    ) -> None:
        self._mount_and_scroll(CardWidget(title=title, step=step, meta=meta, body=body))

    def banner(self, widget: Widget) -> None:
        self._mount_and_scroll(widget)

    def clear_log(self) -> None:
        for child in list(self.children):
            child.remove()

    # --- Internal ---------------------------------------------------------

    def _mount_and_scroll(self, widget: Widget) -> None:
        self.mount(widget)
        # Defer the scroll until after layout settles.
        self.call_after_refresh(self.scroll_end, animate=False)


__all__ = ["LogView"]
