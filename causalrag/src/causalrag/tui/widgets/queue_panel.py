"""QueuePanel — live view of the top-K candidate experiments.

Listens for ``LoopEvent(kind="plan")`` payloads produced by ``master_loop``
and renders the top-5 candidates with score-colored rows. Once a candidate
completes (``LoopEvent(kind="card")`` payload with matching ``candidate_id``
or matching ``hypothesis_id``), its row is struck through.

The panel is intentionally append-only over plan events — it always shows
the *latest* plan snapshot.
"""

from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


def _score_color(score: float) -> str:
    """green > 0.7, yellow > 0.5, red ≤ 0.5."""
    if score > 0.7:
        return "#7ed2e6"  # green/cyan
    if score > 0.5:
        return "#a3b6da"  # yellow/azure
    return "#e08877"  # red


def _candidate_row(c: dict[str, Any], completed: bool) -> tuple[Text, Text, Text, Text, Text]:
    score = float(c.get("score", 0.0))
    color = _score_color(score)
    style_extra = " strike" if completed else ""
    score_t = Text(f"{score:+.2f}", style=f"{color}{style_extra} bold")
    cid_t = Text(str(c.get("id") or c.get("candidate_id") or "?"), style=f"#9ec2ff{style_extra}")
    arrow = Text(
        f"{c.get('treatment', '?')} → {c.get('outcome', '?')}",
        style=f"#cfd6e4{style_extra}",
    )
    estimand_t = Text(str(c.get("estimand_class") or c.get("estimand") or "?"), style=f"#cfd6e4{style_extra}")
    method_t = Text(str(c.get("method") or c.get("recommended_method") or "(auto)"), style=f"#9aa3b5{style_extra}")
    return score_t, cid_t, arrow, estimand_t, method_t


class QueuePanel(Static):
    """Top-5 candidate queue, refreshed on every plan event."""

    DEFAULT_CSS = """
    QueuePanel {
        height: auto;
        max-height: 12;
        padding: 0 1;
        border: round #4d5773;
    }
    """

    candidates: reactive[tuple[dict[str, Any], ...]] = reactive(())
    completed_ids: reactive[frozenset[str]] = reactive(frozenset())

    def __init__(self) -> None:
        super().__init__("", markup=False)
        self._candidates: list[dict[str, Any]] = []
        self._completed_ids: set[str] = set()

    def on_mount(self) -> None:
        self._refresh()

    # --- Public API ------------------------------------------------------

    def update_panel(self, payload: dict[str, Any]) -> None:
        """Apply a ``LoopEvent`` payload.

        ``plan`` events carry ``payload["top"]`` — a ranked list of
        candidates. ``card`` events carry a result row whose
        ``candidate_id``/``id`` we mark as completed (strike-through).
        """
        if not isinstance(payload, dict):
            return
        top = payload.get("top")
        if isinstance(top, list) and top:
            self._candidates = list(top)[:5]
        # Mark completed candidates
        cid = payload.get("candidate_id") or payload.get("id")
        if cid:
            self._completed_ids.add(str(cid))
        self._refresh()

    # Alias matching the spec wording (`app.queue_panel.update(payload)`).
    # Textual's Static.update accepts a renderable; we override here so
    # callers may pass a payload dict OR a renderable for compatibility.
    def update(  # type: ignore[override]
        self, renderable: RenderableType | dict[str, Any] | None = None
    ) -> None:
        if isinstance(renderable, dict):
            self.update_panel(renderable)
            return
        super().update(renderable if renderable is not None else "")

    def render_panel(self) -> RenderableType:
        if not self._candidates:
            return Text(
                "queue · waiting for planner …",
                style="#4d5773",
            )
        t = Table(
            show_header=True,
            show_edge=False,
            box=None,
            pad_edge=False,
            header_style="#4d5773",
            show_lines=False,
            expand=False,
        )
        t.add_column("score", justify="right")
        t.add_column("id", justify="left")
        t.add_column("T → Y", justify="left")
        t.add_column("estimand", justify="left")
        t.add_column("method", justify="left")
        for c in self._candidates:
            cid = str(c.get("id") or c.get("candidate_id") or "")
            completed = cid in self._completed_ids
            row = _candidate_row(c, completed=completed)
            t.add_row(*row)
        return t

    # --- Internal --------------------------------------------------------

    def _refresh(self) -> None:
        super().update(self.render_panel())


__all__ = ["QueuePanel"]
