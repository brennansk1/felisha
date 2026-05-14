"""Estimator leaderboard widget (Sprint 4.4).

When a single hypothesis is run under multiple estimators (the
"robustness" pattern — typical after a red sensitivity verdict
auto-fires a different-family swap), the user wants to see all of them
side-by-side: point, SE, CI, sensitivity chip, energy score, ERUPT.

AutoGluon-Tabular style — a sortable table the user can glance at to
spot disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rich.table import Table
from rich.text import Text
from textual.widgets import Static


@dataclass
class LeaderboardRow:
    """One estimator's result for a given hypothesis."""
    estimator_id: str
    point: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    sensitivity_verdict: str | None  # "green" / "yellow" / "red" / "unknown"
    energy_score: float | None = None  # lower better
    erupt: float | None = None  # higher better
    chain_id: str | None = None
    notes: str | None = None


def _verdict_chip(verdict: str | None) -> Text:
    if verdict is None:
        return Text("—", style="dim")
    glyph_map = {
        "green": ("●", "green"),
        "yellow": ("◐", "yellow"),
        "red": ("○", "red"),
        "errored": ("✗", "red"),
        "unknown": ("?", "dim"),
    }
    glyph, color = glyph_map.get(verdict.lower(), ("?", "dim"))
    return Text(f"{glyph} {verdict}", style=color)


def _format_signed(x: float | None, fmt: str = "+.4f") -> str:
    if x is None:
        return "—"
    return format(x, fmt)


def _format_ci(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    return f"[{low:+.3f}, {high:+.3f}]"


class LeaderboardPanel(Static):
    """Textual widget rendering a sortable estimator leaderboard.

    Update with a list of LeaderboardRow; the panel re-renders to a
    Rich Table sorted by the chosen column (default: energy_score
    ascending, with ties broken by absolute |point| descending).
    """

    DEFAULT_CSS = """
    LeaderboardPanel { height: auto; padding: 0 1; }
    """

    def __init__(
        self,
        *,
        sort_by: str = "energy_score",
        sort_ascending: bool = True,
        max_rows: int = 12,
        **kwargs: Any,
    ) -> None:
        super().__init__("", **kwargs)
        self._rows: list[LeaderboardRow] = []
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.max_rows = max_rows
        self._update_table()

    # ─── public API ────────────────────────────────────────────────

    def set_rows(self, rows: Sequence[LeaderboardRow]) -> None:
        self._rows = list(rows)
        self._update_table()

    def add_row(self, row: LeaderboardRow) -> None:
        self._rows.append(row)
        self._update_table()

    def clear(self) -> None:
        self._rows = []
        self._update_table()

    def set_sort(self, column: str, *, ascending: bool | None = None) -> None:
        if column not in {
            "estimator_id", "point", "se", "p_value",
            "energy_score", "erupt", "sensitivity_verdict",
        }:
            return
        self.sort_by = column
        if ascending is not None:
            self.sort_ascending = ascending
        self._update_table()

    # ─── compatibility shims used by the auto-mode router ──────────

    def update(self, rows: Sequence[LeaderboardRow] | dict[str, Any] | None) -> None:  # type: ignore[override]
        """Accept either a list of rows or a `LoopEvent`-style payload.

        Keeping the same shape as `QueuePanel.update()` so the TUI's
        event router can call `panel.update(payload)` without
        knowing the underlying widget type.
        """
        if rows is None:
            return
        if isinstance(rows, dict):
            cards = rows.get("cards") or rows.get("leaderboard") or rows.get("top")
            if not cards:
                return
            converted = [
                LeaderboardRow(
                    estimator_id=str(c.get("estimator_id", "?")),
                    point=float(c.get("point_estimate", c.get("point", 0.0))),
                    se=c.get("se"),
                    ci_low=c.get("ci_low"),
                    ci_high=c.get("ci_high"),
                    p_value=(
                        float(c["p_value"])
                        if c.get("p_value") not in (None, "NA")
                        else None
                    ),
                    sensitivity_verdict=c.get("sensitivity_verdict"),
                    energy_score=c.get("energy_score"),
                    erupt=c.get("erupt"),
                    chain_id=c.get("chain_id"),
                    notes=c.get("notes"),
                )
                for c in cards
            ]
            self.set_rows(converted)
            return
        self.set_rows(rows)  # type: ignore[arg-type]

    # ─── internal ──────────────────────────────────────────────────

    def _update_table(self) -> None:
        rows = self._sorted_rows()
        super().update(self._render_table(rows))

    def _sorted_rows(self) -> list[LeaderboardRow]:
        key = self.sort_by

        def _val(r: LeaderboardRow) -> Any:
            v = getattr(r, key, None)
            if v is None:
                return float("inf") if self.sort_ascending else float("-inf")
            if key == "sensitivity_verdict":
                rank = {"green": 0, "yellow": 1, "red": 2, "unknown": 3, "errored": 4}
                return rank.get(str(v).lower(), 5)
            return v

        rows = sorted(self._rows, key=_val, reverse=not self.sort_ascending)
        return rows[: self.max_rows]

    def _render_table(self, rows: list[LeaderboardRow]) -> Table:
        t = Table(
            title=f"Estimator leaderboard (sorted by {self.sort_by} "
                  f"{'↑' if self.sort_ascending else '↓'})",
            show_lines=False,
            expand=True,
        )
        t.add_column("estimator", overflow="fold")
        t.add_column("point", justify="right")
        t.add_column("SE", justify="right")
        t.add_column("95% CI", justify="right")
        t.add_column("p", justify="right")
        t.add_column("sensitivity", justify="left")
        t.add_column("energy↓", justify="right")
        t.add_column("ERUPT↑", justify="right")
        t.add_column("chain", justify="left", overflow="fold")
        if not rows:
            t.add_row("(no estimators run yet)", "", "", "", "", "", "", "", "")
            return t
        for r in rows:
            t.add_row(
                r.estimator_id,
                _format_signed(r.point),
                _format_signed(r.se, ".4f") if r.se is not None else "—",
                _format_ci(r.ci_low, r.ci_high),
                _format_signed(r.p_value, ".4g") if r.p_value is not None else "—",
                _verdict_chip(r.sensitivity_verdict),
                _format_signed(r.energy_score, ".4f") if r.energy_score is not None else "—",
                _format_signed(r.erupt, ".4f") if r.erupt is not None else "—",
                r.chain_id or "—",
            )
        return t


__all__ = ["LeaderboardPanel", "LeaderboardRow"]
