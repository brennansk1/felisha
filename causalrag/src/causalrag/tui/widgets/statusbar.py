"""Status bar — phase · step · study · 4 hallucination-guard layers."""

from __future__ import annotations

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


GUARD_LABELS = ("L1", "L2", "L3", "L4")


class StatusBar(Static):
    """Bottom chrome rendering as a single text line.

    Guard values use a tri-state: ``1`` = active and passing (cyan ✓),
    ``0`` = active and warning (azure ⚠), ``-1`` = inactive (dim ·).
    """

    DEFAULT_CSS = ""

    phase: reactive[str] = reactive("0 · idle")
    step: reactive[str] = reactive("")
    study: reactive[str] = reactive("")
    guards: reactive[tuple[int, int, int, int]] = reactive((-1, -1, -1, -1))
    streaming: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__("", markup=False)

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        line = Text()
        line.append("phase ", style="#4d5773")
        line.append(self.phase, style="#cfd6e4")
        if self.step:
            line.append("    step ", style="#4d5773")
            line.append(self.step, style="#cfd6e4")
        if self.study:
            line.append("    study ", style="#4d5773")
            line.append(self.study, style="#cfd6e4")
        line.append("        ")
        if self.streaming:
            line.append("● streaming", style="#5fa8ff")
            line.append("    ")
        line.append("guards ", style="#4d5773")
        for i, state in enumerate(self.guards):
            if state == 1:
                mark, color = "✓", "#7ed2e6"
            elif state == 0:
                mark, color = "⚠", "#a3b6da"
            else:
                mark, color = "·", "#4d5773"
            line.append(f" {mark} {GUARD_LABELS[i]}", style=color)
        self.update(line)

    def watch_phase(self) -> None:
        self._refresh()

    def watch_step(self) -> None:
        self._refresh()

    def watch_study(self) -> None:
        self._refresh()

    def watch_guards(self) -> None:
        self._refresh()

    def watch_streaming(self) -> None:
        self._refresh()

    def set_phase(self, phase: int, label: str | None = None) -> None:
        labels = [
            "0 · idle",
            "1 · discover",
            "2 · feasibility",
            "3 · hypothesize",
            "4 · estimate",
            "5 · sensitivity",
            "6 · report",
        ]
        self.phase = label or labels[max(0, min(phase, len(labels) - 1))]
        if phase >= 1:
            self.guards = (1, 1, 1, 1 if phase != 1 else 0)
        else:
            self.guards = (-1, -1, -1, -1)
