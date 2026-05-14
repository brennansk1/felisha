"""Title bar — mac dots, breadcrumbs, tier/model pills, clock."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.reactive import reactive
from textual.widgets import Static


class TitleBar(Static):
    """Top chrome — three monochrome blue dots, breadcrumb trail, right-aligned
    pills for tier + model + time. The whole bar renders as a single Static
    line that updates reactively when state changes."""

    DEFAULT_CSS = ""

    crumbs: reactive[tuple[str, ...]] = reactive(())
    tier: reactive[str] = reactive("")
    model: reactive[str] = reactive("")
    streaming: reactive[bool] = reactive(False)

    def __init__(
        self,
        crumbs: tuple[str, ...] = (),
        tier: str = "academic",
        model: str = "idle",
    ) -> None:
        super().__init__("", markup=False)
        self._initial_crumbs = crumbs
        self._initial_tier = tier
        self._initial_model = model

    def on_mount(self) -> None:
        self.crumbs = self._initial_crumbs
        self.tier = self._initial_tier
        self.model = self._initial_model
        self._refresh_text()
        self.set_interval(1.0, self._refresh_text)

    def _refresh_text(self) -> None:
        line = Text()
        line.append("● ", style="#3771bf")
        line.append("● ", style="#5fa8ff")
        line.append("● ", style="#9ec2ff")
        line.append("  ")
        line.append("causalrag", style="#cfd6e4")
        for i, c in enumerate(self.crumbs):
            line.append("  ›  ", style="#4d5773")
            is_last = i == len(self.crumbs) - 1
            line.append(c, style="#9ec2ff" if is_last else "#cfd6e4")
        line.append("        ")
        if self.tier:
            line.append("● ", style="#7ed2e6")
            line.append(f"tier · {self.tier}", style="#cfd6e4")
            line.append("    ")
        if self.model:
            line.append("● ", style="#a3b6da" if self.streaming else "#7ed2e6")
            line.append(self.model, style="#cfd6e4")
            line.append("    ")
        line.append(datetime.now().strftime("%H:%M:%S"), style="#4d5773")
        self.update(line)

    def watch_crumbs(self) -> None:
        self._refresh_text()

    def watch_tier(self) -> None:
        self._refresh_text()

    def watch_model(self) -> None:
        self._refresh_text()

    def watch_streaming(self) -> None:
        self._refresh_text()
