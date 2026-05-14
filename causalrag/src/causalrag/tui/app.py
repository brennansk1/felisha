"""CausalRoadmapTUI — the Textual App.

Layout (top → bottom):

    TitleBar             ← dock=top
    LogView              ← scrollable body, takes all remaining height
    ComposerPanel        ← dock=bottom (slash menu + input + hints)
    StatusBar            ← dock=bottom

`/slash` commands run as workers so the UI stays responsive while the
pipeline does its work. Every successful command mutates the StudyProtocol
on disk and updates chrome state (phase, model, tier).
"""

from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from causalrag.core.protocol import StudyProtocol
from causalrag.tui.commands import dispatch
from causalrag.tui.widgets.chain_forest import ChainForestPanel
from causalrag.tui.widgets.composer import COMMANDS, ComposerPanel
from causalrag.tui.widgets.logview import LogView
from causalrag.tui.widgets.queue_panel import QueuePanel
from causalrag.tui.widgets.statusbar import StatusBar
from causalrag.tui.widgets.titlebar import TitleBar


BANNER = """\
╭─── causalrag · TUI ───────────────────────────────────────╮
│  Petersen–van der Laan Causal Roadmap, in a terminal.     │
╰───────────────────────────────────────────────────────────╯"""


class CausalRoadmapTUI(App):
    """The full TUI experience."""

    CSS_PATH = "styles.tcss"
    TITLE = "CausalRoadmap"

    BINDINGS = [
        Binding("ctrl+c", "request_quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_log", "Clear log"),
        Binding("ctrl+k", "focus_input", "Focus input"),
    ]

    def __init__(
        self, project_dir: Path | None = None, auto_mode: bool = False
    ) -> None:
        super().__init__()
        self.project_dir = (project_dir or Path.cwd()).resolve()
        self.protocol: StudyProtocol | None = None
        self.cached_slots = None
        self.cached_profile = None
        self.title_bar: TitleBar = TitleBar(
            crumbs=(self.project_dir.name,),
            tier="academic",
            model="idle",
        )
        self.log_view = LogView()
        self.status_bar = StatusBar()
        self.composer = ComposerPanel()
        self._running_worker = None
        # `--auto` mode mounts the live planner-queue + chain-forest panels
        # so the user can watch the master loop's reasoning in flight.
        self.auto_mode = auto_mode
        self.queue_panel: QueuePanel | None = QueuePanel() if auto_mode else None
        self.chain_forest: ChainForestPanel | None = (
            ChainForestPanel() if auto_mode else None
        )

    def compose(self) -> ComposeResult:
        yield self.title_bar
        if self.queue_panel is not None:
            yield self.queue_panel
        if self.chain_forest is not None:
            yield self.chain_forest
        yield self.log_view
        yield self.composer
        yield self.status_bar

    async def on_mount(self) -> None:
        # Try to load an existing protocol
        proto_path = self.project_dir / "study.causalrag.yaml"
        if proto_path.exists():
            try:
                self.protocol = StudyProtocol.read_yaml(proto_path)
                self.title_bar.tier = self.protocol.tier
            except Exception:
                self.protocol = None

        # Banner
        banner = Static(Text(BANNER, style="#5fa8ff"))
        self.log_view.banner(banner)
        chips_line = Text()
        for c in COMMANDS:
            if c.name in ("/run", "/clear", "/quit", "/exit"):
                continue
            chips_line.append(c.name, style="#9ec2ff")
            chips_line.append("  ", style="")
        self.log_view.banner(Static(chips_line))
        hint = Text(
            "Tip · press / to open the command menu · Tab to autocomplete · ↩ to run",
            style="#4d5773",
        )
        self.log_view.banner(Static(hint))
        self.status_bar.study = self.project_dir.name
        self.status_bar.set_phase(0)
        self.composer.input.focus()

    # --- Public API used by commands -------------------------------------

    def set_project_dir(self, path: Path) -> None:
        self.project_dir = path.resolve()
        self.title_bar.crumbs = (self.project_dir.name,)
        self.status_bar.study = self.project_dir.name

    def set_phase(self, phase: int, label: str | None = None) -> None:
        self.status_bar.set_phase(phase, label)

    def set_streaming(self, streaming: bool) -> None:
        self.status_bar.streaming = streaming
        self.composer.hints.streaming = streaming
        self.title_bar.streaming = streaming
        if streaming:
            self.title_bar.model = "streaming…"
        elif self.cached_slots is not None:
            self.title_bar.model = self.cached_slots.discovery
        else:
            self.title_bar.model = "idle"

    # --- Event handlers --------------------------------------------------

    @on(ComposerPanel.Submit)
    def _on_submit(self, event: ComposerPanel.Submit) -> None:
        line = event.value
        self.log_view.echo(line)
        self._run_command(line)

    @work(exclusive=True)
    async def _run_command(self, line: str) -> None:
        self.set_streaming(True)
        try:
            await dispatch(self, line)
        finally:
            self.set_streaming(False)

    # --- Bindings --------------------------------------------------------

    def action_request_quit(self) -> None:
        self.exit()

    def action_clear_log(self) -> None:
        self.log_view.clear_log()

    def action_focus_input(self) -> None:
        self.composer.input.focus()


def run(project_dir: Path | None = None, auto_mode: bool = False) -> None:
    """Run the TUI. Used by the ``causalrag tui`` entry point.

    When ``auto_mode=True`` (CLI ``--auto`` flag), the live candidate-queue
    and chain-forest panels are mounted above the log so the user can
    watch the master loop's planning + chain bookkeeping in flight.
    """
    app = CausalRoadmapTUI(project_dir=project_dir, auto_mode=auto_mode)
    app.run()


if __name__ == "__main__":
    run()
