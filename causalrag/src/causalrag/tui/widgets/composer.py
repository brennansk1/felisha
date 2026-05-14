"""Composer + slash-menu — the bottom input strip with autocomplete popup.

Behavior mirrored from the design's `live.jsx`:

- Typing `/` opens the menu populated with all commands.
- Typing characters after `/` filters by prefix.
- `Tab` autocompletes the top match (with a trailing space).
- `↑/↓` walk the menu; `Enter` submits.
- `Esc` closes the menu without submitting.
- A hints row below the input shows ``/commands · ↑ history · Tab complete``
  plus a live `● streaming` indicator on the right.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Input, ListItem, ListView, Static

from causalrag.tui.completion import (
    apply_completion,
    common_prefix,
    complete_path,
    needs_path_completion,
)


@dataclass(frozen=True)
class Command:
    name: str       # "/init"
    description: str
    phase: int


COMMANDS: tuple[Command, ...] = (
    Command("/init", "scaffold a new study", 0),
    Command("/doctor", "hardware + ollama diagnostic", 0),
    Command("/discover", "profile · investigator · domain brief · audit", 1),
    Command("/feasibility", "power × MDE grid → admissible pairs", 2),
    Command("/hypothesize", "ranked, scoped hypotheses", 3),
    Command("/estimate", "Roadmap walk · Steps 1–7", 4),
    Command("/sensitivity", "E-value · sensemakr · multiverse", 5),
    Command("/report", "render HTML / PDF / Quarto", 6),
    Command("/run", "full pipeline, one shot (deterministic)", 0),
    Command("/auto", "AUTONOMOUS — LLM proposes K experiments + foundation loop", 0),
    Command("/layout", "toggle --auto mode panels (queue + chain forest)", 0),
    Command("/help", "list of commands", 0),
    Command("/clear", "clear the log view", 0),
    Command("/quit", "exit the TUI", 0),
)


def filter_commands(prefix: str) -> tuple[Command, ...]:
    if not prefix.startswith("/"):
        return ()
    # Slash-menu filter only runs over the command head, not the args.
    head = prefix.split(" ", 1)[0]
    if " " in prefix:
        # User has already finished typing the command; menu hides.
        return ()
    needle = head[1:].lower()
    return tuple(c for c in COMMANDS if c.name[1:].lower().startswith(needle))


class CmdInput(Input):
    """Input widget with custom Tab/Esc/Up/Down handling — the base
    ``Input.Submitted`` is reused (we override only the keystrokes that need
    different semantics for /command autocompletion + history nav)."""

    DEFAULT_CSS = ""

    class Cancelled(Message):
        pass

    class _RequestAutocomplete(Message):
        pass

    class _RequestUp(Message):
        pass

    class _RequestDown(Message):
        pass

    def __init__(self) -> None:
        super().__init__(placeholder="/ to browse · type a /command and press ↩")

    async def on_key(self, event: Key) -> None:
        if event.key == "tab":
            event.stop()
            self.post_message(self._RequestAutocomplete())
        elif event.key == "escape":
            event.stop()
            self.post_message(self.Cancelled())
        elif event.key == "up":
            self.post_message(self._RequestUp())
        elif event.key == "down":
            self.post_message(self._RequestDown())


class SlashMenu(Vertical):
    """Popup list shown above the composer when the input starts with `/`."""

    DEFAULT_CSS = ""

    items: reactive[tuple[Command, ...]] = reactive(())

    class CommandPicked(Message):
        def __init__(self, command: Command) -> None:
            self.command = command
            super().__init__()

    def __init__(self) -> None:
        super().__init__()
        self._list = ListView()

    def compose(self):
        yield self._list

    def watch_items(self, items: tuple[Command, ...]) -> None:
        self._list.clear()
        for c in items:
            line = Text()
            line.append(f"{c.name:<14}", style="#5fa8ff")
            line.append(f"  {c.description}", style="#6b7691")
            line.append(f"   phase {c.phase}", style="#4d5773")
            self._list.append(ListItem(Static(line)))
        self.set_class(bool(items), "-visible")

    @on(ListView.Selected)
    def _on_select(self, event: ListView.Selected) -> None:
        idx = event.list_view.index or 0
        if 0 <= idx < len(self.items):
            self.post_message(self.CommandPicked(self.items[idx]))

    def step(self, direction: int) -> None:
        if not self.items:
            return
        if direction < 0:
            self._list.action_cursor_up()
        else:
            self._list.action_cursor_down()

    def selected_command(self) -> Command | None:
        idx = self._list.index
        if idx is None or not (0 <= idx < len(self.items)):
            return None
        return self.items[idx]


class ComposerHints(Static):
    """One-line hint strip below the input — single Static, updated reactively."""

    DEFAULT_CSS = ""
    streaming: reactive[bool] = reactive(False)
    cassette_status: reactive[str] = reactive("cassette · rec")
    model_label: reactive[str] = reactive("ollama · qwen3:14b")
    # Live "elapsed: 47s" string shown while a worker is running so the
    # operator can see long LLM calls are still ticking, not hung.
    elapsed_seconds: reactive[int] = reactive(0)
    elapsed_label: reactive[str] = reactive("")
    # Transient hint shown after a Tab completion with multiple matches.
    completion_hint: reactive[str] = reactive("")

    def __init__(self) -> None:
        super().__init__("", markup=False)

    def on_mount(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        line = Text()
        # If we have a transient completion-hint preview, render it
        # in place of the usual keybinding row (clearer feedback).
        if self.completion_hint:
            line.append("  match · ", style="#5fa8ff")
            line.append(self.completion_hint, style="#9ec2ff")
            self.update(line)
            return
        for label, key in (
            ("commands", "/"),
            ("history", "↑"),
            ("complete", "Tab"),
            ("clear", "⌃L"),
            ("cancel", "⌃C"),
        ):
            line.append(f" {key} ", style="#9aa3b5")
            line.append(label, style="#4d5773")
            line.append("    ", style="")
        line.append("        ")
        line.append(self.model_label, style="#6b7691")
        line.append("    ", style="")
        line.append(self.cassette_status, style="#6b7691")
        line.append("    ", style="")
        if self.streaming:
            line.append("● streaming", style="#5fa8ff")
            if self.elapsed_seconds > 0:
                tag = self.elapsed_label or "elapsed"
                line.append(
                    f"  · {tag} {self.elapsed_seconds}s",
                    style="#9ec2ff",
                )
        else:
            line.append("idle", style="#4d5773")
        self.update(line)

    def watch_streaming(self) -> None:
        self._refresh()

    def watch_cassette_status(self) -> None:
        self._refresh()

    def watch_model_label(self) -> None:
        self._refresh()

    def watch_elapsed_seconds(self) -> None:
        self._refresh()

    def watch_elapsed_label(self) -> None:
        self._refresh()

    def watch_completion_hint(self) -> None:
        self._refresh()


class ComposerPanel(Vertical):
    """Whole bottom strip: slash menu (collapsible) + input + hints."""

    DEFAULT_CSS = ""

    class Submit(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    HISTORY_LIMIT: ClassVar[int] = 200

    def __init__(self, cwd: Path | None = None) -> None:
        super().__init__()
        self.menu = SlashMenu()
        self.input = CmdInput()
        self.hints = ComposerHints()
        self._history: list[str] = []
        self._history_idx: int | None = None
        # Per-command last-argument memory: maps "/discover" -> last full
        # args string ("data/cohort.csv --treatment T"). Pressing Up on an
        # empty arg slot pulls the last value for the current command.
        self._per_command_args: dict[str, str] = {}
        self._cwd: Path = cwd or Path.cwd()

    def set_cwd(self, cwd: Path) -> None:
        """Update the working directory used for path completion."""
        self._cwd = cwd

    def compose(self):
        yield self.menu
        yield self.input
        yield self.hints

    @on(Input.Changed)
    def _on_changed(self, event: Input.Changed) -> None:
        v = event.value
        self.menu.items = filter_commands(v)
        # Typing dismisses the stale Tab-completion match preview.
        if self.hints.completion_hint:
            self.hints.completion_hint = ""

    @on(Input.Submitted)
    def _on_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        self._history.append(value)
        if len(self._history) > self.HISTORY_LIMIT:
            self._history = self._history[-self.HISTORY_LIMIT :]
        self._history_idx = None
        # Stash per-command arg history so the user can recall the last
        # invocation of "/discover" without scrolling all of history.
        head, _, rest = value.partition(" ")
        if rest:
            self._per_command_args[head.lower()] = rest
        self.input.value = ""
        self.menu.items = ()
        self.post_message(self.Submit(value))

    @on(CmdInput._RequestAutocomplete)
    def _on_autocomplete(self) -> None:
        # Phase 1: filling in the slash command itself.
        if self.menu.items:
            cmd = self.menu.selected_command() or self.menu.items[0]
            self.input.value = f"{cmd.name} "
            self.input.cursor_position = len(self.input.value)
            self.menu.items = ()
            return
        # Phase 2: filling in a file-path argument. We only do this for the
        # commands listed in PATH_ARG_COMMANDS and only when the cursor is
        # sitting on the path slot.
        line = self.input.value
        if not needs_path_completion(line):
            return
        completions = complete_path(line, self._cwd)
        if not completions:
            return
        if len(completions) == 1:
            self.input.value = apply_completion(line, completions[0])
            self.input.cursor_position = len(self.input.value)
            return
        # Multiple matches: extend to the common prefix and show hint.
        shared = common_prefix(completions)
        token = line.rsplit(" ", 1)[-1] if not line.endswith(" ") else ""
        if len(shared) > len(token.rsplit("/", 1)[-1]):
            # Apply the common prefix.
            self.input.value = apply_completion(line, shared)
            self.input.cursor_position = len(self.input.value)
        # Surface the candidates briefly in the hints strip.
        self.hints.completion_hint = "  ".join(completions[:6])

    @on(CmdInput._RequestUp)
    def _on_up(self) -> None:
        if self.menu.items:
            self.menu.step(-1)
            return
        # If the user has typed "/discover " with no args yet, hydrate
        # with the last value they used for that command.
        line = self.input.value
        if line.endswith(" "):
            head = line.split(" ", 1)[0].lower()
            last = self._per_command_args.get(head)
            if last:
                self.input.value = f"{head} {last}"
                self.input.cursor_position = len(self.input.value)
                return
        if not self._history:
            return
        if self._history_idx is None:
            self._history_idx = len(self._history) - 1
        else:
            self._history_idx = max(0, self._history_idx - 1)
        self.input.value = self._history[self._history_idx]
        self.input.cursor_position = len(self.input.value)

    @on(CmdInput._RequestDown)
    def _on_down(self) -> None:
        if self.menu.items:
            self.menu.step(1)
            return
        if self._history_idx is None:
            return
        self._history_idx = min(len(self._history) - 1, self._history_idx + 1)
        self.input.value = self._history[self._history_idx]
        self.input.cursor_position = len(self.input.value)

    @on(CmdInput.Cancelled)
    def _on_cancelled(self) -> None:
        self.menu.items = ()

    @on(SlashMenu.CommandPicked)
    def _on_pick(self, event: SlashMenu.CommandPicked) -> None:
        self.input.value = f"{event.command.name} "
        self.input.cursor_position = len(self.input.value)
        self.menu.items = ()
        self.input.focus()
