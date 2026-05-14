"""MultiPaneLayout — configurable four-region layout for TUI phases.

Sprint 4.1: until this widget existed, only ``--auto`` mode received the
queue + chain-forest panels. The other phases (``discover``, ``estimate``,
``sensitivity``, ``report``) ran in a single-stream log view, which made
it hard to keep the user oriented in the Causal Roadmap step they were on.

``MultiPaneLayout`` is a reusable Textual container that arranges up to
four regions:

    ┌───────────────────────────────────────────────┐
    │  TOP   :: flag chips bar                      │
    ├──────────────────────┬────────────────────────┤
    │  LEFT  :: chain      │  RIGHT :: current walk │
    │           forest     │           / leaderboard│
    │                      │           / queue      │
    ├──────────────────────┴────────────────────────┤
    │  BOTTOM :: streaming log                       │
    └───────────────────────────────────────────────┘

Each *mode* picks a subset of the available slots and a layout strategy:

* ``discover`` — top flag chips + (placeholder for variable-roles in left)
  + log on bottom. No right pane.
* ``estimate`` — leaderboard on the right, walk/queue context on the left,
  log on bottom. Flag chips optional on top.
* ``sensitivity`` — chain forest on the left, leaderboard on the right,
  log on bottom.
* ``auto`` — the full thing: flag chips top, chain forest left, queue panel
  right, log bottom.
* ``report`` — chain forest spans the left, leaderboard the right, log on
  bottom. (No flag chips — the run is done.)

The constructor accepts *any* widget for the slots; callers are responsible
for instantiating their preferred panel (``ChainForestPanel``,
``LeaderboardPanel``, ``QueuePanel``, ``LogView``, ``FlagChipBar``). This
keeps ``multi_pane`` decoupled from the concrete widget implementations and
makes it trivial to swap in a future ``VariableRolesPanel`` for the
discovery phase.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widget import Widget


VALID_MODES: frozenset[str] = frozenset(
    {"discover", "estimate", "sensitivity", "auto", "report"}
)


# Per-mode slot manifest. Each entry lists the slot names that should be
# visible in that mode. Slots not in the manifest are hidden (display=False)
# even if the caller supplied a widget for them — so a single instance can
# be re-used across mode transitions without re-mounting children.
_MODE_SLOTS: dict[str, frozenset[str]] = {
    "discover": frozenset({"flag_chips", "chain_forest", "log_view"}),
    "estimate": frozenset(
        {"flag_chips", "chain_forest", "leaderboard", "log_view"}
    ),
    "sensitivity": frozenset(
        {"flag_chips", "chain_forest", "leaderboard", "log_view"}
    ),
    "auto": frozenset(
        {
            "flag_chips",
            "chain_forest",
            "leaderboard",
            "queue_panel",
            "log_view",
        }
    ),
    "report": frozenset({"chain_forest", "leaderboard", "log_view"}),
}


class MultiPaneLayout(Container):
    """Configurable multi-pane layout for TUI phases.

    Parameters
    ----------
    mode:
        One of ``'discover'``, ``'estimate'``, ``'sensitivity'``, ``'auto'``,
        ``'report'``. Selects which panels are visible.
    flag_chips:
        Optional widget for the top flag-chip bar.
    chain_forest:
        Optional widget for the left chain forest.
    leaderboard:
        Optional widget for the right leaderboard.
    queue_panel:
        Optional widget for the right plan-queue (auto mode only — when both
        ``leaderboard`` and ``queue_panel`` are provided in ``auto`` mode the
        queue stacks above the leaderboard).
    log_view:
        Optional widget for the bottom streaming log.
    """

    DEFAULT_CSS = """
    MultiPaneLayout {
        layout: vertical;
        height: 1fr;
        width: 1fr;
    }
    MultiPaneLayout > #mp-top {
        height: auto;
        min-height: 0;
        width: 1fr;
    }
    MultiPaneLayout > #mp-middle {
        height: 1fr;
        width: 1fr;
    }
    MultiPaneLayout > #mp-middle > #mp-left {
        width: 1fr;
        height: 1fr;
    }
    MultiPaneLayout > #mp-middle > #mp-right {
        width: 1fr;
        height: 1fr;
    }
    MultiPaneLayout > #mp-bottom {
        height: 1fr;
        min-height: 8;
        width: 1fr;
    }
    """

    # Slot names this layout understands. Anything else passed as a kwarg
    # in ``**panels`` is ignored (with a single attribute set on ``self``
    # so tests/inspectors can find it).
    _SLOT_NAMES: tuple[str, ...] = (
        "flag_chips",
        "chain_forest",
        "leaderboard",
        "queue_panel",
        "log_view",
    )

    def __init__(self, mode: str, **panels: Widget | None) -> None:
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(
                f"MultiPaneLayout: unknown mode {mode!r}; "
                f"expected one of {sorted(VALID_MODES)}"
            )
        self.mode = mode
        # Stash widgets on self so compose() can mount them in the right
        # container. ``None`` means "slot empty in this layout".
        self._panels: dict[str, Widget | None] = {
            name: panels.get(name) for name in self._SLOT_NAMES
        }
        # Containers created in compose(); kept as attrs so update_for_mode
        # can flip child visibility without rebuilding.
        self._top: Container | None = None
        self._middle: Horizontal | None = None
        self._left: Vertical | None = None
        self._right: Vertical | None = None
        self._bottom: Container | None = None

    # --- composition -----------------------------------------------------

    def compose(self) -> ComposeResult:  # type: ignore[override]
        self._top = Container(id="mp-top")
        self._left = Vertical(id="mp-left")
        self._right = Vertical(id="mp-right")
        self._middle = Horizontal(self._left, self._right, id="mp-middle")
        self._bottom = Container(id="mp-bottom")
        yield self._top
        yield self._middle
        yield self._bottom

    def on_mount(self) -> None:
        # Mount each provided panel into its target region. Done in
        # ``on_mount`` (rather than ``compose``) because the row/column
        # containers must already be in the DOM before children attach.
        if self._top is not None and self._panels["flag_chips"] is not None:
            self._top.mount(self._panels["flag_chips"])
        if self._left is not None and self._panels["chain_forest"] is not None:
            self._left.mount(self._panels["chain_forest"])
        if self._right is not None:
            # In auto mode we stack queue above leaderboard on the right;
            # in every other mode the right pane is just the leaderboard.
            if self.mode == "auto" and self._panels["queue_panel"] is not None:
                self._right.mount(self._panels["queue_panel"])
            if self._panels["leaderboard"] is not None:
                self._right.mount(self._panels["leaderboard"])
        if self._bottom is not None and self._panels["log_view"] is not None:
            self._bottom.mount(self._panels["log_view"])
        # Now toggle visibility for the chosen mode.
        self._apply_visibility(self.mode)

    # --- public API ------------------------------------------------------

    def update_for_mode(self, mode: str) -> None:
        """Switch to a new mode in-place.

        Children that aren't part of the new mode have ``display`` toggled
        off; new-to-this-mode children are toggled back on. We don't
        remount, so panel state (e.g. accumulated rows in the leaderboard)
        is preserved across mode transitions.
        """
        if mode not in VALID_MODES:
            raise ValueError(
                f"MultiPaneLayout: unknown mode {mode!r}; "
                f"expected one of {sorted(VALID_MODES)}"
            )
        self.mode = mode
        self._apply_visibility(mode)

    def active_panels(self) -> dict[str, Widget]:
        """Return the slot→widget mapping that's currently visible.

        Useful for tests that want to assert which panels participate in
        a given mode without poking at the Textual DOM.
        """
        slots = _MODE_SLOTS[self.mode]
        return {
            name: w
            for name, w in self._panels.items()
            if w is not None and name in slots
        }

    # --- internals -------------------------------------------------------

    def _apply_visibility(self, mode: str) -> None:
        slots = _MODE_SLOTS[mode]
        for name, widget in self._panels.items():
            if widget is None:
                continue
            try:
                widget.display = name in slots
            except Exception:  # pragma: no cover — display is a CSS prop
                pass
        # Hide structural containers that have no visible children in this
        # mode — keeps the bottom log from getting orphan whitespace above it.
        if self._top is not None:
            self._top.display = (
                self._panels["flag_chips"] is not None
                and "flag_chips" in slots
            )
        right_has = (
            (self._panels["leaderboard"] is not None and "leaderboard" in slots)
            or (self._panels["queue_panel"] is not None and "queue_panel" in slots)
        )
        left_has = (
            self._panels["chain_forest"] is not None
            and "chain_forest" in slots
        )
        if self._left is not None:
            self._left.display = left_has
        if self._right is not None:
            self._right.display = right_has
        if self._middle is not None:
            self._middle.display = left_has or right_has
        if self._bottom is not None:
            self._bottom.display = (
                self._panels["log_view"] is not None and "log_view" in slots
            )


__all__ = ["MultiPaneLayout", "VALID_MODES"]
