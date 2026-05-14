"""Unit tests for ``MultiPaneLayout``.

These tests focus on the panel-selection contract: for each declared mode,
the right set of child widgets is wired up and the others are either absent
or hidden. We rely on Textual's app harness only for the cases that need a
mounted DOM (visibility toggles via ``update_for_mode``); pure
"which-panels-are-present" assertions go through the lightweight
``active_panels()`` helper.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from causalrag.tui.widgets.multi_pane import VALID_MODES, MultiPaneLayout


# Sentinel widgets — using ``Static`` avoids pulling in the real panels and
# their event-routing dependencies. The layout treats every slot generically,
# so any ``Widget`` subclass is a faithful stand-in.
def _stub(name: str) -> Static:
    s = Static("", id=f"stub-{name}")
    return s


def _all_panels() -> dict[str, Static]:
    return {
        "flag_chips": _stub("flag_chips"),
        "chain_forest": _stub("chain_forest"),
        "leaderboard": _stub("leaderboard"),
        "queue_panel": _stub("queue_panel"),
        "log_view": _stub("log_view"),
    }


# ── pure-Python contract tests (no app harness needed) ─────────────────────


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        MultiPaneLayout(mode="bogus")


def test_valid_modes_complete() -> None:
    # Spec lists five modes; guard against accidental drops.
    assert VALID_MODES == {
        "discover",
        "estimate",
        "sensitivity",
        "auto",
        "report",
    }


def test_discover_mode_uses_top_left_bottom() -> None:
    panels = _all_panels()
    layout = MultiPaneLayout(mode="discover", **panels)
    active = layout.active_panels()
    # discover shows flag chips top, chain forest left (placeholder for
    # variable-roles), and log on bottom. No leaderboard / queue.
    assert set(active.keys()) == {"flag_chips", "chain_forest", "log_view"}
    assert "leaderboard" not in active
    assert "queue_panel" not in active


def test_estimate_mode_includes_leaderboard_no_queue() -> None:
    panels = _all_panels()
    layout = MultiPaneLayout(mode="estimate", **panels)
    active = layout.active_panels()
    assert "leaderboard" in active
    assert "chain_forest" in active  # walk detail
    assert "log_view" in active
    assert "queue_panel" not in active  # queue is auto-only


def test_sensitivity_mode_shows_forest_and_leaderboard() -> None:
    panels = _all_panels()
    layout = MultiPaneLayout(mode="sensitivity", **panels)
    active = layout.active_panels()
    assert {"chain_forest", "leaderboard", "log_view"} <= set(active.keys())
    assert "queue_panel" not in active


def test_auto_mode_includes_full_forest_and_plan_queue() -> None:
    panels = _all_panels()
    layout = MultiPaneLayout(mode="auto", **panels)
    active = layout.active_panels()
    assert {
        "flag_chips",
        "chain_forest",
        "leaderboard",
        "queue_panel",
        "log_view",
    } <= set(active.keys())


def test_report_mode_drops_flag_chips() -> None:
    panels = _all_panels()
    layout = MultiPaneLayout(mode="report", **panels)
    active = layout.active_panels()
    # Run is done — no live flag chips.
    assert "flag_chips" not in active
    assert {"chain_forest", "leaderboard", "log_view"} <= set(active.keys())


def test_missing_optional_panels_are_silently_ok() -> None:
    # Caller may decline to provide e.g. a leaderboard. The layout should
    # still construct cleanly with only the supplied panels active.
    only_log = MultiPaneLayout(mode="estimate", log_view=_stub("log_view"))
    active = only_log.active_panels()
    assert set(active.keys()) == {"log_view"}


def test_update_for_mode_swaps_active_set_in_place() -> None:
    panels = _all_panels()
    layout = MultiPaneLayout(mode="discover", **panels)
    assert "leaderboard" not in layout.active_panels()
    layout.update_for_mode("estimate")
    assert "leaderboard" in layout.active_panels()
    # Same widget instance is reused — state preservation contract.
    assert layout.active_panels()["leaderboard"] is panels["leaderboard"]
    layout.update_for_mode("auto")
    assert "queue_panel" in layout.active_panels()


def test_update_for_mode_rejects_bogus_mode() -> None:
    layout = MultiPaneLayout(mode="discover")
    with pytest.raises(ValueError):
        layout.update_for_mode("nope")


# ── app-harness test: real Textual mount + display toggling ───────────────


@pytest.mark.asyncio
async def test_mounts_in_app_and_toggles_display() -> None:
    pytest.importorskip("textual")
    panels = _all_panels()
    layout = MultiPaneLayout(mode="discover", **panels)

    class _Harness(App):
        def compose(self) -> ComposeResult:
            yield layout

    app = _Harness()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.05)
        # In discover mode: flag_chips / chain_forest / log_view visible;
        # leaderboard + queue hidden via display=False.
        assert panels["flag_chips"].display is True
        assert panels["chain_forest"].display is True
        assert panels["log_view"].display is True
        assert panels["leaderboard"].display is False
        assert panels["queue_panel"].display is False

        # Switch to auto — queue + leaderboard light up.
        layout.update_for_mode("auto")
        await pilot.pause(0.05)
        assert panels["queue_panel"].display is True
        assert panels["leaderboard"].display is True
        assert panels["chain_forest"].display is True
        assert panels["flag_chips"].display is True

        # Switch to report — flag chips gone, queue gone, leaderboard +
        # forest + log remain.
        layout.update_for_mode("report")
        await pilot.pause(0.05)
        assert panels["flag_chips"].display is False
        assert panels["queue_panel"].display is False
        assert panels["leaderboard"].display is True
        assert panels["chain_forest"].display is True
        assert panels["log_view"].display is True


# pytest-asyncio config (mirrors the surrounding tui/ tests).
def pytest_collection_modifyitems(items):
    for item in items:
        if "asyncio" in item.keywords:
            item.add_marker(pytest.mark.asyncio)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
