"""Tests for the /layout slash command + Ctrl-T binding."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from causalrag.tui.app import CausalRoadmapTUI


def test_app_has_new_bindings() -> None:
    app = CausalRoadmapTUI()
    keys = {b.key for b in app.BINDINGS}
    assert "ctrl+g" in keys
    assert "ctrl+t" in keys


def test_auto_mode_mounts_panels() -> None:
    app = CausalRoadmapTUI(auto_mode=True)
    assert app.queue_panel is not None
    assert app.chain_forest is not None


def test_default_mode_no_panels() -> None:
    app = CausalRoadmapTUI()
    assert app.queue_panel is None
    assert app.chain_forest is None


@pytest.mark.asyncio
async def test_layout_command_toggles_panels() -> None:
    pytest.importorskip("textual")
    with tempfile.TemporaryDirectory() as td:
        app = CausalRoadmapTUI(project_dir=Path(td), auto_mode=True)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            assert app.queue_panel.display is True
            assert app.chain_forest.display is True
            # /layout hide
            for ch in "/layout hide":
                key = ch if ch != " " else "space"
                await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert app.queue_panel.display is False
            assert app.chain_forest.display is False
            # /layout show
            for ch in "/layout show":
                key = ch if ch != " " else "space"
                await pilot.press(key)
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert app.queue_panel.display is True
            assert app.chain_forest.display is True


# pytest-asyncio config
def pytest_collection_modifyitems(items):
    for item in items:
        if "asyncio" in item.keywords:
            item.add_marker(pytest.mark.asyncio)
