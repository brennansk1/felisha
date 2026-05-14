"""Smoke tests for the Textual TUI.

We don't try to assert pixel layouts — Textual's snapshot tooling is the right
hammer for that and it lives in a separate fixture system. Instead we verify
the app boots, the slash-menu filter works, commands dispatch correctly, and
the protocol state mutates as expected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from causalrag.tui.app import CausalRoadmapTUI
from causalrag.tui.widgets.composer import COMMANDS, filter_commands


def test_filter_commands_by_prefix() -> None:
    out = filter_commands("/disc")
    names = [c.name for c in out]
    assert names == ["/discover"]


def test_filter_commands_empty_for_no_slash() -> None:
    assert filter_commands("hello") == ()


def test_filter_commands_lists_all_for_solo_slash() -> None:
    out = filter_commands("/")
    assert len(out) == len(COMMANDS)


def test_app_constructs() -> None:
    app = CausalRoadmapTUI()
    assert app.TITLE == "CausalRoadmap"
    binding_keys = {b.key for b in app.BINDINGS}
    assert "ctrl+c" in binding_keys
    assert "ctrl+l" in binding_keys


@pytest.mark.asyncio
async def test_tui_smoke_help_and_clear() -> None:
    pytest.importorskip("textual")
    with tempfile.TemporaryDirectory() as td:
        app = CausalRoadmapTUI(project_dir=Path(td))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            # /help → 1 cmd echo + 1 card + 1 tip line
            await pilot.press("/")
            for ch in "help":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause(0.3)
            children_after_help = len(list(app.log_view.children))
            assert children_after_help > 3
            # /clear empties the log
            await pilot.press("/")
            for ch in "clear":
                await pilot.press(ch)
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert len(list(app.log_view.children)) <= 1  # the cmd echo for /clear may survive briefly


@pytest.mark.asyncio
async def test_tui_init_scaffolds_project() -> None:
    pytest.importorskip("textual")
    with tempfile.TemporaryDirectory() as td:
        app = CausalRoadmapTUI(project_dir=Path(td))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            await pilot.press("/")
            for ch in "init demo":
                await pilot.press(ch if ch != " " else "space")
            await pilot.press("enter")
            await pilot.pause(0.5)
            assert (app.project_dir / "study.causalrag.yaml").exists()


@pytest.mark.asyncio
async def test_tab_autocompletes_partial_command() -> None:
    pytest.importorskip("textual")
    app = CausalRoadmapTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        await pilot.press("/")
        for ch in "doc":
            await pilot.press(ch)
        await pilot.press("tab")
        await pilot.pause(0.1)
        assert app.composer.input.value == "/doctor "


# pytest-asyncio config
def pytest_collection_modifyitems(items):
    for item in items:
        if "asyncio" in item.keywords:
            item.add_marker(pytest.mark.asyncio)
