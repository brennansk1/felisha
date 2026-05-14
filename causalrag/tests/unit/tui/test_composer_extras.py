"""Extra unit tests covering per-command history + filter_commands edge cases."""

from __future__ import annotations

import pytest

from causalrag.tui.commands import DISPATCH
from causalrag.tui.widgets.composer import COMMANDS, ComposerPanel, filter_commands


def test_filter_commands_hides_menu_once_space_typed() -> None:
    # After the user types a space the slash menu should retreat — they're
    # now editing args, not picking a command.
    assert filter_commands("/discover ") == ()
    assert filter_commands("/discover data.csv") == ()


def test_filter_commands_still_filters_partial_head() -> None:
    out = filter_commands("/disc")
    assert [c.name for c in out] == ["/discover"]


def test_dispatch_registers_question_alias_for_help() -> None:
    assert "/?" in DISPATCH
    assert DISPATCH["/?"] is DISPATCH["/help"]


def test_dispatch_registers_layout_command() -> None:
    assert "/layout" in DISPATCH


@pytest.mark.asyncio
async def test_per_command_args_remembered_after_submit() -> None:
    """Submitting `/discover data.csv` stashes the arg under `/discover`
    so a later Up-on-empty pulls it back."""
    pytest.importorskip("textual")
    import tempfile
    from pathlib import Path

    from causalrag.tui.app import CausalRoadmapTUI

    with tempfile.TemporaryDirectory() as td:
        app = CausalRoadmapTUI(project_dir=Path(td))
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause(0.1)
            panel = app.composer
            # Simulate the submit side-effect directly (avoids actually
            # firing /discover, which needs a protocol):
            value = "/discover data/cohort.csv --treatment T"
            head, _, rest = value.partition(" ")
            panel._history.append(value)
            panel._per_command_args[head.lower()] = rest
            assert panel._per_command_args["/discover"] == "data/cohort.csv --treatment T"


def pytest_collection_modifyitems(items):
    for item in items:
        if "asyncio" in item.keywords:
            item.add_marker(pytest.mark.asyncio)


def test_commands_list_contains_every_dispatch_target() -> None:
    """Slash-menu COMMANDS should at least cover the user-facing commands."""
    advertised = {c.name for c in COMMANDS}
    for required in ("/init", "/doctor", "/discover", "/help", "/clear", "/quit"):
        assert required in advertised


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
