"""Unit tests for ChainForestPanel."""

from __future__ import annotations

import pytest
from rich.console import Console

from causalrag.tui.widgets.chain_forest import ChainForestPanel


def _render(renderable) -> str:
    console = Console(width=160, record=True, file=None)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def test_empty_state() -> None:
    panel = ChainForestPanel()
    text = _render(panel.render_panel())
    assert "waiting" in text.lower()


def test_renders_root_and_two_children_with_indentation() -> None:
    panel = ChainForestPanel()

    # Root experiment (chain_id == its own id)
    panel.update_panel(
        {
            "id": "auto-01",
            "chain_id": "auto-01",
            "parent_id": None,
            "treatment": "drug",
            "outcome": "survival",
            "sensitivity_verdict": "green",
        }
    )
    # First child of auto-01
    panel.update_panel(
        {
            "id": "auto-02",
            "chain_id": "auto-01",
            "parent_id": "auto-01",
            "treatment": "drug",
            "outcome": "survival",
            "sensitivity_verdict": "yellow",
        }
    )
    # Grandchild of auto-02 (still chain auto-01)
    panel.update_panel(
        {
            "id": "auto-03",
            "chain_id": "auto-01",
            "parent_id": "auto-02",
            "treatment": "drug",
            "outcome": "survival",
            "sensitivity_verdict": "red",
        }
    )

    rendered = panel.render_panel()
    text = _render(rendered)

    # All three hypothesis ids must appear.
    assert "auto-01" in text
    assert "auto-02" in text
    assert "auto-03" in text
    # Verdicts.
    assert "green" in text
    assert "yellow" in text
    assert "red" in text

    # Find each line (line-based assertions for indentation).
    lines = [ln for ln in text.splitlines() if "auto-" in ln]
    assert len(lines) == 3
    root_line = next(ln for ln in lines if "auto-01" in ln)
    child_line = next(ln for ln in lines if "auto-02" in ln)
    grand_line = next(ln for ln in lines if "auto-03" in ln)

    # Indentation grows with depth.
    def _lead_spaces(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    root_lead = _lead_spaces(root_line)
    child_lead = _lead_spaces(child_line)
    grand_lead = _lead_spaces(grand_line)
    assert child_lead > root_lead
    assert grand_lead > child_lead

    # Tree connector appears on non-root rows.
    assert "└─" in child_line
    assert "└─" in grand_line


def test_idempotent_on_reemit() -> None:
    panel = ChainForestPanel()
    payload = {
        "id": "auto-01",
        "chain_id": "auto-01",
        "parent_id": None,
        "treatment": "T",
        "outcome": "Y",
        "sensitivity_verdict": "green",
    }
    panel.update_panel(payload)
    panel.update_panel(payload)
    assert len(panel._rows) == 1


def test_handles_missing_chain_id() -> None:
    """A card with no chain_id falls back to its own id as a standalone chain."""
    panel = ChainForestPanel()
    panel.update_panel(
        {
            "id": "auto-01",
            "treatment": "T",
            "outcome": "Y",
            "sensitivity_verdict": "green",
        }
    )
    text = _render(panel.render_panel())
    assert "auto-01" in text


def test_update_accepts_dict_alias() -> None:
    panel = ChainForestPanel()
    panel.update(
        {
            "id": "auto-01",
            "chain_id": "auto-01",
            "parent_id": None,
            "treatment": "T",
            "outcome": "Y",
            "sensitivity_verdict": "green",
        }
    )
    assert len(panel._rows) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
