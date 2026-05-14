"""Unit tests for QueuePanel."""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.table import Table

from causalrag.tui.widgets.queue_panel import QueuePanel, _score_color


def _render(renderable) -> str:
    console = Console(width=160, record=True, file=None)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def test_score_color_thresholds() -> None:
    assert _score_color(0.9) == "#7ed2e6"  # green
    assert _score_color(0.6) == "#a3b6da"  # yellow
    assert _score_color(0.3) == "#e08877"  # red
    assert _score_color(0.5) == "#e08877"  # boundary, ≤ 0.5
    assert _score_color(0.71) == "#7ed2e6"


def test_queue_panel_empty_state() -> None:
    panel = QueuePanel()
    rendered = panel.render_panel()
    text = _render(rendered)
    assert "waiting" in text.lower()


def test_queue_panel_renders_five_candidates() -> None:
    panel = QueuePanel()
    payload = {
        "top": [
            {
                "id": f"c-{i:02d}",
                "treatment": f"T{i}",
                "outcome": f"Y{i}",
                "estimand_class": "ATE",
                "method": "python.dml.linear",
                "score": 0.9 - 0.15 * i,
            }
            for i in range(5)
        ]
    }
    panel.update_panel(payload)
    table = panel.render_panel()
    assert isinstance(table, Table)
    text = _render(table)
    for i in range(5):
        assert f"c-{i:02d}" in text
        assert f"T{i}" in text
        assert f"Y{i}" in text
        assert "ATE" in text
    # Score column has formatted scores
    assert "+0.90" in text
    assert "python.dml.linear" in text


def test_queue_panel_caps_at_five() -> None:
    panel = QueuePanel()
    payload = {
        "top": [
            {
                "id": f"c-{i:02d}",
                "treatment": "T",
                "outcome": "Y",
                "estimand_class": "ATE",
                "method": "m",
                "score": 0.5,
            }
            for i in range(8)
        ]
    }
    panel.update_panel(payload)
    text = _render(panel.render_panel())
    assert "c-04" in text
    # 6th candidate (c-05) must be dropped
    assert "c-05" not in text


def test_queue_panel_strikes_completed() -> None:
    panel = QueuePanel()
    panel.update_panel(
        {
            "top": [
                {
                    "id": "c-aa",
                    "treatment": "T",
                    "outcome": "Y",
                    "estimand_class": "ATE",
                    "method": "m",
                    "score": 0.9,
                }
            ]
        }
    )
    # Feed a card event that marks c-aa completed.
    panel.update_panel({"candidate_id": "c-aa", "id": "auto-01"})
    # Re-render: completed_ids now contains "c-aa"; verify the strike
    # style is applied by inspecting internal state.
    assert "c-aa" in panel._completed_ids


def test_queue_panel_update_accepts_dict() -> None:
    panel = QueuePanel()
    panel.update({"top": [{"id": "c-1", "treatment": "T", "outcome": "Y",
                           "estimand_class": "ATE", "method": "m", "score": 0.8}]})
    text = _render(panel.render_panel())
    assert "c-1" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
