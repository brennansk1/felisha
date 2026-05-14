"""Unit tests for the estimator leaderboard widget (Sprint 4.4)."""

from __future__ import annotations

import pytest

from causalrag.tui.widgets.leaderboard import LeaderboardPanel, LeaderboardRow


def _row(
    name: str = "python.dml.linear",
    point: float = 1.0,
    se: float | None = 0.2,
    verdict: str | None = "green",
    energy: float | None = 0.5,
    erupt: float | None = 0.3,
    chain: str | None = None,
) -> LeaderboardRow:
    return LeaderboardRow(
        estimator_id=name,
        point=point,
        se=se,
        ci_low=point - 2 * (se or 0.0),
        ci_high=point + 2 * (se or 0.0),
        p_value=0.02,
        sensitivity_verdict=verdict,
        energy_score=energy,
        erupt=erupt,
        chain_id=chain,
    )


def test_leaderboard_initial_state_is_empty() -> None:
    p = LeaderboardPanel()
    rendered = p._render_table([])
    # Empty render says so
    assert any("no estimators" in str(c) for c in rendered.columns[0]._cells)


def test_leaderboard_set_rows_renders_table() -> None:
    p = LeaderboardPanel()
    rows = [_row("a", 1.0, energy=0.4), _row("b", 0.5, energy=0.7)]
    p.set_rows(rows)
    sorted_rows = p._sorted_rows()
    # default sort: ascending by energy_score
    assert [r.estimator_id for r in sorted_rows] == ["a", "b"]


def test_leaderboard_sort_by_point_descending() -> None:
    p = LeaderboardPanel(sort_by="point", sort_ascending=False)
    p.set_rows([_row("a", 0.5), _row("b", 2.0), _row("c", 1.0)])
    sorted_rows = p._sorted_rows()
    assert [r.estimator_id for r in sorted_rows] == ["b", "c", "a"]


def test_leaderboard_sort_by_verdict() -> None:
    p = LeaderboardPanel(sort_by="sensitivity_verdict")
    p.set_rows([
        _row("a", verdict="red"),
        _row("b", verdict="green"),
        _row("c", verdict="yellow"),
    ])
    sorted_rows = p._sorted_rows()
    assert [r.estimator_id for r in sorted_rows] == ["b", "c", "a"]


def test_leaderboard_max_rows_caps_display() -> None:
    p = LeaderboardPanel(max_rows=2)
    p.set_rows([_row(f"e{i}", energy=i * 0.1) for i in range(5)])
    sorted_rows = p._sorted_rows()
    assert len(sorted_rows) == 2


def test_leaderboard_handles_none_values() -> None:
    """SE / energy / ERUPT may be None on some estimator paths."""
    p = LeaderboardPanel()
    rows = [
        LeaderboardRow(
            estimator_id="a", point=1.0, se=None, ci_low=None, ci_high=None,
            p_value=None, sensitivity_verdict=None, energy_score=None, erupt=None,
        )
    ]
    p.set_rows(rows)
    rendered = p._render_table(p._sorted_rows())
    # Render should not crash; should contain "—" placeholders
    assert rendered is not None


def test_leaderboard_update_with_payload_dict() -> None:
    """The update() shim should accept the LoopEvent-style dict payload."""
    p = LeaderboardPanel()
    payload = {
        "leaderboard": [
            {
                "estimator_id": "python.dml.linear",
                "point_estimate": 1.5,
                "se": 0.3,
                "ci_low": 0.9,
                "ci_high": 2.1,
                "p_value": 0.01,
                "sensitivity_verdict": "green",
                "energy_score": 0.3,
                "erupt": 0.4,
                "chain_id": "auto-01",
            }
        ]
    }
    p.update(payload)
    assert len(p._rows) == 1
    assert p._rows[0].estimator_id == "python.dml.linear"
    assert p._rows[0].point == 1.5


def test_leaderboard_set_sort_validates_column() -> None:
    p = LeaderboardPanel()
    p.set_sort("point", ascending=False)
    assert p.sort_by == "point"
    assert p.sort_ascending is False
    # bogus column: no-op
    p.set_sort("foobar")
    assert p.sort_by == "point"


def test_leaderboard_clear_resets_state() -> None:
    p = LeaderboardPanel()
    p.set_rows([_row("a"), _row("b")])
    p.clear()
    assert p._rows == []
    rendered = p._render_table([])
    assert any("no estimators" in str(c) for c in rendered.columns[0]._cells)
