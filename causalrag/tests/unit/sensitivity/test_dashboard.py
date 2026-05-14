"""Tests for the unified sensitivity dashboard (Sprint 2.6)."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.result import EstimationResult
from causalrag.sensitivity.dashboard import (
    SensitivityDashboard,
    SensitivityPanel,
    render_sensitivity_dashboard_html,
    run_sensitivity_dashboard,
    _aggregate,
)


# ─────────── Fixtures ────────────────────────────────────────────────────


class _StubCandidate:
    """Light-weight stand-in for CandidateExperiment — only the fields the
    dashboard touches matter, and importing the real master_loop here
    would pull in heavyweight dependencies."""

    def __init__(self, treatment: str, outcome: str) -> None:
        self.treatment = treatment
        self.outcome = outcome


def _make_df(rng_seed: int = 0, n: int = 200) -> pd.DataFrame:
    """Synthetic frame with a continuous outcome and a binary treatment.
    Designed so OLS / sensemakr converges quickly."""
    rng = np.random.default_rng(rng_seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    t = (rng.uniform(0, 1, n) > 0.5).astype(int)
    y = 0.5 * t + 0.3 * x1 + 0.2 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"t": t, "y": y, "x1": x1, "x2": x2})


def _make_result(
    *,
    point: float = 0.5,
    se: float = 0.1,
    n_used: int = 200,
    p_value: float | None = 0.001,
) -> EstimationResult:
    ci_low = point - 1.96 * se
    ci_high = point + 1.96 * se
    return EstimationResult(
        estimator_id="python.dml.linear",
        estimand_class="ATE",
        point_estimate=point,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_used=n_used,
    )


def _make_walk(hyp_id: str = "h-001") -> RoadmapWalk:
    return RoadmapWalk(hypothesis_id=hyp_id)


def _make_protocol() -> StudyProtocol:
    return StudyProtocol(name="test-dashboard")


# ─────────── Smoke + shape ───────────────────────────────────────────────


def test_dashboard_runs_end_to_end_and_returns_panels() -> None:
    df = _make_df()
    result = _make_result()
    walk = _make_walk()
    candidate = _StubCandidate(treatment="t", outcome="y")
    protocol = _make_protocol()

    dash = run_sensitivity_dashboard(
        result=result,
        walk=walk,
        df=df,
        candidate=candidate,
        protocol=protocol,
    )

    assert isinstance(dash, SensitivityDashboard)
    assert dash.hypothesis_id == "h-001"
    # Spec requires ≥ 4 panels; the dashboard wires nine slots in total.
    assert len(dash.panels) >= 4
    # Every panel must declare a stable name + backend.
    for p in dash.panels:
        assert isinstance(p, SensitivityPanel)
        assert p.name
        assert p.backend


def test_dashboard_has_each_named_slot() -> None:
    df = _make_df()
    dash = run_sensitivity_dashboard(
        result=_make_result(),
        walk=_make_walk(),
        df=df,
        candidate=_StubCandidate("t", "y"),
        protocol=_make_protocol(),
    )
    names = {p.name for p in dash.panels}
    for required in (
        "e_value",
        "sensemakr",
        "tipping_point",
        "rosenbaum",
        "manski",
        "negative_control",
        "ovb_chernozhukov",
        "refutation_summary",
    ):
        assert required in names, f"missing panel: {required}"


# ─────────── Failure-isolation ───────────────────────────────────────────


def test_one_panel_failing_does_not_take_down_the_dashboard() -> None:
    df = _make_df()
    # Force the sensemakr panel to fail by patching the wrapper to raise.
    with patch(
        "causalrag.sensitivity.dashboard.run_sensemakr",
        side_effect=RuntimeError("simulated sensemakr outage"),
    ):
        dash = run_sensitivity_dashboard(
            result=_make_result(),
            walk=_make_walk(),
            df=df,
            candidate=_StubCandidate("t", "y"),
            protocol=_make_protocol(),
        )
    # Dashboard still constructed.
    assert isinstance(dash, SensitivityDashboard)
    sm = next(p for p in dash.panels if p.name == "sensemakr")
    assert sm.available is False
    assert "simulated sensemakr outage" in sm.result.get("error", "")
    # Other panels remained functional — at least the e_value should run.
    ev = next(p for p in dash.panels if p.name == "e_value")
    assert ev.available is True


def test_evalue_panel_failure_isolated_to_its_slot() -> None:
    df = _make_df()
    with patch(
        "causalrag.sensitivity.dashboard.evalue_for_estimator",
        side_effect=ValueError("boom"),
    ):
        dash = run_sensitivity_dashboard(
            result=_make_result(),
            walk=_make_walk(),
            df=df,
            candidate=_StubCandidate("t", "y"),
            protocol=_make_protocol(),
        )
    ev = next(p for p in dash.panels if p.name == "e_value")
    assert ev.available is False
    assert "boom" in ev.result.get("error", "")
    # Dashboard still returns a valid aggregate.
    assert dash.aggregate_verdict in ("green", "yellow", "red", "unknown")


# ─────────── Aggregate verdict rules ─────────────────────────────────────


def _panel(name: str, color: str, *, available: bool = True) -> SensitivityPanel:
    return SensitivityPanel(
        name=name,
        backend="test",
        result={},
        verdict_contribution=color,  # type: ignore[arg-type]
        rationale="",
        available=available,
    )


def test_aggregate_three_green_one_yellow_is_yellow() -> None:
    panels = [
        _panel("a", "green"),
        _panel("b", "green"),
        _panel("c", "green"),
        _panel("d", "yellow"),
    ]
    color, _ = _aggregate(panels)
    assert color == "yellow"


def test_aggregate_three_green_one_red_is_red() -> None:
    panels = [
        _panel("a", "green"),
        _panel("b", "green"),
        _panel("c", "green"),
        _panel("d", "red"),
    ]
    color, _ = _aggregate(panels)
    assert color == "red"


def test_aggregate_all_green_is_green() -> None:
    panels = [_panel(n, "green") for n in "abcd"]
    color, _ = _aggregate(panels)
    assert color == "green"


def test_aggregate_only_neutral_and_unavailable_is_unknown() -> None:
    panels = [
        _panel("a", "neutral"),
        _panel("b", "green", available=False),  # unavailable should not contribute
    ]
    color, _ = _aggregate(panels)
    assert color == "unknown"


def test_aggregate_unavailable_panels_excluded_from_verdict() -> None:
    """A red but unavailable panel should NOT poison the verdict — it
    couldn't actually run, so it doesn't add evidence."""
    panels = [
        _panel("a", "green"),
        _panel("b", "green"),
        _panel("c", "red", available=False),
    ]
    color, _ = _aggregate(panels)
    assert color == "green"


def test_aggregate_unknown_only_is_unknown() -> None:
    panels = [
        _panel("a", "unknown"),
        _panel("b", "unknown"),
    ]
    color, _ = _aggregate(panels)
    assert color == "unknown"


# ─────────── HTML rendering ──────────────────────────────────────────────


def test_html_renders_without_crash() -> None:
    df = _make_df()
    dash = run_sensitivity_dashboard(
        result=_make_result(),
        walk=_make_walk(),
        df=df,
        candidate=_StubCandidate("t", "y"),
        protocol=_make_protocol(),
    )
    html = render_sensitivity_dashboard_html(dash)
    assert isinstance(html, str)
    assert "Sensitivity dashboard" in html
    assert dash.hypothesis_id in html
    # Every panel name should appear somewhere in the table body.
    for p in dash.panels:
        assert p.name in html


def test_html_escapes_special_chars_in_rationale() -> None:
    """Manual SensitivityDashboard with characters that must be escaped."""
    dash = SensitivityDashboard(
        hypothesis_id="h-<script>",
        aggregate_verdict="yellow",
        aggregate_rationale="a & b",
        panels=[
            SensitivityPanel(
                name="e_value",
                backend="python.evalue",
                result={},
                verdict_contribution="yellow",
                rationale='evil "quotes" & <tags>',
                available=True,
            )
        ],
    )
    html = render_sensitivity_dashboard_html(dash)
    assert "<script>" not in html  # escaped
    assert "&lt;script&gt;" in html
    assert "&amp;" in html
    assert "&quot;" in html


# ─────────── Protocol-less path ──────────────────────────────────────────


def test_dashboard_works_without_protocol() -> None:
    """The dashboard must not assume a fully-populated StudyProtocol —
    callers from ad-hoc scripts pass only the bare estimate + walk + df."""
    df = _make_df()
    dash = run_sensitivity_dashboard(
        result=_make_result(),
        walk=_make_walk(),
        df=df,
        candidate=_StubCandidate("t", "y"),
        protocol=None,
    )
    assert isinstance(dash, SensitivityDashboard)
    assert len(dash.panels) >= 4
