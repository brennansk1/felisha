"""Tests for the Sprint 5.3 root-cause attribution task."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.core.graph import CausalGraph
from causalrag.tasks.rca import (
    RootCauseReport,
    attribute_metric_change,
)


def _make_synthetic(
    *,
    n: int = 400,
    seed: int = 0,
    x1_after_mean: float = 1.0,
    x2_after_mean: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Y = 2*X1 + 0.5*X2 + noise; only X1 shifts in the 'after' period."""
    rng = np.random.default_rng(seed)
    x1_b = rng.normal(0.0, 1.0, size=n)
    x2_b = rng.normal(0.0, 1.0, size=n)
    noise_b = rng.normal(0.0, 0.1, size=n)
    y_b = 2.0 * x1_b + 0.5 * x2_b + noise_b
    df_before = pd.DataFrame({"X1": x1_b, "X2": x2_b, "Y": y_b})

    x1_a = rng.normal(x1_after_mean, 1.0, size=n)
    x2_a = rng.normal(x2_after_mean, 1.0, size=n)
    noise_a = rng.normal(0.0, 0.1, size=n)
    y_a = 2.0 * x1_a + 0.5 * x2_a + noise_a
    df_after = pd.DataFrame({"X1": x1_a, "X2": x2_a, "Y": y_a})
    return df_before, df_after


# --------------------------------------------------------------------------- #
# Core happy path: DR attribution
# --------------------------------------------------------------------------- #
def test_attribute_metric_change_dr_recovers_x1() -> None:
    df_before, df_after = _make_synthetic(n=400)
    report = attribute_metric_change(
        df_before=df_before,
        df_after=df_after,
        target="Y",
        method="multiply_robust",
    )

    assert isinstance(report, RootCauseReport)
    assert report.target == "Y"
    assert report.method == "multiply_robust"
    # Expected mean shift in Y ≈ 2 * 1.0 + 0.5 * 0.0 = 2.0
    assert report.total_change == pytest.approx(2.0, abs=0.25)

    # Findings include X1 and X2 plus possibly a residual bucket
    nodes = {f.node for f in report.findings}
    assert "X1" in nodes
    assert "X2" in nodes

    contributions = {f.node: f.contribution for f in report.findings}
    x1_share = abs(contributions["X1"]) / max(
        abs(report.total_change), 1e-9
    )
    assert x1_share > 0.70, (
        f"X1 should dominate the attribution; got share={x1_share:.2f}, "
        f"contributions={contributions}"
    )

    # Findings are ranked by absolute contribution
    abs_contribs = [abs(f.contribution) for f in report.findings]
    assert abs_contribs == sorted(abs_contribs, reverse=True)
    assert report.findings[0].rank == 1


# --------------------------------------------------------------------------- #
# auto picks DR for large samples, gcm_anomaly otherwise
# --------------------------------------------------------------------------- #
def test_method_auto_selects_dr_for_large_samples() -> None:
    df_before, df_after = _make_synthetic(n=400)
    report = attribute_metric_change(
        df_before=df_before, df_after=df_after, target="Y", method="auto"
    )
    assert report.method == "multiply_robust"


def test_method_auto_falls_back_to_gcm_for_small_samples() -> None:
    df_before, df_after = _make_synthetic(n=50)
    report = attribute_metric_change(
        df_before=df_before, df_after=df_after, target="Y", method="auto"
    )
    # gcm_anomaly may itself fall back to fallback_regression if dowhy
    # is not installed in the test env; both signal "small-n path".
    assert report.method in {"gcm_anomaly", "fallback_regression"}


# --------------------------------------------------------------------------- #
# Star-graph default when graph is None
# --------------------------------------------------------------------------- #
def test_unknown_graph_uses_star_default() -> None:
    df_before, df_after = _make_synthetic(n=300)
    report = attribute_metric_change(
        df_before=df_before,
        df_after=df_after,
        target="Y",
        graph=None,
        method="multiply_robust",
    )
    nodes = {f.node for f in report.findings}
    # All numeric columns other than Y should be candidates.
    assert "X1" in nodes and "X2" in nodes


def test_graph_filters_candidates_to_ancestors() -> None:
    df_before, df_after = _make_synthetic(n=300)
    # A graph that only declares X1 → Y; X2 is an unrelated isolated node.
    graph = CausalGraph.from_edge_list([("X1", "Y")])
    # X2 needs to be present in the graph as a non-ancestor.
    graph = CausalGraph(
        nodes=("X1", "X2", "Y"),
        edges=graph.edges,
    )
    report = attribute_metric_change(
        df_before=df_before,
        df_after=df_after,
        target="Y",
        graph=graph,
        method="multiply_robust",
    )
    nodes = {f.node for f in report.findings if f.node != "everything_else"}
    assert nodes == {"X1"}


# --------------------------------------------------------------------------- #
# Empty input → empty report with a note
# --------------------------------------------------------------------------- #
def test_empty_before_returns_empty_report() -> None:
    _, df_after = _make_synthetic(n=20)
    df_before = df_after.iloc[0:0].copy()
    report = attribute_metric_change(
        df_before=df_before, df_after=df_after, target="Y"
    )
    assert report.findings == []
    assert report.n_before == 0
    assert report.total_change == 0.0
    assert any("empty" in note.lower() for note in report.notes)


def test_empty_after_returns_empty_report() -> None:
    df_before, _ = _make_synthetic(n=20)
    df_after = df_before.iloc[0:0].copy()
    report = attribute_metric_change(
        df_before=df_before, df_after=df_after, target="Y"
    )
    assert report.findings == []
    assert report.n_after == 0


# --------------------------------------------------------------------------- #
# Column mismatch raises
# --------------------------------------------------------------------------- #
def test_mismatched_columns_raises_value_error() -> None:
    df_before, df_after = _make_synthetic(n=50)
    df_after = df_after.drop(columns=["X2"])
    with pytest.raises(ValueError, match="share columns"):
        attribute_metric_change(
            df_before=df_before, df_after=df_after, target="Y"
        )


def test_missing_target_raises() -> None:
    df_before, df_after = _make_synthetic(n=50)
    with pytest.raises(ValueError, match="target"):
        attribute_metric_change(
            df_before=df_before, df_after=df_after, target="Z"
        )


# --------------------------------------------------------------------------- #
# Sum of contributions ≈ total change
# --------------------------------------------------------------------------- #
def test_contributions_sum_to_total_change() -> None:
    df_before, df_after = _make_synthetic(n=400)
    report = attribute_metric_change(
        df_before=df_before,
        df_after=df_after,
        target="Y",
        method="multiply_robust",
    )
    total = sum(f.contribution for f in report.findings)
    assert total == pytest.approx(report.total_change, abs=1e-6)


# --------------------------------------------------------------------------- #
# Interpretation is non-empty plain text
# --------------------------------------------------------------------------- #
def test_interpretation_mentions_target_and_top_node() -> None:
    df_before, df_after = _make_synthetic(n=400)
    report = attribute_metric_change(
        df_before=df_before,
        df_after=df_after,
        target="Y",
        method="multiply_robust",
    )
    assert "Y" in report.interpretation
    top_node = report.findings[0].node
    assert top_node in report.interpretation
