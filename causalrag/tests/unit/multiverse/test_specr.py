"""Unit tests for the Sprint 6.2 specification curve."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from causalrag.multiverse.specr import (
    SpecCurve,
    SpecResult,
    render_html,
    specification_curve,
)


# --------------------------------------------------------------------------- #
# Synthetic data with known ATE = 2.0
# --------------------------------------------------------------------------- #


def _make_synthetic(n: int = 800, true_ate: float = 2.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)  # irrelevant nuisance — not a confounder
    x4 = rng.normal(size=n)  # irrelevant nuisance — not a confounder
    logits = 0.6 * x1 - 0.4 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(float)
    noise = rng.normal(scale=1.0, size=n)
    y = true_ate * t + 1.5 * x1 + 0.7 * x2 + 0.1 * x3 + noise
    # Spread rows across a synthetic time axis so time-window specs are
    # exercisable in the integration-style test below.
    time = rng.integers(low=0, high=4, size=n)
    return pd.DataFrame(
        {"y": y, "t": t, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "time": time}
    )


# --------------------------------------------------------------------------- #
# Core happy-path: 3 adj sets x 2 estimators x 3 trims = 18 specs
# --------------------------------------------------------------------------- #


def test_specification_curve_recovers_true_ate() -> None:
    """3 x 2 x 3 = 18 specs; ≥80% within 2 SE of truth and significance > 60%."""
    pytest.importorskip("econml")
    pytest.importorskip("statsmodels")
    pytest.importorskip("sklearn")

    df = _make_synthetic(n=500, true_ate=2.0)
    # All three adjustment sets contain the true confounders (x1, x2); they
    # differ in irrelevant-nuisance inclusion. The 18-spec multiverse should
    # therefore concentrate around the true ATE = 2.0.
    adjustment_sets = [
        ("x1", "x2"),
        ("x1", "x2", "x3"),
        ("x1", "x2", "x3", "x4"),
    ]
    estimators = ["python.linear.ols", "python.dml.linear"]

    curve = specification_curve(
        df=df,
        treatment="t",
        outcome="y",
        adjustment_sets=adjustment_sets,
        estimators=estimators,
        trimming_thresholds=(0.0, 0.01, 0.05),
        principled_equivalence=False,
    )

    assert isinstance(curve, SpecCurve)
    assert curve.n_specs == 18, f"expected 18 specs, got {curve.n_specs}"
    assert curve.converged_count >= 16, "almost all specs should converge"
    assert curve.point_curve.shape[0] == curve.converged_count
    assert np.all(np.diff(curve.point_curve) >= -1e-12), "curve must be sorted"

    # ≥80% of converged specs lie within 2 SE of truth.
    within = 0
    for r in curve.results:
        if not r.converged or r.se is None or r.se <= 0:
            continue
        if abs(r.point - 2.0) <= 2.0 * r.se:
            within += 1
    share_within = within / curve.converged_count
    assert share_within >= 0.80, f"share within 2 SE was {share_within:.0%}"

    # Significance share > 0.6.
    assert curve.significance_share > 0.6

    # All converged points should sit on the positive side of zero.
    assert curve.sign_consistency_share >= 0.9

    # Non-principled equivalence → joint test withheld, Bonferroni surfaced.
    assert curve.joint_test_p is None
    assert curve.bonferroni_min_p is not None
    assert "Del Giudice" in curve.interpretation or "principled" in curve.interpretation.lower()


# --------------------------------------------------------------------------- #
# Principled equivalence flips on the Simonsohn joint test
# --------------------------------------------------------------------------- #


def test_principled_equivalence_yields_joint_test() -> None:
    pytest.importorskip("statsmodels")

    df = _make_synthetic(n=400, true_ate=2.0)
    curve = specification_curve(
        df=df,
        treatment="t",
        outcome="y",
        adjustment_sets=[("x1", "x2"), ("x1", "x2", "x3")],
        estimators=["python.linear.ols"],
        trimming_thresholds=(0.0, 0.05),
        principled_equivalence=True,
        random_state=0,
    )
    assert curve.joint_test_p is not None
    assert 0.0 < curve.joint_test_p <= 1.0
    assert "Simonsohn" in curve.interpretation


# --------------------------------------------------------------------------- #
# Time-window specs
# --------------------------------------------------------------------------- #


def test_time_window_specs_filter_rows() -> None:
    pytest.importorskip("statsmodels")

    df = _make_synthetic(n=600, true_ate=2.0)
    curve = specification_curve(
        df=df,
        treatment="t",
        outcome="y",
        adjustment_sets=[("x1", "x2")],
        estimators=["python.linear.ols"],
        trimming_thresholds=(0.0,),
        time_windows=((0, 2), (2, 4)),
        time_column="time",
        principled_equivalence=False,
    )
    assert curve.n_specs == 2
    # Both halves see only a subset, so n_used must be < n total.
    for r in curve.results:
        assert r.converged
        assert r.n_used < len(df)


# --------------------------------------------------------------------------- #
# extra_specs widen the Cartesian product
# --------------------------------------------------------------------------- #


def test_extra_specs_expand_product() -> None:
    pytest.importorskip("statsmodels")

    df = _make_synthetic(n=300, true_ate=2.0)
    curve = specification_curve(
        df=df,
        treatment="t",
        outcome="y",
        adjustment_sets=[("x1", "x2")],
        estimators=["python.linear.ols"],
        trimming_thresholds=(0.0,),
        extra_specs=[{"alpha": 0.05}, {"alpha": 0.10}],
    )
    # 1 x 1 x 1 x 1 x 2 = 2 specs.
    assert curve.n_specs == 2
    extras = sorted(r.spec["extra"]["alpha"] for r in curve.results)
    assert extras == [0.05, 0.10]


# --------------------------------------------------------------------------- #
# Robustness: failing estimator id is captured as non-converged
# --------------------------------------------------------------------------- #


def test_failed_spec_is_recorded_not_raised() -> None:
    pytest.importorskip("statsmodels")

    df = _make_synthetic(n=200, true_ate=2.0)
    curve = specification_curve(
        df=df,
        treatment="t",
        outcome="y",
        adjustment_sets=[("x1", "x2")],
        estimators=["python.linear.ols", "does.not.exist"],
        trimming_thresholds=(0.0,),
    )
    assert curve.n_specs == 2
    failures = [r for r in curve.results if not r.converged]
    assert len(failures) == 1
    assert failures[0].error is not None
    # The OK spec still summarized into the curve.
    assert curve.converged_count == 1


# --------------------------------------------------------------------------- #
# HTML render
# --------------------------------------------------------------------------- #


def test_render_html_emits_self_contained_fragment() -> None:
    pytest.importorskip("statsmodels")

    df = _make_synthetic(n=300, true_ate=2.0)
    curve = specification_curve(
        df=df,
        treatment="t",
        outcome="y",
        adjustment_sets=[("x1", "x2"), ("x1",)],
        estimators=["python.linear.ols"],
        trimming_thresholds=(0.0, 0.05),
    )
    html = render_html(curve, title="Test specr")
    assert isinstance(html, str) and len(html) > 200
    assert "<svg" in html
    assert "Test specr" in html
    assert "specification rank" in html
    assert "significance share" in html
    # Every spec id appears in the table.
    for r in curve.results:
        if r.converged:
            assert r.spec_id in html


# --------------------------------------------------------------------------- #
# Empty / pathological inputs raise early
# --------------------------------------------------------------------------- #


def test_specification_curve_rejects_empty_inputs() -> None:
    df = _make_synthetic(n=100)
    with pytest.raises(ValueError):
        specification_curve(
            df=df,
            treatment="t",
            outcome="y",
            adjustment_sets=[],
            estimators=["python.linear.ols"],
        )
    with pytest.raises(ValueError):
        specification_curve(
            df=df,
            treatment="t",
            outcome="y",
            adjustment_sets=[("x1",)],
            estimators=[],
        )


# --------------------------------------------------------------------------- #
# SpecResult.significant helper
# --------------------------------------------------------------------------- #


def test_specresult_significant_helper() -> None:
    r = SpecResult(
        spec_id="s",
        spec={},
        point=1.0,
        se=0.1,
        ci_low=0.8,
        ci_high=1.2,
        converged=True,
    )
    assert r.significant() is True
    r2 = SpecResult(
        spec_id="s",
        spec={},
        point=0.05,
        se=0.5,
        ci_low=-1.0,
        ci_high=1.1,
        converged=True,
    )
    assert r2.significant() is False
    r3 = SpecResult(
        spec_id="s",
        spec={},
        point=float("nan"),
        se=None,
        ci_low=None,
        ci_high=None,
        converged=False,
    )
    assert r3.significant() is False
