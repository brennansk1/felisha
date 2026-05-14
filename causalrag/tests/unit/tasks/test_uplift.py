"""Tests for Sprint 5.5 — uplift / Qini / AUUC / policy targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.tasks.uplift import (
    TargetingReport,
    UpliftCurve,
    _cumulative_uplift_curve,
    _trapz_area,
    build_targeting_report,
    policy_tree,
)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------


def _make_heterogeneous(n: int = 800, seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic dataset with strong positive heterogeneity in CATE.

    Half the population has CATE = +2, the other half CATE = 0. The
    oracle CATE perfectly identifies which half is which, so top-50%
    targeting captures essentially all the gain.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    is_high = (x1 > 0).astype(np.float64)
    # Randomized treatment (P(T=1)=0.5) — keeps the propensity story trivial.
    t = (rng.uniform(size=n) < 0.5).astype(np.float64)
    # Y = baseline + tau(X) * T + noise; tau = 2 when x1>0 else 0.
    tau = 2.0 * is_high
    y = 0.5 * x1 + tau * t + rng.normal(scale=0.2, size=n)
    # Oracle CATE predictions (perfect targeting signal).
    cate = tau.copy()
    return dict(x1=x1, t=t, y=y, cate=cate, tau=tau)


def _make_random_cate(n: int = 600, seed: int = 1) -> dict[str, np.ndarray]:
    """Same DGP as ``_make_heterogeneous`` but with a *random* CATE signal
    (no information about who actually benefits)."""
    base = _make_heterogeneous(n=n, seed=seed)
    rng = np.random.default_rng(seed + 999)
    base["cate"] = rng.normal(size=n)
    return base


# ---------------------------------------------------------------------------
# Pure-Python curve math (no estimator dependencies)
# ---------------------------------------------------------------------------


def test_cumulative_uplift_curve_endpoints() -> None:
    """Curve starts at (0, 0) and ends at (1, n * total_ATT)."""
    rng = np.random.default_rng(0)
    n = 200
    cate = rng.normal(size=n)
    y = rng.normal(size=n)
    t = (rng.uniform(size=n) < 0.5).astype(np.float64)
    fraction, lift, total_att = _cumulative_uplift_curve(cate, y, t)
    assert fraction[0] == 0.0
    assert lift[0] == 0.0
    assert fraction[-1] == pytest.approx(1.0)
    # Last lift value is n * (mean Y|T=1 - mean Y|T=0).
    expected_total = n * (y[t > 0.5].mean() - y[t < 0.5].mean())
    assert lift[-1] == pytest.approx(expected_total)
    assert total_att == pytest.approx(expected_total / n)


def test_trapz_area_on_known_shape() -> None:
    """Trapezoid area of y=x on [0,1] is 0.5."""
    x = np.linspace(0, 1, 101)
    y = x.copy()
    assert _trapz_area(x, y) == pytest.approx(0.5, abs=1e-6)


def test_trapz_area_single_point_is_zero() -> None:
    assert _trapz_area(np.array([0.0]), np.array([0.0])) == 0.0


# ---------------------------------------------------------------------------
# build_targeting_report — heterogeneity + Qini > 0
# ---------------------------------------------------------------------------


def test_targeting_report_recovers_heterogeneity() -> None:
    data = _make_heterogeneous(n=800, seed=0)
    report = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
    )
    assert isinstance(report, TargetingReport)
    assert report.n_total == 800
    # Oracle CATE: top 50% are exactly the responders, so policy treats
    # roughly half the population.
    assert report.n_recommended_treat > 0
    assert report.n_recommended_treat < 800
    # AUUC strictly beats the random baseline.
    auuc_random = _trapz_area(
        report.qini.fraction_treated, report.qini.random_baseline
    )
    assert report.qini.auuc > auuc_random
    # Qini coefficient is positive (>= ~0.3 with perfect targeting on
    # this DGP — leave a margin for finite-sample noise).
    assert report.qini.qini_coefficient > 0.2
    # Top-decile ATT is substantially larger than bottom-decile ATT.
    quantiles = sorted(report.quantile_atts.keys())
    top_att = report.quantile_atts[quantiles[0]]
    bot_att = report.quantile_atts[quantiles[-1]]
    assert top_att > bot_att


def test_targeting_report_top_half_recovers_total_att() -> None:
    """With oracle CATE, the cumulative uplift at the top-50% mark
    should equal essentially the entire sample-wide ATT gain."""
    data = _make_heterogeneous(n=1000, seed=2)
    report = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
    )
    n = report.n_total
    frac = report.qini.fraction_treated
    lift = report.qini.lift
    # Find the lift at f = 0.5 by linear interpolation.
    lift_at_half = float(np.interp(0.5, frac, lift))
    total_gain = lift[-1]
    # Top-50% should capture >=85% of the total gain.
    assert lift_at_half / total_gain >= 0.85


# ---------------------------------------------------------------------------
# Random CATE => Qini ~ 0
# ---------------------------------------------------------------------------


def test_random_cate_gives_qini_near_zero() -> None:
    data = _make_random_cate(n=1000, seed=3)
    report = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
    )
    # Random ranking should leave Qini close to zero (within sampling noise).
    assert abs(report.qini.qini_coefficient) < 0.25


# ---------------------------------------------------------------------------
# Threshold behaviour
# ---------------------------------------------------------------------------


def test_threshold_changes_recommended_count() -> None:
    data = _make_heterogeneous(n=800, seed=4)
    r0 = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
        threshold=0.0,
    )
    r_high = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
        threshold=0.5,
    )
    # With CATE in {0, 2}, threshold=0 treats the +2 half (~400), and
    # threshold=0.5 still treats the +2 half — counts equal here. Check
    # threshold > 2 treats nobody.
    r_none = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
        threshold=10.0,
    )
    assert r_none.n_recommended_treat == 0
    # And threshold of -10 treats everyone.
    r_all = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
        threshold=-10.0,
    )
    assert r_all.n_recommended_treat == 800
    # Monotonicity: as threshold rises, recommended count is non-increasing.
    assert r0.n_recommended_treat >= r_high.n_recommended_treat >= r_none.n_recommended_treat
    assert r_high.threshold_used == 0.5


# ---------------------------------------------------------------------------
# Input-validation paths
# ---------------------------------------------------------------------------


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        build_targeting_report(
            cate_predictions=np.zeros(10),
            observed_outcomes=np.zeros(9),
            observed_treatments=np.zeros(10),
        )


def test_non_binary_treatment_raises() -> None:
    with pytest.raises(ValueError, match="binary"):
        build_targeting_report(
            cate_predictions=np.zeros(10),
            observed_outcomes=np.zeros(10),
            observed_treatments=np.array([0, 1, 2] * 3 + [0]),
        )


def test_propensity_override_changes_epv() -> None:
    """Passing an explicit propensity reweights the EPV calculation."""
    data = _make_heterogeneous(n=400, seed=5)
    report_default = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
    )
    report_custom = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
        propensity=np.full(400, 0.7),  # over-emphasizes control units
    )
    # The EPV numbers differ (we don't care about direction — just that
    # the propensity input actually flows through the IPW formula).
    assert report_default.expected_policy_value != report_custom.expected_policy_value


# ---------------------------------------------------------------------------
# policy_tree — econml backend available in this env
# ---------------------------------------------------------------------------


def test_policy_tree_econml_backend() -> None:
    """When econml is installed, policy_tree returns an econml-backed
    PolicyTreeResult and the predict_fn maps rows to {0, 1}."""
    econml_policy = pytest.importorskip("econml.policy")
    assert econml_policy is not None
    data = _make_heterogeneous(n=400, seed=6)
    X = pd.DataFrame({"x1": data["x1"]})
    result = policy_tree(X=X, cate_predictions=data["cate"], max_depth=3)
    assert result is not None
    assert result.backend == "econml"
    assert result.feature_names == ["x1"]
    assert result.tree is not None
    # predict_fn should produce binary recommendations and treat the
    # positive-x1 half (where CATE = +2).
    pred = result.predict_fn(X)
    assert set(np.unique(pred).tolist()).issubset({0, 1})
    # Most positive-x1 rows should be recommended for treatment.
    treat_rate_high = pred[data["x1"] > 0].mean()
    treat_rate_low = pred[data["x1"] <= 0].mean()
    assert treat_rate_high > treat_rate_low


def test_policy_tree_returns_none_when_unavailable(monkeypatch) -> None:
    """If neither econml.policy nor R policytree is importable, the
    function returns None instead of raising."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name.startswith("econml.policy") or name == "policytree":
            raise ImportError(f"simulated absence of {name}")
        if name.startswith("rpy2"):
            raise ImportError(f"simulated absence of {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    X = pd.DataFrame({"x1": np.linspace(-1, 1, 50)})
    cate = np.linspace(-1, 1, 50)
    result = policy_tree(X=X, cate_predictions=cate, max_depth=2)
    assert result is None


def test_policy_tree_validates_inputs() -> None:
    with pytest.raises(TypeError, match="DataFrame"):
        policy_tree(X=np.zeros((10, 2)), cate_predictions=np.zeros(10))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length"):
        policy_tree(
            X=pd.DataFrame({"x": np.zeros(10)}),
            cate_predictions=np.zeros(9),
        )


# ---------------------------------------------------------------------------
# UpliftCurve dataclass smoke test
# ---------------------------------------------------------------------------


def test_uplift_curve_arrays_are_aligned() -> None:
    data = _make_heterogeneous(n=200, seed=7)
    report = build_targeting_report(
        cate_predictions=data["cate"],
        observed_outcomes=data["y"],
        observed_treatments=data["t"],
    )
    curve: UpliftCurve = report.qini
    n_points = len(curve.fraction_treated)
    assert n_points == len(curve.lift)
    assert n_points == len(curve.random_baseline)
    # Random baseline is monotone in fraction (when total_ATT > 0).
    diffs = np.diff(curve.random_baseline)
    if curve.random_baseline[-1] > 0:
        assert np.all(diffs >= -1e-9)
