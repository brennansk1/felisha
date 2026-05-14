"""Tests for the proximal-CI two-stage regression estimator.

These exercise the headline claim of proximal causal inference: when there
is an unmeasured confounder ``U`` of treatment and outcome, but the data
include a *pair* of valid proxies (a negative-control exposure NCE and a
negative-control outcome NCO), the ATE is point-identified, while a naive
regression that simply omits ``U`` is systematically biased.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.core.protocol import StudyProtocol
from causalrag.estimators.python.proximal import ProximalRegressionEstimator


def _synthesize_proximal(
    n: int,
    true_ate: float = 1.5,
    seed: int = 0,
    confound_strength: float = 2.0,
) -> pd.DataFrame:
    """Synthetic dataset with a latent confounder ``U`` plus an (NCE, NCO) pair.

    DAG:
      U  -> T,  U -> Y,  U -> W (NCE),  U -> Z (NCO)
      X  -> T,  X -> Y                (observed confounder)
      T  -> Y                         (the causal effect we want to recover)

    Crucially, T does NOT cause NCE and NCO does NOT directly cause Y, so
    the (W, Z) pair satisfies the negative-control conditions. A regression
    of Y on T and X (omitting U) is biased by ``confound_strength`` times
    the U-on-T coefficient; the proximal estimator should erase that bias.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    x = rng.normal(size=n)
    # Treatment driven by U and X (binary via logistic).
    logits = 1.2 * u + 0.5 * x
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(np.float64)
    # NCE: function of U only (plus noise). Not a function of T.
    w = 1.0 * u + 0.4 * rng.normal(size=n)
    # NCO: function of U only (plus noise). Different coefficient so that
    # the (W, Z) pair is linearly independent.
    z = 0.8 * u + 0.4 * rng.normal(size=n)
    # Outcome: true ATE * T + X + confound_strength * U + noise.
    y = (
        true_ate * t
        + 0.7 * x
        + confound_strength * u
        + 0.5 * rng.normal(size=n)
    )
    return pd.DataFrame(
        {"y": y, "t": t, "x": x, "nce": w, "nco": z, "u": u}
    )


def _naive_ols_ate(df: pd.DataFrame) -> float:
    """Naive OLS of Y on T and observed X — should be biased by latent U."""
    from numpy.linalg import lstsq

    n = len(df)
    design = np.column_stack(
        [np.ones(n), df["t"].to_numpy(), df["x"].to_numpy()]
    )
    coef, *_ = lstsq(design, df["y"].to_numpy(), rcond=None)
    return float(coef[1])


# ---------------------------------------------------------------------------
# Headline test: proximal recovers, naive fails
# ---------------------------------------------------------------------------

def test_proximal_recovers_ate_while_naive_regression_is_biased() -> None:
    true_ate = 1.5
    df = _synthesize_proximal(n=1000, true_ate=true_ate, seed=7)

    est = ProximalRegressionEstimator(
        treatment="t",
        outcome="y",
        confounders=("x",),
        negative_control_exposure="nce",
        negative_control_outcome="nco",
        n_folds=5,
        bootstrap_iterations=80,
        seed=42,
    )
    est.fit(df, StudyProtocol(name="proximal-smoke"))
    result = est.estimate()

    # Proximal point is close to truth.
    assert result.estimator_id == "python.proximal.regression"
    assert result.estimand_class == "ATE"
    assert result.n_used == 1000
    assert result.se is not None and result.se > 0
    assert result.ci_low is not None and result.ci_high is not None
    # The headline claim: the truth lies within ~2 bootstrap SE.
    assert abs(result.point_estimate - true_ate) < 2.0 * result.se, (
        f"proximal point {result.point_estimate} too far from {true_ate} "
        f"(SE={result.se})"
    )

    # And the naive regression must be visibly biased — bigger error than
    # the proximal estimator, and well outside the proximal SE band.
    naive = _naive_ols_ate(df)
    naive_err = abs(naive - true_ate)
    prox_err = abs(result.point_estimate - true_ate)
    assert naive_err > 0.3, (
        f"naive OLS not biased enough to make this test meaningful "
        f"(naive={naive}, truth={true_ate})"
    )
    assert prox_err < naive_err, (
        f"proximal ({prox_err}) should beat naive ({naive_err})"
    )


# ---------------------------------------------------------------------------
# Configuration / guard tests
# ---------------------------------------------------------------------------

def test_missing_nce_or_nco_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="negative_control"):
        ProximalRegressionEstimator(
            treatment="t",
            outcome="y",
            confounders=("x",),
            negative_control_exposure="",
            negative_control_outcome="nco",
        )
    with pytest.raises(ValueError, match="negative_control"):
        ProximalRegressionEstimator(
            treatment="t",
            outcome="y",
            confounders=("x",),
            negative_control_exposure="nce",
            negative_control_outcome="",
        )


def test_missing_proxy_column_in_data_raises() -> None:
    df = _synthesize_proximal(n=400, seed=1).drop(columns=["nce"])
    est = ProximalRegressionEstimator(
        treatment="t",
        outcome="y",
        confounders=("x",),
        negative_control_exposure="nce",
        negative_control_outcome="nco",
        bootstrap_iterations=10,
    )
    with pytest.raises(ValueError, match="nce"):
        est.fit(df, StudyProtocol(name="proximal-missing-col"))


def test_min_sample_size_guard_at_n_200() -> None:
    df = _synthesize_proximal(n=200, seed=2)
    est = ProximalRegressionEstimator(
        treatment="t",
        outcome="y",
        confounders=("x",),
        negative_control_exposure="nce",
        negative_control_outcome="nco",
        bootstrap_iterations=10,
    )
    with pytest.raises(ValueError, match="300"):
        est.fit(df, StudyProtocol(name="proximal-tiny"))


# ---------------------------------------------------------------------------
# Cross-fit vs single-fit behaviour
# ---------------------------------------------------------------------------

def test_cross_fitting_is_used_and_recorded_in_diagnostics() -> None:
    df = _synthesize_proximal(n=600, seed=3)
    est = ProximalRegressionEstimator(
        treatment="t",
        outcome="y",
        confounders=("x",),
        negative_control_exposure="nce",
        negative_control_outcome="nco",
        n_folds=5,
        bootstrap_iterations=20,
        seed=11,
    )
    est.fit(df, StudyProtocol(name="proximal-cv"))
    res = est.estimate()
    assert res.diagnostics["cross_fit_used"] is True
    assert res.diagnostics["n_folds"] == 5
    # Single-fit baseline is recorded alongside.
    assert res.diagnostics["point_single_fit"] is not None
    # Diagnose surfaces the stage-1 proxy strength.
    diag = est.diagnose()
    assert "nce_stage1_r2" in diag
    assert 0.0 <= diag["nce_stage1_r2"] <= 1.0


def test_cross_fit_point_differs_from_single_fit_point_on_average() -> None:
    """The whole point of cross-fitting is to give a *different* (less
    plug-in-biased) point estimate than a single fit. Across many seeds the
    two estimators should track the truth equally well in expectation, but
    on any one sample they differ — we assert that the spread of (cross-fit
    minus single-fit) is non-trivially nonzero across seeds.
    """
    deltas: list[float] = []
    truth = 1.5
    for seed in range(8):
        df = _synthesize_proximal(n=500, true_ate=truth, seed=100 + seed)
        est = ProximalRegressionEstimator(
            treatment="t",
            outcome="y",
            confounders=("x",),
            negative_control_exposure="nce",
            negative_control_outcome="nco",
            n_folds=5,
            bootstrap_iterations=0,  # don't waste time bootstrapping here
            seed=seed,
        )
        est.fit(df, StudyProtocol(name="proximal-cv-vs-single"))
        res = est.estimate()
        single = res.diagnostics["point_single_fit"]
        deltas.append(res.point_estimate - float(single))
    arr = np.asarray(deltas, dtype=np.float64)
    # The two estimators are not literally identical across seeds.
    assert np.std(arr) > 0.0


# ---------------------------------------------------------------------------
# Registry / refutation
# ---------------------------------------------------------------------------

def test_estimator_is_registered() -> None:
    from causalrag.core.registry import get_registry

    reg = get_registry()
    entry = reg.get(ProximalRegressionEstimator.id)
    assert entry.factory is ProximalRegressionEstimator
    assert "ATE" in entry.supported_estimands
    assert entry.min_sample_size == 300


def test_refute_swaps_proxies_and_reports_delta() -> None:
    df = _synthesize_proximal(n=500, seed=5)
    est = ProximalRegressionEstimator(
        treatment="t",
        outcome="y",
        confounders=("x",),
        negative_control_exposure="nce",
        negative_control_outcome="nco",
        bootstrap_iterations=0,
        seed=9,
    )
    est.fit(df, StudyProtocol(name="proximal-refute"))
    refutation = est.refute()
    assert "ate_original_proxies" in refutation
    assert "ate_swapped_proxies" in refutation
    assert "swap_delta" in refutation
