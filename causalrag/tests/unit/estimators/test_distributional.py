"""Tests for distributional / quantile-regression estimators (Sprint 7.7).

Design of the synthetic DGP
---------------------------
Treatment ``T`` is binary with propensity depending on ``x``. Under T=0, the
outcome is N(mu0, sigma0^2). Under T=1, the outcome is drawn from the same
location but with a *much heavier upper tail* — specifically a Gaussian with
an additive exponential right-tail bump. This way the average effect is
small, the median (tau=0.5) effect is small, but the tau=0.9 effect is
large. Firpo's RIF UQPE at tau=0.9 must dominate Firpo at tau=0.5, and the
CFM QTE curve must be monotone non-decreasing in tau over the chosen grid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from causalrag.core.protocol import StudyProtocol  # noqa: E402
from causalrag.core.registry import get_registry  # noqa: E402
from causalrag.estimators.python.distributional import (  # noqa: E402
    CFVCounterfactualDistribution,
    DiNardoFortinLemieuxReweighting,
    FirpoRIFQuantileEstimator,
)


def _synthesize_tail_shift(n: int = 1500, seed: int = 11) -> pd.DataFrame:
    """T shifts the upper tail of Y far more than the median.

    Y(0) ~ N(0, 1).
    Y(1) = N(0, 1) + Exp(rate=0.5)  (a positive, heavy upper tail).

    So E[Y(1) - Y(0)] ≈ 2, but Q_0.5(Y(1)) - Q_0.5(Y(0)) is only ~ ln(2)/0.5
    ≈ 1.39 of an exponential plus a tiny shift, whereas Q_0.9(Y(1)) -
    Q_0.9(Y(0)) is much larger (Exp(0.5) has 0.9-quantile ≈ 4.6, vs ~ 1.28
    for the Gaussian — a gap of several units once you condition on the
    higher-tail behaviour).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.3 * x1 - 0.2 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(float)
    base = rng.normal(loc=0.0, scale=1.0, size=n)
    bump = rng.exponential(scale=1.0 / 0.5, size=n)  # mean 2, heavy tail
    # Y is base + t * bump + a small confounder effect (x1) so adjustment matters.
    y = base + t * bump + 0.3 * x1
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


# ---------------------------------------------------------------------------
# Registration / catalog
# ---------------------------------------------------------------------------

def test_distributional_estimators_register_in_catalog() -> None:
    reg = get_registry()
    ids = {entry.id for entry in reg.all()}
    assert "python.firpo.rif_quantile" in ids
    assert "python.cfvm.counterfactual_dist" in ids
    assert "python.dfl.reweighting" in ids


# ---------------------------------------------------------------------------
# Firpo 2007 RIF
# ---------------------------------------------------------------------------

def test_firpo_rif_quantile_returns_well_shaped_result() -> None:
    df = _synthesize_tail_shift(n=800)
    est = FirpoRIFQuantileEstimator(
        treatment="t",
        outcome="y",
        confounders=("x1", "x2"),
        tau=0.5,
        random_state=7,
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()
    assert result.estimator_id == "python.firpo.rif_quantile"
    assert result.estimand_class == "QUANTILE_TREATMENT_EFFECT"
    assert result.n_used > 0
    assert result.se is not None and result.se > 0
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low < result.ci_high
    assert "tau" in result.diagnostics and result.diagnostics["tau"] == 0.5
    assert result.diagnostics["q_tau"] is not None


def test_firpo_rejects_tau_outside_unit_interval() -> None:
    with pytest.raises(ValueError):
        FirpoRIFQuantileEstimator(
            treatment="t", outcome="y", confounders=(), tau=0.0
        )
    with pytest.raises(ValueError):
        FirpoRIFQuantileEstimator(
            treatment="t", outcome="y", confounders=(), tau=1.0
        )


def test_firpo_rif_detects_larger_effect_at_high_quantile() -> None:
    """Key Sprint 7.7 acceptance criterion: tau=0.9 ⇒ larger effect than tau=0.5."""
    df = _synthesize_tail_shift(n=2000)
    est_med = FirpoRIFQuantileEstimator(
        treatment="t", outcome="y", confounders=("x1", "x2"), tau=0.5, random_state=7
    ).fit(df, StudyProtocol(name="smoke"))
    est_hi = FirpoRIFQuantileEstimator(
        treatment="t", outcome="y", confounders=("x1", "x2"), tau=0.9, random_state=7
    ).fit(df, StudyProtocol(name="smoke"))

    res_med = est_med.estimate()
    res_hi = est_hi.estimate()
    # Heavy upper tail of Y(1) — the 0.9-effect must exceed the median effect.
    assert res_hi.point_estimate > res_med.point_estimate
    # And by a meaningful margin (the upper tail is ~ exp(rate=0.5)).
    assert (res_hi.point_estimate - res_med.point_estimate) > 0.5


# ---------------------------------------------------------------------------
# CFM 2013 counterfactual distribution
# ---------------------------------------------------------------------------

def test_cfvm_returns_qte_curve_and_monotone() -> None:
    df = _synthesize_tail_shift(n=2000)
    est = CFVCounterfactualDistribution(
        treatment="t",
        outcome="y",
        confounders=("x1", "x2"),
        tau_grid=(0.1, 0.25, 0.5, 0.75, 0.9),
        n_thresholds=30,
        random_state=7,
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()
    assert result.estimator_id == "python.cfvm.counterfactual_dist"
    assert result.estimand_class == "COUNTERFACTUAL_DISTRIBUTION"

    curve = est.qte_curve()
    qte = curve["qte"]
    assert qte.shape == (5,)
    # Monotone non-decreasing — under heavier upper tail in T=1 the QTE
    # curve should grow (weakly) with tau. Allow a small numerical slack
    # at adjacent grid points since logistic distribution regression at
    # discrete thresholds can wiggle by ~ 0.05.
    diffs = np.diff(qte)
    # Mostly monotone with tiny slack.
    assert np.all(diffs >= -0.1)
    # Strict global increase from tau=0.1 to tau=0.9.
    assert qte[-1] > qte[0]
    # qte at 0.9 should clearly exceed qte at 0.5.
    assert qte[-1] > qte[2]

    # Diagnostics surface the full curve.
    assert "qte_curve" in result.diagnostics
    assert len(result.diagnostics["qte_curve"]) == 5


def test_cfvm_rejects_invalid_tau_grid() -> None:
    with pytest.raises(ValueError):
        CFVCounterfactualDistribution(
            treatment="t", outcome="y", confounders=(), tau_grid=()
        )
    with pytest.raises(ValueError):
        CFVCounterfactualDistribution(
            treatment="t", outcome="y", confounders=(), tau_grid=(0.0, 0.5)
        )
    with pytest.raises(ValueError):
        CFVCounterfactualDistribution(
            treatment="t", outcome="y", confounders=(), tau_grid=(0.5, 1.0)
        )


# ---------------------------------------------------------------------------
# DiNardo-Fortin-Lemieux
# ---------------------------------------------------------------------------

def test_dfl_returns_curve_and_detects_tail_shift() -> None:
    df = _synthesize_tail_shift(n=2000)
    est = DiNardoFortinLemieuxReweighting(
        treatment="t",
        outcome="y",
        confounders=("x1", "x2"),
        tau=0.9,
        tau_grid=(0.1, 0.25, 0.5, 0.75, 0.9),
        random_state=7,
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()
    assert result.estimator_id == "python.dfl.reweighting"
    assert result.estimand_class == "COUNTERFACTUAL_DISTRIBUTION"

    # Point estimate is the QTE at tau = 0.9 — must be positive and large.
    assert result.point_estimate > 1.0

    curve = est.qte_curve()
    assert "qte" in curve
    # Upper-tail effect bigger than median effect.
    qte = curve["qte"]
    tau = curve["tau"]
    idx_90 = int(np.argmin(np.abs(tau - 0.9)))
    idx_50 = int(np.argmin(np.abs(tau - 0.5)))
    assert qte[idx_90] > qte[idx_50]


def test_dfl_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        DiNardoFortinLemieuxReweighting(
            treatment="t", outcome="y", confounders=(), tau=0.0
        )
    with pytest.raises(ValueError):
        DiNardoFortinLemieuxReweighting(
            treatment="t", outcome="y", confounders=(), trim=0.6
        )


def test_dfl_handles_no_covariates() -> None:
    """No covariates ⇒ identity reweighting; smoke-test only."""
    df = _synthesize_tail_shift(n=600)
    est = DiNardoFortinLemieuxReweighting(
        treatment="t", outcome="y", confounders=(), tau=0.5, random_state=7
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()
    assert np.isfinite(result.point_estimate)
