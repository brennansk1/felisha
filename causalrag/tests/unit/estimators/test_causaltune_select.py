"""Tests for causaltune-style ground-truth-free estimator selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.estimators.causaltune_select import (
    LeaderboardEntry,
    energy_distance,
    energy_score,
    erupt,
    select_best_estimator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantCATE:
    """Estimator stub with a fixed CATE for every row."""

    def __init__(self, value: float, name: str = "const") -> None:
        self._value = float(value)
        self.id = name

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self._value, dtype=np.float64)


class _OracleCATE:
    """Estimator stub that returns the oracle CATE.

    Treats the first column of X as the modifier: CATE = beta_0 + beta_1 * X[:, 0].
    """

    def __init__(self, beta_0: float, beta_1: float, name: str = "oracle") -> None:
        self.beta_0 = float(beta_0)
        self.beta_1 = float(beta_1)
        self.id = name

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self.beta_0 + self.beta_1 * X[:, 0]


def _make_synth(
    n: int = 800, seed: int = 0, tau_fn=None
) -> pd.DataFrame:
    """Synthetic dataset with covariate x1, binary T, outcome y.

    Y = 1 + 0.5*x1 + tau(x1)*T + noise, T ~ Bernoulli(sigmoid(0.5*x1)).
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    p = 1.0 / (1.0 + np.exp(-0.5 * x1))
    t = (rng.uniform(size=n) < p).astype(float)
    if tau_fn is None:
        tau_fn = lambda x: 1.0 + 0.0 * x  # constant CATE = 1
    tau = tau_fn(x1)
    y = 1.0 + 0.5 * x1 + tau * t + rng.normal(0.0, 0.3, size=n)
    return pd.DataFrame({"x1": x1, "t": t, "y": y})


# ---------------------------------------------------------------------------
# energy_distance
# ---------------------------------------------------------------------------


def test_energy_distance_zero_on_identical_samples() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    assert energy_distance(x, x) == pytest.approx(0.0, abs=1e-9)


def test_energy_distance_positive_on_different_samples() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=400)
    y = rng.normal(3.0, 1.0, size=400)
    d_diff = energy_distance(x, y)
    d_same = energy_distance(x, x[: len(y)])
    assert d_diff > d_same
    assert d_diff > 0.5


def test_energy_distance_inf_on_empty() -> None:
    assert energy_distance(np.array([]), np.array([1.0])) == float("inf")
    assert energy_distance(np.array([1.0]), np.array([])) == float("inf")


# ---------------------------------------------------------------------------
# energy_score: correctly-specified beats misspecified
# ---------------------------------------------------------------------------


def test_energy_score_prefers_correct_estimator() -> None:
    # True CATE = 1 everywhere.
    df = _make_synth(n=800, seed=1, tau_fn=lambda x: np.ones_like(x))
    good = _ConstantCATE(1.0, name="good")
    bad = _ConstantCATE(-5.0, name="bad")
    es_good = energy_score(good, df, "t", "y", ("x1",))
    es_bad = energy_score(bad, df, "t", "y", ("x1",))
    assert es_good < es_bad


# ---------------------------------------------------------------------------
# ERUPT: oracle policy beats random / all-treat
# ---------------------------------------------------------------------------


def test_erupt_higher_for_oracle_policy() -> None:
    # Subgroup A (x1 > 0) has positive CATE (+2). Subgroup B (x1 <= 0)
    # has negative CATE (-2). Oracle policy: treat A, withhold from B.
    def tau_fn(x):
        return np.where(x > 0, 2.0, -2.0)

    df = _make_synth(n=1200, seed=2, tau_fn=tau_fn)
    oracle = _OracleCATE(beta_0=0.0, beta_1=10.0)  # picks sign of x1
    # Inverted (anti-oracle): treats B, withholds from A.
    anti = _OracleCATE(beta_0=0.0, beta_1=-10.0, name="anti")
    er_oracle = erupt(oracle, df, "t", "y", ("x1",))
    er_anti = erupt(anti, df, "t", "y", ("x1",))
    assert er_oracle > er_anti


# ---------------------------------------------------------------------------
# select_best_estimator
# ---------------------------------------------------------------------------


def test_select_best_estimator_picks_good_one() -> None:
    df = _make_synth(n=800, seed=3, tau_fn=lambda x: np.ones_like(x))
    good = _ConstantCATE(1.0, name="good")
    mediocre = _ConstantCATE(0.5, name="mediocre")
    bad = _ConstantCATE(-3.0, name="bad")
    result = select_best_estimator(
        [good, mediocre, bad], df, "t", "y", ("x1",)
    )
    assert result["best_name"] == "good"
    assert result["best"] is good
    lb = result["leaderboard"]
    assert len(lb) == 3
    assert all(isinstance(e, LeaderboardEntry) for e in lb)


def test_select_best_estimator_accepts_mapping() -> None:
    df = _make_synth(n=600, seed=4, tau_fn=lambda x: np.ones_like(x))
    cands = {
        "a": _ConstantCATE(1.0),
        "b": _ConstantCATE(-2.0),
    }
    result = select_best_estimator(cands, df, "t", "y", ("x1",))
    assert result["best_name"] == "a"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_df_returns_inf_and_neg_inf() -> None:
    df = pd.DataFrame({"x1": [], "t": [], "y": []})
    est = _ConstantCATE(1.0)
    assert energy_score(est, df, "t", "y", ("x1",)) == float("inf")
    assert erupt(est, df, "t", "y", ("x1",)) == float("-inf")


def test_all_treated_df_returns_degenerate_values() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=100),
            "t": np.ones(100),
            "y": rng.normal(size=100),
        }
    )
    est = _ConstantCATE(1.0)
    assert energy_score(est, df, "t", "y", ("x1",)) == float("inf")
    assert erupt(est, df, "t", "y", ("x1",)) == float("-inf")


def test_no_treatment_variance_df_returns_degenerate() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=100),
            "t": np.zeros(100),
            "y": rng.normal(size=100),
        }
    )
    est = _ConstantCATE(1.0)
    assert energy_score(est, df, "t", "y", ("x1",)) == float("inf")
    assert erupt(est, df, "t", "y", ("x1",)) == float("-inf")


def test_select_best_empty_candidates() -> None:
    df = _make_synth(n=200, seed=5)
    result = select_best_estimator([], df, "t", "y", ("x1",))
    assert result["best"] is None
    assert result["best_name"] is None
    assert result["leaderboard"] == []
