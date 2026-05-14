"""Shared fixtures for synthetic-dataset integration tests.

These tests run against benchmark datasets with known data-generating processes
(Lalonde NSW, IHDP-flavored synthetic, ACIC-flavored synthetic, M-bias collider
injection, high-dim sparse-truth). The Lalonde data is real; everything else is
generated in-process with a fixed seed so the "ground truth" is exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def lalonde_nsw() -> pd.DataFrame:
    """Lalonde NSW (Dehejia-Wahba sample) via causaldata."""
    causaldata = pytest.importorskip("causaldata")
    ds = causaldata.nsw_mixtape.load_pandas().data
    return ds.drop(columns=["data_id"]).reset_index(drop=True)


@pytest.fixture
def ihdp_synthetic() -> tuple[pd.DataFrame, float]:
    """IHDP-flavored semi-synthetic: real-shape covariates, synthetic outcome
    with a known average treatment effect (≈ 4.0 in the Hill (2011) Response A
    setup; we use a simplified version here for reproducibility).

    Returns ``(df, true_ate)``.
    """
    rng = np.random.default_rng(11)
    n = 747  # IHDP sample size
    p_cont = 6
    p_bin = 19
    x_cont = rng.normal(size=(n, p_cont))
    x_bin = rng.binomial(1, 0.5, size=(n, p_bin))
    treat = rng.binomial(1, 1 / (1 + np.exp(-0.3 * x_cont.sum(axis=1))), size=n)
    # Outcome: heterogeneous CATE driven by the first two continuous covariates
    cate_per_row = 3.0 + 1.5 * x_cont[:, 0] + 0.5 * x_cont[:, 1]
    base = 2.0 * x_cont.sum(axis=1) + 0.3 * x_bin.sum(axis=1)
    y = base + treat * cate_per_row + rng.normal(scale=1.0, size=n)
    true_ate = float(cate_per_row.mean())
    df = pd.DataFrame(
        {
            **{f"x_cont_{i}": x_cont[:, i] for i in range(p_cont)},
            **{f"x_bin_{i}": x_bin[:, i] for i in range(p_bin)},
            "treat": treat.astype(float),
            "y": y,
        }
    )
    return df, true_ate


@pytest.fixture
def acic_synthetic() -> tuple[pd.DataFrame, float]:
    """ACIC-flavored synthetic: ~30 covariates, sparse-true DGP, known ATE."""
    rng = np.random.default_rng(2026)
    n = 2000
    p = 30
    x = rng.normal(size=(n, p))
    relevant = [0, 3, 7, 15]
    relevance = np.zeros(p)
    relevance[relevant] = [1.2, -0.8, 0.6, -0.4]
    propensity_logit = x @ relevance / 2
    p_treat = 1 / (1 + np.exp(-propensity_logit))
    treat = (rng.uniform(size=n) < p_treat).astype(float)
    true_ate = 2.5
    y = (
        x @ relevance
        + treat * true_ate
        + rng.normal(scale=1.0, size=n)
    )
    df = pd.DataFrame(
        {**{f"x{i}": x[:, i] for i in range(p)}, "treat": treat, "y": y}
    )
    return df, true_ate


@pytest.fixture
def m_bias_collider() -> tuple[pd.DataFrame, float]:
    """M-bias DGP: U1 → X ← U2, T → X, X → Y. Adjusting for X is a collider
    error and biases the estimate. The pipeline should catch this.

    Returns ``(df, true_ate)``.
    """
    rng = np.random.default_rng(7)
    n = 1000
    u1 = rng.normal(size=n)
    u2 = rng.normal(size=n)
    treat = (rng.uniform(size=n) < 0.5).astype(float)
    # X is a post-treatment collider: caused by U1, U2, and treat
    x_collider = 0.7 * u1 + 0.7 * u2 + 0.5 * treat + rng.normal(scale=0.3, size=n)
    true_ate = 1.5
    y = treat * true_ate + 0.5 * u2 + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame(
        {
            "u1_observed": u1,  # a true confounder PROXY (observed)
            "x_collider": x_collider,  # the trap — looks confounder-like
            "treat": treat,
            "y": y,
        }
    )
    return df, true_ate


@pytest.fixture
def high_dim_sparse() -> tuple[pd.DataFrame, float, list[str]]:
    """High-dim sparse-truth: p ≈ sqrt(n), only 3 of p covariates relevant.

    Returns ``(df, true_ate, relevant_columns)``.
    """
    rng = np.random.default_rng(42)
    n = 500
    p = 60
    x = rng.normal(size=(n, p))
    relevant_idx = [2, 11, 37]
    coefs = np.zeros(p)
    coefs[relevant_idx] = [1.0, -0.8, 0.6]
    propensity_logit = x @ coefs / 2
    treat = (rng.uniform(size=n) < 1 / (1 + np.exp(-propensity_logit))).astype(float)
    true_ate = 1.8
    y = x @ coefs + treat * true_ate + rng.normal(size=n)
    df = pd.DataFrame(
        {**{f"x{i}": x[:, i] for i in range(p)}, "treat": treat, "y": y}
    )
    return df, true_ate, [f"x{i}" for i in relevant_idx]


@pytest.fixture
def survival_synthetic() -> pd.DataFrame:
    """Right-censored survival outcome with paired (time, event) columns."""
    rng = np.random.default_rng(3)
    n = 600
    age = rng.normal(loc=60, scale=10, size=n)
    treat = (rng.uniform(size=n) < 0.5).astype(int)
    base_hazard = np.exp(-2.0 - 0.3 * treat + 0.05 * (age - 60))
    true_time = rng.exponential(1 / base_hazard)
    censor_time = rng.exponential(scale=5.0, size=n)
    time = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(int)
    return pd.DataFrame(
        {"age": age, "treat": treat.astype(float), "overall_time_days": time, "overall_event": event}
    )


@pytest.fixture
def mixed_types_dataset() -> pd.DataFrame:
    """Synthetic with mixed types: strings, dates, booleans, categoricals,
    free text, identifiers. Stresses the auto_preprocess pipeline."""
    rng = np.random.default_rng(99)
    n = 400
    return pd.DataFrame(
        {
            "patient_id": [f"PT-{i:05d}" for i in range(n)],
            "enrollment_date": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
            "site": rng.choice(["site_A", "site_B", "site_C", "site_D"], size=n),
            "active": rng.choice([True, False], size=n),
            "notes": ["free-form clinical note text " + str(i) for i in range(n)],
            "age": rng.normal(loc=50, scale=12, size=n),
            "income": rng.gamma(shape=2.0, scale=20000, size=n),  # skewed
            "treat": rng.binomial(1, 0.5, size=n).astype(float),
            "y": rng.normal(size=n),
        }
    )
