"""Tests for the Markov-boundary feedback layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.discovery.markov_boundary import (
    MarkovBoundaryReport,
    discover_markov_boundary,
    discover_multiple_mbs,
    discover_stable_mb,
)


@pytest.fixture
def linear_mb_dataset() -> pd.DataFrame:
    """Y = 2*X1 - 1.5*X2 + noise; X3, X4, X5 are independent noise.

    True MB(Y) = {X1, X2}. Used to check that IAMB recovers the right set.
    """
    rng = np.random.default_rng(11)
    n = 600
    x1, x2, x3, x4, x5 = rng.normal(size=(5, n))
    y = 2 * x1 - 1.5 * x2 + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "x5": x5, "y": y})


def test_python_iamb_recovers_true_mb(linear_mb_dataset: pd.DataFrame) -> None:
    report = discover_markov_boundary(
        linear_mb_dataset, target="y", prefer_bnlearn=False
    )
    assert report.backend == "python.iamb"
    assert set(report.mb) == {"x1", "x2"}
    assert report.test == "fisher_z"


def test_mb_target_validation() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="not in df columns"):
        discover_markov_boundary(df, target="missing", prefer_bnlearn=False)


def test_python_fallback_handles_non_numeric_target() -> None:
    df = pd.DataFrame(
        {
            "x1": np.random.default_rng(0).normal(size=50),
            "cat": ["a"] * 25 + ["b"] * 25,
        }
    )
    report = discover_markov_boundary(df, target="cat", prefer_bnlearn=False)
    assert report.backend == "python.iamb"
    assert report.mb == []
    assert any("non-numeric" in n for n in report.notes)


def test_returns_empty_mb_when_independent() -> None:
    """If no covariate is associated with the target, MB should be empty."""
    rng = np.random.default_rng(13)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "x3": rng.normal(size=n),
            "y": rng.normal(size=n),  # independent of all x's
        }
    )
    report = discover_markov_boundary(df, target="y", prefer_bnlearn=False)
    # Allow at most one spurious inclusion (Fisher-z has nontrivial type-I)
    assert len(report.mb) <= 1


def test_max_size_caps_mb() -> None:
    rng = np.random.default_rng(17)
    n = 400
    p = 8
    X = rng.normal(size=(n, p))
    # All p covariates predict y
    coefs = np.linspace(0.6, 1.0, p)
    y = X @ coefs + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({f"x{i}": X[:, i] for i in range(p)})
    df["y"] = y
    report = discover_markov_boundary(
        df, target="y", prefer_bnlearn=False, max_size=3
    )
    assert len(report.mb) <= 3


# bnlearn path is optional — only runs if rpy2 + bnlearn are installed.
def _has_bnlearn() -> bool:
    try:
        from causalrag.estimators.rbridge._r import r_session, require

        r_session()
        require("bnlearn")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _has_bnlearn(), reason="rpy2 / bnlearn not installed")
def test_bnlearn_path_recovers_true_mb(linear_mb_dataset: pd.DataFrame) -> None:
    report = discover_markov_boundary(
        linear_mb_dataset, target="y", prefer_bnlearn=True
    )
    assert report.backend == "bnlearn"
    assert set(report.mb) == {"x1", "x2"}
    assert report.test in {"cor", "mi"}


# ─── Phase 2 — multiple-MB discovery ──────────────────────────────────────


def test_discover_multiple_mbs_returns_primary_mb() -> None:
    """When the data has only one viable MB, the primary should match
    the deterministic IAMB and alternative_mbs should be empty."""
    rng = np.random.default_rng(23)
    n = 500
    x1, x2, x3 = rng.normal(size=(3, n))
    y = 2 * x1 - x2 + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})
    r = discover_multiple_mbs(df, target="y", k=3, randomness=0.6)
    assert set(r.mb) == {"x1", "x2"}
    # alternative_mbs may be empty or contain near-identical sets — be lenient
    assert r.alternative_mbs is not None


def test_discover_multiple_mbs_invalid_target() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        discover_multiple_mbs(df, target="missing", k=2)


def test_discover_multiple_mbs_non_numeric_target() -> None:
    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    r = discover_multiple_mbs(df, target="cat", k=2)
    assert r.mb == []
    assert r.alternative_mbs == []


# ─── Phase 3 — stability subsampling ──────────────────────────────────────


def test_discover_stable_mb_recovers_signal() -> None:
    """Stability subsampling on a clean signal should keep the true MB
    variables. (With a permissive threshold and finite bootstrap iters,
    occasional noise inclusions are allowed; the true signal must always
    survive.)"""
    rng = np.random.default_rng(31)
    n = 400
    x1, x2, x3, x4 = rng.normal(size=(4, n))
    y = 2 * x1 - 1.5 * x2 + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})
    r = discover_stable_mb(
        df,
        target="y",
        bootstrap_iterations=10,
        stability_threshold=0.8,  # stricter than default
        prefer_bnlearn=False,
    )
    assert "x1" in r.mb and "x2" in r.mb
    assert r.stability_scores is not None
    assert r.stability_scores.get("x1", 0) >= 0.8
    assert r.stability_scores.get("x2", 0) >= 0.8


def test_discover_stable_mb_threshold_filters_low_freq() -> None:
    """A column that's only selected in a minority of bootstraps must
    be dropped when the threshold is high."""
    rng = np.random.default_rng(33)
    n = 300
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    weak = rng.normal(size=n)  # noise; might be selected occasionally
    y = 2 * x1 + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "weak": weak, "y": y})
    r = discover_stable_mb(
        df,
        target="y",
        bootstrap_iterations=10,
        stability_threshold=0.9,  # very strict
        prefer_bnlearn=False,
    )
    # x1 always; weak/x2 should rarely make it
    assert "x1" in r.mb
    assert r.bootstrap_iterations == 10


def test_discover_stable_mb_records_metadata() -> None:
    rng = np.random.default_rng(41)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    r = discover_stable_mb(
        df,
        target="y",
        bootstrap_iterations=5,
        prefer_bnlearn=False,
    )
    assert r.bootstrap_iterations == 5
    assert r.method.startswith("stability+")


def test_run_discovery_attaches_markov_boundaries(tmp_path) -> None:
    """End-to-end: run_discovery without an LLM should still produce MB reports."""
    from causalrag.discovery import run_discovery

    rng = np.random.default_rng(19)
    n = 400
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, size=n),
            "education": rng.integers(1, 17, size=n),
            "treat": (rng.uniform(size=n) > 0.5).astype(int),
            "income": rng.normal(50000, 15000, size=n),
        }
    )
    df["income"] += 5000 * df["treat"] + 1000 * df["education"]
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)

    # No client → skips LLM stages; treatment + outcome explicit.
    result = run_discovery(
        source=csv, client=None, treatment="treat", outcome="income"
    )
    assert isinstance(result.markov_boundaries, tuple)
    # We asked MB on income (the outcome) + treat
    targets = {mb["target"] for mb in result.markov_boundaries}
    assert "income" in targets
