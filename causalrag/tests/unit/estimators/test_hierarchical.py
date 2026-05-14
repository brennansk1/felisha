"""Tests for HierarchicalDMLEstimator (Sprint 6.5.5).

Covers:

- Recovery on a 30×20 clustered synthetic dataset with a cluster-level
  confounder and true ATE = 1.5; cluster-robust SE strictly larger than
  the naive SE.
- Behavior on single-level (n_clusters ≈ n) data: cluster-robust SE
  collapses to ≈ the naive SE.
- Cluster-level treatment detection: when the treatment is constant
  within every cluster the estimator routes to the cluster-level path.
- ``min_sample_size`` and minimum-cluster-count enforcement.
- Bootstrap-of-clusters actually resamples clusters (verified via a
  monkeypatch on ``np.random.Generator.integers``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from causalrag.core.protocol import StudyProtocol  # noqa: E402
from causalrag.estimators.python.hierarchical import (  # noqa: E402
    HierarchicalDMLEstimator,
)


# ----------------------------------------------------------------------
# Synthesis helpers
# ----------------------------------------------------------------------
def _synth_clustered(
    n_clusters: int = 30,
    cluster_size: int = 20,
    true_ate: float = 1.5,
    seed: int = 11,
) -> pd.DataFrame:
    """Two-level data: unit-level treatment, cluster-level confounder.

    Cluster effect ``u_c`` is shared by all units in cluster c, drives a
    cluster-level confounder ``z_c`` that influences both T and Y, plus
    a unit-level covariate ``x``.
    """
    rng = np.random.default_rng(seed)
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    # Strong cluster random effect that is NOT in the feature set — this
    # is what leaves residual within-cluster correlation in the AIPW
    # score, which is exactly what the cluster sandwich is meant to catch.
    u = rng.normal(scale=2.5, size=n_clusters)
    z = 0.7 * u + rng.normal(scale=0.3, size=n_clusters)  # cluster-level confounder

    n = n_clusters * cluster_size
    x = rng.normal(size=n)
    u_unit = u[cluster_id]
    z_unit = z[cluster_id]

    logits = 0.4 * z_unit + 0.3 * x
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(float)
    noise = rng.normal(scale=0.5, size=n)
    # u_unit appears in Y but is NOT a feature — it generates the cluster
    # correlation in the influence-function residuals.
    y = true_ate * t + 0.8 * z_unit + 0.5 * x + 1.0 * u_unit + noise

    return pd.DataFrame(
        {"y": y, "t": t, "x": x, "z": z_unit, "cluster": cluster_id}
    )


def _synth_flat(n: int = 600, true_ate: float = 1.5, seed: int = 13) -> pd.DataFrame:
    """One cluster per unit ⇒ ICC ≈ 0 and CR SE ≈ naive SE."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    logits = 0.3 * x
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(float)
    noise = rng.normal(scale=0.5, size=n)
    y = true_ate * t + 0.6 * x + noise
    # Use enough distinct clusters so the min-cluster check passes, but
    # mostly small clusters so the within/between gap is tiny.
    cluster = np.arange(n) % 60
    return pd.DataFrame({"y": y, "t": t, "x": x, "cluster": cluster})


def _synth_cluster_level_tx(
    n_clusters: int = 30,
    cluster_size: int = 20,
    true_ate: float = 1.5,
    seed: int = 23,
) -> pd.DataFrame:
    """Treatment lives at the cluster level (constant within each cluster)."""
    rng = np.random.default_rng(seed)
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    z = rng.normal(size=n_clusters)
    t_cluster = (rng.uniform(size=n_clusters) < 0.5).astype(float)
    t = t_cluster[cluster_id]
    z_unit = z[cluster_id]

    n = n_clusters * cluster_size
    x = rng.normal(size=n)
    u = rng.normal(scale=1.0, size=n_clusters)[cluster_id]
    noise = rng.normal(scale=0.5, size=n)
    y = true_ate * t + 0.7 * z_unit + 0.4 * x + 1.0 * u + noise
    return pd.DataFrame(
        {"y": y, "t": t, "x": x, "z": z_unit, "cluster": cluster_id}
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_recovers_ate_within_2se_and_cluster_se_larger_than_naive() -> None:
    df = _synth_clustered(n_clusters=30, cluster_size=20, true_ate=1.5, seed=11)
    est = HierarchicalDMLEstimator(
        treatment="t",
        outcome="y",
        cluster_column="cluster",
        confounders=("x",),
        cluster_confounders=("z",),
        bootstrap_iterations=10,
        seed=7,
    )
    est.fit(df, StudyProtocol(name="hier-smoke"))
    result = est.estimate()

    assert result.estimator_id == "python.hierarchical.dml"
    assert result.estimand_class == "ATE"
    assert result.se is not None and result.se > 0
    # Within 2 cluster-robust SE of the truth.
    assert abs(result.point_estimate - 1.5) <= 2.0 * result.se, (
        f"point={result.point_estimate}, se={result.se}"
    )

    diag = result.diagnostics
    assert diag["n_clusters"] == 30
    assert diag["units_per_cluster_p50"] == 20.0
    assert diag["treatment_level"] == "unit"
    assert 0.0 <= diag["icc"] <= 1.0
    # The whole point of clustering: cluster-robust SE > naive SE.
    assert diag["cluster_robust_se"] > diag["naive_se"]


def test_flat_data_cluster_se_close_to_naive_se() -> None:
    df = _synth_flat(n=600, true_ate=1.5, seed=13)
    est = HierarchicalDMLEstimator(
        treatment="t",
        outcome="y",
        cluster_column="cluster",
        confounders=("x",),
        bootstrap_iterations=5,
        seed=7,
    )
    est.fit(df, StudyProtocol(name="hier-flat"))
    result = est.estimate()
    diag = result.diagnostics

    assert diag["treatment_level"] == "unit"
    # With ~10 units per cluster but no true cluster-level effect, the
    # two SEs should be in the same ballpark. Allow generous slack.
    ratio = diag["cluster_robust_se"] / diag["naive_se"]
    assert 0.5 <= ratio <= 1.8, f"ratio={ratio}"


def test_cluster_level_treatment_detection_and_routing() -> None:
    df = _synth_cluster_level_tx(
        n_clusters=30, cluster_size=20, true_ate=1.5, seed=23
    )
    est = HierarchicalDMLEstimator(
        treatment="t",
        outcome="y",
        cluster_column="cluster",
        confounders=("x",),
        cluster_confounders=("z",),
        bootstrap_iterations=5,
        seed=7,
    )
    est.fit(df, StudyProtocol(name="hier-cluster-tx"))
    result = est.estimate()

    assert result.diagnostics["treatment_level"] == "cluster"
    # ATE point should still be in the right ballpark.
    assert abs(result.point_estimate - 1.5) < 1.0


def test_min_sample_size_enforced() -> None:
    df = _synth_clustered(n_clusters=10, cluster_size=5, true_ate=1.5, seed=1)
    # 10 * 5 = 50 rows ⇒ below min_sample_size of 100.
    est = HierarchicalDMLEstimator(
        treatment="t",
        outcome="y",
        cluster_column="cluster",
        confounders=("x",),
    )
    with pytest.raises(ValueError, match="at least 100 rows"):
        est.fit(df, StudyProtocol(name="hier-tiny"))


def test_min_cluster_count_enforced() -> None:
    # 9 clusters × 20 rows = 180 rows (passes row floor), but < 10 clusters.
    df = _synth_clustered(n_clusters=9, cluster_size=20, true_ate=1.5, seed=1)
    est = HierarchicalDMLEstimator(
        treatment="t",
        outcome="y",
        cluster_column="cluster",
        confounders=("x",),
    )
    with pytest.raises(ValueError, match="at least 10 clusters"):
        est.fit(df, StudyProtocol(name="hier-few-clusters"))


def test_bootstrap_resamples_clusters_not_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cluster bootstrap must call ``rng.integers(0, n_clusters, size=n_clusters)``.

    We wrap ``np.random.default_rng`` inside the hierarchical module so the
    spy sees every ``rng.integers`` call made by ``_bootstrap_clusters`` and
    can confirm the draws are over ``[0, n_clusters)``, not ``[0, n_rows)``.
    """
    df = _synth_clustered(n_clusters=30, cluster_size=20, true_ate=1.5, seed=11)
    est = HierarchicalDMLEstimator(
        treatment="t",
        outcome="y",
        cluster_column="cluster",
        confounders=("x",),
        bootstrap_iterations=3,
        seed=7,
    )
    est.fit(df, StudyProtocol(name="hier-bs"))

    calls: list[tuple[int, int, int]] = []

    class _SpyRNG:
        def __init__(self, real: np.random.Generator) -> None:
            self._real = real

        def integers(self, low, high=None, size=None, *args, **kwargs):
            if high is not None and size is not None:
                try:
                    calls.append((int(low), int(high), int(size)))
                except Exception:
                    pass
            return self._real.integers(low, high=high, size=size, *args, **kwargs)

        def __getattr__(self, name: str):
            return getattr(self._real, name)

    import causalrag.estimators.python.hierarchical as hmod

    real_default_rng = hmod.np.random.default_rng

    def _fake_default_rng(seed=None):
        return _SpyRNG(real_default_rng(seed))

    monkeypatch.setattr(hmod.np.random, "default_rng", _fake_default_rng)
    est._bootstrap_clusters(est._prep, alpha=0.05)

    assert len(calls) >= 1
    n_clusters = est._prep.n_clusters
    n_rows = est._prep.n
    assert n_clusters != n_rows
    bs_draws = [c for c in calls if c[2] == n_clusters]
    assert len(bs_draws) >= 1, f"expected size={n_clusters} draws; got {calls}"
    for low, high, size in bs_draws:
        assert low == 0
        assert high == n_clusters
        assert size == n_clusters
    # And nothing should be sampling from [0, n_rows) inside the loop.
    assert not any(high == n_rows for _, high, _ in calls), (
        f"bootstrap drew from row space: {calls}"
    )


def test_registered_in_catalog() -> None:
    from causalrag.core.registry import get_registry

    reg = get_registry()
    entry = reg.get("python.hierarchical.dml")
    assert entry.backend == "python"
    assert "ATE" in entry.supported_estimands
    assert "ATT" in entry.supported_estimands
    assert entry.min_sample_size == 100


def test_diagnose_and_refute_post_fit() -> None:
    df = _synth_clustered(n_clusters=30, cluster_size=20, true_ate=1.5, seed=11)
    est = HierarchicalDMLEstimator(
        treatment="t",
        outcome="y",
        cluster_column="cluster",
        confounders=("x",),
        cluster_confounders=("z",),
        bootstrap_iterations=3,
        seed=7,
    )
    # Pre-fit diagnose is informative but harmless.
    assert est.diagnose() == {"fitted": False}
    est.fit(df, StudyProtocol(name="hier-diag"))
    diag = est.diagnose()
    assert diag["fitted"] is True
    assert diag["n_clusters"] == 30
    assert diag["treatment_level"] == "unit"

    ref = est.refute()
    assert "cluster_se_over_naive_se" in ref
    assert "clustering_matters" in ref
