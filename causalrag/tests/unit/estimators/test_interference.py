"""Tests for network-interference estimators (PDD §33 / Sprint 6.5.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.core.interference import InterferenceGraph
from causalrag.core.registry import get_registry
from causalrag.estimators.python.interference import (
    AronowSamiiEstimator,
    SavjeAronowHudgensEstimator,
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def test_estimators_registered() -> None:
    reg = get_registry()
    ids = {e.id for e in reg.all()}
    assert "python.interference.aronow_samii" in ids
    assert "python.interference.savje" in ids


# ---------------------------------------------------------------------------
# Aronow-Samii partial-interference DGP recovery
# ---------------------------------------------------------------------------
def _make_partial_interference_panel(
    *,
    n_clusters: int = 10,
    cluster_size: int = 10,
    direct_effect: float = 2.0,
    spill_effect: float = 0.5,
    seed: int = 7,
) -> tuple[pd.DataFrame, InterferenceGraph, float]:
    """Synthetic panel: ``n_clusters`` clusters of ``cluster_size`` units.

    Within each cluster, treatment is Bernoulli(0.5). Each unit's
    outcome is:
        Y_i = base + direct_effect * T_i + spill_effect * mean(T_{-i in cluster}) + noise.

    Returns df, graph, and the ground-truth direct effect.
    """
    rng = np.random.default_rng(seed)
    n = n_clusters * cluster_size
    clusters: dict[int, int] = {}
    for c in range(n_clusters):
        for j in range(cluster_size):
            clusters[c * cluster_size + j] = c
    t = rng.binomial(1, 0.5, size=n).astype(float)
    y = np.zeros(n, dtype=float)
    for c in range(n_clusters):
        members = list(range(c * cluster_size, (c + 1) * cluster_size))
        t_c = t[members]
        for i_idx, i in enumerate(members):
            others = np.delete(t_c, i_idx)
            spill = float(others.mean()) if len(others) else 0.0
            y[i] = 1.0 + direct_effect * t[i] + spill_effect * spill + rng.normal(0, 0.5)
    df = pd.DataFrame({"T": t, "Y": y})
    g = InterferenceGraph.from_clusters(clusters, n_units=n)
    return df, g, direct_effect


def test_aronow_samii_recovers_direct_effect() -> None:
    df, g, true_te = _make_partial_interference_panel(seed=11)
    est = AronowSamiiEstimator(
        treatment="T", outcome="Y", interference_graph=g
    ).fit(df)
    res = est.estimate()
    assert res.estimand_class == "ATE"
    # Within 2 SE of the truth.
    assert res.se is not None and res.se > 0
    assert abs(res.point_estimate - true_te) <= 2.0 * res.se
    assert res.diagnostics["interference_kind"] == "partial"
    assert res.diagnostics["n_informative_clusters"] >= 8


def test_aronow_samii_requires_partial_kind() -> None:
    g = InterferenceGraph.from_edge_list(
        n_units=4, edges=[(0, 1)], interference_kind="general"
    )
    with pytest.raises(ValueError, match="partial"):
        AronowSamiiEstimator(treatment="T", outcome="Y", interference_graph=g)


def test_aronow_samii_missing_graph_errors_clearly() -> None:
    with pytest.raises(ValueError, match="interference_graph"):
        AronowSamiiEstimator(treatment="T", outcome="Y")


def test_aronow_samii_data_row_count_mismatch() -> None:
    g = InterferenceGraph.from_clusters({0: 0, 1: 0, 2: 1, 3: 1})
    df = pd.DataFrame({"T": [0, 1, 0], "Y": [0.0, 1.0, 0.5]})  # 3 rows ≠ 4 units
    est = AronowSamiiEstimator(treatment="T", outcome="Y", interference_graph=g)
    with pytest.raises(ValueError, match="row ordering"):
        est.fit(df)


def test_aronow_samii_no_informative_cluster_errors() -> None:
    # All-treated cluster + all-control cluster: no within-cluster contrast.
    clusters = {0: 0, 1: 0, 2: 1, 3: 1}
    g = InterferenceGraph.from_clusters(clusters)
    df = pd.DataFrame({"T": [1, 1, 0, 0], "Y": [3.0, 3.2, 1.1, 1.0]})
    est = AronowSamiiEstimator(treatment="T", outcome="Y", interference_graph=g)
    with pytest.raises(ValueError, match="unidentified"):
        est.fit(df)


# ---------------------------------------------------------------------------
# Sävje-Aronow-Hudgens on Erdős-Rényi network
# ---------------------------------------------------------------------------
def _erdos_renyi_edges(n: int, p_edge: float, rng: np.random.Generator) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                edges.append((i, j))
    return edges


def test_savje_on_erdos_renyi_gives_sensible_eate() -> None:
    rng = np.random.default_rng(42)
    n = 200
    edges = _erdos_renyi_edges(n, p_edge=0.03, rng=rng)
    g = InterferenceGraph.from_edge_list(
        n_units=n, edges=edges, interference_kind="general"
    )
    direct = 1.5
    spill = 0.4
    p = 0.5
    t = rng.binomial(1, p, size=n).astype(float)
    exposure = g.exposure_vector(t)
    y = 0.5 + direct * t + spill * exposure + rng.normal(0, 0.5, size=n)
    df = pd.DataFrame({"T": t, "Y": y})
    est = SavjeAronowHudgensEstimator(
        treatment="T", outcome="Y", interference_graph=g
    ).fit(df)
    res = est.estimate()
    assert res.estimand_class == "ATE"
    # The corrected estimator should land in a sensible range — within
    # 3 SE of the structural direct effect (the EATE under this design
    # is direct + spill * E[exposure] = direct + spill * p, but the
    # Sävje residualisation strips the spillover contamination so the
    # consistent target is `direct` itself).
    assert res.se is not None and res.se > 0
    # Sanity range: estimator must not be wildly off.
    assert 0.5 < res.point_estimate < 2.5
    # And within 3 SE of direct effect.
    assert abs(res.point_estimate - direct) <= 3.0 * res.se
    assert res.diagnostics["interference_kind"] == "general"
    assert res.diagnostics["p_hat"] == pytest.approx(t.mean())


def test_savje_missing_graph_errors_clearly() -> None:
    with pytest.raises(ValueError, match="interference_graph"):
        SavjeAronowHudgensEstimator(treatment="T", outcome="Y")


def test_savje_data_row_count_mismatch() -> None:
    g = InterferenceGraph.from_edge_list(n_units=5, edges=[(0, 1)])
    df = pd.DataFrame({"T": [0, 1, 0], "Y": [0.0, 1.0, 0.5]})
    est = SavjeAronowHudgensEstimator(treatment="T", outcome="Y", interference_graph=g)
    with pytest.raises(ValueError, match="row ordering"):
        est.fit(df)


def test_savje_propensity_zero_raises() -> None:
    g = InterferenceGraph.from_edge_list(n_units=4, edges=[(0, 1), (2, 3)])
    df = pd.DataFrame({"T": [0, 0, 0, 0], "Y": [1.0, 2.0, 3.0, 4.0]})
    est = SavjeAronowHudgensEstimator(treatment="T", outcome="Y", interference_graph=g)
    with pytest.raises(ValueError, match="propensity"):
        est.fit(df)


def test_savje_accepts_partial_graph() -> None:
    # Partial kind also supported (as a special case).
    g = InterferenceGraph.from_clusters({0: 0, 1: 0, 2: 1, 3: 1})
    rng = np.random.default_rng(0)
    t = np.array([1, 0, 1, 0], dtype=float)
    y = 0.5 + 1.0 * t + rng.normal(0, 0.1, size=4)
    df = pd.DataFrame({"T": t, "Y": y})
    est = SavjeAronowHudgensEstimator(treatment="T", outcome="Y", interference_graph=g).fit(df)
    res = est.estimate()
    assert np.isfinite(res.point_estimate)
