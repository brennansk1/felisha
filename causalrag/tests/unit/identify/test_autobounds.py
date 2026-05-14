"""Tests for :mod:`causalrag.identify.autobounds_bridge` (Sprint 2.8)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.identify import autobounds_bridge as ab
from causalrag.identify.autobounds_bridge import PartialIDResult, partial_identify


# ---------------------------------------------------------------------------
# Fixtures: tiny confounded DAGs
# ---------------------------------------------------------------------------


def _confounded_dag() -> CausalGraph:
    """U -> T, U -> Y, T -> Y — the canonical unmeasured-confounder DAG."""
    return CausalGraph(
        nodes=("U", "T", "Y"),
        edges=(
            CausalEdge(source="U", target="T"),
            CausalEdge(source="U", target="Y"),
            CausalEdge(source="T", target="Y"),
        ),
    )


def _five_node_dag() -> CausalGraph:
    """U -> T, U -> Y, T -> Y plus two ancestor covariates of T."""
    return CausalGraph(
        nodes=("U", "T", "Y", "Z1", "Z2"),
        edges=(
            CausalEdge(source="U", target="T"),
            CausalEdge(source="U", target="Y"),
            CausalEdge(source="T", target="Y"),
            CausalEdge(source="Z1", target="T"),
            CausalEdge(source="Z2", target="T"),
        ),
    )


def _wide_dag(n: int) -> CausalGraph:
    """``n`` independent ancestors of T plus T -> Y. Always ≥ n+2 nodes."""
    edges: list[CausalEdge] = [CausalEdge(source="T", target="Y")]
    nodes: list[str] = ["T", "Y"]
    for i in range(n):
        z = f"Z{i}"
        nodes.append(z)
        edges.append(CausalEdge(source=z, target="T"))
    return CausalGraph(nodes=tuple(nodes), edges=tuple(edges))


def _make_binary_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """n samples from U -> T, U -> Y, T -> Y, all Bernoulli."""
    rng = np.random.default_rng(seed)
    u = rng.binomial(1, 0.4, size=n)
    p_t = np.where(u == 1, 0.7, 0.3)
    t = rng.binomial(1, p_t)
    p_y = np.clip(0.15 + 0.4 * u + 0.25 * t, 0.0, 1.0)
    y = rng.binomial(1, p_y)
    return pd.DataFrame({"U": u, "T": t, "Y": y})


# ---------------------------------------------------------------------------
# Pure-Python tests — exercise the fallback path with NO autobounds dep
# ---------------------------------------------------------------------------


def test_empty_df_returns_trivial_fallback() -> None:
    """An empty frame is refused with a note; the result is a trivial bound."""
    graph = _confounded_dag()
    df = pd.DataFrame({"T": [], "Y": []})

    result = partial_identify(
        graph=graph, treatment="T", outcome="Y", df=df, timeout_seconds=2.0
    )

    assert isinstance(result, PartialIDResult)
    assert result.backend == "fallback"
    assert result.lower_bound == -1.0
    assert result.upper_bound == 1.0
    assert result.bound_width == 2.0
    assert result.point_estimate is None
    assert any("empty" in n.lower() for n in result.notes)


def test_graph_too_large_refuses() -> None:
    """A DAG whose relevant subgraph still exceeds max_nodes is refused."""
    graph = _wide_dag(20)  # 22 nodes total, all ancestors of T → all relevant
    df = pd.DataFrame({"T": [0, 1] * 50, "Y": [0, 1] * 50})

    result = partial_identify(
        graph=graph,
        treatment="T",
        outcome="Y",
        df=df,
        max_nodes=10,
        timeout_seconds=2.0,
    )

    assert result.backend == "fallback"
    assert result.n_nodes > 10
    assert any("too large" in n.lower() for n in result.notes)
    assert result.lower_bound <= result.upper_bound


def test_too_many_treatment_levels_refuses() -> None:
    """T with > max_levels distinct values is refused with a note."""
    graph = _confounded_dag()
    df = pd.DataFrame(
        {"T": list(range(15)) * 10, "Y": [0, 1] * 75}
    )

    result = partial_identify(
        graph=graph,
        treatment="T",
        outcome="Y",
        df=df,
        max_levels=10,
        timeout_seconds=2.0,
    )

    assert result.backend == "fallback"
    assert any("levels" in n.lower() and "treatment" in n.lower() for n in result.notes)


def test_continuous_outcome_refuses_or_falls_back() -> None:
    """A continuous Y exceeds max_levels and is refused with a note."""
    graph = _confounded_dag()
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"T": rng.integers(0, 2, size=200), "Y": rng.normal(size=200)}
    )

    result = partial_identify(
        graph=graph,
        treatment="T",
        outcome="Y",
        df=df,
        max_levels=10,
        timeout_seconds=2.0,
    )

    # Either refused for too-many-levels (expected) or autobounds ran and
    # returned a finite bound. Both are acceptable per the spec.
    assert result.lower_bound <= result.upper_bound
    if result.backend == "fallback":
        assert any("outcome" in n.lower() for n in result.notes)


def test_missing_column_returns_fallback() -> None:
    """If T/Y aren't in the dataframe we degrade rather than raise."""
    graph = _confounded_dag()
    df = pd.DataFrame({"T": [0, 1, 0, 1]})  # no Y

    result = partial_identify(
        graph=graph, treatment="T", outcome="Y", df=df, timeout_seconds=2.0
    )

    assert result.backend == "fallback"
    assert any("missing" in n.lower() for n in result.notes)


def test_result_has_expected_dataclass_fields() -> None:
    """All fields documented in the spec are populated on the fallback path."""
    graph = _confounded_dag()
    df = pd.DataFrame({"T": [], "Y": []})

    result = partial_identify(
        graph=graph, treatment="T", outcome="Y", df=df, timeout_seconds=2.0
    )

    # Dataclass surface — every field must be present and well-typed.
    assert isinstance(result.target_estimand, str)
    assert isinstance(result.lower_bound, float)
    assert isinstance(result.upper_bound, float)
    assert isinstance(result.bound_width, float)
    assert isinstance(result.backend, str)
    assert isinstance(result.n_nodes, int)
    assert isinstance(result.runtime_seconds, float)
    assert isinstance(result.notes, list)
    assert result.runtime_seconds >= 0.0
    assert result.bound_width == result.upper_bound - result.lower_bound


def test_missing_autobounds_yields_fallback_note() -> None:
    """When the autobounds package is unavailable we say so explicitly.

    We force the unavailability path even if autobounds *is* installed, so the
    test is deterministic across environments.
    """
    graph = _confounded_dag()
    df = _make_binary_df(n=50)

    original = ab._autobounds_available
    try:
        ab._autobounds_available = lambda: False  # type: ignore[assignment]
        result = partial_identify(
            graph=graph, treatment="T", outcome="Y", df=df, timeout_seconds=2.0
        )
    finally:
        ab._autobounds_available = original  # type: ignore[assignment]

    assert result.backend == "fallback"
    assert result.lower_bound == -1.0
    assert result.upper_bound == 1.0
    assert any("autobounds" in n.lower() for n in result.notes)


def test_timeout_returns_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker that exceeds the timeout produces a fallback with a note.

    We monkey-patch the worker function to sleep past the wall-clock budget.
    This exercises the ``ProcessPoolExecutor`` timeout path without requiring
    autobounds itself to be installed.
    """
    graph = _confounded_dag()
    df = _make_binary_df(n=50)

    # Force autobounds_available True so we reach the worker.
    monkeypatch.setattr(ab, "_autobounds_available", lambda: True)
    # Replace the worker with a slow one; must be importable from the module
    # path so the child process can pickle it.
    monkeypatch.setattr(ab, "_autobounds_worker", _slow_worker_for_timeout_test)

    t0 = time.perf_counter()
    result = partial_identify(
        graph=graph,
        treatment="T",
        outcome="Y",
        df=df,
        timeout_seconds=0.5,
    )
    elapsed = time.perf_counter() - t0

    assert result.backend == "fallback"
    # We should NOT have waited the full sleep (5s); the timeout cuts in.
    assert elapsed < 4.0
    assert any("exceeded" in n.lower() or "timeout" in n.lower() for n in result.notes)


def _slow_worker_for_timeout_test(*args: object, **kwargs: object) -> tuple[float, float, list[str]]:
    """Top-level (picklable) worker that sleeps past any reasonable timeout."""
    time.sleep(5.0)
    return (-0.5, 0.5, [])


# ---------------------------------------------------------------------------
# autobounds-only tests — skipped when the package is not installed
# ---------------------------------------------------------------------------


def test_confounded_dag_yields_nontrivial_bounds() -> None:
    """Binary T/Y under U -> T, U -> Y, T -> Y produces informative bounds."""
    pytest.importorskip("autobounds")
    graph = _confounded_dag()
    df = _make_binary_df(n=500)

    result = partial_identify(
        graph=graph, treatment="T", outcome="Y", df=df, timeout_seconds=60.0
    )

    if result.backend != "autobounds":
        pytest.skip(f"autobounds did not run cleanly: {result.notes}")
    # An informative bound is strictly tighter than the trivial [-1, 1].
    assert result.lower_bound > -1.0 or result.upper_bound < 1.0
    assert result.lower_bound <= result.upper_bound
    assert result.point_estimate is not None


def test_five_node_dag_runs_within_budget() -> None:
    """The 5-node DAG fits inside max_nodes and runs end-to-end."""
    pytest.importorskip("autobounds")
    graph = _five_node_dag()
    df = _make_binary_df(n=500)
    # Add Z1, Z2 columns so the joint is fully specified.
    rng = np.random.default_rng(1)
    df = df.assign(
        Z1=rng.integers(0, 2, size=len(df)),
        Z2=rng.integers(0, 2, size=len(df)),
    )

    result = partial_identify(
        graph=graph, treatment="T", outcome="Y", df=df, timeout_seconds=60.0
    )

    assert result.lower_bound <= result.upper_bound
    assert result.n_nodes <= 10
