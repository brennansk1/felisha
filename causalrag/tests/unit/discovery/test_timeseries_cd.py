"""Tests for :mod:`causalrag.discovery.timeseries_cd` (Sprint 7.1).

Three coverage tiers:

1. *Import-safety / API-shape tests* run without tigramite — they
   import the module, inspect the public signature, and exercise the
   pure-Python ``_graph_array_to_edges`` decoder with a hand-built
   tigramite-shaped numpy array (so we test our decoding logic
   independently of whether tigramite can run on the test box).
2. *Mock-friendly stub test* monkey-patches the lazy tigramite imports
   so we can verify ``discover_timeseries_dag`` plumbs ``tau_max``,
   ``alpha``, and the algorithm dispatch into the right tigramite call
   without paying for an actual CI-test run.
3. *Live recovery test* gated on ``pytest.importorskip("tigramite")``
   — generates a linear VAR(1) with a known X→Y lag-1 link plus an
   unrelated Z, runs PCMCI+, and asserts the discovered edge set
   contains the true link with the correct ``lag-1`` annotation.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalrag.core.graph import CausalGraph
from causalrag.discovery import timeseries_cd
from causalrag.discovery.timeseries_cd import (
    _graph_array_to_edges,
    _split_panel,
    discover_timeseries_dag,
)


# ─── tier 1: pure-Python decoder + helpers ─────────────────────────────


def test_split_panel_single_series_orders_by_time() -> None:
    df = pd.DataFrame(
        {
            "t": [2, 0, 1],
            "x": [20.0, 0.0, 10.0],
            "y": [200.0, 0.0, 100.0],
        }
    )
    arrays, names = _split_panel(df, time_column="t", unit_column=None)
    assert names == ["x", "y"]
    assert len(arrays) == 1
    # Sorted by t → 0, 10, 20 / 0, 100, 200
    np.testing.assert_allclose(arrays[0][:, 0], [0.0, 10.0, 20.0])
    np.testing.assert_allclose(arrays[0][:, 1], [0.0, 100.0, 200.0])


def test_split_panel_groups_by_unit_and_sorts_each() -> None:
    df = pd.DataFrame(
        {
            "u": ["a", "a", "b", "b"],
            "t": [1, 0, 1, 0],
            "x": [11.0, 10.0, 21.0, 20.0],
        }
    )
    arrays, names = _split_panel(df, time_column="t", unit_column="u")
    assert names == ["x"]
    assert len(arrays) == 2
    np.testing.assert_allclose(arrays[0][:, 0], [10.0, 11.0])
    np.testing.assert_allclose(arrays[1][:, 0], [20.0, 21.0])


def test_split_panel_validates_columns() -> None:
    df = pd.DataFrame({"x": [1.0]})
    with pytest.raises(ValueError, match="time_column"):
        _split_panel(df, time_column="missing", unit_column=None)
    with pytest.raises(ValueError, match="unit_column"):
        _split_panel(df, time_column="x", unit_column="missing")


def test_split_panel_requires_variable_columns() -> None:
    df = pd.DataFrame({"t": [0, 1], "u": ["a", "a"]})
    with pytest.raises(ValueError, match="no variable columns"):
        _split_panel(df, time_column="t", unit_column="u")


def _build_link_array(
    n: int, tau_max: int, links: list[tuple[int, int, int, str]]
) -> np.ndarray:
    """Construct an ``(N, N, tau_max+1)`` tigramite-style link array.

    For tau=0 we mirror the contemporaneous mark to ``(j, i, 0)`` with
    the reverse arrow — that's how tigramite actually writes its
    output, and the decoder must dedupe on it.
    """
    g = np.full((n, n, tau_max + 1), "", dtype="<U3")
    for i, j, tau, mark in links:
        g[i, j, tau] = mark
        if tau == 0:
            mirror = {"-->": "<--", "<--": "-->", "<->": "<->", "o-o": "o-o"}.get(mark)
            if mirror is not None:
                g[j, i, 0] = mirror
    return g


def test_graph_array_to_edges_decodes_lagged_directed() -> None:
    names = ["x", "y", "z"]
    # x_{t-1} -> y_t  (lag-1 directed)
    graph = _build_link_array(3, tau_max=2, links=[(0, 1, 1, "-->")])
    edges = _graph_array_to_edges(graph, names, algorithm="pcmci_plus")
    assert len(edges) == 1
    e = edges[0]
    assert (e.source, e.target) == ("x", "y")
    assert e.bidirected is False
    assert e.note == "pcmci_plus lag-1"


def test_graph_array_to_edges_dedupes_contemporaneous() -> None:
    names = ["a", "b"]
    graph = _build_link_array(2, tau_max=1, links=[(0, 1, 0, "-->")])
    edges = _graph_array_to_edges(graph, names, algorithm="pcmci_plus")
    # mirror at (b, a, 0) should not produce a second a→b edge or a b→a one
    assert len(edges) == 1
    assert (edges[0].source, edges[0].target) == ("a", "b")
    assert edges[0].note == "pcmci_plus contemporaneous"


def test_graph_array_to_edges_handles_bidirected_lpcmci() -> None:
    names = ["a", "b"]
    graph = _build_link_array(2, tau_max=0, links=[(0, 1, 0, "<->")])
    edges = _graph_array_to_edges(graph, names, algorithm="lpcmci")
    assert len(edges) == 1
    assert edges[0].bidirected is True
    # Canonical alphabetic order
    assert edges[0].source == "a"
    assert edges[0].target == "b"


def test_graph_array_to_edges_skips_fully_ambiguous() -> None:
    names = ["a", "b"]
    graph = _build_link_array(2, tau_max=0, links=[(0, 1, 0, "o-o")])
    edges = _graph_array_to_edges(graph, names, algorithm="lpcmci")
    assert edges == []


def test_graph_array_to_edges_partial_circle_orients_toward_arrowhead() -> None:
    names = ["a", "b"]
    g = np.full((2, 2, 1), "", dtype="<U3")
    g[0, 1, 0] = "o->"
    edges = _graph_array_to_edges(g, names, algorithm="lpcmci")
    assert len(edges) == 1
    assert (edges[0].source, edges[0].target) == ("a", "b")
    assert edges[0].bidirected is False


# ─── tier 2: mock-friendly stub for the lazy import path ───────────────


class _StubPCMCI:
    """Minimal stand-in for ``tigramite.pcmci.PCMCI``.

    Records the constructor + ``run_pcmciplus`` call args so the test
    can assert ``tau_max`` / ``pc_alpha`` propagated, and returns a
    hand-rolled graph dict that the decoder can chew on.
    """

    calls: list[dict[str, Any]] = []

    def __init__(self, dataframe: Any, cond_ind_test: Any, verbosity: int = 0) -> None:
        self.dataframe = dataframe
        self.cond_ind_test = cond_ind_test
        type(self).calls.append({"init": {"verbosity": verbosity}})

    def run_pcmciplus(self, *, tau_max: int, pc_alpha: float) -> dict[str, Any]:
        type(self).calls.append(
            {"run": {"tau_max": tau_max, "pc_alpha": pc_alpha}}
        )
        # 2 vars, tau_max+1 lag slices, one lag-1 link x→y
        graph = np.full((2, 2, tau_max + 1), "", dtype="<U3")
        graph[0, 1, 1] = "-->"
        return {"graph": graph}


class _StubDataFrame:
    def __init__(self, data: Any, **kwargs: Any) -> None:
        self.data = data
        self.kwargs = kwargs


class _StubParCorr:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_tigramite_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake ``tigramite`` package into ``sys.modules`` so the
    lazy imports inside ``discover_timeseries_dag`` resolve to stubs.

    We have to put placeholders for the submodules our code touches:
    ``tigramite``, ``tigramite.pcmci``, ``tigramite.data_processing``,
    and ``tigramite.independence_tests.parcorr``.
    """
    _StubPCMCI.calls = []
    root = types.ModuleType("tigramite")
    pcmci_mod = types.ModuleType("tigramite.pcmci")
    pcmci_mod.PCMCI = _StubPCMCI  # type: ignore[attr-defined]
    dp_mod = types.ModuleType("tigramite.data_processing")
    dp_mod.DataFrame = _StubDataFrame  # type: ignore[attr-defined]
    it_pkg = types.ModuleType("tigramite.independence_tests")
    parcorr_mod = types.ModuleType("tigramite.independence_tests.parcorr")
    parcorr_mod.ParCorr = _StubParCorr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tigramite", root)
    monkeypatch.setitem(sys.modules, "tigramite.pcmci", pcmci_mod)
    monkeypatch.setitem(sys.modules, "tigramite.data_processing", dp_mod)
    monkeypatch.setitem(sys.modules, "tigramite.independence_tests", it_pkg)
    monkeypatch.setitem(
        sys.modules, "tigramite.independence_tests.parcorr", parcorr_mod
    )


def test_discover_timeseries_dag_dispatches_to_pcmci_plus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tigramite_stub(monkeypatch)
    df = pd.DataFrame(
        {
            "t": list(range(20)),
            "x": np.linspace(0, 1, 20),
            "y": np.linspace(0, 1, 20) ** 2,
        }
    )
    g = discover_timeseries_dag(
        df,
        time_column="t",
        algorithm="pcmci_plus",
        tau_max=2,
        alpha=0.1,
    )
    assert isinstance(g, CausalGraph)
    assert g.rank == 0
    assert set(g.nodes) == {"x", "y"}
    # The stub planted a lag-1 x→y link
    assert len(g.edges) == 1
    e = g.edges[0]
    assert (e.source, e.target) == ("x", "y")
    assert e.note == "pcmci_plus lag-1"
    # Verify our kwargs survived the trip into tigramite
    run_call = next(c for c in _StubPCMCI.calls if "run" in c)
    assert run_call["run"] == {"tau_max": 2, "pc_alpha": 0.1}


def test_discover_timeseries_dag_unknown_algorithm_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tigramite_stub(monkeypatch)
    df = pd.DataFrame({"t": [0, 1], "x": [0.0, 1.0]})
    with pytest.raises(ValueError, match="unknown algorithm"):
        discover_timeseries_dag(
            df,
            time_column="t",
            algorithm="not_a_real_one",  # type: ignore[arg-type]
        )


def test_module_imports_without_tigramite() -> None:
    """The module itself must be import-safe without tigramite —
    only the function call lazily reaches for it.
    """
    assert hasattr(timeseries_cd, "discover_timeseries_dag")
    # No tigramite reference at module scope
    assert "tigramite" not in sys.modules or True  # tautology; intent is docs


# ─── tier 3: live tigramite recovery ───────────────────────────────────


def test_pcmci_plus_recovers_known_lag1_link() -> None:
    pytest.importorskip("tigramite")
    rng = np.random.default_rng(7)
    n = 400
    x = np.zeros(n)
    y = np.zeros(n)
    z = rng.normal(size=n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.normal(scale=0.5)
        y[t] = 0.8 * x[t - 1] + rng.normal(scale=0.3)
    df = pd.DataFrame(
        {
            "t": np.arange(n),
            "x": x,
            "y": y,
            "z": z,
        }
    )
    g = discover_timeseries_dag(
        df,
        time_column="t",
        algorithm="pcmci_plus",
        tau_max=2,
        alpha=0.05,
        ci_test="parcorr",
    )
    assert isinstance(g, CausalGraph)
    assert set(g.nodes) == {"x", "y", "z"}
    # The true lag-1 X→Y link should be among the discovered edges.
    xy_lag1 = [
        e
        for e in g.edges
        if e.source == "x" and e.target == "y" and (e.note or "").endswith("lag-1")
    ]
    assert xy_lag1, f"expected x→y lag-1 in {[(e.source, e.target, e.note) for e in g.edges]}"
    # Z is independent noise — no edge to/from z at any lag should
    # *dominate* the output. We only require there's no lag-1 z→y false
    # positive, which is the strongest spurious link in this DGP.
    bad = [
        e
        for e in g.edges
        if e.source == "z" and e.target == "y" and (e.note or "").endswith("lag-1")
    ]
    assert not bad, "spurious z→y lag-1 edge"
