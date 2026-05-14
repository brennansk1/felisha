"""Tests for the time-varying / panel DAG layering (Sprint 6.5.3)."""

from __future__ import annotations

import pytest

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.core.temporal_graph import TimeIndexedNode, TimeVaryingDAG


def _base_txy() -> CausalGraph:
    """Base graph: X -> T, X -> Y, T -> Y. X confounder, T treatment, Y outcome."""
    return CausalGraph(
        nodes=("X", "T", "Y"),
        edges=(
            CausalEdge(source="X", target="T"),
            CausalEdge(source="X", target="Y"),
            CausalEdge(source="T", target="Y"),
        ),
        roles={
            "X": VariableRole.CONFOUNDER,
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
        },
    )


def test_unroll_T3_has_9_nodes() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3,
                        treatment_outcome_lags=(0,))
    g = tv.materialise()
    assert set(g.nodes) == {
        "X_t0", "X_t1", "X_t2",
        "T_t0", "T_t1", "T_t2",
        "Y_t0", "Y_t1", "Y_t2",
    }
    assert len(g.nodes) == 9


def test_contemporaneous_treatment_outcome_edges() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3,
                        treatment_outcome_lags=(0,))
    g = tv.materialise()
    edge_pairs = {(e.source, e.target) for e in g.edges if not e.bidirected}
    for t in range(3):
        assert (f"T_t{t}", f"Y_t{t}") in edge_pairs


def test_lag1_confounder_persistence() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3,
                        treatment_outcome_lags=(0,))
    g = tv.materialise()
    edge_pairs = {(e.source, e.target) for e in g.edges}
    assert ("X_t0", "X_t1") in edge_pairs
    assert ("X_t1", "X_t2") in edge_pairs
    # No skip-lag.
    assert ("X_t0", "X_t2") not in edge_pairs


def test_treatment_outcome_lags_0_and_1() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3,
                        treatment_outcome_lags=(0, 1))
    g = tv.materialise()
    edge_pairs = {(e.source, e.target) for e in g.edges}
    assert ("T_t0", "Y_t0") in edge_pairs
    assert ("T_t0", "Y_t1") in edge_pairs
    # lag-2 not in the list
    assert ("T_t0", "Y_t2") not in edge_pairs
    # lag-1 at t=1 -> t=2
    assert ("T_t1", "Y_t2") in edge_pairs
    # lag-1 starting at t=2 would go to t=3 which does not exist
    assert ("T_t2", "Y_t3") not in edge_pairs


def test_confounder_persistence_false_drops_autoregression() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3,
                        treatment_outcome_lags=(0,),
                        confounder_persistence=False)
    g = tv.materialise()
    edge_pairs = {(e.source, e.target) for e in g.edges}
    assert ("X_t0", "X_t1") not in edge_pairs
    assert ("X_t1", "X_t2") not in edge_pairs


def test_bidirected_edges_propagate_per_time() -> None:
    base = CausalGraph(
        nodes=("U", "Y", "T"),
        edges=(
            CausalEdge(source="U", target="Y", bidirected=True),
            CausalEdge(source="T", target="Y"),
        ),
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "U": VariableRole.UNMEASURED_CONFOUNDER_CANDIDATE,
        },
    )
    tv = TimeVaryingDAG(base_graph=base, n_periods=3,
                        treatment_outcome_lags=(0,))
    g = tv.materialise()
    bi_pairs = {
        frozenset({e.source, e.target}) for e in g.edges if e.bidirected
    }
    for t in range(3):
        assert frozenset({f"U_t{t}", f"Y_t{t}"}) in bi_pairs
    assert len(bi_pairs) == 3


def test_roles_inherited_per_time_copy() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=2,
                        treatment_outcome_lags=(0,))
    g = tv.materialise()
    for t in range(2):
        assert g.roles[f"X_t{t}"] is VariableRole.CONFOUNDER
        assert g.roles[f"T_t{t}"] is VariableRole.TREATMENT
        assert g.roles[f"Y_t{t}"] is VariableRole.OUTCOME


def test_time_indexed_nodes_returns_all_copies() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3,
                        treatment_outcome_lags=(0,))
    nodes = tv.time_indexed_nodes("X")
    assert nodes == (
        TimeIndexedNode(name="X", time_index=0, qualified_name="X_t0"),
        TimeIndexedNode(name="X", time_index=1, qualified_name="X_t1"),
        TimeIndexedNode(name="X", time_index=2, qualified_name="X_t2"),
    )


def test_time_indexed_nodes_unknown_base_raises() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=2)
    with pytest.raises(KeyError):
        tv.time_indexed_nodes("nope")


def test_adjustment_set_at_time_2() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3,
                        treatment_outcome_lags=(0, 1))
    adj = tv.adjustment_set_at_time(2)
    # Confounders up to t=2.
    assert "X_t0" in adj and "X_t1" in adj and "X_t2" in adj
    # Prior treatments strictly before t=2.
    assert "T_t0" in adj and "T_t1" in adj
    assert "T_t2" not in adj
    # Prior outcomes strictly before t=2.
    assert "Y_t0" in adj and "Y_t1" in adj
    assert "Y_t2" not in adj


def test_adjustment_set_at_time_zero() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3)
    adj = tv.adjustment_set_at_time(0)
    # Only contemporaneous confounder at t=0.
    assert adj == frozenset({"X_t0"})


def test_adjustment_set_out_of_range() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=2)
    with pytest.raises(ValueError):
        tv.adjustment_set_at_time(5)
    with pytest.raises(ValueError):
        tv.adjustment_set_at_time(-1)


def test_materialised_graph_is_acyclic() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=4,
                        treatment_outcome_lags=(0, 1, 2))
    g = tv.materialise()
    assert isinstance(g, CausalGraph)
    assert g.is_acyclic()


def test_materialise_is_idempotent_cached() -> None:
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=2)
    g1 = tv.materialise()
    g2 = tv.materialise()
    assert g1 is g2  # cached


def test_n_periods_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TimeVaryingDAG(base_graph=_base_txy(), n_periods=0)


def test_non_treatment_outcome_edges_use_contemporaneous_only() -> None:
    """X -> T is not a treatment->outcome edge, so it should be a plain
    contemporaneous copy at every time index, not lagged."""
    tv = TimeVaryingDAG(base_graph=_base_txy(), n_periods=3,
                        treatment_outcome_lags=(0, 1))
    g = tv.materialise()
    edge_pairs = {(e.source, e.target) for e in g.edges}
    for t in range(3):
        assert (f"X_t{t}", f"T_t{t}") in edge_pairs
    # No X_t0 -> T_t1 (it's not a treatment->outcome edge).
    assert ("X_t0", "T_t1") not in edge_pairs
