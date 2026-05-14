"""Tests for :mod:`causalrag.identify.decomposition` (Sprint 6.5.6)."""

from __future__ import annotations

from pydantic import ConfigDict

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.identify.decomposition import (
    c_components,
    d_separation_prune,
    extract_relevant_subgraph,
    summarise_dag,
)


class BidirectedEdge(CausalEdge):
    """Test-only subclass that carries a ``bidirected`` flag.

    The production :class:`CausalEdge` does not yet model bidirected edges; the
    decomposition module reads ``getattr(edge, "bidirected", False)`` so this
    subclass is enough to exercise the c-component logic in unit tests.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bidirected: bool = True


def _chain(n: int) -> CausalGraph:
    """Build a directed chain X0 -> X1 -> ... -> X{n-1}."""
    edges = tuple(CausalEdge(source=f"X{i}", target=f"X{i+1}") for i in range(n - 1))
    nodes = tuple(f"X{i}" for i in range(n))
    return CausalGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# c_components
# ---------------------------------------------------------------------------


def test_c_components_empty_graph() -> None:
    assert c_components(CausalGraph.empty()) == []


def test_c_components_single_node() -> None:
    g = CausalGraph(nodes=("A",), edges=())
    assert c_components(g) == [frozenset({"A"})]


def test_c_components_pure_dag_all_singletons() -> None:
    g = _chain(10)
    comps = c_components(g)
    assert len(comps) == 10
    assert all(len(c) == 1 for c in comps)
    # Every node accounted for exactly once.
    assert set().union(*comps) == set(g.nodes)


def test_c_components_disconnected_directed_components_still_singletons() -> None:
    g = CausalGraph(
        nodes=("A", "B", "C", "D"),
        edges=(
            CausalEdge(source="A", target="B"),
            CausalEdge(source="C", target="D"),
        ),
    )
    comps = c_components(g)
    # No bidirected edges → 4 singletons regardless of directed connectivity.
    assert len(comps) == 4


def test_c_components_100_node_dag_with_4_latents() -> None:
    """A 100-node chain with 4 bidirected edges → exactly 4 non-singleton
    c-components, the remaining 92 nodes singletons."""
    n = 100
    directed_edges = [
        CausalEdge(source=f"X{i}", target=f"X{i+1}") for i in range(n - 1)
    ]
    # Four bidirected edges over disjoint node pairs.
    bidir_pairs = [(5, 6), (20, 21), (50, 51), (80, 81)]
    bidir_edges = [
        BidirectedEdge(source=f"X{a}", target=f"X{b}") for a, b in bidir_pairs
    ]
    g = CausalGraph(
        nodes=tuple(f"X{i}" for i in range(n)),
        edges=tuple(directed_edges + bidir_edges),
    )

    comps = c_components(g)
    non_singletons = [c for c in comps if len(c) > 1]
    singletons = [c for c in comps if len(c) == 1]

    assert len(non_singletons) == 4
    assert all(len(c) == 2 for c in non_singletons)
    assert len(singletons) == n - 8
    # Sanity: components partition the node set.
    assert set().union(*comps) == set(g.nodes)


# ---------------------------------------------------------------------------
# extract_relevant_subgraph
# ---------------------------------------------------------------------------


def test_extract_returns_empty_when_treatment_missing() -> None:
    g = _chain(5)
    sub = extract_relevant_subgraph(g, treatment="missing", outcome="X4")
    assert sub.nodes == ()
    assert sub.edges == ()


def test_extract_small_chain_keeps_t_to_y_path() -> None:
    g = _chain(5)  # X0 -> X1 -> X2 -> X3 -> X4
    sub = extract_relevant_subgraph(g, treatment="X1", outcome="X3")
    # Keep ancestors of T (X0), the T-Y path (X1, X2, X3), drop X4.
    assert set(sub.nodes) == {"X0", "X1", "X2", "X3"}
    assert "X4" not in sub.nodes


def test_extract_100_node_chain_narrow_window() -> None:
    """T and Y near one end of a 100-node chain → only ~10 nodes survive."""
    g = _chain(100)  # X0 -> ... -> X99
    sub = extract_relevant_subgraph(g, treatment="X90", outcome="X95")
    kept = set(sub.nodes)
    # Ancestors of X90 = X0..X89; plus path X90..X95 = 6 more = 96.
    # Use the OTHER end of the chain so the relevant slice is small.
    sub2 = extract_relevant_subgraph(g, treatment="X3", outcome="X8")
    kept2 = set(sub2.nodes)
    # Ancestors of T (X0..X2) + path X3..X8 (6 nodes) = 9 nodes total, ≤ 10.
    assert len(kept2) <= 10
    assert {"X3", "X4", "X5", "X6", "X7", "X8"}.issubset(kept2)
    assert "X50" not in kept2
    assert "X99" not in kept2
    # The first scenario (T late, Y later) is intentionally large — sanity-check.
    assert len(kept) > 90


def test_extract_keeps_adjustment_set_ancestors() -> None:
    # Z -> T -> Y, plus W -> Z. Ancestors of Z should be retained.
    g = CausalGraph(
        nodes=("W", "Z", "T", "Y", "Junk"),
        edges=(
            CausalEdge(source="W", target="Z"),
            CausalEdge(source="Z", target="T"),
            CausalEdge(source="T", target="Y"),
        ),
    )
    sub = extract_relevant_subgraph(
        g, treatment="T", outcome="Y", adjustment_set={"Z"}
    )
    assert "W" in sub.nodes
    assert "Junk" not in sub.nodes


def test_extract_pulls_in_c_component_partners() -> None:
    # A <-> B bidirected; A is an ancestor of T. B has no directed link but
    # should be retained because it shares a c-component with A.
    g = CausalGraph(
        nodes=("A", "B", "T", "Y", "Off"),
        edges=(
            CausalEdge(source="A", target="T"),
            CausalEdge(source="T", target="Y"),
            BidirectedEdge(source="A", target="B"),
        ),
    )
    sub = extract_relevant_subgraph(g, treatment="T", outcome="Y")
    assert {"A", "B", "T", "Y"}.issubset(set(sub.nodes))
    assert "Off" not in sub.nodes


# ---------------------------------------------------------------------------
# d_separation_prune
# ---------------------------------------------------------------------------


def test_prune_removes_redundant_adjuster() -> None:
    # Two confounders Z1, Z2 each open a backdoor; a third "redundant" R is
    # an ancestor of Z1 only — once Z1 is in the set, R adds nothing.
    #   R -> Z1 -> T
    #         \-> Y
    #   Z2 -> T
    #   Z2 -> Y
    g = CausalGraph(
        nodes=("R", "Z1", "Z2", "T", "Y"),
        edges=(
            CausalEdge(source="R", target="Z1"),
            CausalEdge(source="Z1", target="T"),
            CausalEdge(source="Z1", target="Y"),
            CausalEdge(source="Z2", target="T"),
            CausalEdge(source="Z2", target="Y"),
        ),
    )
    pruned = d_separation_prune(g, "T", "Y", {"R", "Z1", "Z2"})
    # Z1 and Z2 are required; R is redundant.
    assert "Z1" in pruned
    assert "Z2" in pruned
    assert "R" not in pruned


def test_prune_leaves_required_adjusters_alone() -> None:
    # Single backdoor through Z; Z must stay.
    g = CausalGraph(
        nodes=("Z", "T", "Y"),
        edges=(
            CausalEdge(source="Z", target="T"),
            CausalEdge(source="Z", target="Y"),
        ),
    )
    pruned = d_separation_prune(g, "T", "Y", {"Z"})
    assert pruned == frozenset({"Z"})


def test_prune_returns_input_when_not_separating() -> None:
    # T directly causes Y; no Z can block. Function should return the input
    # unchanged rather than fabricate a separation claim.
    g = CausalGraph(
        nodes=("T", "Y", "Q"),
        edges=(CausalEdge(source="T", target="Y"),),
    )
    pruned = d_separation_prune(g, "T", "Y", {"Q"})
    assert pruned == frozenset({"Q"})


def test_prune_handles_empty_candidate_set() -> None:
    g = _chain(3)
    pruned = d_separation_prune(g, "X0", "X2", set())
    assert pruned == frozenset()


# ---------------------------------------------------------------------------
# summarise_dag
# ---------------------------------------------------------------------------


def test_summarise_empty_graph() -> None:
    s = summarise_dag(CausalGraph.empty())
    assert s["n_nodes"] == 0
    assert s["n_edges"] == 0
    assert s["max_in_degree"] == 0
    assert s["max_out_degree"] == 0
    assert s["n_c_components"] == 0
    assert s["has_bidirected_edges"] is False
    assert s["n_strongly_connected_components"] == 0


def test_summarise_chain_and_bidirected_flag() -> None:
    g = _chain(5)
    s = summarise_dag(g)
    assert s["n_nodes"] == 5
    assert s["n_edges"] == 4
    assert s["max_in_degree"] == 1
    assert s["max_out_degree"] == 1
    assert s["n_c_components"] == 5
    assert s["has_bidirected_edges"] is False

    g2 = CausalGraph(
        nodes=("A", "B", "C"),
        edges=(
            CausalEdge(source="A", target="B"),
            BidirectedEdge(source="B", target="C"),
        ),
    )
    s2 = summarise_dag(g2)
    assert s2["has_bidirected_edges"] is True
    # A is singleton; B-C form one c-component.
    assert s2["n_c_components"] == 2
