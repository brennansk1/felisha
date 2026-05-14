"""Tests for core.dag_constructors (sprint 6.5.8)."""

from __future__ import annotations

import networkx as nx
import pytest

from causalrag.core.dag_constructors import (
    build_backdoor_dag,
    build_frontdoor_dag,
    build_iv_dag,
    build_mediator_chain_dag,
    build_proximal_dag,
)
from causalrag.core.graph import CausalGraph
from causalrag.core.roles import VariableRole


def _directed_only(g: CausalGraph) -> nx.DiGraph:
    """Return an nx.DiGraph view containing only the directed (non-bidirected) edges."""
    h: nx.DiGraph = nx.DiGraph()
    for n in g.nodes:
        h.add_node(n)
    for e in g.edges:
        if not e.bidirected:
            h.add_edge(e.source, e.target)
    return h


def _has_directed_edge(g: CausalGraph, u: str, v: str) -> bool:
    return any(
        (e.source == u and e.target == v and not e.bidirected) for e in g.edges
    )


def _has_bidirected_edge(g: CausalGraph, u: str, v: str) -> bool:
    return any(
        e.bidirected and {e.source, e.target} == {u, v} for e in g.edges
    )


# --------------------------------------------------------------------------- #
# backdoor                                                                    #
# --------------------------------------------------------------------------- #


def test_backdoor_dag_basic_topology() -> None:
    g = build_backdoor_dag(
        treatment="T",
        outcome="Y",
        confounders=("C1", "C2"),
    )
    assert isinstance(g, CausalGraph)
    assert _has_directed_edge(g, "T", "Y")
    for c in ("C1", "C2"):
        assert _has_directed_edge(g, c, "T")
        assert _has_directed_edge(g, c, "Y")
    assert g.roles["T"] is VariableRole.TREATMENT
    assert g.roles["Y"] is VariableRole.OUTCOME
    assert g.roles["C1"] is VariableRole.CONFOUNDER
    assert nx.is_directed_acyclic_graph(_directed_only(g))


def test_backdoor_dag_modifiers_touch_outcome_only() -> None:
    g = build_backdoor_dag(
        treatment="T",
        outcome="Y",
        confounders=(),
        modifiers=("M",),
    )
    assert _has_directed_edge(g, "M", "Y")
    assert not _has_directed_edge(g, "M", "T")
    assert g.roles["M"] is VariableRole.EFFECT_MODIFIER


def test_backdoor_dag_latent_confounder_emits_bidirected() -> None:
    g = build_backdoor_dag(
        treatment="T",
        outcome="Y",
        latent_confounders=("U",),
    )
    assert _has_bidirected_edge(g, "T", "Y")
    assert nx.is_directed_acyclic_graph(_directed_only(g))


# --------------------------------------------------------------------------- #
# IV                                                                          #
# --------------------------------------------------------------------------- #


def test_iv_dag_exclusion_restriction() -> None:
    g = build_iv_dag(treatment="T", outcome="Y", instrument="Z")
    assert _has_directed_edge(g, "Z", "T")
    assert _has_directed_edge(g, "T", "Y")
    # exclusion: Z has no direct edge of any kind to Y
    assert not _has_directed_edge(g, "Z", "Y")
    assert not _has_bidirected_edge(g, "Z", "Y")
    assert _has_bidirected_edge(g, "T", "Y")
    assert g.roles["Z"] is VariableRole.INSTRUMENT
    assert nx.is_directed_acyclic_graph(_directed_only(g))


def test_iv_dag_can_drop_latent() -> None:
    g = build_iv_dag(
        treatment="T",
        outcome="Y",
        instrument="Z",
        latent_treatment_outcome_confounder=False,
    )
    assert not _has_bidirected_edge(g, "T", "Y")


def test_iv_dag_confounders_feed_both_t_and_y() -> None:
    g = build_iv_dag(
        treatment="T", outcome="Y", instrument="Z", confounders=("C",),
    )
    assert _has_directed_edge(g, "C", "T")
    assert _has_directed_edge(g, "C", "Y")


# --------------------------------------------------------------------------- #
# front-door                                                                  #
# --------------------------------------------------------------------------- #


def test_frontdoor_dag_topology() -> None:
    g = build_frontdoor_dag(treatment="T", outcome="Y", mediator="M")
    # M has an incoming from T and an outgoing to Y
    assert _has_directed_edge(g, "T", "M")
    assert _has_directed_edge(g, "M", "Y")
    # The direct T -> Y edge IS present (front-door identifies despite it)
    assert _has_directed_edge(g, "T", "Y")
    assert _has_bidirected_edge(g, "T", "Y")
    assert g.roles["M"] is VariableRole.MEDIATOR
    assert nx.is_directed_acyclic_graph(_directed_only(g))


# --------------------------------------------------------------------------- #
# mediator chain                                                              #
# --------------------------------------------------------------------------- #


def test_mediator_chain_three_mediators_has_four_directed_chain_edges() -> None:
    g = build_mediator_chain_dag(
        treatment="T",
        outcome="Y",
        mediators=("M1", "M2", "M3"),
    )
    expected = [("T", "M1"), ("M1", "M2"), ("M2", "M3"), ("M3", "Y")]
    for u, v in expected:
        assert _has_directed_edge(g, u, v), f"missing chain edge {u} -> {v}"
    # And exactly four chain edges (no extras between chain members)
    chain_edges = {
        (e.source, e.target)
        for e in g.edges
        if not e.bidirected and (e.source, e.target) in set(expected + [(v, u) for (u, v) in expected])
    }
    assert chain_edges == set(expected)
    for m in ("M1", "M2", "M3"):
        assert g.roles[m] is VariableRole.MEDIATOR
    assert nx.is_directed_acyclic_graph(_directed_only(g))


def test_mediator_chain_requires_at_least_one_mediator() -> None:
    with pytest.raises(ValueError):
        build_mediator_chain_dag(treatment="T", outcome="Y", mediators=())


def test_mediator_chain_with_confounders_feeds_each_mediator() -> None:
    g = build_mediator_chain_dag(
        treatment="T",
        outcome="Y",
        mediators=("M1", "M2"),
        confounders=("C",),
    )
    for node in ("T", "Y", "M1", "M2"):
        assert _has_directed_edge(g, "C", node)


# --------------------------------------------------------------------------- #
# proximal                                                                    #
# --------------------------------------------------------------------------- #


def test_proximal_dag_has_bidirected_for_nce_and_nco() -> None:
    g = build_proximal_dag(
        treatment="T",
        outcome="Y",
        negative_control_exposure="W",
        negative_control_outcome="Zp",
    )
    assert _has_directed_edge(g, "T", "Y")
    # NCE shares latent with T; NCO shares latent with Y
    assert _has_bidirected_edge(g, "W", "T")
    assert _has_bidirected_edge(g, "Zp", "Y")
    # And the T <-> Y latent itself
    assert _has_bidirected_edge(g, "T", "Y")
    assert g.roles["W"] is VariableRole.NEGATIVE_CONTROL
    assert g.roles["Zp"] is VariableRole.NEGATIVE_CONTROL
    assert nx.is_directed_acyclic_graph(_directed_only(g))


# --------------------------------------------------------------------------- #
# round-trip acyclicity                                                       #
# --------------------------------------------------------------------------- #


def test_all_constructors_acyclic_after_dropping_bidirected() -> None:
    graphs = [
        build_backdoor_dag(
            treatment="T",
            outcome="Y",
            confounders=("C1", "C2"),
            modifiers=("M",),
            latent_confounders=("U",),
        ),
        build_iv_dag(
            treatment="T",
            outcome="Y",
            instrument="Z",
            confounders=("C",),
        ),
        build_frontdoor_dag(
            treatment="T",
            outcome="Y",
            mediator="M",
            confounders=("C",),
        ),
        build_mediator_chain_dag(
            treatment="T",
            outcome="Y",
            mediators=("M1", "M2", "M3"),
            confounders=("C",),
        ),
        build_proximal_dag(
            treatment="T",
            outcome="Y",
            negative_control_exposure="W",
            negative_control_outcome="Zp",
            confounders=("C",),
        ),
    ]
    for g in graphs:
        assert nx.is_directed_acyclic_graph(_directed_only(g)), (
            f"directed projection has a cycle for {g}"
        )
