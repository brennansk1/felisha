"""Tests for core.effect_modifier_topology (sprint 6.5.9)."""

from __future__ import annotations

import pytest

from causalrag.core.dag_constructors import build_backdoor_dag
from causalrag.core.effect_modifier_topology import (
    build_dag_with_modifiers,
    is_effect_modifier,
    modifiers_of,
)
from causalrag.core.graph import CausalGraph
from causalrag.core.roles import VariableRole


def _has_directed_edge(g: CausalGraph, u: str, v: str) -> bool:
    return any(
        e.source == u and e.target == v and not e.bidirected for e in g.edges
    )


def _has_bidirected_edge(g: CausalGraph, u: str, v: str) -> bool:
    return any(
        e.bidirected and {e.source, e.target} == {u, v} for e in g.edges
    )


def _ty_directed_edge(g: CausalGraph, t: str, y: str):
    for e in g.edges:
        if e.source == t and e.target == y and not e.bidirected:
            return e
    return None


# --------------------------------------------------------------------------- #
# build_dag_with_modifiers                                                     #
# --------------------------------------------------------------------------- #


def test_build_dag_with_modifiers_returns_causal_graph() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        confounders=("C1",),
        modifiers=("M1",),
    )
    assert isinstance(g, CausalGraph)
    assert set(g.nodes) == {"T", "Y", "C1", "M1"}


def test_modifier_has_edge_to_outcome_only() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        modifiers=("M1", "M2"),
    )
    for m in ("M1", "M2"):
        assert _has_directed_edge(g, m, "Y"), f"missing {m} -> Y"
        assert not _has_directed_edge(g, m, "T"), f"unexpected {m} -> T"


def test_confounder_has_edges_to_both_t_and_y() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        confounders=("C1", "C2"),
    )
    for c in ("C1", "C2"):
        assert _has_directed_edge(g, c, "T")
        assert _has_directed_edge(g, c, "Y")


def test_treatment_outcome_edge_present() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        modifiers=("M1",),
    )
    assert _has_directed_edge(g, "T", "Y")


def test_ty_edge_note_lists_every_modifier_in_order() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        modifiers=("Age", "Sex", "Site"),
    )
    edge = _ty_directed_edge(g, "T", "Y")
    assert edge is not None
    assert edge.note is not None
    assert "moderated by" in edge.note
    # ordering preserved
    assert edge.note == "moderated by Age, Sex, Site"
    for m in ("Age", "Sex", "Site"):
        assert m in edge.note


def test_ty_edge_note_absent_when_no_modifiers() -> None:
    g = build_dag_with_modifiers(treatment="T", outcome="Y")
    edge = _ty_directed_edge(g, "T", "Y")
    assert edge is not None
    assert edge.note is None


def test_roles_assigned_correctly() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        confounders=("C1",),
        modifiers=("M1", "M2"),
    )
    assert g.roles["T"] is VariableRole.TREATMENT
    assert g.roles["Y"] is VariableRole.OUTCOME
    assert g.roles["C1"] is VariableRole.CONFOUNDER
    assert g.roles["M1"] is VariableRole.EFFECT_MODIFIER
    assert g.roles["M2"] is VariableRole.EFFECT_MODIFIER


def test_duplicate_modifiers_deduped_preserving_order() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        modifiers=("M1", "M2", "M1"),
    )
    assert modifiers_of(g) == ("M1", "M2")
    edge = _ty_directed_edge(g, "T", "Y")
    assert edge is not None and edge.note == "moderated by M1, M2"


def test_latent_confounders_flag_adds_bidirected_edge() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        modifiers=("M1",),
        latent_confounders=True,
    )
    assert _has_bidirected_edge(g, "T", "Y")
    # without the flag, no bidirected edge
    g2 = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        modifiers=("M1",),
    )
    assert not _has_bidirected_edge(g2, "T", "Y")


def test_graph_is_acyclic() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        confounders=("C1", "C2"),
        modifiers=("M1", "M2"),
    )
    assert g.is_acyclic()


# --------------------------------------------------------------------------- #
# distinguishability from confounder topology                                  #
# --------------------------------------------------------------------------- #


def test_modifier_topology_differs_from_confounder_topology() -> None:
    """A modifier yields M->Y only; a confounder yields C->T and C->Y."""
    mod_graph = build_dag_with_modifiers(
        treatment="T", outcome="Y", modifiers=("X",)
    )
    conf_graph = build_dag_with_modifiers(
        treatment="T", outcome="Y", confounders=("X",)
    )
    # In the modifier graph, X has no edge to T.
    assert _has_directed_edge(mod_graph, "X", "Y")
    assert not _has_directed_edge(mod_graph, "X", "T")
    # In the confounder graph, X has edges to both T and Y.
    assert _has_directed_edge(conf_graph, "X", "Y")
    assert _has_directed_edge(conf_graph, "X", "T")
    # And the role assignment differs.
    assert mod_graph.roles["X"] is VariableRole.EFFECT_MODIFIER
    assert conf_graph.roles["X"] is VariableRole.CONFOUNDER


# --------------------------------------------------------------------------- #
# is_effect_modifier                                                           #
# --------------------------------------------------------------------------- #


def test_is_effect_modifier_true_for_modifier() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        modifiers=("M1", "M2"),
    )
    assert is_effect_modifier(g, "M1") is True
    assert is_effect_modifier(g, "M2") is True


def test_is_effect_modifier_false_for_confounder() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        confounders=("C1",),
        modifiers=("M1",),
    )
    assert is_effect_modifier(g, "C1") is False


def test_is_effect_modifier_false_for_treatment_and_outcome() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        modifiers=("M1",),
    )
    assert is_effect_modifier(g, "T") is False
    assert is_effect_modifier(g, "Y") is False


def test_is_effect_modifier_false_for_unknown_node() -> None:
    g = build_dag_with_modifiers(treatment="T", outcome="Y", modifiers=("M1",))
    assert is_effect_modifier(g, "does_not_exist") is False


def test_is_effect_modifier_against_backdoor_dag_constructor() -> None:
    """The 6.5.8 backdoor builder uses the same role; ensure predicate
    recognises modifiers it produces too."""
    g = build_backdoor_dag(
        treatment="T",
        outcome="Y",
        confounders=("C1",),
        modifiers=("M1",),
    )
    assert is_effect_modifier(g, "M1") is True
    assert is_effect_modifier(g, "C1") is False


def test_is_effect_modifier_rejects_role_only_node_without_y_edge() -> None:
    """A node with EFFECT_MODIFIER role but no outgoing edge to Y is
    NOT considered an effect modifier — the predicate is structural."""
    from causalrag.core.graph import CausalEdge

    g = CausalGraph(
        nodes=("T", "Y", "M"),
        edges=(CausalEdge(source="T", target="Y"),),
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "M": VariableRole.EFFECT_MODIFIER,
        },
    )
    assert is_effect_modifier(g, "M") is False


def test_is_effect_modifier_rejects_node_with_edge_to_treatment() -> None:
    """A node with EFFECT_MODIFIER role AND an edge to T fails the
    structural test — it would be functioning as a confounder."""
    from causalrag.core.graph import CausalEdge

    g = CausalGraph(
        nodes=("T", "Y", "M"),
        edges=(
            CausalEdge(source="T", target="Y"),
            CausalEdge(source="M", target="Y"),
            CausalEdge(source="M", target="T"),
        ),
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "M": VariableRole.EFFECT_MODIFIER,
        },
    )
    assert is_effect_modifier(g, "M") is False


# --------------------------------------------------------------------------- #
# modifiers_of                                                                 #
# --------------------------------------------------------------------------- #


def test_modifiers_of_returns_ordered_list() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        confounders=("C1",),
        modifiers=("M1", "M2", "M3"),
    )
    assert modifiers_of(g) == ("M1", "M2", "M3")


def test_modifiers_of_empty_when_none() -> None:
    g = build_dag_with_modifiers(
        treatment="T", outcome="Y", confounders=("C1",)
    )
    assert modifiers_of(g) == ()


def test_modifiers_of_excludes_other_roles() -> None:
    g = build_dag_with_modifiers(
        treatment="T",
        outcome="Y",
        confounders=("C1", "C2"),
        modifiers=("M1",),
    )
    out = modifiers_of(g)
    assert "C1" not in out
    assert "C2" not in out
    assert "T" not in out
    assert "Y" not in out
    assert out == ("M1",)
