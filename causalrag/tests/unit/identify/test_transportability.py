"""Tests for :mod:`causalrag.identify.transportability` (Sprint 6.4)."""

from __future__ import annotations

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.identify.transportability import (
    SelectionDiagram,
    TransportabilityResult,
    transportability_identify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph(edges: list[tuple[str, str]]) -> CausalGraph:
    return CausalGraph.from_edge_list(edges)


# ---------------------------------------------------------------------------
# SelectionDiagram construction
# ---------------------------------------------------------------------------


def test_selection_diagram_from_user_spec_empty() -> None:
    g = _graph([("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, [])
    assert diag.selection_nodes == ()
    assert diag.differing_variables == ()
    assert diag.target_population_label == "target"


def test_selection_diagram_from_user_spec_filters_unknown_vars() -> None:
    g = _graph([("T", "Y"), ("Z", "T"), ("Z", "Y")])
    diag = SelectionDiagram.from_user_spec(g, ["Z", "NOT_A_NODE", "Y"])
    assert "Z" in diag.differing_variables
    assert "Y" in diag.differing_variables
    assert "NOT_A_NODE" not in diag.differing_variables
    # Selection nodes are named S__<var>.
    assert all(s.startswith("S__") for s in diag.selection_nodes)


def test_selection_diagram_from_user_spec_dedupes() -> None:
    g = _graph([("T", "Y"), ("Z", "T"), ("Z", "Y")])
    diag = SelectionDiagram.from_user_spec(g, ["Z", "Z", "Z"])
    assert diag.selection_nodes == ("S__Z",)


# ---------------------------------------------------------------------------
# Source = target → always transportable
# ---------------------------------------------------------------------------


def test_no_selection_nodes_transports_directly() -> None:
    # Confounded T -> Y with Z as a parent of both.
    g = _graph([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, [])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="Y"
    )
    assert isinstance(result, TransportabilityResult)
    assert result.transportable is True
    assert result.method == "direct"
    assert result.transport_formula is not None
    assert "do(T=T)" in result.transport_formula
    assert result.auxiliary_variables_needed == ()


def test_simple_two_node_no_confounder_transports() -> None:
    g = _graph([("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, [])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="Y"
    )
    assert result.transportable is True
    assert result.method == "direct"


# ---------------------------------------------------------------------------
# Selection on irrelevant nodes
# ---------------------------------------------------------------------------


def test_selection_on_non_confounder_still_transportable() -> None:
    # W is a side variable that doesn't influence T or Y.
    g = _graph([("Z", "T"), ("Z", "Y"), ("T", "Y"), ("W", "Q")])
    diag = SelectionDiagram.from_user_spec(g, ["W"])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="Y"
    )
    assert result.transportable is True
    assert result.method == "direct"
    assert any("does not affect" in n for n in result.notes)


def test_selection_on_descendant_of_outcome_irrelevant() -> None:
    # D is a downstream descendant of Y, not in the formula.
    g = _graph([("T", "Y"), ("Y", "D")])
    diag = SelectionDiagram.from_user_spec(g, ["D"])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="Y"
    )
    assert result.transportable is True
    assert result.method == "direct"


# ---------------------------------------------------------------------------
# Selection on a parent of treatment → auxiliary data required
# ---------------------------------------------------------------------------


def test_selection_on_treatment_parent_requires_aux_data() -> None:
    # Z is the only confounder, a parent of T (and Y). Selection on Z
    # means the marginal P(Z) differs in the target.
    g = _graph([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, ["Z"])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="Y"
    )
    assert result.transportable is False
    assert result.method == "auxiliary_data_required"
    assert "Z" in result.auxiliary_variables_needed
    assert result.transport_formula is None


def test_selection_on_treatment_parent_with_aux_data_transports() -> None:
    g = _graph([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, ["Z"])
    result = transportability_identify(
        diagram=diag,
        treatment="T",
        outcome="Y",
        auxiliary_observed=("Z",),
    )
    assert result.transportable is True
    assert result.method == "auxiliary_data_required"
    assert result.auxiliary_variables_needed == ("Z",)
    assert result.transport_formula is not None
    # Formula should reference the target-population marginal of Z.
    assert "P^target(Z)" in result.transport_formula


# ---------------------------------------------------------------------------
# Selection on the outcome's mechanism → non-transportable
# ---------------------------------------------------------------------------


def test_selection_on_outcome_is_non_transportable() -> None:
    g = _graph([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, ["Y"])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="Y"
    )
    assert result.transportable is False
    assert result.method == "non_transportable"
    assert result.transport_formula is None
    assert any("outcome" in n.lower() for n in result.notes)


# ---------------------------------------------------------------------------
# Selection on treatment alone → still transports
# ---------------------------------------------------------------------------


def test_selection_on_treatment_alone_transports() -> None:
    # P(T) doesn't appear in P(Y|do(T)).
    g = _graph([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, ["T"])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="Y"
    )
    assert result.transportable is True
    assert result.method == "direct"


# ---------------------------------------------------------------------------
# Bareinboim 2014 motivating example: smoking → cancer with selection on
# a confounder.
# ---------------------------------------------------------------------------


def test_bareinboim_2014_smoking_cancer_confounder_selection() -> None:
    # Genotype G confounds smoking S and cancer C; the two populations
    # (e.g., LA and NYC) have different prevalences of G — this is the
    # canonical Bareinboim-Pearl example.
    g = _graph([("G", "S"), ("G", "C"), ("S", "C")])
    diag = SelectionDiagram.from_user_spec(g, ["G"])
    # Without target marginal of G: not transportable.
    result_no_aux = transportability_identify(
        diagram=diag, treatment="S", outcome="C"
    )
    assert result_no_aux.transportable is False
    assert result_no_aux.method == "auxiliary_data_required"
    assert result_no_aux.auxiliary_variables_needed == ("G",)
    # With target marginal of G: transportable, formula uses P^target(G).
    result_aux = transportability_identify(
        diagram=diag,
        treatment="S",
        outcome="C",
        auxiliary_observed=("G",),
    )
    assert result_aux.transportable is True
    assert result_aux.method == "auxiliary_data_required"
    assert result_aux.transport_formula is not None
    assert "P^target(G)" in result_aux.transport_formula
    assert "do(T=S)" in result_aux.transport_formula


# ---------------------------------------------------------------------------
# Edge cases / robustness
# ---------------------------------------------------------------------------


def test_missing_treatment_returns_non_transportable() -> None:
    g = _graph([("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, [])
    result = transportability_identify(
        diagram=diag, treatment="NOT_THERE", outcome="Y"
    )
    assert result.transportable is False
    assert result.method == "non_transportable"
    assert any("treatment" in w for w in result.warnings)


def test_missing_outcome_returns_non_transportable() -> None:
    g = _graph([("T", "Y")])
    diag = SelectionDiagram.from_user_spec(g, [])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="NOT_THERE"
    )
    assert result.transportable is False
    assert result.method == "non_transportable"


def test_outcome_not_descendant_of_treatment_transports() -> None:
    # T and Y are unrelated — effect is trivially zero / identifiable.
    g = CausalGraph(
        nodes=("T", "Y"),
        edges=(CausalEdge(source="T", target="X"),),
    )
    # Re-build with Y as a true sink not reachable from T.
    g2 = _graph([("T", "X"), ("Z", "Y")])
    diag = SelectionDiagram.from_user_spec(g2, [])
    result = transportability_identify(
        diagram=diag, treatment="T", outcome="Y"
    )
    assert result.transportable is True


def test_result_dataclass_defaults() -> None:
    r = TransportabilityResult(
        transportable=True,
        transport_formula="x",
        method="direct",
    )
    assert r.auxiliary_variables_needed == ()
    assert r.warnings == []
    assert r.notes == []
