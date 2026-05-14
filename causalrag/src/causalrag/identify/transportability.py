"""Bareinboim-Pearl transportability identification (Sprint 6.4).

When the target population differs from the source population (gated by
``TARGET_POPULATION_DIFFERS`` upstream), the standard ID algorithm is not
sufficient: a quantity identifiable in the source distribution may fail to
be identifiable in the target. Bareinboim & Pearl (2014; JAIR 51:1-43) and
Correa & Bareinboim (2020) introduce *selection diagrams* — DAGs augmented
with selection nodes ``S_i`` whose only edges are ``S_i -> X_i`` — and a
do-calculus algorithm (``sID``) that decides identifiability of the target
estimand from a mixture of source observational/experimental data and
limited target-population data.

This module implements a pragmatic, *graphical-shortcut* form of ``sID``
sufficient for Felisha's pipeline. It does not perform full do-calculus
rewriting; instead it inspects which selection nodes touch the
identification-relevant ancestry of the outcome to decide one of three
outcomes:

* ``direct`` — the source-population effect transports unchanged.
* ``auxiliary_data_required`` — the effect transports if the target
  population supplies marginals of the differing variables.
* ``non_transportable`` — even with auxiliary marginals the effect is
  unidentifiable in the target.

The implementation is intentionally pure and side-effect-free; wiring into
``q5_identify`` / ``master_loop`` is a separate ticket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

from causalrag.core.graph import CausalGraph


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionDiagram:
    """Bareinboim selection diagram.

    A selection diagram augments the source-population DAG with selection
    nodes ``S_i``; a directed edge ``S_i -> X_i`` indicates that ``X_i``
    is distributed differently in the target than in the source (i.e. the
    mechanism producing ``X_i`` is not invariant across populations). The
    base graph itself is taken to be common to both populations.
    """

    base_graph: CausalGraph
    selection_nodes: tuple[str, ...] = ()
    target_population_label: str = "target"

    @classmethod
    def from_user_spec(
        cls,
        base_graph: CausalGraph,
        differs_at: list[str],
    ) -> "SelectionDiagram":
        """Build a diagram from the variables flagged as differing.

        ``differs_at`` lists the *base-graph* variable names whose
        mechanism is believed to differ between source and target. The
        selection node names returned use the convention ``S__<var>``.
        Variables not present in ``base_graph.nodes`` are silently
        dropped — callers are expected to validate first.
        """
        seen: set[str] = set()
        nodes: list[str] = []
        base_nodes = set(base_graph.nodes)
        for v in differs_at:
            if v in base_nodes and v not in seen:
                seen.add(v)
                nodes.append(v)
        selection = tuple(f"S__{v}" for v in nodes)
        return cls(base_graph=base_graph, selection_nodes=selection)

    @property
    def differing_variables(self) -> tuple[str, ...]:
        """Base-graph variables that the selection nodes point to."""
        return tuple(
            s.removeprefix("S__") if s.startswith("S__") else s
            for s in self.selection_nodes
        )


@dataclass
class TransportabilityResult:
    transportable: bool
    transport_formula: str | None
    method: Literal["direct", "auxiliary_data_required", "non_transportable"]
    auxiliary_variables_needed: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _source_digraph(graph: CausalGraph) -> nx.DiGraph:
    """NetworkX directed view of the *source* graph (no selection nodes)."""
    dg: nx.DiGraph = nx.DiGraph()
    dg.add_nodes_from(graph.nodes)
    for e in graph.edges:
        if not getattr(e, "bidirected", False):
            dg.add_edge(e.source, e.target)
    return dg


def _is_source_identifiable(
    graph: CausalGraph, treatment: str, outcome: str
) -> bool:
    """Pragmatic source-population identifiability check.

    Felisha's full ID algorithm (``q5_identify``) is owned by another
    ticket; here we use a conservative graphical surrogate:

    * If either treatment or outcome is missing from the graph → not ID.
    * If outcome is not a descendant of treatment → trivially ID (effect
      is zero / disconnected).
    * Otherwise → the source effect is taken to be identifiable when
      backdoor adjustment by the parents of ``treatment`` is admissible
      (parents themselves are not descendants of treatment, which is the
      standard Pearl backdoor admissibility condition).

    This is sufficient for the transportability decision because the
    transportability layer only needs to know whether the *source*
    estimand exists before asking whether it transports.
    """
    if treatment not in graph.nodes or outcome not in graph.nodes:
        return False
    dg = _source_digraph(graph)
    if treatment not in dg or outcome not in dg:
        return False
    if outcome not in nx.descendants(dg, treatment):
        # Trivially identifiable: do(T) leaves Y unaffected.
        return True
    descendants_t = nx.descendants(dg, treatment) | {treatment}
    for p in dg.predecessors(treatment):
        if p in descendants_t:
            # Backdoor adjuster is itself a descendant of treatment.
            return False
    return True


def _adjustment_parents(graph: CausalGraph, treatment: str) -> tuple[str, ...]:
    """Parents of treatment in the source graph — the backdoor adjusters."""
    dg = _source_digraph(graph)
    if treatment not in dg:
        return ()
    return tuple(sorted(dg.predecessors(treatment)))


def _trial_targets(
    graph: CausalGraph, treatment: str, outcome: str
) -> set[str]:
    """Nodes whose mechanism appears in the source-identification formula.

    For Pearl's backdoor formula
        P(Y | do(T)) = Σ_Z P(Y | T, Z) P(Z)
    where ``Z`` = parents(T), the mechanisms involved are:

    * P(Y | T, Z): the conditional outcome mechanism — depends on the
      stable behaviour of Y *and* on Z (Z appearing in the conditioning
      set means Z's distribution among controls feeds the formula via
      its support, but only its *marginal* P(Z) enters as a free term).
    * P(Z): the marginal of the adjusters.

    A selection node on a node in this set means the formula's terms are
    not invariant across populations, hence the source formula cannot be
    transported verbatim.
    """
    relevant: set[str] = {outcome, treatment}
    relevant.update(_adjustment_parents(graph, treatment))
    return relevant


def _format_transport_formula(
    treatment: str,
    outcome: str,
    adjusters: tuple[str, ...],
    aux_marginal_vars: tuple[str, ...],
    target_label: str,
) -> str:
    """Render the do-calculus expression for the target estimand.

    ``aux_marginal_vars`` are the adjuster variables whose marginal must
    be drawn from the *target* population (because the source marginal
    differs); all remaining adjusters use the source marginal.
    """
    if not adjusters:
        # No confounding: P^*(y | do(t)) = P(y | t) from the source.
        if target_label and aux_marginal_vars:
            return (
                f"P^{target_label}(Y={outcome} | do(T={treatment})) = "
                f"P(Y={outcome} | T={treatment})"
            )
        return f"P(Y={outcome} | do(T={treatment})) = P(Y={outcome} | T={treatment})"

    z_list = ", ".join(adjusters)
    aux = set(aux_marginal_vars)
    # Split adjusters into source-marginal vs target-marginal terms.
    source_terms = " ".join(
        f"P({z})" for z in adjusters if z not in aux
    )
    target_terms = " ".join(
        f"P^{target_label}({z})" for z in adjusters if z in aux
    )
    marginal = " ".join(t for t in (source_terms, target_terms) if t)
    lhs_pop = f"P^{target_label}" if aux else "P"
    return (
        f"{lhs_pop}(Y={outcome} | do(T={treatment})) = "
        f"Σ_{{{z_list}}} P(Y={outcome} | T={treatment}, {z_list}) {marginal}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def transportability_identify(
    *,
    diagram: SelectionDiagram,
    treatment: str,
    outcome: str,
    auxiliary_observed: tuple[str, ...] = (),
) -> TransportabilityResult:
    """Bareinboim-Pearl ``sID``-style transportability decision.

    See module docstring for the algorithm sketch. The function never
    raises on bad inputs; it returns a result whose ``warnings`` list
    explains what went wrong (e.g. treatment missing from the graph).
    """
    warnings: list[str] = []
    notes: list[str] = []

    graph = diagram.base_graph

    if treatment not in graph.nodes:
        warnings.append(f"treatment {treatment!r} not in graph")
    if outcome not in graph.nodes:
        warnings.append(f"outcome {outcome!r} not in graph")
    if warnings:
        return TransportabilityResult(
            transportable=False,
            transport_formula=None,
            method="non_transportable",
            auxiliary_variables_needed=(),
            warnings=warnings,
            notes=notes,
        )

    # 1. Source-population ID.
    if not _is_source_identifiable(graph, treatment, outcome):
        notes.append(
            "Source-population effect P(Y|do(T)) is not identifiable; "
            "transportability is moot."
        )
        return TransportabilityResult(
            transportable=False,
            transport_formula=None,
            method="non_transportable",
            auxiliary_variables_needed=(),
            warnings=warnings,
            notes=notes,
        )

    adjusters = _adjustment_parents(graph, treatment)
    target_label = diagram.target_population_label

    # 2. Identify which selection nodes touch the identification ancestry.
    differing = set(diagram.differing_variables)
    formula_targets = _trial_targets(graph, treatment, outcome)

    # Selection nodes that *don't* affect any formula term are irrelevant.
    invariant_selectors = {v for v in differing if v not in formula_targets}
    binding_selectors = differing & formula_targets

    if invariant_selectors:
        notes.append(
            "Selection on "
            + ", ".join(sorted(invariant_selectors))
            + " does not affect the source identification formula and is ignored."
        )

    # 3. Direct transportability — no binding selectors.
    if not binding_selectors:
        formula = _format_transport_formula(
            treatment=treatment,
            outcome=outcome,
            adjusters=adjusters,
            aux_marginal_vars=(),
            target_label=target_label,
        )
        notes.append(
            "No selection node intersects the identification formula; the "
            "source estimand transports directly."
        )
        return TransportabilityResult(
            transportable=True,
            transport_formula=formula,
            method="direct",
            auxiliary_variables_needed=(),
            warnings=warnings,
            notes=notes,
        )

    # 4. Selection on the outcome itself — the conditional outcome
    # mechanism P(Y | T, Z) is not invariant, and no marginal of any
    # observable variable can fix it. Non-transportable.
    if outcome in binding_selectors:
        notes.append(
            f"Selection node on outcome {outcome!r}: the outcome mechanism "
            "is not invariant across populations and cannot be recovered "
            "from auxiliary marginals."
        )
        return TransportabilityResult(
            transportable=False,
            transport_formula=None,
            method="non_transportable",
            auxiliary_variables_needed=(),
            warnings=warnings,
            notes=notes,
        )

    # 5. Selection on the treatment itself only affects P(T), which never
    # appears in P(Y | do(T)). Treat as invariant.
    if binding_selectors == {treatment}:
        formula = _format_transport_formula(
            treatment=treatment,
            outcome=outcome,
            adjusters=adjusters,
            aux_marginal_vars=(),
            target_label=target_label,
        )
        notes.append(
            "Selection on treatment alone does not enter P(Y|do(T)); "
            "transports directly."
        )
        return TransportabilityResult(
            transportable=True,
            transport_formula=formula,
            method="direct",
            auxiliary_variables_needed=(),
            warnings=warnings,
            notes=notes,
        )

    # 6. Binding selectors fall on adjuster variables (parents of T).
    # The source marginal P(Z) for those is not valid in the target;
    # we need the target marginal.
    adjuster_selectors = tuple(
        sorted(v for v in binding_selectors if v in set(adjusters))
    )
    other = tuple(
        sorted(v for v in binding_selectors if v not in set(adjusters))
    )
    if other:
        notes.append(
            "Binding selectors on "
            + ", ".join(other)
            + " fall outside the adjuster set; not yet handled."
        )
        return TransportabilityResult(
            transportable=False,
            transport_formula=None,
            method="non_transportable",
            auxiliary_variables_needed=(),
            warnings=warnings,
            notes=notes,
        )

    have = set(auxiliary_observed)
    missing = tuple(v for v in adjuster_selectors if v not in have)
    if missing:
        notes.append(
            "Auxiliary target-population data required on "
            + ", ".join(missing)
            + " (marginal distribution P^"
            + target_label
            + " over the differing adjusters)."
        )
        return TransportabilityResult(
            transportable=False,
            transport_formula=None,
            method="auxiliary_data_required",
            auxiliary_variables_needed=adjuster_selectors,
            warnings=warnings,
            notes=notes,
        )

    # We have the marginals — render a transport formula that uses
    # target marginals for the binding adjusters and source marginals for
    # the rest.
    formula = _format_transport_formula(
        treatment=treatment,
        outcome=outcome,
        adjusters=adjusters,
        aux_marginal_vars=adjuster_selectors,
        target_label=target_label,
    )
    notes.append(
        "Effect is transportable using target marginals on "
        + ", ".join(adjuster_selectors)
        + "."
    )
    return TransportabilityResult(
        transportable=True,
        transport_formula=formula,
        method="auxiliary_data_required",
        auxiliary_variables_needed=adjuster_selectors,
        warnings=warnings,
        notes=notes,
    )


__all__ = [
    "SelectionDiagram",
    "TransportabilityResult",
    "transportability_identify",
]
