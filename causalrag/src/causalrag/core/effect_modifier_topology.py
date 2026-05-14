"""Effect-modifier topology rendering (PDD §33 sprint 6.5.9).

Effect modifiers (M) sit on the *causal* side of the outcome equation but
are not assignment determinants. Under the previous rendering they either
collapsed into the confounder set (introducing spurious M -> T edges and
biasing the adjustment set) or fell out of the DAG entirely. The
canonical topology is::

    confounder_i -> T
    confounder_i -> Y
    modifier_j   -> Y           (no  modifier_j -> T)
    T            -> Y           (note: 'moderated by M1, M2, ...')

This module renders that topology and exposes helpers for downstream
identification / CATE code to recover the ordered modifier list and to
test individual nodes for the effect-modifier pattern.

See also ``core/dag_constructors.py`` (sprint 6.5.8) which provides
``build_backdoor_dag`` accepting modifiers in the same shape.
"""

from __future__ import annotations

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole


def _dedup_preserve(items: tuple[str, ...]) -> tuple[str, ...]:
    """Return items with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)


def _ordered_nodes(*groups: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for n in group:
            if n not in seen:
                seen.add(n)
                out.append(n)
    return tuple(out)


def build_dag_with_modifiers(
    *,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...] = (),
    modifiers: tuple[str, ...] = (),
    latent_confounders: bool = False,
) -> CausalGraph:
    """DAG with effect modifiers carrying EFFECT_MODIFIER role.

    Topology::

        confounder_i -> T
        confounder_i -> Y
        modifier_j   -> Y           (modifier influences Y conditional on T)
        T            -> Y           (the moderated edge; the ``note`` reads
                                     'moderated by M1, M2, ...')

    No ``modifier -> T`` edge is emitted — modifiers do not determine
    assignment. When ``latent_confounders`` is True a single bidirected
    T <-> Y edge is added to flag residual unmeasured confounding.

    The ordering of ``modifiers`` is preserved: it shows up in the
    moderation note, in the node ordering, and in
    :func:`modifiers_of` results.
    """
    confounders = _dedup_preserve(confounders)
    modifiers = _dedup_preserve(modifiers)

    nodes = _ordered_nodes((treatment, outcome), confounders, modifiers)

    roles: dict[str, VariableRole] = {
        treatment: VariableRole.TREATMENT,
        outcome: VariableRole.OUTCOME,
    }
    for c in confounders:
        roles[c] = VariableRole.CONFOUNDER
    for m in modifiers:
        roles[m] = VariableRole.EFFECT_MODIFIER

    edges: list[CausalEdge] = []
    for c in confounders:
        edges.append(CausalEdge(source=c, target=treatment))
        edges.append(CausalEdge(source=c, target=outcome))
    for m in modifiers:
        edges.append(CausalEdge(source=m, target=outcome))

    if modifiers:
        note = "moderated by " + ", ".join(modifiers)
    else:
        note = None
    edges.append(CausalEdge(source=treatment, target=outcome, note=note))

    if latent_confounders:
        edges.append(
            CausalEdge(
                source=treatment,
                target=outcome,
                bidirected=True,
                note="latent T-Y confounder (effect-modifier topology)",
            )
        )

    return CausalGraph(nodes=nodes, edges=tuple(edges), roles=roles, rank=1)


def is_effect_modifier(graph: CausalGraph, node: str) -> bool:
    """Return True iff ``node`` matches the effect-modifier pattern.

    A node is an effect modifier when:

    1. it carries the ``EFFECT_MODIFIER`` role; and
    2. it has at least one outgoing directed edge whose target is an
       OUTCOME-roled node; and
    3. it has NO outgoing directed edge to any TREATMENT-roled node.

    Bidirected edges are ignored — they encode latent confounding, not
    directed moderation.
    """
    if graph.roles.get(node) is not VariableRole.EFFECT_MODIFIER:
        return False

    outcomes = {n for n, r in graph.roles.items() if r is VariableRole.OUTCOME}
    treatments = {n for n, r in graph.roles.items() if r is VariableRole.TREATMENT}

    has_edge_to_outcome = False
    has_edge_to_treatment = False
    for e in graph.edges:
        if e.bidirected or e.source != node:
            continue
        if e.target in outcomes:
            has_edge_to_outcome = True
        if e.target in treatments:
            has_edge_to_treatment = True

    return has_edge_to_outcome and not has_edge_to_treatment


def modifiers_of(graph: CausalGraph) -> tuple[str, ...]:
    """Return the ordered list of nodes with the EFFECT_MODIFIER role.

    Order follows ``graph.nodes`` so it matches the construction order
    used by :func:`build_dag_with_modifiers`.
    """
    return tuple(
        n for n in graph.nodes
        if graph.roles.get(n) is VariableRole.EFFECT_MODIFIER
    )


__all__ = [
    "build_dag_with_modifiers",
    "is_effect_modifier",
    "modifiers_of",
]
