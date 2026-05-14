"""Pure DAG-topology constructors for common identification patterns.

These helpers produce :class:`CausalGraph` objects with appropriate
role assignments. They are intentionally side-effect-free: callers
(notably ``master_loop._build_graph_for_proposal``) compose them with
brief metadata to materialise the topology corresponding to a proposed
identification strategy (backdoor, IV, front-door, multi-mediator
chain, proximal).

See PDD §13 (core/graph.py) and §33 sprint 6.5.8.
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


def build_backdoor_dag(
    *,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...] = (),
    modifiers: tuple[str, ...] = (),
    latent_confounders: tuple[str, ...] = (),
) -> CausalGraph:
    """Standard backdoor DAG: confounders -> T, confounders -> Y, T -> Y.

    Latent confounders enter as BIDIRECTED edges: each name in
    ``latent_confounders`` becomes a bidirected edge T <-> Y (i.e., a
    latent confounder of T and Y). Modifiers get the EFFECT_MODIFIER
    role; they have an edge to Y but not to T.
    """
    confounders = _dedup_preserve(confounders)
    modifiers = _dedup_preserve(modifiers)
    latent_confounders = _dedup_preserve(latent_confounders)

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
    edges.append(CausalEdge(source=treatment, target=outcome))

    for latent in latent_confounders:
        edges.append(
            CausalEdge(
                source=treatment,
                target=outcome,
                bidirected=True,
                note=f"latent confounder: {latent}",
            )
        )

    return CausalGraph(nodes=nodes, edges=tuple(edges), roles=roles, rank=1)


def build_iv_dag(
    *,
    treatment: str,
    outcome: str,
    instrument: str,
    confounders: tuple[str, ...] = (),
    latent_treatment_outcome_confounder: bool = True,
) -> CausalGraph:
    """Canonical IV DAG: Z -> T -> Y plus T <-> Y bidirected.

    The instrument has no direct edge to Y (exclusion restriction)
    and is independent of the latent confounder. Optional measured
    confounders feed into both T and Y.
    """
    confounders = _dedup_preserve(confounders)
    nodes = _ordered_nodes((instrument, treatment, outcome), confounders)

    roles: dict[str, VariableRole] = {
        instrument: VariableRole.INSTRUMENT,
        treatment: VariableRole.TREATMENT,
        outcome: VariableRole.OUTCOME,
    }
    for c in confounders:
        roles[c] = VariableRole.CONFOUNDER

    edges: list[CausalEdge] = [
        CausalEdge(source=instrument, target=treatment),
        CausalEdge(source=treatment, target=outcome),
    ]
    for c in confounders:
        edges.append(CausalEdge(source=c, target=treatment))
        edges.append(CausalEdge(source=c, target=outcome))

    if latent_treatment_outcome_confounder:
        edges.append(
            CausalEdge(
                source=treatment,
                target=outcome,
                bidirected=True,
                note="latent T-Y confounder (IV)",
            )
        )

    return CausalGraph(nodes=nodes, edges=tuple(edges), roles=roles, rank=1)


def build_frontdoor_dag(
    *,
    treatment: str,
    outcome: str,
    mediator: str,
    confounders: tuple[str, ...] = (),
    latent_treatment_outcome_confounder: bool = True,
) -> CausalGraph:
    """Canonical front-door DAG: T -> M -> Y, T -> Y direct, plus T <-> Y.

    The mediator must be unconfounded by U. Front-door identifies the
    T -> Y effect via M despite the latent U.
    """
    confounders = _dedup_preserve(confounders)
    nodes = _ordered_nodes((treatment, mediator, outcome), confounders)

    roles: dict[str, VariableRole] = {
        treatment: VariableRole.TREATMENT,
        mediator: VariableRole.MEDIATOR,
        outcome: VariableRole.OUTCOME,
    }
    for c in confounders:
        roles[c] = VariableRole.CONFOUNDER

    edges: list[CausalEdge] = [
        CausalEdge(source=treatment, target=mediator),
        CausalEdge(source=mediator, target=outcome),
        CausalEdge(source=treatment, target=outcome),
    ]
    for c in confounders:
        edges.append(CausalEdge(source=c, target=treatment))
        edges.append(CausalEdge(source=c, target=outcome))

    if latent_treatment_outcome_confounder:
        edges.append(
            CausalEdge(
                source=treatment,
                target=outcome,
                bidirected=True,
                note="latent T-Y confounder (front-door)",
            )
        )

    return CausalGraph(nodes=nodes, edges=tuple(edges), roles=roles, rank=1)


def build_mediator_chain_dag(
    *,
    treatment: str,
    outcome: str,
    mediators: tuple[str, ...],
    confounders: tuple[str, ...] = (),
) -> CausalGraph:
    """Multi-mediator chain: T -> M_1 -> M_2 -> ... -> M_k -> Y.

    Confounders feed into T, Y, and every mediator on the chain.
    """
    if not mediators:
        raise ValueError("mediator_chain_dag requires at least one mediator")
    mediators = _dedup_preserve(mediators)
    confounders = _dedup_preserve(confounders)

    nodes = _ordered_nodes((treatment,), mediators, (outcome,), confounders)

    roles: dict[str, VariableRole] = {
        treatment: VariableRole.TREATMENT,
        outcome: VariableRole.OUTCOME,
    }
    for m in mediators:
        roles[m] = VariableRole.MEDIATOR
    for c in confounders:
        roles[c] = VariableRole.CONFOUNDER

    edges: list[CausalEdge] = []
    chain = (treatment, *mediators, outcome)
    for u, v in zip(chain[:-1], chain[1:], strict=True):
        edges.append(CausalEdge(source=u, target=v))

    for c in confounders:
        edges.append(CausalEdge(source=c, target=treatment))
        edges.append(CausalEdge(source=c, target=outcome))
        for m in mediators:
            edges.append(CausalEdge(source=c, target=m))

    return CausalGraph(nodes=nodes, edges=tuple(edges), roles=roles, rank=1)


def build_proximal_dag(
    *,
    treatment: str,
    outcome: str,
    negative_control_exposure: str,
    negative_control_outcome: str,
    confounders: tuple[str, ...] = (),
) -> CausalGraph:
    """Liu-Tchetgen-Tchetgen 2024 proximal DAG.

    Topology: T -> Y; a latent U with U <-> T and U <-> Y; NCE is a
    child of U (d-separated from Y given U); NCO is a child of U
    (d-separated from T given U). The two bidirected edges are
    represented by attaching NCE and NCO to T and Y respectively via
    bidirected edges (their shared latent U is implicit).
    """
    confounders = _dedup_preserve(confounders)
    nodes = _ordered_nodes(
        (treatment, outcome, negative_control_exposure, negative_control_outcome),
        confounders,
    )

    roles: dict[str, VariableRole] = {
        treatment: VariableRole.TREATMENT,
        outcome: VariableRole.OUTCOME,
        negative_control_exposure: VariableRole.NEGATIVE_CONTROL,
        negative_control_outcome: VariableRole.NEGATIVE_CONTROL,
    }
    for c in confounders:
        roles[c] = VariableRole.CONFOUNDER

    edges: list[CausalEdge] = [
        CausalEdge(source=treatment, target=outcome),
    ]
    for c in confounders:
        edges.append(CausalEdge(source=c, target=treatment))
        edges.append(CausalEdge(source=c, target=outcome))

    # Latent U is represented by bidirected edges. NCE shares U with T;
    # NCO shares U with Y; T and Y also share U.
    edges.extend(
        [
            CausalEdge(
                source=treatment,
                target=outcome,
                bidirected=True,
                note="latent U: shared cause of T and Y (proximal)",
            ),
            CausalEdge(
                source=negative_control_exposure,
                target=treatment,
                bidirected=True,
                note="latent U: NCE shares latent with T (proximal)",
            ),
            CausalEdge(
                source=negative_control_outcome,
                target=outcome,
                bidirected=True,
                note="latent U: NCO shares latent with Y (proximal)",
            ),
        ]
    )

    return CausalGraph(nodes=nodes, edges=tuple(edges), roles=roles, rank=1)


__all__ = [
    "build_backdoor_dag",
    "build_iv_dag",
    "build_frontdoor_dag",
    "build_mediator_chain_dag",
    "build_proximal_dag",
]
