"""Step 5 — Assess identifiability via DoWhy.

PDD §10.5 + design principle 8 ("Identifiability is a hard gate"). Given a
CausalEstimand, a CausalGraph, and an ObservedDataSpec, ask DoWhy whether the
estimand is identifiable from the chosen DAG under our assumptions. Output is
a typed :class:`IdentificationResult` that downstream Steps 6/7 read; a
non-identifiable result blocks estimation by default.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.graph import CausalGraph
from causalrag.core.roles import VariableRole

IdStrategy = Literal[
    "backdoor",
    "frontdoor",
    "iv",
    "mediation",
    "do-calculus",
    "non-identifiable",
    "unsupported",
]


class IdentificationResult(BaseModel):
    """Step 5 output.

    ``identifiable`` is the gate: if False, Step 7 refuses to run unless the
    analyst sets ``allow_nonidentifiable=True`` (logged as an override).
    """

    model_config = ConfigDict(extra="forbid")

    identifiable: bool
    strategy: IdStrategy
    adjustment_set: tuple[str, ...] = ()
    instrument: str | None = None
    mediator: str | None = None
    estimand_expression: str | None = None
    notes: list[str] = Field(default_factory=list)
    dowhy_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    weak: bool = False
    diagnostics: dict[str, Any] = Field(default_factory=dict)


_SUPPORTED: frozenset[EstimandClass] = frozenset(
    {
        EstimandClass.ATE,
        EstimandClass.ATT,
        EstimandClass.ATC,
        EstimandClass.CATE,
        EstimandClass.LATE,
        EstimandClass.NDE,
        EstimandClass.NIE,
    }
)


def identify_effect(
    estimand: CausalEstimand,
    graph: CausalGraph,
    df: pd.DataFrame | None = None,
    *,
    candidate_graphs: tuple[CausalGraph, ...] | None = None,
) -> IdentificationResult:
    """Ask DoWhy whether ``estimand`` is identifiable from ``graph``.

    The data frame is optional; DoWhy needs a frame at construction time, so
    when ``df`` is None we synthesize a single-row frame with the right column
    names. This is purely a graph operation — no estimation happens here.

    If ``candidate_graphs`` is provided, identification is attempted against
    each candidate (in their existing order — assumed to be ranked best-first)
    and the highest-ranked DAG that yields identifiability is returned, with
    ``diagnostics["chose_dag"]`` set to the chosen index. ``graph`` is the
    default/fallback when no candidate yields an identifiable strategy.
    """
    if candidate_graphs:
        last_result: IdentificationResult | None = None
        for idx, cand in enumerate(candidate_graphs):
            res = identify_effect(estimand, cand, df=df)
            if res.identifiable:
                res.diagnostics["chose_dag"] = idx
                return res
            last_result = res
        # No candidate identifiable — fall back to the supplied default graph
        # and annotate diagnostics.
        fallback = identify_effect(estimand, graph, df=df)
        fallback.diagnostics["chose_dag"] = None
        fallback.diagnostics["n_candidates_tried"] = len(candidate_graphs)
        if last_result is not None and not fallback.identifiable:
            return fallback
        return fallback

    if estimand.klass not in _SUPPORTED:
        return IdentificationResult(
            identifiable=False,
            strategy="unsupported",
            notes=[f"Estimand class {estimand.klass.value} is not supported by Step 5 yet."],
        )

    needed = _required_columns(estimand, graph)
    if df is None:
        df = pd.DataFrame({c: [0.0] for c in needed})
    else:
        missing = [c for c in needed if c not in df.columns]
        if missing:
            return IdentificationResult(
                identifiable=False,
                strategy="non-identifiable",
                notes=[f"Required columns missing from data: {missing}"],
            )

    try:
        from dowhy import CausalModel
    except ImportError as e:
        raise RuntimeError(
            "Step 5 requires DoWhy: pip install 'causalrag[estimators]'"
        ) from e

    gml = _graph_to_gml(graph)
    try:
        model = CausalModel(
            data=df,
            treatment=estimand.treatment,
            outcome=estimand.outcome,
            graph=gml,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=False)
    except Exception as e:  # DoWhy raises for ill-formed DAGs / missing nodes
        return IdentificationResult(
            identifiable=False,
            strategy="non-identifiable",
            notes=[f"DoWhy identify_effect raised: {type(e).__name__}: {e}"],
        )

    expr = str(getattr(identified, "estimand_expression", "") or "")
    strategy, adjustment, instrument, mediator = _interpret(identified, estimand)
    identifiable = strategy != "non-identifiable"

    notes: list[str] = []
    warnings: list[str] = []
    diagnostics: dict[str, Any] = {}
    weak = False

    if strategy == "iv" and estimand.klass != EstimandClass.LATE:
        notes.append(
            "DoWhy proposed an IV strategy but the requested estimand is not LATE; "
            "consider revising the estimand."
        )

    if strategy == "backdoor":
        filtered, drops, weak_flag, filter_warnings = _filter_adjustment_set(
            adjustment, graph, estimand
        )
        diagnostics["dropped_descendants"] = sorted(drops["descendants"])
        diagnostics["dropped_mediators"] = sorted(drops["mediators"])
        diagnostics["dropped_colliders"] = sorted(drops["colliders"])
        diagnostics["original_adjustment_set"] = list(adjustment)
        warnings.extend(filter_warnings)
        adjustment = filtered
        if weak_flag:
            weak = True
            warnings.append(
                "no remaining confounders after collider/descendant filtering — "
                "verify the DAG is correct"
            )

    return IdentificationResult(
        identifiable=identifiable,
        strategy=strategy,
        adjustment_set=tuple(adjustment),
        instrument=instrument,
        mediator=mediator,
        estimand_expression=expr or None,
        notes=notes,
        dowhy_metadata={
            "estimand_type": getattr(identified, "estimand_type", None),
        },
        warnings=warnings,
        weak=weak,
        diagnostics=diagnostics,
    )


# --- Helpers -----------------------------------------------------------------


def _required_columns(estimand: CausalEstimand, graph: CausalGraph) -> tuple[str, ...]:
    needed = {estimand.treatment, estimand.outcome, *estimand.modifiers}
    if estimand.mediator:
        needed.add(estimand.mediator)
    if estimand.instrument:
        needed.add(estimand.instrument)
    needed.update(graph.nodes)
    return tuple(sorted(needed))


def _graph_to_gml(graph: CausalGraph) -> str:
    """Emit a minimal GML string that DoWhy can parse."""
    lines = ["graph [", "  directed 1"]
    node_idx = {n: i for i, n in enumerate(graph.nodes)}
    for n, i in node_idx.items():
        lines.append(f'  node [ id {i} label "{n}" ]')
    for edge in graph.edges:
        s = node_idx.get(edge.source)
        t = node_idx.get(edge.target)
        if s is None or t is None:
            continue
        lines.append(f"  edge [ source {s} target {t} ]")
    lines.append("]")
    return "\n".join(lines)


def _interpret(
    identified: Any, estimand: CausalEstimand
) -> tuple[IdStrategy, tuple[str, ...], str | None, str | None]:
    """Coerce DoWhy's IdentifiedEstimand into our typed strategy tag.

    DoWhy's API across 0.10–0.14 exposes ``backdoor_variables``,
    ``frontdoor_variables``, and ``instrumental_variables`` as either dicts or
    lists. We probe defensively.
    """
    estimands = getattr(identified, "estimands", {}) or {}
    if estimands.get("backdoor") and getattr(identified, "backdoor_variables", None):
        adj = _flatten(identified.backdoor_variables)
        return "backdoor", tuple(sorted(set(adj))), None, None
    if estimands.get("iv") and getattr(identified, "instrumental_variables", None):
        iv = _flatten(identified.instrumental_variables)
        return "iv", (), (iv[0] if iv else None), None
    if estimands.get("frontdoor") and getattr(identified, "frontdoor_variables", None):
        fd = _flatten(identified.frontdoor_variables)
        return "frontdoor", tuple(sorted(set(fd))), None, (fd[0] if fd else None)
    if estimand.mediator and estimand.klass in {EstimandClass.NDE, EstimandClass.NIE}:
        return "mediation", (), None, estimand.mediator
    return "non-identifiable", (), None, None


def _filter_adjustment_set(
    adjustment: tuple[str, ...],
    graph: CausalGraph,
    estimand: CausalEstimand,
) -> tuple[tuple[str, ...], dict[str, set[str]], bool, list[str]]:
    """Apply the collider/descendant/mediator safety filters to ``adjustment``.

    Returns the filtered tuple, a dict of {category: dropped-set}, a ``weak``
    flag (True when the original set was non-empty but is now empty), and a
    list of human-readable warnings for any drop.
    """
    original = set(adjustment)
    treatment = estimand.treatment
    outcome = estimand.outcome

    descendants_of_T: frozenset[str] = (
        graph.descendants(treatment) if treatment in graph.nodes else frozenset()
    )
    mediators: set[str] = set(graph.variables_with_role(VariableRole.MEDIATOR))
    role_colliders: set[str] = set(graph.variables_with_role(VariableRole.COLLIDER))
    structural_colliders: set[str] = (
        set(graph.colliders_between(treatment, outcome))
        if (treatment in graph.nodes and outcome in graph.nodes)
        else set()
    )
    colliders = role_colliders | structural_colliders

    # Categorize mediators first (more specific than "descendant"), then
    # remaining descendants, then colliders. Mediators and structural colliders
    # are typically descendants of T too, so this ordering yields the most
    # informative diagnostic bucket.
    dropped_mediators = original & mediators
    dropped_colliders = (original & colliders) - dropped_mediators
    dropped_descendants = (original & set(descendants_of_T)) - dropped_mediators - dropped_colliders

    # T and Y themselves should never be in the adjustment set
    dropped_descendants.discard(treatment)
    dropped_descendants.discard(outcome)

    filtered = tuple(
        sorted(
            original
            - dropped_descendants
            - dropped_mediators
            - dropped_colliders
            - {treatment, outcome}
        )
    )

    warnings: list[str] = []
    if dropped_descendants:
        warnings.append(
            f"Dropped descendants of treatment {treatment!r} from adjustment set: "
            f"{sorted(dropped_descendants)} (would block mediated effects)."
        )
    if dropped_mediators:
        warnings.append(
            f"Dropped known mediators from adjustment set: "
            f"{sorted(dropped_mediators)} (mediators lie on the causal path)."
        )
    if dropped_colliders:
        warnings.append(
            f"Dropped colliders on T-Y paths from adjustment set: "
            f"{sorted(dropped_colliders)} (conditioning would induce M-bias)."
        )

    weak = bool(original) and not filtered

    drops = {
        "descendants": dropped_descendants,
        "mediators": dropped_mediators,
        "colliders": dropped_colliders,
    }
    return filtered, drops, weak, warnings


def _flatten(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for v in value.values():
            if isinstance(v, (list, tuple)):
                out.extend(v)
            elif isinstance(v, str):
                out.append(v)
        return out
    if isinstance(value, (list, tuple)):
        out_l: list[str] = []
        for v in value:
            if isinstance(v, (list, tuple)):
                out_l.extend(v)
            else:
                out_l.append(str(v))
        return out_l
    return []
