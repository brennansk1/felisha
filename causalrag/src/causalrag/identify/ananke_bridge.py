"""Sprint 6.1 — second identification engine via ananke / Y0.

This module provides a parallel identification path alongside DoWhy
(:mod:`causalrag.roadmap.q5_identify`). The optional dependency
``ananke-causal`` is *not* pinned in ``pyproject.toml``; when missing we fall
back to a pure-Python implementation that handles the cases the existing
:mod:`causalrag.identify.decomposition` helpers already cover (backdoor on
DAGs, simple front-door, and c-component decomposition for ADMG
non-identifiability). The intent is that callers can always invoke
:func:`ananke_identify` and either get a full Tian-Shpitser verdict (when
ananke is installed) or a conservative fallback annotated
``backend="fallback"``.

The dataclass :class:`AnankeIDResult` mirrors the fields requested in the
sprint plan and is intentionally a plain dataclass (rather than a pydantic
model) so it stays decoupled from the existing
:class:`~causalrag.roadmap.q5_identify.IdentificationResult` schema.

:func:`reconcile` is a thin helper that compares an ``IdentificationResult``
(DoWhy) and an :class:`AnankeIDResult` (ananke) and reports whether the two
engines agree. Disagreement is flagged when (a) one engine identifies and
the other does not, or (b) both identify via a backdoor strategy but with
different adjustment sets.

The ananke API surface has shifted across versions (``OneLineID``,
``OneLineAID``, ``identify``, ``ID``, …) so the bridge probes defensively
and never imports ananke at module load time.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Literal

from causalrag.core.graph import CausalGraph
from causalrag.identify.decomposition import c_components, extract_relevant_subgraph

EstimandClassStr = str
MethodStr = Literal[
    "backdoor", "frontdoor", "iv", "mediation", "tian-id", "idc", "transport", "none"
]
BackendStr = Literal["ananke", "fallback"]


@dataclass
class AnankeIDResult:
    """Structured output of :func:`ananke_identify`.

    Mirrors the DoWhy ``IdentificationResult`` where possible while adding
    ADMG-specific fields (c-component decomposition, free-form do-calculus
    proof steps, ``backend`` tag).
    """

    identified: bool
    estimand_class: EstimandClassStr
    estimand_expression: str | None
    adjustment_set: tuple[str, ...]
    method: MethodStr
    c_component_decomposition: tuple[frozenset[str], ...]
    proof_steps: list[str] = field(default_factory=list)
    weak: bool = False
    warnings: list[str] = field(default_factory=list)
    backend: BackendStr = "fallback"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ananke_identify(
    *,
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    estimand_class: EstimandClassStr = "ATE",
) -> AnankeIDResult:
    """Run ananke's identification on ``graph`` for ``treatment`` → ``outcome``.

    Conversion: :class:`CausalGraph` (directed + ``bidirected``-flagged
    edges) → ``ananke.graphs.ADMG``. Then we attempt, in order:

    1. ``ananke.identification.OneLineID(admg, [T], [Y]).id()``
    2. ``ananke.identification.OneLineID(admg, [T], [Y]).functional()``
    3. ``ananke.identification.identify(admg, {T}, {Y})``
    4. Module-level ``ananke.identification.ID``-like callables.

    If none of the probes succeeds, or if the ``ananke`` package is not
    installed, the function falls back to the pure-Python heuristics in
    :mod:`causalrag.identify.decomposition` and returns a result annotated
    ``backend="fallback"``.
    """
    # Validate inputs early — these are user-supplied identifiers; we want
    # the bridge to behave deterministically rather than throw deep inside
    # ananke.
    if treatment not in graph.nodes:
        return _fallback_result(
            graph,
            treatment,
            outcome,
            estimand_class,
            extra_warnings=[f"treatment {treatment!r} not in graph"],
            identifiable_override=False,
        )
    if outcome not in graph.nodes:
        return _fallback_result(
            graph,
            treatment,
            outcome,
            estimand_class,
            extra_warnings=[f"outcome {outcome!r} not in graph"],
            identifiable_override=False,
        )
    if treatment == outcome:
        return _fallback_result(
            graph,
            treatment,
            outcome,
            estimand_class,
            extra_warnings=["treatment == outcome"],
            identifiable_override=False,
        )

    ananke_mod = _try_import_ananke()
    if ananke_mod is None:
        return _fallback_result(graph, treatment, outcome, estimand_class)

    try:
        return _ananke_path(ananke_mod, graph, treatment, outcome, estimand_class)
    except Exception as exc:  # noqa: BLE001 — defensive boundary
        # Any failure inside ananke should degrade to the fallback, with the
        # exception type+message captured as a warning.
        result = _fallback_result(graph, treatment, outcome, estimand_class)
        result.warnings.append(
            f"ananke path raised {type(exc).__name__}: {exc}; using fallback"
        )
        return result


def reconcile(dowhy_result: Any, ananke_result: AnankeIDResult) -> dict[str, Any]:
    """Compare a DoWhy ``IdentificationResult`` with an :class:`AnankeIDResult`.

    The DoWhy side is typed as ``Any`` to avoid importing pydantic here; we
    read only the ``identifiable`` and ``adjustment_set`` attributes.

    Returns
    -------
    dict
        ``{"agree": bool, "primary": "dowhy" | "ananke",
           "disagreement_note": str | None}``. ``primary`` is whichever
        engine produced an identifiable verdict; ties default to ``"dowhy"``.
    """
    dowhy_identified = bool(getattr(dowhy_result, "identifiable", False))
    dowhy_adj = tuple(sorted(getattr(dowhy_result, "adjustment_set", ()) or ()))
    dowhy_strategy = getattr(dowhy_result, "strategy", None)

    ananke_identified = bool(ananke_result.identified)
    ananke_adj = tuple(sorted(ananke_result.adjustment_set))

    # Verdict mismatch.
    if dowhy_identified != ananke_identified:
        primary: Literal["dowhy", "ananke"] = "dowhy" if dowhy_identified else "ananke"
        note = (
            f"identifiability mismatch: dowhy={dowhy_identified} "
            f"(strategy={dowhy_strategy}), ananke={ananke_identified} "
            f"(method={ananke_result.method}, backend={ananke_result.backend})"
        )
        return {"agree": False, "primary": primary, "disagreement_note": note}

    # Both unidentifiable — trivial agreement, but flag the no-go situation.
    if not dowhy_identified and not ananke_identified:
        return {"agree": True, "primary": "dowhy", "disagreement_note": None}

    # Both identified — compare adjustment sets when both engines went the
    # backdoor route. If methods differ (e.g. one chose backdoor, the other
    # front-door), we report agreement on the verdict but note the methodic
    # divergence.
    both_backdoor = dowhy_strategy == "backdoor" and ananke_result.method == "backdoor"
    if both_backdoor and dowhy_adj != ananke_adj:
        return {
            "agree": False,
            "primary": "dowhy",
            "disagreement_note": (
                f"adjustment sets differ: dowhy={dowhy_adj} vs ananke={ananke_adj}"
            ),
        }
    if dowhy_strategy and ananke_result.method != "none" and (
        dowhy_strategy != ananke_result.method
    ):
        return {
            "agree": True,
            "primary": "dowhy",
            "disagreement_note": (
                f"both identifiable but via different methods: "
                f"dowhy={dowhy_strategy} vs ananke={ananke_result.method}"
            ),
        }
    return {"agree": True, "primary": "dowhy", "disagreement_note": None}


# ---------------------------------------------------------------------------
# Ananke conversion + probing
# ---------------------------------------------------------------------------


def _try_import_ananke() -> Any | None:
    """Best-effort import of ``ananke``. Returns the module or None."""
    try:
        return importlib.import_module("ananke")
    except Exception:  # noqa: BLE001 — ananke is optional
        return None


def _graph_to_admg(ananke_mod: Any, graph: CausalGraph) -> Any:
    """Convert :class:`CausalGraph` into ``ananke.graphs.ADMG``."""
    admg_cls = importlib.import_module("ananke.graphs").ADMG
    vertices = list(graph.nodes)
    di_edges = [(e.source, e.target) for e in graph.edges if not e.bidirected]
    bi_edges = [(e.source, e.target) for e in graph.edges if e.bidirected]
    # ADMG signature: ADMG(vertices, di_edges, bi_edges). Some versions take
    # positional, others keyword; try positional first.
    try:
        return admg_cls(vertices, di_edges, bi_edges)
    except TypeError:
        return admg_cls(vertices=vertices, di_edges=di_edges, bi_edges=bi_edges)


def _ananke_path(
    ananke_mod: Any,
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    estimand_class: EstimandClassStr,
) -> AnankeIDResult:
    """Run the actual ananke probes. Raises on unrecoverable errors."""
    ident_mod = importlib.import_module("ananke.identification")
    admg = _graph_to_admg(ananke_mod, graph)

    proof_steps: list[str] = [
        f"Converted CausalGraph to ADMG with "
        f"{len(list(getattr(admg, 'vertices', graph.nodes)))} vertices, "
        f"{sum(1 for e in graph.edges if not e.bidirected)} directed edges, "
        f"{sum(1 for e in graph.edges if e.bidirected)} bidirected edges."
    ]

    # Probe 1: OneLineID class with .id() / .functional()
    identified = False
    expression: str | None = None
    method: MethodStr = "tian-id"

    one_line = getattr(ident_mod, "OneLineID", None) or getattr(
        ident_mod, "OneLineAID", None
    )
    if one_line is not None:
        try:
            instance = one_line(admg, [treatment], [outcome])
        except TypeError:
            # Some versions expect sets.
            instance = one_line(admg, {treatment}, {outcome})
        # .id() commonly returns a bool; .functional() the expression string.
        id_fn = getattr(instance, "id", None) or getattr(instance, "identify", None)
        if callable(id_fn):
            verdict = id_fn()
            identified = bool(verdict) if not isinstance(verdict, str) else True
            proof_steps.append(f"OneLineID.id() returned: {verdict!r}")
            if isinstance(verdict, str):
                expression = verdict
        func_fn = getattr(instance, "functional", None)
        if callable(func_fn):
            try:
                expr = func_fn()
                expression = str(expr) if expr is not None else expression
                if expression:
                    proof_steps.append(f"OneLineID.functional(): {expression}")
                    identified = True
            except Exception as exc:  # noqa: BLE001
                proof_steps.append(
                    f"OneLineID.functional() raised {type(exc).__name__}: {exc}"
                )
    else:
        # Probe 2: module-level identify / ID
        for cand_name in ("identify", "ID", "id_alg"):
            cand = getattr(ident_mod, cand_name, None)
            if callable(cand):
                try:
                    out = cand(admg, {treatment}, {outcome})
                    identified = bool(out)
                    expression = str(out) if isinstance(out, str) else None
                    proof_steps.append(f"{cand_name}() returned: {out!r}")
                    break
                except Exception as exc:  # noqa: BLE001
                    proof_steps.append(
                        f"{cand_name}() raised {type(exc).__name__}: {exc}"
                    )

    # Method refinement: if there's no bidirected edge between T and Y and
    # backdoor variables exist, label as backdoor; otherwise tian-id.
    adjustment_set: tuple[str, ...] = ()
    if identified and not _has_bidirected_between(graph, treatment, outcome):
        backdoor_candidate = _backdoor_from_dag(graph, treatment, outcome)
        if backdoor_candidate is not None:
            method = "backdoor"
            adjustment_set = backdoor_candidate
            proof_steps.append(
                f"Inferred backdoor adjustment set from DAG ancestors: "
                f"{adjustment_set}"
            )
        else:
            method = "frontdoor" if _has_frontdoor(graph, treatment, outcome) else "tian-id"

    comps = tuple(c_components(graph))

    return AnankeIDResult(
        identified=identified,
        estimand_class=estimand_class,
        estimand_expression=expression,
        adjustment_set=adjustment_set,
        method=method if identified else "none",
        c_component_decomposition=comps,
        proof_steps=proof_steps,
        weak=False,
        warnings=[],
        backend="ananke",
    )


# ---------------------------------------------------------------------------
# Pure-Python fallback (works without ananke installed)
# ---------------------------------------------------------------------------


def _fallback_result(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    estimand_class: EstimandClassStr,
    *,
    extra_warnings: list[str] | None = None,
    identifiable_override: bool | None = None,
) -> AnankeIDResult:
    """Conservative identifiability verdict using only the in-tree helpers."""
    warnings = list(extra_warnings or [])
    warnings.append(
        "ananke not available — used pure-Python fallback "
        "(handles backdoor on DAGs, simple front-door, ADMG non-id via T<->Y)"
    )

    comps = tuple(c_components(graph))
    proof_steps: list[str] = [
        f"Fallback engine: graph has {len(graph.nodes)} nodes, "
        f"{len(graph.edges)} edges, {len(comps)} c-component(s)."
    ]

    if identifiable_override is False:
        return AnankeIDResult(
            identified=False,
            estimand_class=estimand_class,
            estimand_expression=None,
            adjustment_set=(),
            method="none",
            c_component_decomposition=comps,
            proof_steps=proof_steps,
            weak=False,
            warnings=warnings,
            backend="fallback",
        )

    if treatment not in graph.nodes or outcome not in graph.nodes or treatment == outcome:
        return AnankeIDResult(
            identified=False,
            estimand_class=estimand_class,
            estimand_expression=None,
            adjustment_set=(),
            method="none",
            c_component_decomposition=comps,
            proof_steps=proof_steps,
            weak=False,
            warnings=warnings,
            backend="fallback",
        )

    # Non-identifiability heuristic: hedge T<->Y is an unblockable backdoor.
    if _has_bidirected_between(graph, treatment, outcome):
        proof_steps.append(
            f"Found bidirected edge {treatment} <-> {outcome}: "
            f"hedge ⇒ T → Y not identifiable from observational distribution."
        )
        return AnankeIDResult(
            identified=False,
            estimand_class=estimand_class,
            estimand_expression=None,
            adjustment_set=(),
            method="none",
            c_component_decomposition=comps,
            proof_steps=proof_steps,
            weak=False,
            warnings=warnings,
            backend="fallback",
        )

    # Backdoor attempt: ancestors of T and Y that are not descendants of T.
    adjustment_set = _backdoor_from_dag(graph, treatment, outcome)
    if adjustment_set is not None:
        proof_steps.append(
            f"Backdoor adjustment set (parents of T excluding descendants of T): "
            f"{adjustment_set}"
        )
        expr = (
            f"sum_z P({outcome} | {treatment}, "
            f"{', '.join(adjustment_set) if adjustment_set else '∅'}) "
            f"* P({', '.join(adjustment_set) if adjustment_set else '∅'})"
        )
        return AnankeIDResult(
            identified=True,
            estimand_class=estimand_class,
            estimand_expression=expr,
            adjustment_set=adjustment_set,
            method="backdoor",
            c_component_decomposition=comps,
            proof_steps=proof_steps,
            weak=not bool(adjustment_set),
            warnings=warnings,
            backend="fallback",
        )

    # Front-door attempt for the canonical T -> M -> Y with T<->Y hedge case
    # is unreachable here because we already early-returned on T<->Y. Detect
    # a structural front-door anyway (T -> M -> Y with no T<->Y).
    if _has_frontdoor(graph, treatment, outcome):
        proof_steps.append("Structural front-door pattern T -> M -> Y detected.")
        return AnankeIDResult(
            identified=True,
            estimand_class=estimand_class,
            estimand_expression=None,
            adjustment_set=(),
            method="frontdoor",
            c_component_decomposition=comps,
            proof_steps=proof_steps,
            weak=False,
            warnings=warnings,
            backend="fallback",
        )

    proof_steps.append("No backdoor or front-door pattern recognised by fallback.")
    return AnankeIDResult(
        identified=False,
        estimand_class=estimand_class,
        estimand_expression=None,
        adjustment_set=(),
        method="none",
        c_component_decomposition=comps,
        proof_steps=proof_steps,
        weak=False,
        warnings=warnings,
        backend="fallback",
    )


# ---------------------------------------------------------------------------
# Pure-graph helpers (no ananke dependency)
# ---------------------------------------------------------------------------


def _has_bidirected_between(graph: CausalGraph, a: str, b: str) -> bool:
    return any(
        e.bidirected and {e.source, e.target} == {a, b} for e in graph.edges
    )


def _backdoor_from_dag(
    graph: CausalGraph, treatment: str, outcome: str
) -> tuple[str, ...] | None:
    """Return parents-of-T ∩ ancestors-of-Y, dropping descendants of T.

    Returns None when treatment or outcome are missing, or when there is a
    direct bidirected edge T <-> Y (signalling hedge non-identifiability).
    The empty tuple is a valid return — it means *no adjustment needed*,
    which still satisfies the backdoor criterion (e.g. RCT-like graph).
    """
    if _has_bidirected_between(graph, treatment, outcome):
        return None
    # Restrict to relevant subgraph for efficiency on big DAGs.
    sub = extract_relevant_subgraph(graph, treatment, outcome)
    if treatment not in sub.nodes or outcome not in sub.nodes:
        return None
    desc_t = sub.descendants(treatment)
    parents_t = set(sub.parents(treatment))
    # Only count nodes that are also ancestors of Y (otherwise they're
    # irrelevant for identifying T -> Y).
    dg = sub.to_networkx()
    import networkx as nx

    try:
        anc_y = nx.ancestors(dg, outcome)
    except Exception:  # noqa: BLE001
        return None
    candidates = (parents_t & anc_y) - desc_t - {treatment, outcome}
    return tuple(sorted(candidates))


def _has_frontdoor(graph: CausalGraph, treatment: str, outcome: str) -> bool:
    """Detect T -> M -> Y for some M with no other path bypassing M.

    A coarse structural check sufficient for the fallback: there exists an
    M such that ``T -> M`` and ``M -> Y`` are edges and ``Y`` has no
    direct edge from ``T``.
    """
    direct_t_y = any(
        not e.bidirected and e.source == treatment and e.target == outcome
        for e in graph.edges
    )
    if direct_t_y:
        return False
    for e in graph.edges:
        if e.bidirected or e.source != treatment:
            continue
        mediator = e.target
        if any(
            (not e2.bidirected) and e2.source == mediator and e2.target == outcome
            for e2 in graph.edges
        ):
            return True
    return False
