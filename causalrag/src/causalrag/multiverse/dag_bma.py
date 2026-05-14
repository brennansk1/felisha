"""Multiverse-of-DAGs Bayesian model averaging (Sprint 6.3).

Causal discovery rarely yields a single, definitive DAG: it returns a
ranked list of plausible candidates, often with bootstrap stability
probabilities attached. Each candidate may license a different
identification strategy (or none at all), and the point estimate that
flows out of Step 7 is therefore *conditional on the chosen DAG*.

This module operationalises Bayesian model averaging over those
candidate DAGs (PDD §11.1, design principle "triangulate, don't
cherry-pick"). For each candidate G_i with posterior weight w_i we:

1. Run Step 5 identification on G_i.
2. If identifiable, run Step 7 estimation to obtain (point_i, se_i).
3. Combine across DAGs with the standard BMA point/variance formulas:

       point_BMA = Σ w_i · point_i
       SE_BMA    = sqrt( Σ w_i · (SE_i^2 + (point_i - point_BMA)^2) )

   The variance formula is the law of total variance: within-DAG
   sampling variance plus between-DAG disagreement variance. Both
   pieces are essential — using SE_i alone would understate the
   epistemic uncertainty contributed by DAG ambiguity.

When ``bootstrapped_cd_posterior`` is omitted, weights default to a
uniform 1/k over identifiable DAGs; otherwise they are renormalised
over identifiable DAGs so the weights inside the BMA sum to 1.

A coarse ``consensus_verdict`` is derived from the spread of the
identifiable findings: ``"consensus"`` when the points agree on sign
and the relative spread is small, ``"split"`` when identifiable DAGs
disagree on sign, and ``"non_identifiable"`` when fewer than half of
the candidate DAGs admit identification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Protocol

import pandas as pd

from causalrag.core.estimand import CausalEstimand
from causalrag.core.graph import CausalGraph
from causalrag.core.protocol import StudyProtocol


ConsensusVerdict = Literal["consensus", "split", "non_identifiable"]


@dataclass
class DAGBMAFinding:
    """Per-DAG row in a :class:`DAGBMAReport`.

    ``posterior_weight`` is the *normalised* weight used inside the BMA
    aggregate (i.e. it sums to 1 across identifiable findings). DAGs
    that fail identification carry ``posterior_weight == 0.0`` and
    contribute no mass to the average.
    """

    dag_index: int
    dag_rank: int
    identifiable: bool
    point: float | None
    se: float | None
    posterior_weight: float
    notes: list[str] = field(default_factory=list)


@dataclass
class DAGBMAReport:
    findings: list[DAGBMAFinding]
    n_dags: int
    n_identifiable: int
    bma_point: float
    bma_se: float
    bma_ci_low: float | None
    bma_ci_high: float | None
    consensus_verdict: ConsensusVerdict
    interpretation: str


# --- Identification / estimation hooks ---------------------------------------
#
# Step 5 + Step 7 are imported lazily because they pull in DoWhy/EconML.
# Tests inject a lightweight ``run_single`` callable to avoid the heavy
# dependency chain.


class _SingleRunner(Protocol):
    """Callable that runs id + estimation against one DAG.

    Returns ``(identifiable, point, se, notes)``. ``point`` and ``se``
    must be ``None`` when ``identifiable`` is False or estimation
    fails. ``notes`` is a free-form list of human-readable strings
    propagated to :class:`DAGBMAFinding.notes`.
    """

    def __call__(
        self,
        graph: CausalGraph,
        df: pd.DataFrame,
        estimand: CausalEstimand,
        estimator_id: str,
    ) -> tuple[bool, float | None, float | None, list[str]]:
        ...


def _default_run_single(
    graph: CausalGraph,
    df: pd.DataFrame,
    estimand: CausalEstimand,
    estimator_id: str,
) -> tuple[bool, float | None, float | None, list[str]]:
    """Real Step 5 + Step 7 against a single DAG.

    Imports are deferred so the module is importable in test envs that
    lack DoWhy/EconML.
    """
    from causalrag.roadmap.q5_identify import identify_effect
    from causalrag.roadmap.q7_estimate import estimate as run_estimate

    notes: list[str] = []
    try:
        ident = identify_effect(estimand, graph, df=df)
    except Exception as e:  # defensive — DoWhy can raise on odd DAGs
        return False, None, None, [f"identify failed: {type(e).__name__}: {e}"]

    if not ident.identifiable:
        notes.append(f"non-identifiable: {ident.strategy}")
        return False, None, None, notes

    try:
        result = run_estimate(
            df=df,
            estimand=estimand,
            identification=ident,
            protocol=StudyProtocol(name="dag_bma"),
            confounders=ident.adjustment_set,
            prefer=estimator_id,
        )
    except Exception as e:
        notes.append(f"estimate failed: {type(e).__name__}: {e}")
        return False, None, None, notes

    return True, float(result.point_estimate), result.se, notes


# --- BMA core ----------------------------------------------------------------


def _normalise_weights(
    candidate_graphs: Iterable[CausalGraph],
    posterior: dict[int, float] | None,
) -> list[float]:
    """Return a per-DAG (non-normalised across identifiability) prior
    weight vector. Renormalisation over *identifiable* DAGs happens
    after Step 5 runs."""
    graphs = list(candidate_graphs)
    k = len(graphs)
    if k == 0:
        return []
    if posterior is None:
        return [1.0 / k] * k
    raw = [float(posterior.get(i, 0.0)) for i in range(k)]
    total = sum(raw)
    if total <= 0.0:
        # Posterior provided but all-zero for the candidate set — fall
        # back to uniform so we still produce a meaningful report.
        return [1.0 / k] * k
    return [r / total for r in raw]


def _consensus_verdict(
    findings: list[DAGBMAFinding],
    n_total: int,
    bma_point: float,
) -> tuple[ConsensusVerdict, str]:
    """Classify the spread of identifiable findings.

    Rules:

    - ``non_identifiable`` when fewer than half the candidate DAGs are
      identifiable. (Majority of the multiverse can't even answer the
      question — the BMA point is reported but flagged.)
    - ``split`` when identifiable DAGs disagree on the *sign* of the
      effect, or the weighted relative spread is large.
    - ``consensus`` otherwise.

    The "relative spread" is the weighted standard deviation of points
    divided by the absolute BMA point (with a small floor to keep it
    well-behaved near zero).
    """
    ident = [f for f in findings if f.identifiable and f.point is not None]
    n_ident = len(ident)
    if n_total == 0 or n_ident * 2 < n_total:
        return (
            "non_identifiable",
            f"Only {n_ident} of {n_total} candidate DAGs admit identification; "
            "DAG-averaged effect is reported but the multiverse cannot reach "
            "a confident verdict.",
        )

    signs = {1 if f.point > 0 else (-1 if f.point < 0 else 0) for f in ident}
    # Treat zero-sign as agreeing with whichever non-zero sign is present.
    if 1 in signs and -1 in signs:
        return (
            "split",
            f"{n_ident} identifiable DAGs disagree on the sign of the effect; "
            "BMA point hides genuine structural ambiguity.",
        )

    # Weighted spread relative to |BMA point|.
    var_between = sum(
        f.posterior_weight * (f.point - bma_point) ** 2 for f in ident
    )
    spread = math.sqrt(max(var_between, 0.0))
    denom = max(abs(bma_point), 1e-9)
    if spread / denom > 0.5:
        return (
            "split",
            f"{n_ident} identifiable DAGs agree on sign but disagree in magnitude "
            f"(weighted spread {spread:.3g} vs BMA point {bma_point:.3g}).",
        )

    return (
        "consensus",
        f"{n_ident} of {n_total} candidate DAGs are identifiable and agree "
        f"(BMA point {bma_point:.3g}, SE captured between- and within-DAG "
        "uncertainty).",
    )


def dag_bma(
    *,
    candidate_graphs: Iterable[CausalGraph],
    df: pd.DataFrame,
    estimand: CausalEstimand,
    bootstrapped_cd_posterior: dict[int, float] | None = None,
    estimator_id: str = "python.dml.linear",
    run_single: _SingleRunner | None = None,
) -> DAGBMAReport:
    """Run Bayesian model averaging across candidate DAGs.

    Parameters
    ----------
    candidate_graphs:
        Ranked iterable of candidate DAGs (top-K from causal discovery).
        Order is preserved on the resulting findings via ``dag_index``;
        each finding also carries the ``CausalGraph.rank`` value.
    df:
        Observed data passed through to Step 5 / Step 7 unchanged.
    estimand:
        Target counterfactual quantity.
    bootstrapped_cd_posterior:
        Optional mapping ``{dag_index: posterior_mass}`` from bootstrap
        causal-discovery stability. When omitted, weights are uniform
        ``1/k``.
    estimator_id:
        Registry id forwarded to Step 7's ``prefer=`` (default
        ``"python.dml.linear"`` for back-compat with §33 sprint outputs).
    run_single:
        Test hook. When provided, used in place of the default Step 5 +
        Step 7 runner. Production callers leave this as ``None``.
    """
    graphs = list(candidate_graphs)
    n_dags = len(graphs)
    if n_dags == 0:
        return DAGBMAReport(
            findings=[],
            n_dags=0,
            n_identifiable=0,
            bma_point=0.0,
            bma_se=0.0,
            bma_ci_low=None,
            bma_ci_high=None,
            consensus_verdict="non_identifiable",
            interpretation="No candidate DAGs supplied; nothing to average.",
        )

    runner = run_single or _default_run_single
    prior_weights = _normalise_weights(graphs, bootstrapped_cd_posterior)

    raw: list[tuple[CausalGraph, bool, float | None, float | None, list[str], float]] = []
    for idx, (graph, w) in enumerate(zip(graphs, prior_weights)):
        identifiable, point, se, notes = runner(graph, df, estimand, estimator_id)
        raw.append((graph, identifiable, point, se, notes, w))

    # Renormalise weights over *identifiable* DAGs so BMA mass sums to 1.
    ident_mass = sum(
        w for (_, identifiable, point, _, _, w) in raw
        if identifiable and point is not None
    )

    findings: list[DAGBMAFinding] = []
    for idx, (graph, identifiable, point, se, notes, w) in enumerate(raw):
        if identifiable and point is not None and ident_mass > 0.0:
            norm_w = w / ident_mass
        else:
            norm_w = 0.0
        findings.append(
            DAGBMAFinding(
                dag_index=idx,
                dag_rank=getattr(graph, "rank", idx + 1),
                identifiable=identifiable and point is not None,
                point=point if identifiable else None,
                se=se if identifiable else None,
                posterior_weight=norm_w,
                notes=list(notes),
            )
        )

    n_identifiable = sum(1 for f in findings if f.identifiable)

    # BMA point + SE — only over identifiable findings.
    if n_identifiable == 0 or ident_mass <= 0.0:
        bma_point = 0.0
        bma_se = 0.0
        bma_ci_low: float | None = None
        bma_ci_high: float | None = None
    else:
        bma_point = sum(
            f.posterior_weight * f.point  # type: ignore[operator]
            for f in findings
            if f.identifiable
        )
        # Law of total variance: within-DAG SE^2 + between-DAG (point - mean)^2.
        var_terms: list[float] = []
        any_se = False
        for f in findings:
            if not f.identifiable:
                continue
            within = (f.se ** 2) if (f.se is not None and math.isfinite(f.se)) else 0.0
            if f.se is not None and math.isfinite(f.se):
                any_se = True
            between = (f.point - bma_point) ** 2  # type: ignore[operator]
            var_terms.append(f.posterior_weight * (within + between))
        var_total = sum(var_terms)
        bma_se = math.sqrt(max(var_total, 0.0))
        if any_se and bma_se > 0.0:
            bma_ci_low = bma_point - 1.96 * bma_se
            bma_ci_high = bma_point + 1.96 * bma_se
        else:
            # No SE available from any identifiable DAG — CI undefined.
            bma_ci_low = None
            bma_ci_high = None

    verdict, interpretation = _consensus_verdict(findings, n_dags, bma_point)

    return DAGBMAReport(
        findings=findings,
        n_dags=n_dags,
        n_identifiable=n_identifiable,
        bma_point=bma_point,
        bma_se=bma_se,
        bma_ci_low=bma_ci_low,
        bma_ci_high=bma_ci_high,
        consensus_verdict=verdict,
        interpretation=interpretation,
    )


__all__ = ["DAGBMAFinding", "DAGBMAReport", "dag_bma"]
