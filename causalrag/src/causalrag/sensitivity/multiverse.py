"""Multiverse / specification-curve sensitivity (PDD §11.1).

Runs the same hypothesis across the full Cartesian product of:

- candidate DAGs (cross-DAG triangulation)
- estimators that satisfy the situation flags (cross-estimator triangulation)
- selection methods (post-double / correlation-pruning / none)

Returns one row per (dag, estimator, selection) cell with point + CI. The
report renderer plots this as a forest / specification curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from causalrag.core.estimand import CausalEstimand
from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalGraph
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import get_registry
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q7_estimate import estimate as run_estimate


@dataclass
class MultiverseRow:
    dag_rank: int
    estimator_id: str
    selection: str
    point_estimate: float
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    n_used: int
    notes: str | None = None


@dataclass
class MultiverseResultFull:
    rows: list[MultiverseRow] = field(default_factory=list)
    headline_point: float = 0.0
    headline_ci: tuple[float, float] | None = None
    rows_passing_sign_test: int = 0
    rows_total: int = 0

    def summary(self) -> dict[str, Any]:
        points = [r.point_estimate for r in self.rows]
        if not points:
            return {}
        points_sorted = sorted(points)
        n = len(points_sorted)
        return {
            "n": n,
            "min": points_sorted[0],
            "max": points_sorted[-1],
            "median": points_sorted[n // 2],
            "fraction_same_sign": float(
                sum(1 for p in points if (p > 0) == (self.headline_point > 0)) / n
            ),
        }


def run_multiverse(
    df: pd.DataFrame,
    estimand: CausalEstimand,
    protocol: StudyProtocol,
    *,
    confounders: tuple[str, ...],
    selection_methods: tuple[str, ...] = ("auto", "post_double_selection", "correlation_pruning", "none"),
    max_estimators: int = 4,
) -> MultiverseResultFull:
    """Run the same hypothesis across the multiverse of candidate DAGs ×
    estimators × selection methods. Capped by ``max_estimators`` to keep
    runtime reasonable."""
    flags = frozenset(protocol.flags)
    candidates_for = get_registry().candidates_for(
        estimand=estimand.klass.value,
        required=flags,
        n=len(df),
    )
    estimator_ids = [c.id for c in candidates_for[:max_estimators]]
    dags = protocol.candidate_graphs or (
        protocol.discovery.candidate_graphs if protocol.discovery else ()
    )
    if not dags:
        dags = (
            CausalGraph.from_edge_list(
                [(estimand.treatment, estimand.outcome)]
            ),
        )

    rows: list[MultiverseRow] = []
    headline_point: float | None = None
    headline_ci: tuple[float, float] | None = None
    for dag in dags:
        ident = identify_effect(estimand, dag, df=df)
        if not ident.identifiable:
            continue
        for est_id in estimator_ids:
            for sel in selection_methods:
                try:
                    result = run_estimate(
                        df=df,
                        estimand=estimand,
                        identification=ident,
                        protocol=protocol,
                        confounders=confounders,
                        flags=set(flags),
                        prefer=est_id,
                        selection=sel,  # type: ignore[arg-type]
                    )
                except Exception as e:
                    rows.append(
                        MultiverseRow(
                            dag_rank=dag.rank,
                            estimator_id=est_id,
                            selection=sel,
                            point_estimate=float("nan"),
                            ci_low=None,
                            ci_high=None,
                            p_value=None,
                            n_used=0,
                            notes=f"failed: {type(e).__name__}",
                        )
                    )
                    continue
                rows.append(
                    MultiverseRow(
                        dag_rank=dag.rank,
                        estimator_id=result.estimator_id,
                        selection=sel,
                        point_estimate=result.point_estimate,
                        ci_low=result.ci_low,
                        ci_high=result.ci_high,
                        p_value=result.p_value,
                        n_used=result.n_used,
                    )
                )
                if headline_point is None:
                    headline_point = result.point_estimate
                    if result.ci_low is not None and result.ci_high is not None:
                        headline_ci = (result.ci_low, result.ci_high)

    n_total = len([r for r in rows if not (r.point_estimate != r.point_estimate)])
    n_same_sign = 0
    if headline_point is not None:
        n_same_sign = sum(
            1
            for r in rows
            if r.point_estimate == r.point_estimate
            and (r.point_estimate > 0) == (headline_point > 0)
        )
    return MultiverseResultFull(
        rows=rows,
        headline_point=headline_point or 0.0,
        headline_ci=headline_ci,
        rows_passing_sign_test=n_same_sign,
        rows_total=n_total,
    )


__all__ = ["MultiverseRow", "MultiverseResultFull", "run_multiverse"]
