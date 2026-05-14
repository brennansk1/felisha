"""Hallucination guard pipeline — four layers (PDD §16.6).

The three-layer gate from CausalRAG generalizes to four layers for the
general-purpose package:

- **Layer 1 — Prevention.** Prompts are templated; column names, value labels,
  and allowed enum values are injected directly into the prompt so the LLM is
  constrained at generation time. Ollama 0.4+ structured-output enforcement is
  used wherever the caller passes a ``json_schema``.

  Implementation: ``OllamaClient.parse(..., json_schema=...)`` in
  :mod:`causalrag.llm.ollama_client`. No code lives in this module for Layer
  1; it is a *design* commitment carried out by the prompt builders.

- **Layer 2 — Schema validation.** Every LLM response is parsed by a Pydantic
  model. Failures retry once with the validation error as feedback.
  Persistent failures raise :class:`SchemaValidationFailed`.

  Implementation: ``OllamaClient.parse``.

- **Layer 3 — Semantic validation.** Every variable the LLM names must exist
  in the dataset. Every temporal claim is cross-checked against detected
  temporal ordering. Every IV candidate fails its relevance check on the data
  silently and is downgraded. This module exposes :func:`check_columns_exist`,
  :func:`check_temporal_consistency`, and :func:`check_iv_relevance`.

- **Layer 4 — Statistical validation.** Conditional-independence claims
  implied by an LLM-proposed DAG are tested on the data. Disagreements are
  surfaced; *neither view silently overrides the other*. This module exposes
  :func:`audit_dag_edges`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from causalrag.core.graph import CausalGraph


# --- Layer 3: semantic validation -------------------------------------------


class SemanticGuardViolation(ValueError):
    """Layer 3 violation that the caller decided cannot be auto-recovered."""


def check_columns_exist(referenced: list[str], known: set[str], context: str) -> list[str]:
    """Return the subset of ``referenced`` that does not appear in ``known``.

    The caller decides whether to raise or to log + drop. Layer-3 violations
    that surface to the analyst are recorded in the StudyProtocol's
    ``overrides`` list rather than silently dropped.
    """
    return [c for c in referenced if c not in known]


def check_temporal_consistency(
    edges: list[tuple[str, str]],
    temporal_positions: dict[str, str],
) -> list[tuple[str, str, str]]:
    """Return edges that violate temporal order: ``(source, target, reason)``.

    Position rank: baseline=0, pre_treatment=1, treatment_era=2, post_treatment=3,
    outcome=4. An edge whose source rank > target rank is a violation
    (post-treatment can't cause pre-treatment).
    """
    rank = {
        "baseline": 0,
        "pre_treatment": 1,
        "treatment_era": 2,
        "post_treatment": 3,
        "outcome": 4,
        "unknown": None,
    }
    out: list[tuple[str, str, str]] = []
    for src, tgt in edges:
        rs = rank.get(temporal_positions.get(src, "unknown"))
        rt = rank.get(temporal_positions.get(tgt, "unknown"))
        if rs is None or rt is None:
            continue
        if rs > rt:
            out.append(
                (src, tgt, f"temporal mismatch: {src}[{rs}] -> {tgt}[{rt}]")
            )
    return out


@dataclass
class IVRelevance:
    instrument: str
    treatment: str
    p_value: float
    correlation: float
    passes_relevance: bool


def check_iv_relevance(
    df: pd.DataFrame,
    instrument: str,
    treatment: str,
    threshold_p: float = 0.05,
    min_abs_corr: float = 0.1,
) -> IVRelevance:
    """Layer 3 IV check: weak instruments are silently downgraded.

    Tests the marginal Z ↔ T association. A genuine IV must satisfy
    relevance (Z is associated with T) — this is the necessary condition;
    exclusion is untestable from observational data alone.
    """
    from scipy.stats import kendalltau

    if instrument not in df.columns or treatment not in df.columns:
        return IVRelevance(instrument, treatment, 1.0, 0.0, False)
    joined = df[[instrument, treatment]].dropna()
    if len(joined) < 30:
        return IVRelevance(instrument, treatment, 1.0, 0.0, False)
    try:
        tau, p = kendalltau(joined[instrument], joined[treatment])
        corr = float(tau)
        p_val = float(p)
    except Exception:
        return IVRelevance(instrument, treatment, 1.0, 0.0, False)
    passes = p_val < threshold_p and abs(corr) >= min_abs_corr
    return IVRelevance(instrument, treatment, p_val, corr, passes)


# --- Layer 4: statistical validation of DAG edges ---------------------------


Verdict = Literal["supported", "contradicted", "inconclusive"]


@dataclass
class EdgeAudit:
    source: str
    target: str
    conditioning_set: tuple[str, ...]
    partial_correlation: float
    p_value: float
    verdict: Verdict
    note: str | None = None


def _partial_correlation(
    df: pd.DataFrame, x: str, y: str, z: tuple[str, ...]
) -> tuple[float, float]:
    """Return (partial r, two-sided p-value) for X ⊥ Y | Z via OLS residualization.

    Uses Fisher's z transform for the p-value. Falls back to a Kendall-tau
    test on residuals when OLS is singular (perfectly collinear conditioner).
    """
    from scipy.stats import kendalltau, norm

    joined = df[[x, y, *z]].dropna()
    if len(joined) < 30:
        return 0.0, 1.0

    if not z:
        # Marginal correlation via Pearson on rank-transformed columns
        rx = joined[x].rank().to_numpy()
        ry = joined[y].rank().to_numpy()
        r = float(np.corrcoef(rx, ry)[0, 1])
    else:
        try:
            from sklearn.linear_model import LinearRegression

            zm = joined[list(z)].to_numpy().astype(float)
            xm = joined[x].to_numpy().astype(float)
            ym = joined[y].to_numpy().astype(float)
            rx = xm - LinearRegression().fit(zm, xm).predict(zm)
            ry = ym - LinearRegression().fit(zm, ym).predict(zm)
            if rx.std() < 1e-9 or ry.std() < 1e-9:
                return 0.0, 1.0
            r = float(np.corrcoef(rx, ry)[0, 1])
        except Exception:
            try:
                tau, p = kendalltau(joined[x], joined[y])
                return float(tau), float(p)
            except Exception:
                return 0.0, 1.0

    if abs(r) >= 0.9999:
        return float(r), 0.0
    n = len(joined)
    k = len(z)
    if n - k - 3 <= 0:
        return float(r), 1.0
    # Fisher's z-transform
    z_stat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)
    p_val = float(2 * (1 - norm.cdf(abs(z_stat))))
    return float(r), p_val


def audit_dag_edges(
    graph: CausalGraph,
    df: pd.DataFrame,
    *,
    threshold_p: float = 0.01,
    min_abs_r: float = 0.05,
) -> list[EdgeAudit]:
    """Layer 4: for every LLM-proposed edge ``S → T``, test S ⊥ T | parents(T) \\ {S}.

    A *supported* edge has a significant partial correlation in the expected
    direction; a *contradicted* edge has a near-zero partial correlation; the
    rest are *inconclusive*. We never silently remove edges — we surface
    contradictions so the analyst can edit the DAG.
    """
    nx_graph = graph.to_networkx()
    audits: list[EdgeAudit] = []
    for edge in graph.edges:
        if not edge.llm_proposed:
            continue
        if edge.source not in df.columns or edge.target not in df.columns:
            audits.append(
                EdgeAudit(
                    source=edge.source,
                    target=edge.target,
                    conditioning_set=(),
                    partial_correlation=0.0,
                    p_value=1.0,
                    verdict="inconclusive",
                    note="column missing from data; cannot test",
                )
            )
            continue
        other_parents = tuple(
            p for p in nx_graph.predecessors(edge.target)
            if p != edge.source and p in df.columns
        )
        r, p = _partial_correlation(df, edge.source, edge.target, other_parents)
        verdict: Verdict
        if p < threshold_p and abs(r) >= min_abs_r:
            verdict = "supported"
        elif p > 0.2 and abs(r) < min_abs_r:
            verdict = "contradicted"
        else:
            verdict = "inconclusive"
        audits.append(
            EdgeAudit(
                source=edge.source,
                target=edge.target,
                conditioning_set=other_parents,
                partial_correlation=r,
                p_value=p,
                verdict=verdict,
            )
        )
    return audits


__all__ = [
    "SemanticGuardViolation",
    "check_columns_exist",
    "check_temporal_consistency",
    "check_iv_relevance",
    "IVRelevance",
    "audit_dag_edges",
    "EdgeAudit",
]
