"""Phase 1 — discovery agent (PDD §7).

This package orchestrates Stages 1a (connector ingestion), 1b (deterministic
profiler), 1c (LLM investigator), and 1d (flag emission). Stage 1e (domain
expert brief) lands in Week 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import DiscoveryReport
from causalrag.core.roles import VariableSpec
from causalrag.data.connectors import Connector, from_uri
from causalrag.data.flags import emit_from_profile
from causalrag.data.profiler import DatasetProfile, profile_dataframe
from causalrag.core.graph import CausalGraph
from causalrag.discovery.expert import (
    ConfounderContradiction,
    DomainExpertBrief,
    brief_to_candidate_graphs,
    run_domain_expert,
)
from causalrag.discovery.investigator import (
    InvestigatorReport,
    run_investigator,
    to_variable_specs,
)
from causalrag.llm.guards import EdgeAudit, audit_dag_edges
from causalrag.llm.ollama_client import OllamaClient


@dataclass
class DiscoveryResult:
    """In-process artifact bundling all Phase 1 outputs.

    The serializable projection is :class:`DiscoveryReport` (which lives on the
    StudyProtocol); we keep the raw frame + statistical profile alongside for
    downstream phases in the same session.
    """

    dataframe: pd.DataFrame
    profile: DatasetProfile
    investigator: InvestigatorReport | None
    expert: DomainExpertBrief | None
    candidate_graphs: tuple[CausalGraph, ...]
    confounder_audit: tuple[ConfounderContradiction, ...]
    dag_audit: tuple[EdgeAudit, ...]
    flags: set[DataFlag]
    columns: tuple[VariableSpec, ...]
    source_describe: dict[str, Any]
    research_question: str | None

    def to_report(self) -> DiscoveryReport:
        return DiscoveryReport(
            columns=self.columns,
            flags=self.flags,
            domain_brief=self.expert.domain_summary if self.expert else None,
            candidate_graphs=self.candidate_graphs,
        )


def run_discovery(
    *,
    source: str | Path | Connector | pd.DataFrame,
    research_question: str | None = None,
    client: OllamaClient | None = None,
    expert_client: OllamaClient | None = None,
    treatment: str | None = None,
    outcome: str | None = None,
    k_dags: int = 3,
) -> DiscoveryResult:
    """Run Stages 1a → 1e end-to-end.

    ``client`` runs Stage 1c (investigator). ``expert_client`` runs Stage 1e
    (domain expert); when omitted the investigator client is reused. Either
    may be ``None``: a None ``client`` skips both LLM stages and emits a
    profile-only result.
    """
    df, source_describe = _ingest(source)
    profile = profile_dataframe(df)

    investigator_report: InvestigatorReport | None = None
    expert_brief: DomainExpertBrief | None = None
    candidate_graphs: tuple[CausalGraph, ...] = ()
    contradictions: tuple[ConfounderContradiction, ...] = ()

    if client is not None:
        investigator_report, _ = run_investigator(
            df=df, profile=profile, client=client, research_question=research_question
        )
        columns = to_variable_specs(profile, investigator_report)
        expert_target = expert_client or client
        expert_brief, contradiction_list, _ = run_domain_expert(
            df=df,
            profile=profile,
            investigator=investigator_report,
            client=expert_target,
            research_question=research_question,
            k=k_dags,
        )
        candidate_graphs = brief_to_candidate_graphs(expert_brief)
        contradictions = tuple(contradiction_list)
        dag_audit: tuple[EdgeAudit, ...] = ()
        if candidate_graphs:
            audits: list[EdgeAudit] = []
            for g in candidate_graphs:
                audits.extend(audit_dag_edges(g, df))
            dag_audit = tuple(audits)
    else:
        columns = to_variable_specs(profile, _empty_investigator(profile))
        dag_audit = ()

    # If T/Y weren't passed in explicitly, infer them from the investigator's
    # role assignments. This fires the T/Y-aware detectors
    # (BINARY_TREATMENT, IMBALANCED_TREATMENT, BOUNDED_OUTCOME, etc.)
    # automatically in master mode where the caller doesn't know which
    # column is which yet.
    if (treatment is None or outcome is None) and investigator_report is not None:
        from causalrag.core.roles import VariableRole

        for var in columns:
            if treatment is None and var.role is VariableRole.TREATMENT:
                treatment = var.name
            elif outcome is None and var.role is VariableRole.OUTCOME:
                outcome = var.name

    flags = emit_from_profile(
        profile, treatment=treatment, outcome=outcome, df=df
    )

    # Missingness diagnostic — runs deterministically once T/Y are known.
    # The report lands on DiscoveryResult so downstream callers can see
    # whether to suggest MICE / IPCW / refuse, and HEAVY_MISSINGNESS is
    # promoted to a flag if the diagnostic says so.
    try:
        from causalrag.data.missingness import diagnose_missingness

        missingness_report = diagnose_missingness(df, treatment=treatment, outcome=outcome)
        # If max per-column missing >= 20%, surface HEAVY_MISSINGNESS.
        if missingness_report.per_column_rate and max(missingness_report.per_column_rate.values()) >= 0.20:
            flags.add(DataFlag.HEAVY_MISSINGNESS)
    except Exception:
        missingness_report = None

    # Merge in flags that the LLM brief unambiguously implies
    # (MEDIATOR_PROPOSED, EFFECT_MODIFICATION_OF_INTEREST,
    # INSTRUMENTAL_CANDIDATE_PRESENT).
    if expert_brief is not None:
        from causalrag.discovery.expert import flags_from_brief

        flags |= flags_from_brief(expert_brief, investigator=investigator_report)

    return DiscoveryResult(
        dataframe=df,
        profile=profile,
        investigator=investigator_report,
        expert=expert_brief,
        candidate_graphs=candidate_graphs,
        confounder_audit=contradictions,
        dag_audit=dag_audit,
        flags=flags,
        columns=columns,
        source_describe=source_describe,
        research_question=research_question,
    )


def _ingest(
    source: str | Path | Connector | pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(source, pd.DataFrame):
        return source, {"source": "<dataframe>", "n_rows": len(source)}
    if isinstance(source, Connector):
        connector = source
    else:
        connector = from_uri(source)
    table = connector.to_arrow()
    return table.to_pandas(), connector.describe()


def _empty_investigator(profile: DatasetProfile) -> InvestigatorReport:
    from causalrag.discovery.investigator import InvestigatorColumn

    return InvestigatorReport(
        domain_tag="other",
        columns=[
            InvestigatorColumn(
                column=c.name,
                domain_meaning=f"(no LLM enrichment; logical dtype {c.logical_dtype})",
                temporal_position="unknown",
            )
            for c in profile.columns
        ],
    )


__all__ = ["DiscoveryResult", "run_discovery"]
