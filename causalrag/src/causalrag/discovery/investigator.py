"""Stage 1c — LLM investigator role (PDD §7.3).

Given the deterministic ``DatasetProfile`` plus a 10-row sample, asks the LLM
to produce per-column semantic metadata: ``domain_meaning``, ``domain_tag``,
``value_interpretation``, ``temporal_position``, ``watch_for``. Responses are
strictly validated by Pydantic — failures retry with the validation error fed
back (PDD §16.6 Layer 2).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.roles import VariableRole, VariableSpec
from causalrag.data.profiler import ColumnProfile, DatasetProfile
from causalrag.llm.honesty import with_honesty
from causalrag.llm.ollama_client import LLMResponse, OllamaClient

TemporalPosition = Literal[
    "baseline",
    "pre_treatment",
    "treatment_era",
    "post_treatment",
    "outcome",
    "unknown",
]

DomainTag = Literal[
    "clinical",
    "financial",
    "marketing",
    "education",
    "manufacturing",
    "social_science",
    "web_analytics",
    "environmental",
    "other",
]

_WATCH_TAGS = frozenset(
    {
        "high_missing",
        "low_variance",
        "suspected_identifier",
        "suspected_target_leakage",
        "derived_from_outcome",
        "post_treatment_proxy",
        "contains_pii",
    }
)


class InvestigatorColumn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    domain_meaning: str = Field(..., min_length=1, max_length=500)
    domain_tag: DomainTag = "other"
    value_interpretation: str | None = None
    temporal_position: TemporalPosition = "unknown"
    watch_for: list[str] = Field(default_factory=list)
    proposed_role: VariableRole | None = None


class InvestigatorReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain_tag: DomainTag = "other"
    columns: list[InvestigatorColumn]

    def column(self, name: str) -> InvestigatorColumn | None:
        for c in self.columns:
            if c.column == name:
                return c
        return None


_SYSTEM_PROMPT = (
    "You are a senior data scientist preparing input for a Petersen-van der Laan "
    "Causal Roadmap analysis. You are in the *investigator* role — column-by-column "
    "semantic understanding only. The downstream pipeline will use your output to "
    "propose DAGs and select estimators, so precision matters more than coverage.\n\n"
    "Domain tag must be chosen from this exact set: clinical, financial, marketing, "
    "education, manufacturing, social_science, web_analytics, environmental, other. "
    "Map carefully — labor-economics evaluation data is social_science, NOT web_analytics. "
    "Health-claims data is clinical. Stock/credit/transaction data is financial. "
    "Use 'other' if genuinely unclear; never guess between two plausible tags.\n\n"
    "Your job, for every column:\n"
    "1. Explain in 1-3 sentences what the column actually measures (units, "
    "   collection mechanism, when in the data-generating process it is realized).\n"
    "2. Tag its temporal position relative to a hypothetical treatment.\n"
    "3. Tag warnings about its analytic suitability.\n"
    "4. Propose a tentative variable role — but only when the temporal position "
    "   and domain context support it. Do not propose 'treatment' or 'outcome' "
    "   on a hunch; downstream stages will refine.\n\n"
    "DO NOT:\n"
    "- Adjust your role proposals for columns whose temporal position is "
    "  'post_treatment' or 'outcome' as confounders — those would induce collider "
    "  or M-bias when adjusted on. Tag them with watch_for='post_treatment_proxy' "
    "  or 'derived_from_outcome' instead.\n"
    "- Propose instruments without explicit reason; an instrument must plausibly "
    "  affect treatment without affecting the outcome except through treatment.\n"
    "- Include identifiers (patient_id, uuid, row_no) as analytic variables; "
    "  tag them with watch_for='suspected_identifier'.\n"
    "- Hallucinate column meanings when the name is ambiguous; prefer "
    "  temporal_position='unknown' and a short 'I cannot reliably interpret this' "
    "  domain_meaning.\n\n"
    "Return ONLY a JSON object that conforms to the provided schema. No prose "
    "outside the JSON, no markdown fences. watch_for tags must be drawn from "
    "this exact set: " + ", ".join(sorted(_WATCH_TAGS)) + "."
)


def _profile_summary_for_prompt(profile: DatasetProfile, max_top: int = 5) -> str:
    lines = [f"# Dataset profile: {profile.n_rows} rows × {profile.n_cols} columns"]
    for c in profile.columns:
        bits = [
            f"- {c.name}: dtype={c.dtype}, logical={c.logical_dtype}, "
            f"missing={c.missing_rate:.0%}, unique={c.cardinality}"
        ]
        if c.logical_dtype in {"continuous", "count", "ordinal"} and c.mean is not None:
            bits.append(
                f"  range=[{c.min}, {c.max}], median={c.p50}, mean≈{c.mean:.3f}"
            )
        if c.top_values:
            top = ", ".join(f"{k}={v}" for k, v in c.top_values[:max_top])
            bits.append(f"  top: {top}")
        if c.suspected_identifier:
            bits.append("  HINT: suspected identifier column")
        if c.suspected_time_column:
            bits.append("  HINT: suspected time-to-event column")
        if c.suspected_event_indicator:
            bits.append("  HINT: suspected event indicator (binary)")
        lines.append("\n".join(bits))
    if profile.censoring_pairs:
        lines.append("# Suspected censoring pairs (time, event):")
        for t, e in profile.censoring_pairs:
            lines.append(f"- ({t}, {e})")
    if profile.string_formats:
        lines.append("# Inferred string formats (use to set watch_for):")
        for col, fmt in profile.string_formats.items():
            lines.append(f"- {col}: {fmt}")
    return "\n".join(lines)


def _sample_for_prompt(df: pd.DataFrame, n: int = 10) -> str:
    if df.empty:
        return "(empty dataset)"
    if len(df) > n:
        sample = df.sample(n=n, random_state=0)
    else:
        sample = df
    return sample.to_csv(index=False)


def _build_prompt(
    profile: DatasetProfile, df: pd.DataFrame, research_question: str | None
) -> str:
    parts = [
        "## Statistical profile (deterministic — authoritative for dtypes)",
        _profile_summary_for_prompt(profile),
        "",
        "## 10-row sample (CSV) — use only to disambiguate value labels",
        _sample_for_prompt(df),
    ]
    if research_question:
        parts.extend(
            [
                "",
                "## User research question (use to bias temporal_position and "
                "proposed_role inferences, NOT to invent columns)",
                research_question,
            ]
        )
    parts.append(
        "\n## Task\nReturn a JSON object with keys `domain_tag` (one tag for the "
        "whole dataset) and `columns` (a list of per-column objects, one entry per "
        "profile column, in the same order).\n\n"
        "Reasoning checklist before you write each row:\n"
        "  a) What does the column name + sample values suggest the field measures?\n"
        "  b) When in the data-generating timeline is it realized?\n"
        "  c) If a research question is provided, does the column plausibly serve "
        "as treatment, outcome, confounder, mediator, or effect modifier — or none?\n"
        "  d) Are there hazards (post-treatment proxy, identifier, leakage, PII)?\n"
        "Encode your conclusions in the JSON; do not include the reasoning itself "
        "in the output."
    )
    return "\n".join(parts)


def run_investigator(
    *,
    df: pd.DataFrame,
    profile: DatasetProfile,
    client: OllamaClient,
    research_question: str | None = None,
) -> tuple[InvestigatorReport, LLMResponse]:
    prompt = _build_prompt(profile, df, research_question)
    response = client.parse(
        prompt=prompt,
        schema=InvestigatorReport,
        system=with_honesty(_SYSTEM_PROMPT),
        json_schema=InvestigatorReport.model_json_schema(),
    )
    report = response.parsed
    assert isinstance(report, InvestigatorReport)
    _validate_columns_match(report, profile)
    return report, response


def _validate_columns_match(report: InvestigatorReport, profile: DatasetProfile) -> None:
    """Semantic Layer 3 check (PDD §16.6): every column the LLM names must
    exist in the profile, and no profile column may be silently dropped."""
    profile_names = [c.name for c in profile.columns]
    report_names = [c.column for c in report.columns]
    extra = set(report_names) - set(profile_names)
    if extra:
        raise ValueError(
            f"Investigator response referenced columns not in the profile: {sorted(extra)}"
        )
    # Drop or pad — never silently. If the LLM dropped some, surface that.
    missing = set(profile_names) - set(report_names)
    if missing:
        # Add stub entries flagged as 'unknown' so downstream code never
        # encounters a missing key. Provenance: this is a Layer 2 fallback.
        for name in profile_names:
            if name in missing:
                report.columns.append(
                    InvestigatorColumn(
                        column=name,
                        domain_meaning=f"(LLM omitted column {name!r}; defaulted to unknown)",
                        temporal_position="unknown",
                        proposed_role=None,
                    )
                )


def to_variable_specs(
    profile: DatasetProfile,
    report: InvestigatorReport,
) -> tuple[VariableSpec, ...]:
    """Project the investigator report + deterministic dtypes into VariableSpec
    tuples consumable by the StudyProtocol."""
    out: list[VariableSpec] = []
    for col in profile.columns:
        info = report.column(col.name)
        role = (info.proposed_role if info else None) or VariableRole.AUXILIARY
        out.append(
            VariableSpec(
                name=col.name,
                role=role,
                dtype=col.dtype,
                nullable=col.missing_rate > 0,
                semantic_description=info.domain_meaning if info else None,
                measured_at=info.temporal_position if info else "unknown",
                analyst_override=False,
            )
        )
    return tuple(out)
