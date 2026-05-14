"""Master-mode hypothesis generation — propose K diverse experiments.

When the analyst hands the pipeline a dataset with no treatment/outcome
specified, the master-mode generator behaves like a senior causal-
inference statistician:

1. Reads the discovery report + domain expert brief.
2. Identifies the variables most likely to play each role (treatment,
   outcome, mediator, instrument, effect modifier).
3. Proposes K diverse hypotheses spanning estimand types (ATE, CATE,
   NDE/NIE, LATE, RMST contrast, MTP/dose-response, MTP/mixture).
4. Ranks them by **impact × identifiability × power**.
5. Returns a HypothesisQueue ready for the per-hypothesis Roadmap walk.

When an LLM client is provided we use the reasoning model to propose
hypotheses (with the full method catalog injected so it knows what's
possible). Otherwise we fall back to a deterministic generator that
covers the main estimand types using the discovery-tagged roles.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.protocol import Hypothesis, StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.estimators.catalog import catalog_markdown
from causalrag.hypothesize.automated import (
    HypothesisProposal,
    proposals_to_hypotheses,
)
from causalrag.llm.honesty import with_honesty
from causalrag.llm.ollama_client import OllamaClient


class MasterHypothesis(BaseModel):
    """One hypothesis proposed by the master generator."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., description="1 is the highest-impact hypothesis")
    research_question: str = Field(
        ..., description="One-sentence plain English"
    )
    treatment: str
    outcome: str
    modifiers: list[str] = Field(default_factory=list)
    mediator: str | None = None
    instrument: str | None = None
    estimand_class: str = Field(..., description="ATE / CATE / NDE / NIE / LATE / RMST_CONTRAST / MODIFIED_TREATMENT_POLICY")
    counterfactual: bool = False
    recommended_method: str | None = Field(
        default=None, description="Estimator id from the catalog (e.g. rbridge.lmtp.shift)"
    )
    impact_rationale: str
    identifiability_rationale: str
    power_rationale: str


class MasterQueue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypotheses: list[MasterHypothesis]


_MASTER_SYSTEM_PROMPT = (
    "You are a senior causal-inference statistician with 20 years of "
    "applied experience. The analyst has handed you a dataset with no "
    "research question and asked you to propose the K most-impactful "
    "experiments that the data can support.\n\n"
    "You will be given the discovery report (per-column profile + LLM "
    "investigator role assignments + domain expert brief) and the full "
    "method catalog. Your job:\n\n"
    "1. Identify the most plausible (treatment, outcome) pairs the data "
    "   can support — not just one. Consider every column that could "
    "   serve as a treatment.\n"
    "2. Propose K diverse hypotheses spanning multiple estimand types:\n"
    "   - ATE for the headline causal effect\n"
    "   - CATE when an effect modifier is present\n"
    "   - NDE/NIE when a mediator is plausible\n"
    "   - RMST contrast when an outcome is right-censored survival\n"
    "   - MTP/dose-response when treatment is continuous\n"
    "   - Mixture exposure when multiple treatments are jointly relevant\n"
    "   - LATE when an instrument is named\n"
    "3. For each hypothesis recommend the best estimator id from the "
    "   catalog (e.g. 'rbridge.grf.causal_survival_forest' for censored "
    "   CATE, 'rbridge.lmtp.shift' for dose-response).\n"
    "4. Rank by impact × identifiability × power (in that order of "
    "   priority).\n\n"
    "RULES:\n"
    "- Diversity matters: don't propose 5 ATEs on the same outcome.\n"
    "- Don't propose hypotheses the data can't identify (e.g., no IV → "
    "  no LATE; no mediator → no NDE/NIE).\n"
    "- Every column you reference must appear in the investigator report.\n"
    "- Be specific about WHY each hypothesis matters (impact_rationale), "
    "  WHY it is identifiable (identifiability_rationale), and WHY it "
    "  has power on this dataset (power_rationale).\n\n"
    "METHOD CATALOG (the estimators you can recommend):\n"
    "{CATALOG_TABLE}\n\n"
    "Return ONLY a JSON object conforming to the schema. Output exactly "
    "K hypotheses."
)


def run_master_hypothesize(
    *,
    protocol: StudyProtocol,
    df: pd.DataFrame | None,
    client: OllamaClient,
    k: int = 5,
) -> tuple[list[Hypothesis], list[MasterHypothesis]]:
    """Use the reasoning LLM to propose K diverse hypotheses.

    Returns a tuple of (Hypothesis objects ready for the protocol queue,
    MasterHypothesis objects with full rationale text for the report).
    """
    if protocol.discovery is None:
        raise ValueError("Master hypothesize requires a completed discovery phase.")

    prompt = _build_prompt(protocol, k=k)
    system = _MASTER_SYSTEM_PROMPT.replace("{CATALOG_TABLE}", catalog_markdown())
    response = client.parse(
        prompt=prompt,
        schema=MasterQueue,
        system=with_honesty(system),
        json_schema=MasterQueue.model_json_schema(),
    )
    queue = response.parsed
    assert isinstance(queue, MasterQueue)

    valid_columns = {v.name for v in protocol.discovery.columns}
    hypotheses: list[Hypothesis] = []
    master_records: list[MasterHypothesis] = []
    for mh in queue.hypotheses:
        if mh.treatment not in valid_columns or mh.outcome not in valid_columns:
            continue
        if mh.mediator and mh.mediator not in valid_columns:
            mh.mediator = None
        if mh.instrument and mh.instrument not in valid_columns:
            mh.instrument = None
        mh.modifiers = [m for m in mh.modifiers if m in valid_columns]

        try:
            klass = EstimandClass(mh.estimand_class.upper())
        except ValueError:
            klass = EstimandClass.ATE
        est = CausalEstimand.model_validate(
            {
                "class": klass,
                "treatment": mh.treatment,
                "outcome": mh.outcome,
                "modifiers": tuple(mh.modifiers),
                "mediator": mh.mediator,
                "instrument": mh.instrument,
                "formal_expression": _formal_for(klass),
            }
        )
        rationale = (
            f"[{mh.rank}] {mh.research_question}\n"
            f"  Impact: {mh.impact_rationale}\n"
            f"  Identifiability: {mh.identifiability_rationale}\n"
            f"  Power: {mh.power_rationale}\n"
            f"  Recommended estimator: {mh.recommended_method or 'auto-select'}"
        )
        hypotheses.append(
            Hypothesis(
                id=f"master-{mh.rank:02d}",
                treatment=mh.treatment,
                outcome=mh.outcome,
                modifiers=tuple(mh.modifiers),
                counterfactual=mh.counterfactual,
                rationale=rationale,
                impact_score=max(0.0, 1.0 - (mh.rank - 1) * 0.10),  # rank-1 → 1.0
                estimand=est,
            )
        )
        master_records.append(mh)
    return hypotheses, master_records


def _formal_for(klass: EstimandClass) -> str:
    mapping = {
        EstimandClass.ATE: "E[Y(1) - Y(0)]",
        EstimandClass.ATT: "E[Y(1) - Y(0) | T=1]",
        EstimandClass.ATC: "E[Y(1) - Y(0) | T=0]",
        EstimandClass.CATE: "E[Y(1) - Y(0) | X=x]",
        EstimandClass.LATE: "Local ATE among compliers (Wald)",
        EstimandClass.RMST_CONTRAST: "E[min(T_surv, τ)|A=1] - E[min(T_surv, τ)|A=0]",
        EstimandClass.NDE: "Natural Direct Effect",
        EstimandClass.NIE: "Natural Indirect Effect",
        EstimandClass.MODIFIED_TREATMENT_POLICY: "E[Y(δ(A))]",
    }
    return mapping.get(klass, f"{klass.value} estimand")


def _build_prompt(protocol: StudyProtocol, k: int) -> str:
    if protocol.discovery is None:
        return ""
    lines = [
        f"## Dataset: {protocol.dataset.source if protocol.dataset else 'unknown'}",
        f"## n_rows: {protocol.dataset.n_rows if protocol.dataset else '?'}, "
        f"n_cols: {protocol.dataset.n_cols if protocol.dataset else '?'}",
        "",
        "## Variables (from the Stage-1c LLM investigator)",
    ]
    for v in protocol.discovery.columns:
        lines.append(
            f"  - **{v.name}** ({v.dtype}): role={v.role.value}, "
            f"temporal={v.measured_at or 'unknown'}, "
            f"description={v.semantic_description or '—'}"
        )
    lines.append("")
    lines.append("## Data flags emitted")
    for f in sorted(protocol.flags, key=lambda x: x.value):
        lines.append(f"  - {f.value}")
    if protocol.discovery.domain_brief:
        lines.append("")
        lines.append("## Domain expert brief")
        lines.append(protocol.discovery.domain_brief[:1500])
    if protocol.research_question:
        lines.append("")
        lines.append(f"## Optional user prompt: {protocol.research_question}")
    lines.append("")
    lines.append(
        f"## Task\nPropose exactly {k} diverse, identifiable, well-powered "
        "hypotheses ranked by impact × identifiability × power. Span estimand "
        "types where the data supports them. Recommend a specific estimator "
        "id from the catalog for each hypothesis."
    )
    return "\n".join(lines)


__all__ = ["MasterHypothesis", "MasterQueue", "run_master_hypothesize"]
