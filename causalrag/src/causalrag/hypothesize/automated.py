"""Automated hypothesis generation (PDD §9.2).

Uses the reasoning ("hypothesize") model to propose a ranked queue of K
falsifiable hypotheses given the feasibility report + domain expert brief.
Each proposal includes an impact score (weighting coverage × power × novelty
× no-post-treatment × counterfactual-share per §9.3).

Falls back to a deterministic generator when no LLM client is supplied:
emit one ATE per admissible pair plus one CATE-on-strongest-modifier
hypothesis per outcome.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.protocol import FeasibilityReport, Hypothesis, StudyProtocol
from causalrag.discovery.expert import DomainExpertBrief
from causalrag.llm.honesty import with_honesty
from causalrag.llm.ollama_client import OllamaClient


class HypothesisProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treatment: str
    outcome: str
    modifiers: list[str] = Field(default_factory=list)
    counterfactual: bool = False
    estimand_class: str = "ATE"
    rationale: str
    impact_score: float = Field(..., ge=0.0, le=1.0)


class HypothesisQueue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypotheses: list[HypothesisProposal]


_SYSTEM_PROMPT = (
    "You are a senior causal-inference methodologist proposing a ranked set "
    "of falsifiable hypotheses for a study. You will be given a feasibility "
    "report (which (treatment, outcome) pairs have adequate power) and the "
    "Stage-1e domain expert brief.\n\n"
    "Propose at most 6 hypotheses ordered by impact. Impact is a weighted "
    "combination of:\n"
    "  - clinical / economic importance of the effect\n"
    "  - statistical power on the data at hand\n"
    "  - novelty (would this surprise a domain expert?)\n"
    "  - identifiability strength (backdoor with observed adjustment > IV > "
    "    frontdoor > mediation)\n"
    "Avoid post-treatment variables as modifiers. Surface CATE proposals only "
    "when the brief lists effect modifiers explicitly.\n\n"
    "Return ONLY a JSON object with a `hypotheses` array. Every column you "
    "reference must appear in the feasibility report's admissible_pairs or "
    "the expert brief's effect_modifiers."
)


def _impact_score(
    pair: tuple[str, str],
    feasibility: FeasibilityReport | None,
    in_brief: bool,
) -> float:
    score = 0.5
    if feasibility and pair in feasibility.admissible_pairs:
        score += 0.3
    if in_brief:
        score += 0.2
    return min(score, 1.0)


def deterministic_proposals(
    protocol: StudyProtocol,
    brief: DomainExpertBrief | None,
    counterfactual_ratio: float = 0.30,
    include_underpowered: bool = False,
) -> list[HypothesisProposal]:
    """LLM-free fallback. Produces one ATE proposal per admissible pair plus
    one CATE proposal per outcome using the brief's top effect modifier.

    When admissible is empty (feasibility flagged everything underpowered)
    we fall back to the (treatment, outcome) candidates from the discovery
    report so the pipeline doesn't dead-end the analyst. The power caveat
    propagates through to the report.
    """
    feasibility = protocol.feasibility
    proposals: list[HypothesisProposal] = []
    admissible = list(feasibility.admissible_pairs) if feasibility else []
    if not admissible and protocol.discovery is not None:
        from causalrag.core.roles import VariableRole

        treatments = [
            v.name for v in protocol.discovery.columns if v.role is VariableRole.TREATMENT
        ]
        outcomes = [
            v.name for v in protocol.discovery.columns if v.role is VariableRole.OUTCOME
        ]
        admissible = [(t, y) for t in treatments for y in outcomes]
    brief_pairs = set()
    if brief:
        for t in brief.treatments:
            for y in brief.outcomes:
                brief_pairs.add((t.column, y.column))
    for t, y in admissible:
        proposals.append(
            HypothesisProposal(
                treatment=t,
                outcome=y,
                estimand_class="ATE",
                rationale=f"Headline ATE of {t} on {y} (admissible by feasibility filter).",
                impact_score=_impact_score((t, y), feasibility, (t, y) in brief_pairs),
            )
        )
    if brief and brief.effect_modifiers and admissible:
        t0, y0 = admissible[0]
        mod = brief.effect_modifiers[0]
        proposals.append(
            HypothesisProposal(
                treatment=t0,
                outcome=y0,
                modifiers=[mod],
                estimand_class="CATE",
                rationale=f"CATE of {t0} on {y0} stratified by {mod}.",
                impact_score=_impact_score((t0, y0), feasibility, True) * 0.9,
            )
        )
    # Counterfactual share — promote a fraction to counterfactual targets
    if counterfactual_ratio > 0 and proposals:
        n_cf = max(1, int(round(counterfactual_ratio * len(proposals))))
        for p in proposals[:n_cf]:
            p.counterfactual = True
    return proposals


def run_automated(
    *,
    protocol: StudyProtocol,
    brief: DomainExpertBrief | None,
    client: OllamaClient | None = None,
    counterfactual_ratio: float = 0.30,
) -> list[HypothesisProposal]:
    """Top-level entry. Uses LLM when ``client`` is provided, else the
    deterministic fallback."""
    if client is None or brief is None:
        return deterministic_proposals(protocol, brief, counterfactual_ratio)

    prompt = _build_prompt(protocol, brief, counterfactual_ratio)
    response = client.parse(
        prompt=prompt,
        schema=HypothesisQueue,
        system=with_honesty(_SYSTEM_PROMPT),
        json_schema=HypothesisQueue.model_json_schema(),
    )
    queue = response.parsed
    assert isinstance(queue, HypothesisQueue)
    # Light semantic validation
    valid_columns = {v.name for v in (protocol.discovery.columns if protocol.discovery else ())}
    cleaned: list[HypothesisProposal] = []
    for p in queue.hypotheses:
        if p.treatment not in valid_columns or p.outcome not in valid_columns:
            continue
        p.modifiers = [m for m in p.modifiers if m in valid_columns]
        cleaned.append(p)
    return cleaned or deterministic_proposals(protocol, brief, counterfactual_ratio)


def _build_prompt(
    protocol: StudyProtocol,
    brief: DomainExpertBrief,
    counterfactual_ratio: float,
) -> str:
    parts = ["## Domain summary", brief.domain_summary, ""]
    if protocol.feasibility:
        parts.append("## Admissible pairs (from feasibility filter)")
        for t, y in protocol.feasibility.admissible_pairs:
            parts.append(f"  - {t} → {y}")
    parts.append("\n## Domain warnings")
    for w in brief.identification_warnings:
        parts.append(f"  - {w}")
    parts.append("\n## Effect modifiers (from brief)")
    for m in brief.effect_modifiers:
        parts.append(f"  - {m}")
    parts.append("\n## Unmeasured confounders to keep in mind")
    for u in brief.unmeasured_confounders:
        parts.append(f"  - {u.name}: {u.reason}")
    parts.append(
        f"\n## Task\nPropose at most 6 hypotheses. Aim for "
        f"~{counterfactual_ratio:.0%} of them to be counterfactual / stochastic-"
        f"intervention queries. Return JSON conforming to the provided schema."
    )
    return "\n".join(parts)


def proposals_to_hypotheses(
    proposals: list[HypothesisProposal],
) -> list[Hypothesis]:
    out: list[Hypothesis] = []
    for i, p in enumerate(proposals):
        try:
            klass = EstimandClass(p.estimand_class.upper())
        except ValueError:
            klass = EstimandClass.ATE
        est = CausalEstimand.model_validate(
            {
                "class": klass,
                "treatment": p.treatment,
                "outcome": p.outcome,
                "modifiers": tuple(p.modifiers),
                "formal_expression": "E[Y(1) - Y(0)]" if klass == EstimandClass.ATE else f"{klass.value} expression",
            }
        )
        out.append(
            Hypothesis(
                id=f"auto-{i + 1:02d}",
                treatment=p.treatment,
                outcome=p.outcome,
                modifiers=tuple(p.modifiers),
                counterfactual=p.counterfactual,
                rationale=p.rationale,
                impact_score=p.impact_score,
                estimand=est,
            )
        )
    return out


__all__ = [
    "HypothesisProposal",
    "HypothesisQueue",
    "deterministic_proposals",
    "proposals_to_hypotheses",
    "run_automated",
]
