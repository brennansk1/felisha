"""Step 8 — Interpret the result (PDD §10.8).

Translates the numerical estimate plus sensitivity verdict back into the
language of the original research question. When an LLM client is provided,
asks the reasoning model for a structured narrative; otherwise produces a
defensible template-based summary that names the strategy, magnitude, and
robustness honestly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.estimand import CausalEstimand
from causalrag.core.result import EstimationResult
from causalrag.llm.honesty import with_honesty
from causalrag.llm.ollama_client import OllamaClient
from causalrag.roadmap.q5_identify import IdentificationResult
from causalrag.sensitivity.verdict import SensitivityVerdict


class Interpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str = Field(..., description="One-sentence plain-English conclusion")
    magnitude: str = Field(..., description="Interpretation of effect size in domain units")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions the analyst signed")
    robustness: str = Field(..., description="What sensitivity says about the conclusion")
    caveats: list[str] = Field(default_factory=list)


_SYSTEM_PROMPT = (
    "You are a senior methodologist writing the Step 8 interpretation of a "
    "Petersen-van der Laan Causal Roadmap analysis. Given a causal estimand, "
    "the identification strategy used, the point estimate + CI + p-value, "
    "and the sensitivity verdict, produce a structured Interpretation in "
    "the analyst-facing voice.\n\n"
    "Rules:\n"
    "- Do not say 'causes' if the verdict color is red or yellow.\n"
    "- Quantify magnitude in the outcome's units, not Cohen's d.\n"
    "- List the assumptions explicitly (positivity, no-unmeasured-"
    "  confounding, consistency, SUTVA) when relevant.\n"
    "- Be explicit about what sensitivity rules out and what it doesn't.\n"
    "Return ONLY a JSON object conforming to the provided schema."
)


def _template_interpretation(
    estimand: CausalEstimand,
    identification: IdentificationResult,
    estimate: EstimationResult,
    verdict: SensitivityVerdict | None,
) -> Interpretation:
    sign = "+" if estimate.point_estimate >= 0 else "−"
    headline = (
        f"{estimand.treatment} produces a {sign}{abs(estimate.point_estimate):.4f} unit shift "
        f"in {estimand.outcome} ({estimate.estimand_class}, "
        f"strategy={identification.strategy}, n={estimate.n_used:,})."
    )
    ci_part = (
        f"95% CI [{estimate.ci_low:+.4f}, {estimate.ci_high:+.4f}]."
        if estimate.ci_low is not None and estimate.ci_high is not None
        else ""
    )
    p_part = f" p-value = {estimate.p_value:.3g}." if estimate.p_value is not None else ""
    magnitude = (
        f"Point estimate {estimate.point_estimate:+.4f} in {estimand.outcome} units. {ci_part}{p_part}"
    )
    assumptions = [
        "Positivity: every covariate stratum has both treated and untreated units.",
        "Consistency: T=t implies the observed Y is Y(t).",
        "Conditional ignorability under the chosen adjustment set.",
        "SUTVA: no interference between units.",
    ]
    if identification.strategy == "iv":
        assumptions += [
            "IV relevance: Z is associated with T (validated empirically in Layer 3).",
            "IV exclusion: Z affects Y only through T (untestable; analyst-asserted).",
        ]
    if verdict is None:
        robustness = "Sensitivity has not been run."
    elif verdict.color == "green":
        robustness = (
            f"Robust to unmeasured confounding ({verdict.rationale}). "
            "A large unobserved confounder would be required to overturn the conclusion."
        )
    elif verdict.color == "yellow":
        robustness = (
            f"Moderately robust ({verdict.rationale}). "
            "Treat the magnitude with caution; the optimistic CI bound is fragile."
        )
    else:
        robustness = (
            f"FRAGILE ({verdict.rationale}). "
            "A modest unmeasured confounder could nullify this effect."
        )
    caveats: list[str] = []
    if estimate.diagnostics.get("overlap"):
        pos = estimate.diagnostics["overlap"].get("positivity", {})
        if pos.get("verdict") != "green":
            caveats.append(f"Positivity {pos.get('verdict', '?')}: {pos.get('note', '')}.")
    if estimate.refutations:
        n_passed = estimate.refutations.get("n_passed", 0)
        if n_passed < 3:
            caveats.append(f"Refutations: only {n_passed}/3 passed.")
    return Interpretation(
        headline=headline,
        magnitude=magnitude,
        assumptions=assumptions,
        robustness=robustness,
        caveats=caveats,
    )


def interpret(
    *,
    estimand: CausalEstimand,
    identification: IdentificationResult,
    estimate: EstimationResult,
    verdict: SensitivityVerdict | None = None,
    client: OllamaClient | None = None,
) -> Interpretation:
    """Produce the Step 8 interpretation. LLM-driven when client is provided;
    template-driven otherwise."""
    if client is None:
        return _template_interpretation(estimand, identification, estimate, verdict)
    prompt = _build_prompt(estimand, identification, estimate, verdict)
    response = client.parse(
        prompt=prompt,
        schema=Interpretation,
        system=with_honesty(_SYSTEM_PROMPT),
        json_schema=Interpretation.model_json_schema(),
    )
    interp = response.parsed
    assert isinstance(interp, Interpretation)
    return interp


def _build_prompt(
    estimand: CausalEstimand,
    identification: IdentificationResult,
    estimate: EstimationResult,
    verdict: SensitivityVerdict | None,
) -> str:
    lines = [
        "## Estimand",
        f"  class: {estimand.klass.value}",
        f"  treatment: {estimand.treatment}",
        f"  outcome: {estimand.outcome}",
        f"  formal: {estimand.formal_expression}",
        "",
        "## Identification",
        f"  strategy: {identification.strategy}",
        f"  identifiable: {identification.identifiable}",
        f"  adjustment set: {list(identification.adjustment_set)}",
        "",
        "## Estimate",
        f"  estimator: {estimate.estimator_id}",
        f"  point: {estimate.point_estimate}",
        f"  ci: [{estimate.ci_low}, {estimate.ci_high}]",
        f"  p-value: {estimate.p_value}",
        f"  n_used: {estimate.n_used}",
    ]
    if verdict is not None:
        lines += [
            "",
            "## Sensitivity",
            f"  color: {verdict.color}",
            f"  rationale: {verdict.rationale}",
        ]
    lines.append("\nReturn ONLY a JSON Interpretation per the schema.")
    return "\n".join(lines)


__all__ = ["Interpretation", "interpret"]
