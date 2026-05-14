"""Anomaly / sanity-check audit — LLM-assisted detector for subtle
wrong-shape estimator outputs.

The current pipeline can silently emit implausibly large effects, CIs so
wide they're uninformative, p-values inconsistent with their CIs,
sign-flips against the naive estimate, saturated propensities, or
near-zero effective sample sizes. Each pattern is a different "wrong
shape", which makes pattern-matching with an LLM a useful complement to
the deterministic checks below.

Design:

- Deterministic pre-screen runs FIRST. Its flags are unconditional.
- If an :class:`OllamaClient` is supplied, the LLM is asked for additional
  qualitative flags (magnitude implausibility, sign-flips vs naive,
  refutation divergence, overfit risk). Its flags are merged with the
  pre-screen flags; deterministic flags are never dropped.
- Recommendation is deterministic when severe pre-screen flags fire;
  otherwise the LLM's recommendation wins.
- Failure-safe: if the LLM call raises, fall back to deterministic-only.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.protocol import RoadmapWalk
from causalrag.core.result import EstimationResult
from causalrag.llm.ollama_client import OllamaClient


class AnomalyAuditFlag(str, Enum):
    IMPLAUSIBLE_MAGNITUDE = "implausible_magnitude"
    CI_TOO_WIDE = "ci_too_wide"
    P_VALUE_INCONSISTENT_WITH_CI = "p_value_inconsistent_with_ci"
    SIGN_FLIP_VS_NAIVE = "sign_flip_vs_naive"
    NEAR_ZERO_N_USED = "near_zero_n_used"
    SATURATED_PROPENSITY = "saturated_propensity"
    REFUTATION_DIVERGENCE = "refutation_divergence"
    OVERFIT_RISK = "overfit_risk"


Recommendation = Literal["accept", "rerun_with_different_estimator", "disqualify"]


class AnomalyAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    flags: list[AnomalyAuditFlag] = Field(default_factory=list)
    rationale_per_flag: dict[str, str] = Field(default_factory=dict)
    recommendation: Recommendation = "accept"
    overall_note: str = ""


class _LLMAuditPayload(BaseModel):
    """Schema we ask the LLM to return. Kept permissive on extras so we can
    evolve the prompt without bumping the cassette format."""

    model_config = ConfigDict(extra="ignore")

    flags: list[AnomalyAuditFlag] = Field(default_factory=list)
    rationale_per_flag: dict[str, str] = Field(default_factory=dict)
    recommendation: Recommendation = "accept"
    overall_note: str = ""


_SYSTEM_PROMPT = (
    "You are a senior referee reviewing the output of a causal estimator. "
    "Review the numbers below for SUBTLE wrong-shape patterns. You are NOT "
    "re-doing the analysis — you are sniffing for: implausibly large effects "
    "given domain (e.g., a drug that doubles 5-year survival), sign-flips vs "
    "the naive correlation, signs of an over-fit per-row CATE forest, "
    "refutation results that diverge wildly from the main estimate, etc.\n\n"
    "Use the provided pre-screen flags as context — they are already in the "
    "audit; you do not need to repeat them, but you may. Add qualitative "
    "flags that the deterministic checks cannot catch.\n\n"
    "Return ONLY a JSON object matching the AnomalyAudit schema."
)


# --- Deterministic pre-screen ------------------------------------------------


def _prescreen(
    result: EstimationResult, naive_estimate: float | None
) -> tuple[list[AnomalyAuditFlag], dict[str, str]]:
    flags: list[AnomalyAuditFlag] = []
    rationale: dict[str, str] = {}

    # NEAR_ZERO_N_USED
    if result.n_used < 30:
        flags.append(AnomalyAuditFlag.NEAR_ZERO_N_USED)
        rationale[AnomalyAuditFlag.NEAR_ZERO_N_USED.value] = (
            f"n_used={result.n_used} is below the 30-row sanity floor; "
            "standard errors are not trustworthy."
        )

    # CI_TOO_WIDE
    if (
        result.ci_low is not None
        and result.ci_high is not None
        and result.point_estimate != 0.0
    ):
        width = abs(result.ci_high - result.ci_low)
        if width > 10.0 * abs(result.point_estimate):
            flags.append(AnomalyAuditFlag.CI_TOO_WIDE)
            rationale[AnomalyAuditFlag.CI_TOO_WIDE.value] = (
                f"CI width {width:.4g} is >10x the point estimate "
                f"{result.point_estimate:.4g}; effectively uninformative."
            )

    # P_VALUE_INCONSISTENT_WITH_CI
    if (
        result.ci_low is not None
        and result.ci_high is not None
        and result.p_value is not None
    ):
        ci_excludes_zero = (result.ci_low > 0.0) or (result.ci_high < 0.0)
        p_significant = result.p_value <= 0.05
        if ci_excludes_zero != p_significant:
            flags.append(AnomalyAuditFlag.P_VALUE_INCONSISTENT_WITH_CI)
            rationale[AnomalyAuditFlag.P_VALUE_INCONSISTENT_WITH_CI.value] = (
                f"CI [{result.ci_low:.4g}, {result.ci_high:.4g}] "
                f"{'excludes' if ci_excludes_zero else 'includes'} 0 "
                f"but p={result.p_value:.4g} "
                f"{'is' if p_significant else 'is not'} significant at 0.05."
            )

    # SATURATED_PROPENSITY
    overlap = result.diagnostics.get("overlap") if result.diagnostics else None
    if isinstance(overlap, dict):
        p_max = overlap.get("p_max")
        p_min = overlap.get("p_min")
        if (isinstance(p_max, int | float) and p_max > 0.99) or (
            isinstance(p_min, int | float) and p_min < 0.01
        ):
            flags.append(AnomalyAuditFlag.SATURATED_PROPENSITY)
            rationale[AnomalyAuditFlag.SATURATED_PROPENSITY.value] = (
                f"Propensity range p_min={p_min} / p_max={p_max} hits the "
                "[0.01, 0.99] guard; positivity is effectively violated."
            )

    # REFUTATION_DIVERGENCE
    refs = result.refutations or {}
    refs_list: list[dict[str, Any]] = []
    if isinstance(refs.get("tests"), list):
        refs_list = [r for r in refs["tests"] if isinstance(r, dict)]
    else:
        for v in refs.values():
            if isinstance(v, dict):
                refs_list.append(v)
    for ref in refs_list:
        delta = ref.get("delta_in_se_units")
        if isinstance(delta, int | float) and abs(delta) > 3.0:
            flags.append(AnomalyAuditFlag.REFUTATION_DIVERGENCE)
            rationale[AnomalyAuditFlag.REFUTATION_DIVERGENCE.value] = (
                f"Refutation {ref.get('name', '<unnamed>')} shifted the "
                f"estimate by {delta:.2f} SE — beyond the 3-SE tolerance."
            )
            break

    # SIGN_FLIP_VS_NAIVE — deterministic when naive is provided
    if naive_estimate is not None and naive_estimate != 0.0 and result.point_estimate != 0.0:
        if (naive_estimate > 0) != (result.point_estimate > 0):
            flags.append(AnomalyAuditFlag.SIGN_FLIP_VS_NAIVE)
            rationale[AnomalyAuditFlag.SIGN_FLIP_VS_NAIVE.value] = (
                f"Naive estimate {naive_estimate:+.4g} and adjusted estimate "
                f"{result.point_estimate:+.4g} have opposite signs."
            )

    return flags, rationale


# --- Recommendation logic ----------------------------------------------------


def _deterministic_recommendation(
    flags: list[AnomalyAuditFlag], n_used: int
) -> Recommendation | None:
    """Returns a recommendation if pre-screen severity demands it, else None."""
    if n_used < 10:
        return "disqualify"
    if AnomalyAuditFlag.IMPLAUSIBLE_MAGNITUDE in flags:
        return "rerun_with_different_estimator"
    if AnomalyAuditFlag.SIGN_FLIP_VS_NAIVE in flags:
        return "rerun_with_different_estimator"
    return None


# --- Prompt builder ----------------------------------------------------------


def _build_prompt(
    *,
    result: EstimationResult,
    walk: RoadmapWalk,
    treatment: str,
    outcome: str,
    naive_estimate: float | None,
    domain_brief: str | None,
    prescreen_flags: list[AnomalyAuditFlag],
    prescreen_rationale: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append("## Target")
    lines.append(f"  treatment: {treatment}")
    lines.append(f"  outcome: {outcome}")
    lines.append(f"  hypothesis_id: {walk.hypothesis_id}")
    if walk.q1_question:
        lines.append(f"  question: {walk.q1_question}")
    if domain_brief:
        lines.append("")
        lines.append("## Domain brief")
        lines.append(domain_brief.strip())
    lines.append("")
    lines.append("## Estimate")
    lines.append(f"  estimator: {result.estimator_id}")
    lines.append(f"  point: {result.point_estimate}")
    lines.append(f"  se: {result.se}")
    lines.append(f"  ci: [{result.ci_low}, {result.ci_high}]")
    lines.append(f"  p-value: {result.p_value}")
    lines.append(f"  n_used: {result.n_used}")
    if naive_estimate is not None:
        lines.append(f"  naive (unadjusted) estimate: {naive_estimate}")
    if result.diagnostics:
        lines.append("")
        lines.append("## Diagnostics")
        for k, v in result.diagnostics.items():
            lines.append(f"  {k}: {v}")
    if result.refutations:
        lines.append("")
        lines.append("## Refutations")
        for k, v in result.refutations.items():
            lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("## Deterministic pre-screen (already flagged)")
    if prescreen_flags:
        for f in prescreen_flags:
            lines.append(f"  - {f.value}: {prescreen_rationale.get(f.value, '')}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append(
        "Return ONLY a JSON object with fields {flags, rationale_per_flag, "
        "recommendation, overall_note} per the AnomalyAudit schema. Add new "
        "qualitative flags only; do not repeat pre-screen rationale verbatim."
    )
    return "\n".join(lines)


# --- Public API --------------------------------------------------------------


def audit_for_anomalies(
    *,
    result: EstimationResult,
    walk: RoadmapWalk,
    treatment: str,
    outcome: str,
    naive_estimate: float | None = None,
    domain_brief: str | None = None,
    client: OllamaClient | None = None,
) -> AnomalyAudit:
    """Run the deterministic pre-screen and optionally consult the LLM.

    Failure-safe: any exception from the LLM call collapses to the
    deterministic-only audit, never raises.
    """
    det_flags, det_rationale = _prescreen(result, naive_estimate)

    if client is None:
        rec = _deterministic_recommendation(det_flags, result.n_used) or "accept"
        return AnomalyAudit(
            flags=list(det_flags),
            rationale_per_flag=dict(det_rationale),
            recommendation=rec,
            overall_note=(
                "Deterministic pre-screen only (no LLM client supplied)."
                if not det_flags
                else f"Deterministic pre-screen flagged {len(det_flags)} issue(s)."
            ),
        )

    prompt = _build_prompt(
        result=result,
        walk=walk,
        treatment=treatment,
        outcome=outcome,
        naive_estimate=naive_estimate,
        domain_brief=domain_brief,
        prescreen_flags=det_flags,
        prescreen_rationale=det_rationale,
    )

    try:
        response = client.parse(
            prompt=prompt,
            schema=_LLMAuditPayload,
            system=_SYSTEM_PROMPT,
            json_schema=_LLMAuditPayload.model_json_schema(),
        )
        payload = response.parsed
        assert isinstance(payload, _LLMAuditPayload)
    except Exception as exc:  # pragma: no cover - failure-safe path
        rec = _deterministic_recommendation(det_flags, result.n_used) or "accept"
        return AnomalyAudit(
            flags=list(det_flags),
            rationale_per_flag=dict(det_rationale),
            recommendation=rec,
            overall_note=(
                f"LLM audit failed ({type(exc).__name__}); deterministic-only result."
            ),
        )

    # Merge: deterministic flags preserved, LLM additions deduped.
    merged_flags: list[AnomalyAuditFlag] = list(det_flags)
    seen = set(merged_flags)
    for f in payload.flags:
        if f not in seen:
            merged_flags.append(f)
            seen.add(f)

    merged_rationale = dict(det_rationale)
    for k, v in payload.rationale_per_flag.items():
        # Don't let the LLM overwrite the deterministic rationale.
        if k not in merged_rationale:
            merged_rationale[k] = v

    # Recommendation: deterministic severity wins; otherwise LLM's choice.
    forced = _deterministic_recommendation(merged_flags, result.n_used)
    recommendation: Recommendation = forced if forced is not None else payload.recommendation

    overall_note = payload.overall_note or (
        f"{len(merged_flags)} flag(s) total ({len(det_flags)} deterministic)."
    )

    return AnomalyAudit(
        flags=merged_flags,
        rationale_per_flag=merged_rationale,
        recommendation=recommendation,
        overall_note=overall_note,
    )


__all__ = [
    "AnomalyAudit",
    "AnomalyAuditFlag",
    "audit_for_anomalies",
]
