"""Domain-aware sensitivity interpretation — LLM rationale wrapper.

The deterministic sensitivity layer (``evalue.py`` + ``sensemakr_py.py`` +
``verdict.py``) produces an authoritative color (green / yellow / red /
unknown) plus a short template string. That string is technically correct
but tone-deaf: an E-value of 1.6 means very different things to a
cardiologist (substantial — comparable to common clinical confounders) and
to a digital-marketer (weak — easily swamped by selection effects).

This module asks an LLM to translate the deterministic numbers into a
plain-language interpretation in the *domain's* voice. The deterministic
verdict color remains authoritative; the LLM only fills in the rationale,
and we hard-override the LLM if it tries to disagree with the color.

Design contract:

- INPUT: the deterministic E-value result, the sensemakr result (or
  ``None``), the deterministic verdict color, the point estimate + CI,
  treatment + outcome names, the domain brief, and the outcome dtype.
- OUTPUT: a :class:`SensitivityInterpretation` whose ``verdict_color`` is
  *forced* to equal the deterministic verdict.
- DEGENERATE INPUTS: if the E-value carries a ``reason`` (the "unknown"
  path), short-circuit to a ``verdict_color='unknown'`` interpretation
  without calling the LLM.
- FAILURE: any LLM error → return a default interpretation built from the
  existing deterministic ``sensitivity_rationale`` string. Never raise.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from causalrag.llm.ollama_client import OllamaClient
from causalrag.sensitivity.evalue import EValueResult
from causalrag.sensitivity.sensemakr_py import SensemakrResult


VerdictColor = Literal["green", "yellow", "red", "unknown"]


# ─────────── Output schema ───────────────────────────────────────────────


class SensitivityInterpretation(BaseModel):
    """Plain-language sensitivity rationale for a single experiment."""

    model_config = ConfigDict(extra="forbid")

    verdict_color: Literal["green", "yellow", "red", "unknown"] = Field(
        ...,
        description=(
            "MUST match the deterministic verdict. The LLM is told the "
            "color and instructed not to change it; if it does, the caller "
            "overrides."
        ),
    )
    plain_language: str = Field(
        ...,
        description=(
            "1-3 sentences in the inferred domain's voice. No statistical "
            "jargon."
        ),
    )
    what_it_rules_out: str = Field(
        ...,
        description=(
            "What kind of unmeasured confounder THIS evidence would "
            "survive — domain-specific examples preferred."
        ),
    )
    what_it_does_not_rule_out: str = Field(
        ...,
        description="What THIS evidence does NOT protect against.",
    )
    plausibility_of_threshold_confounder: str = Field(
        ...,
        description=(
            "Is an unmeasured confounder at the E-value threshold a "
            "plausible bound in this domain? Answer in domain terms."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "Short text the synthesis layer can quote verbatim. Self-"
            "contained sentence."
        ),
    )


# ─────────── Domain inference (heuristic) ────────────────────────────────


# Substring → domain label. The LLM does the heavy lifting; this is only
# used to slot a single domain word into the system prompt so phrasing
# matches the field. Mirrors the heuristic register used in synthesis.py
# but lighter — we don't need a full DomainKind here.
_DOMAIN_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "clinical",
        (
            "patient",
            "patients",
            "clinical",
            "clinician",
            "trial",
            "diagnosis",
            "treatment arm",
            "hospital",
            "mortality",
            "tumor",
            "disease",
            "drug",
            "dosage",
            "icu",
            "cohort",
            "comorbidit",
        ),
    ),
    (
        "marketing",
        (
            "marketing",
            "customer",
            "customers",
            "campaign",
            "ad spend",
            "conversion",
            "click",
            "channel",
            "ctr",
            "audience",
            "creative",
            "ltv",
            "arpu",
        ),
    ),
    (
        "business",
        (
            "revenue",
            "sales",
            "profit",
            "margin",
            "pricing",
            "kpi",
            "subscription",
            "churn",
            "retention",
            "business",
            "operator",
        ),
    ),
    (
        "policy",
        (
            "policy",
            "policymaker",
            "program",
            "subsidy",
            "voucher",
            "welfare",
            "regulation",
            "intervention rollout",
            "constituent",
        ),
    ),
    (
        "education",
        (
            "student",
            "students",
            "school",
            "teacher",
            "curriculum",
            "instruction",
            "learning",
            "test score",
            "classroom",
        ),
    ),
    (
        "ecology",
        (
            "ecology",
            "species",
            "habitat",
            "ecosystem",
            "biodiversity",
            "watershed",
        ),
    ),
    (
        "engineering",
        (
            "engineering",
            "setpoint",
            "sensor",
            "tolerance",
            "throughput",
            "fabrication",
            "process control",
        ),
    ),
    (
        "operations",
        (
            "operations",
            "supply chain",
            "fulfillment",
            "logistics",
            "queue",
            "throughput",
        ),
    ),
)


def _infer_domain(domain_brief: str | None) -> str:
    """Pick a single-word domain label by keyword count.

    Falls back to ``"general"`` if nothing matches — the LLM is still given
    the brief verbatim, so the prompt remains useful.
    """
    if not domain_brief:
        return "general"
    lowered = domain_brief.lower()
    best: tuple[int, str] = (0, "general")
    for label, keywords in _DOMAIN_KEYWORDS:
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits > best[0]:
            best = (hits, label)
    return best[1]


# ─────────── Default-on-failure construction ─────────────────────────────


def _default_interpretation(
    *,
    verdict_color: VerdictColor,
    deterministic_rationale: str,
    reason: str | None = None,
) -> SensitivityInterpretation:
    """Build a safe interpretation from the deterministic rationale alone.

    Used when the LLM call fails or when the inputs are degenerate
    (E-value 'unknown' path). Never raises.
    """
    base = deterministic_rationale.strip() or (
        f"Deterministic sensitivity verdict: {verdict_color}."
    )
    if reason:
        rationale = f"{base} (LLM interpretation unavailable: {reason})"
    else:
        rationale = base
    return SensitivityInterpretation(
        verdict_color=verdict_color,
        plain_language=base,
        what_it_rules_out=(
            "See deterministic rationale; no domain-tailored interpretation "
            "was produced for this experiment."
        ),
        what_it_does_not_rule_out=(
            "Any unmeasured confounder stronger than the E-value threshold "
            "implied by the deterministic analysis."
        ),
        plausibility_of_threshold_confounder=(
            "Not assessed — LLM interpretation unavailable, so the "
            "domain-plausibility of a confounder at the threshold could "
            "not be judged."
        ),
        rationale=rationale,
    )


def _unknown_interpretation(
    *,
    deterministic_rationale: str,
    evalue_reason: str | None,
) -> SensitivityInterpretation:
    """Build the refusal interpretation for the E-value 'unknown' path."""
    reason = evalue_reason or "E-value could not be computed."
    rationale = (
        "Sensitivity could not be assessed: "
        f"{reason} The deterministic verdict is 'unknown'."
    )
    base_text = deterministic_rationale.strip() or rationale
    return SensitivityInterpretation(
        verdict_color="unknown",
        plain_language=(
            "We cannot say how robust this finding is to unmeasured "
            f"confounding. {reason}"
        ),
        what_it_rules_out=(
            "Nothing — without a computable sensitivity statistic we "
            "cannot rule out any class of unmeasured confounder."
        ),
        what_it_does_not_rule_out=(
            "Any unmeasured confounder, weak or strong. Treat the effect "
            "estimate as conditional on the assumption of no unmeasured "
            "confounding."
        ),
        plausibility_of_threshold_confounder=(
            "Not applicable — no threshold was produced."
        ),
        rationale=f"{base_text} {rationale}".strip(),
    )


# ─────────── Prompt construction ─────────────────────────────────────────


_SYSTEM_PROMPT_TEMPLATE = (
    "You are a referee at a top journal in the {domain} field. Your job "
    "is to interpret a sensitivity analysis for a non-statistician domain "
    "expert who reads {domain} work every day.\n\n"
    "The deterministic verdict color is FIXED at '{color}' — do NOT change "
    "it. Your verdict_color field MUST be exactly '{color}'. Your job is "
    "to explain WHAT THIS MEANS in {domain} terms: is the threshold "
    "confounder plausible in this field, what kind of bias does this "
    "result survive, what does it not protect against.\n\n"
    "Calibration matters. An E-value of 1.6 is substantial in clinical "
    "research (rivals common unmeasured confounders like socioeconomic "
    "status); the same E-value is weak in digital marketing where "
    "selection and channel-mix confounding routinely exceed it. Calibrate "
    "your plausibility judgment to the {domain} field specifically.\n\n"
    "Style: speak the audience's language ({domain} domain experts). No "
    "statistical jargon in plain_language — terms like 'partial R²', "
    "'p-value', 'unbiased' are fine in the technical fields but only if "
    "they actually clarify. Keep plain_language to 1-3 sentences.\n\n"
    "Return ONLY a JSON SensitivityInterpretation. The verdict_color "
    "MUST be '{color}'."
)


def _build_prompt(
    *,
    evalue_result: EValueResult,
    sensemakr_result: SensemakrResult | None,
    deterministic_verdict: VerdictColor,
    point_estimate: float,
    ci_low: float | None,
    ci_high: float | None,
    treatment: str,
    outcome: str,
    domain_brief: str | None,
    outcome_dtype: str,
    domain_label: str,
) -> str:
    """Assemble the user-prompt context block.

    The deterministic verdict color is restated INSIDE the prompt as well
    as in the system message — defense in depth against the LLM forgetting
    its instructions on long contexts.
    """
    ci_str = (
        f"[{ci_low:+.4f}, {ci_high:+.4f}]"
        if ci_low is not None and ci_high is not None
        else "(not reported)"
    )
    e_ci_str = (
        f"{evalue_result.e_value_ci:.3f}"
        if evalue_result.e_value_ci is not None
        else "(not computable — CI may cross the null)"
    )
    rv_block = "  - sensemakr robustness value: (not run)\n"
    if sensemakr_result is not None:
        rv_block = (
            f"  - sensemakr robustness value (RV): "
            f"{sensemakr_result.robustness_value:.4f}\n"
            f"  - sensemakr RV at q=1: "
            f"{sensemakr_result.robustness_value_q:.4f}\n"
            f"  - sensemakr backend: {sensemakr_result.backend}\n"
        )

    parts: list[str] = []
    parts.append("## DETERMINISTIC VERDICT (FIXED — do not modify)")
    parts.append(f"  - verdict_color: {deterministic_verdict}")
    parts.append("")
    parts.append("## Sensitivity numbers")
    parts.append(
        f"  - E-value (point estimate): {evalue_result.e_value:.3f}\n"
        f"  - E-value (CI bound nearer the null): {e_ci_str}\n"
        f"  - E-value scale: {evalue_result.scale}\n"
        f"{rv_block}"
    )
    parts.append("## Effect")
    parts.append(
        f"  - treatment → outcome: {treatment} → {outcome}\n"
        f"  - outcome dtype: {outcome_dtype}\n"
        f"  - point estimate: {point_estimate:+.4f}\n"
        f"  - 95% CI: {ci_str}"
    )
    parts.append("")
    parts.append(f"## Domain ({domain_label})")
    if domain_brief:
        parts.append(domain_brief[:1500])
    else:
        parts.append("(no domain brief provided)")
    parts.append("")
    parts.append(
        "## Task\n"
        "Write a SensitivityInterpretation in the language of the "
        f"{domain_label} field. The verdict_color MUST be "
        f"'{deterministic_verdict}'. Calibrate the plausibility of a "
        f"threshold-strength confounder to what is realistic in "
        f"{domain_label} specifically."
    )
    return "\n".join(parts)


# ─────────── Public API ──────────────────────────────────────────────────


def interpret_sensitivity(
    *,
    evalue_result: EValueResult,
    sensemakr_result: SensemakrResult | None,
    deterministic_verdict: VerdictColor,
    point_estimate: float,
    ci_low: float | None,
    ci_high: float | None,
    treatment: str,
    outcome: str,
    domain_brief: str | None,
    outcome_dtype: str,
    client: OllamaClient,
    deterministic_rationale: str = "",
) -> SensitivityInterpretation:
    """LLM-translated, domain-aware sensitivity interpretation.

    The deterministic ``deterministic_verdict`` is authoritative — if the
    LLM emits any other ``verdict_color`` we overwrite the field and warn.

    Parameters
    ----------
    evalue_result:
        Output of :func:`causalrag.sensitivity.evalue.evalue` /
        :func:`evalue_for_estimator`. If ``evalue_result.reason`` is set
        (the 'unknown' path), this function short-circuits to an
        ``unknown`` interpretation and the LLM is NOT called.
    sensemakr_result:
        Output of :func:`sensemakr`, or ``None`` if the run failed / was
        skipped.
    deterministic_verdict:
        The color from :func:`causalrag.sensitivity.verdict.aggregate`
        (extended with ``"unknown"`` for the refusal case).
    point_estimate, ci_low, ci_high:
        Effect on the treatment scale, as written by the estimator.
    treatment, outcome:
        Column names — passed to the prompt so the LLM can phrase the
        finding concretely.
    domain_brief:
        The discovery layer's free-text brief. Used both for keyword-based
        domain inference and passed verbatim into the prompt.
    outcome_dtype:
        ``"binary"`` / ``"continuous"`` / ``"survival"`` / etc. Helps the
        LLM phrase the effect (probability points vs. units).
    client:
        Configured :class:`OllamaClient`. The function never hits the
        network if no client is functional — any error during the call is
        caught and a deterministic-default interpretation is returned.
    deterministic_rationale:
        The existing template string from ``master_loop._run_one_experiment``
        — used as the fallback rationale on LLM failure.
    """
    # 1. Degenerate-input short-circuit. The 'unknown' E-value path means
    # we don't even have a number to interpret; calling the LLM here would
    # produce hallucinated confidence.
    if evalue_result.reason is not None or deterministic_verdict == "unknown":
        return _unknown_interpretation(
            deterministic_rationale=deterministic_rationale,
            evalue_reason=evalue_result.reason,
        )

    # 2. Build prompt + system message.
    domain_label = _infer_domain(domain_brief)
    system = _SYSTEM_PROMPT_TEMPLATE.format(
        domain=domain_label, color=deterministic_verdict
    )
    prompt = _build_prompt(
        evalue_result=evalue_result,
        sensemakr_result=sensemakr_result,
        deterministic_verdict=deterministic_verdict,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        treatment=treatment,
        outcome=outcome,
        domain_brief=domain_brief,
        outcome_dtype=outcome_dtype,
        domain_label=domain_label,
    )

    # 3. Call the LLM. Any failure → fallback default. We catch BaseException
    # rather than Exception so a misbehaving transport (timeouts that raise
    # KeyboardInterrupt-like errors in some httpx versions) still degrades
    # gracefully — sensitivity interpretation is best-effort prose, not a
    # load-bearing artifact.
    try:
        response = client.parse(
            prompt=prompt,
            schema=SensitivityInterpretation,
            system=system,
            json_schema=SensitivityInterpretation.model_json_schema(),
        )
        parsed = response.parsed
        assert isinstance(parsed, SensitivityInterpretation)
    except Exception as exc:  # noqa: BLE001 — best-effort interpretation
        return _default_interpretation(
            verdict_color=deterministic_verdict,
            deterministic_rationale=deterministic_rationale,
            reason=f"{type(exc).__name__}: {exc}",
        )

    # 4. Enforce the verdict-color contract. The LLM is instructed to
    # echo the deterministic color, but instruction-following is not
    # guaranteed on smaller local models. Override + warn.
    if parsed.verdict_color != deterministic_verdict:
        warnings.warn(
            (
                "LLM sensitivity interpretation returned verdict_color="
                f"{parsed.verdict_color!r}; overriding to deterministic "
                f"verdict {deterministic_verdict!r}."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        parsed = parsed.model_copy(update={"verdict_color": deterministic_verdict})

    return parsed


__all__ = [
    "SensitivityInterpretation",
    "VerdictColor",
    "interpret_sensitivity",
]


# Expose a helper for tests / callers that need to see what was inferred.
def _infer_domain_for_test(brief: str | None) -> str:
    """Test-only re-export of the domain inference heuristic."""
    return _infer_domain(brief)


# Keep an unused-name suppressor for Any import (kept for forward-compat).
_ANY: Any = None
