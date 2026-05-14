"""Domain-agnostic insights synthesis layer.

After the master loop completes, this module translates the
statistical results of every completed experiment into an executive
synthesis written in the language of the dataset's domain. A clinical
dataset produces clinical implications and care-pathway suggestions;
a sales dataset produces business impact and operator actions; an
ecology dataset produces ecological interpretations and follow-up
study designs; a policy dataset produces policy recommendations; a
physics or engineering dataset produces mechanistic conclusions and
design implications.

The reasoning LLM does the translation. We give it: (a) the inferred
domain context (from the dataset's domain_brief built earlier in the
pipeline), (b) per-experiment statistical results, (c) per-experiment
magnitude conversions that are appropriate for that outcome type
(continuous / binary / count / monetary / survival), and (d) honest
uncertainty signals (CI width, sensitivity verdict, refutation
status). The LLM produces a ranked list of plain-language findings
with the audience and register appropriate to the domain.

Output is an :class:`ExecutiveSynthesis` — one TL;DR sentence plus a
ranked list of :class:`Insight`. The HTML report renders this at the
top, before the technical Roadmap walks.
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.estimand import EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.llm.honesty import with_honesty
from causalrag.llm.ollama_client import OllamaClient


# ─────────── Domain hint (LLM may override) ──────────────────────────────


DomainKind = Literal[
    "business",
    "clinical",
    "policy",
    "social_science",
    "ecology",
    "physical_science",
    "engineering",
    "education",
    "marketing",
    "operations",
    "other",
]


_DOMAIN_HINTS: dict[DomainKind, tuple[str, str]] = {
    # (audience description, action-language guidance)
    "business": (
        "an operator / executive who will decide where to allocate budget",
        "Recommend concrete operational actions (pilots, rollouts, budget shifts).",
    ),
    "clinical": (
        "a clinician / clinical researcher who will judge whether to change practice",
        "Phrase implications as care-pathway considerations; never prescribe. "
        "Flag generalizability limits and the need for confirmatory trials.",
    ),
    "policy": (
        "a policymaker or program evaluator weighing intervention design",
        "Frame as program-design implications and follow-up evaluation needs. "
        "Be explicit about external validity and equity considerations.",
    ),
    "social_science": (
        "a social scientist interpreting a behavioural mechanism",
        "Frame as evidence for / against a hypothesized mechanism and as "
        "directions for confirmatory study.",
    ),
    "ecology": (
        "an ecologist / environmental scientist interpreting field data",
        "Frame as ecological interpretation plus suggested manipulative or "
        "longitudinal follow-up.",
    ),
    "physical_science": (
        "a scientist interpreting an experimental or observational measurement",
        "Frame as mechanistic interpretation and follow-up measurement.",
    ),
    "engineering": (
        "an engineer deciding a design parameter or operating setpoint",
        "Frame as design / setpoint implications and the test that would "
        "validate the change.",
    ),
    "education": (
        "an educator / program designer judging an instructional intervention",
        "Frame as instructional-design implications and learner-subgroup "
        "follow-up.",
    ),
    "marketing": (
        "a marketer deciding targeting, message, or channel",
        "Frame as targeting / channel implications and a holdout test that "
        "would confirm.",
    ),
    "operations": (
        "an operations leader deciding a process change",
        "Frame as process-change implications and the rollout / measurement "
        "plan that would validate.",
    ),
    "other": (
        "a domain expert who knows the field but did not run this analysis",
        "Frame the implication in the natural language of the field; suggest "
        "the next study that would tighten the conclusion.",
    ),
}


# ─────────── Output schema ───────────────────────────────────────────────


class Insight(BaseModel):
    """One actionable finding, in the language of the dataset's domain."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., ge=1, description="1 is the highest-priority finding")
    hypothesis_id: str
    headline: str = Field(
        ...,
        description=(
            "One-sentence plain-language finding written for the domain "
            "audience. Quantified using ONLY the magnitudes the pipeline "
            "provided. No statistical jargon."
        ),
    )
    quantified_effect: str = Field(
        ...,
        description=(
            "Numeric statement of the effect in domain-appropriate units "
            "(currency, percentage points, days of survival, units sold, "
            "mg/L, students, etc.). If the CI crosses zero, say so."
        ),
    )
    domain_implication: str = Field(
        ...,
        description=(
            "What this means in the domain — written for the audience the "
            "pipeline inferred (clinician / operator / policymaker / "
            "scientist / engineer / educator / etc.)."
        ),
    )
    suggested_next_step: str = Field(
        ...,
        description=(
            "Concrete next action appropriate to the domain — a pilot, a "
            "confirmatory study, a design change, a follow-up measurement, "
            "a randomized trial, a policy memo, etc."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "high = significant + green sensitivity + refutations pass; "
            "medium = significant + yellow sensitivity OR refutations mixed; "
            "low = CI crosses zero OR red sensitivity OR refutations failed."
        ),
    )
    caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Honest caveats specific to this finding (small sample, "
            "unmeasured confounding plausible, boundary effect only, "
            "extrapolation outside support, etc.)."
        ),
    )
    estimator_used: str
    unit_category: Literal[
        "monetary", "rate", "time", "count", "concentration", "continuous"
    ] | None = Field(
        default=None,
        description=(
            "Optional domain-aware override of the outcome's natural unit "
            "category. If the regex / dtype / flag heuristic miscategorizes "
            "the column (e.g., 'value_lbs' is mass, not currency), set this "
            "to the correct kind."
        ),
    )
    unit_category_rationale: str | None = Field(
        default=None,
        description=(
            "One-sentence justification for unit_category when overriding "
            "the heuristic. REQUIRED if unit_category is set."
        ),
    )


class ExecutiveSynthesis(BaseModel):
    """Top-of-report synthesis in the dataset's domain language."""

    model_config = ConfigDict(extra="forbid")

    inferred_domain: DomainKind = Field(
        ...,
        description="Best-guess domain the dataset belongs to. Drives audience and register.",
    )
    tldr: str = Field(
        ...,
        description="A single sentence — the most important thing this analysis found.",
    )
    findings: list[Insight]
    overall_caveats: list[str] = Field(
        default_factory=list,
        description=(
            "Limitations that apply across all findings (e.g., 'all results "
            "are observational; an interventional study would strengthen "
            "causal interpretation')."
        ),
    )
    validation_warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Deterministic-validation warnings appended after the LLM call "
            "(fabricated hypothesis ids, wrong estimator names, confidence "
            "downgrades, etc.). Not produced by the LLM."
        ),
    )


# ─────────── Magnitude conversion (domain-agnostic) ──────────────────────


_MONETARY_PATTERNS = re.compile(
    r"(revenue|sales|spend|price|cost|gmv|ltv|arpu|payment|charge|fee|"
    r"income|profit|margin|usd|dollar|eur|gbp|yen|salary|wage)",
    re.I,
)
_RATE_OR_BINARY = re.compile(
    r"(converted|churned|retained|click|subscribed|engaged|active|"
    r"purchased|attrited|response|adverse|mortality|incidence|"
    r"prevalence|positive|failure|success|rate)",
    re.I,
)
_COUNT_PATTERNS = re.compile(
    r"(count|n_|num_|sessions|orders|visits|events|admissions|"
    r"hospitalizations|attempts|trials)",
    re.I,
)
_TIME_PATTERNS = re.compile(
    r"(days|hours|minutes|seconds|months|years|duration|time_to|"
    r"survival|tenure|age)",
    re.I,
)
_CONCENTRATION_PATTERNS = re.compile(
    r"(mg|kg|ng|pg|μg|ug|ppb|ppm|mol|conc|level|titer|titre)",
    re.I,
)


def _lookup_column_spec(
    protocol: StudyProtocol | None, name: str
) -> Any | None:
    """Find the VariableSpec for ``name`` in the protocol if available."""
    if protocol is None:
        return None
    sources: list[Any] = []
    if protocol.dataset is not None and protocol.dataset.columns:
        sources.extend(protocol.dataset.columns)
    if protocol.discovery is not None and protocol.discovery.columns:
        sources.extend(protocol.discovery.columns)
    for spec in sources:
        if getattr(spec, "name", None) == name:
            return spec
    return None


def _classify_outcome_units(
    name: str,
    flags: frozenset[DataFlag],
    *,
    protocol: StudyProtocol | None = None,
) -> str:
    """Heuristic categorization of the outcome's natural units.

    Consults — in priority order — (1) outcome-type flags, (2) the
    column's dtype + per-row profile (min/max/pct_zeros) from
    ``protocol.dataset.columns`` if present, and (3) the regex
    fallback on the column name.
    """
    n = name.lower()

    # (1) Flags take precedence over name regex — they were emitted by
    # the profiler which actually looked at the data.
    if DataFlag.RIGHT_CENSORED_OUTCOME in flags:
        return "time"
    if DataFlag.BINARY_OUTCOME in flags:
        return "rate"
    if DataFlag.COUNT_OUTCOME in flags:
        return "count"

    # (2) Dtype + profile — only consulted to *disqualify* the monetary
    # / concentration regex when the dtype clearly contradicts (e.g. a
    # column named "cost_to_complete" that is actually boolean).
    spec = _lookup_column_spec(protocol, name)
    dtype: str | None = None
    profile: dict[str, Any] | None = None
    if spec is not None:
        dtype = getattr(spec, "dtype", None)
        prof = getattr(spec, "profile", None)
        if isinstance(prof, dict):
            profile = prof
        elif prof is not None:
            try:
                profile = dict(prof)
            except Exception:
                profile = None

    # Boolean / 0-1 integer columns regardless of name → rate
    if dtype in {"bool", "boolean"}:
        return "rate"
    if profile is not None:
        try:
            mn = profile.get("min")
            mx = profile.get("max")
            if (
                mn is not None
                and mx is not None
                and float(mn) >= 0.0
                and float(mx) <= 1.0
                and dtype in {"int64", "int32", "int", "float64", "float32", "float"}
            ):
                # 0/1 indicator regardless of column-name regex
                return "rate"
        except (TypeError, ValueError):
            pass

    # (3) Regex fallback on the name — but only fire monetary /
    # concentration when the dtype is numeric (object/datetime can't be
    # money).
    numeric_dtype = dtype is None or dtype.startswith(
        ("int", "float", "Int", "Float", "number")
    )
    if numeric_dtype and _MONETARY_PATTERNS.search(n):
        return "monetary"
    if _RATE_OR_BINARY.search(n):
        return "rate"
    if _TIME_PATTERNS.search(n):
        return "time"
    if _COUNT_PATTERNS.search(n):
        return "count"
    if numeric_dtype and _CONCENTRATION_PATTERNS.search(n):
        return "concentration"
    return "continuous"


def _population_scale(
    *,
    estimand_klass: str | None,
    n_used: int,
    n_treated: int | None,
    n_control: int | None,
) -> tuple[int, str, str | None]:
    """Pick the right population count to multiply a per-subject effect by.

    Returns ``(count, key_suffix, caveat)``. ``key_suffix`` becomes the
    bare name of the aggregated field (e.g. ``expected_count_ate``).
    ``caveat`` is a short note that callers should attach when the
    population is uncertain.
    """
    if estimand_klass == EstimandClass.ATT.value:
        if n_treated is not None and n_treated > 0:
            return n_treated, "att_n_treated", None
        return (
            n_used,
            "uncertain_population",
            (
                "Estimand is ATT but n_treated was not recorded; aggregate "
                "uses n_used as a fallback and may overstate the count."
            ),
        )
    if estimand_klass == EstimandClass.ATC.value:
        if n_control is not None and n_control > 0:
            return n_control, "atc_n_control", None
        return (
            n_used,
            "uncertain_population",
            (
                "Estimand is ATC but n_control was not recorded; aggregate "
                "uses n_used as a fallback and may overstate the count."
            ),
        )
    if estimand_klass == EstimandClass.ATE.value:
        return n_used, "ate_n_used", None
    return (
        n_used,
        "uncertain_population",
        (
            f"Estimand class {estimand_klass!r} does not have a "
            "standard population-scale interpretation; aggregate uses "
            "n_used and may not represent the policy-relevant count."
        ),
    )


def _magnitude(
    outcome_name: str,
    point: float,
    n_used: int,
    flags: frozenset[DataFlag],
    *,
    protocol: StudyProtocol | None = None,
    estimand_klass: str | None = None,
    n_treated: int | None = None,
    n_control: int | None = None,
) -> dict[str, Any]:
    """Compute defensible magnitude conversions for the LLM to quote.

    Pure math from (point estimate, n, outcome name). We do not invent
    annualization, market sizing, dose-response extrapolation, or
    monetization that isn't directly derivable from the data."""
    kind = _classify_outcome_units(outcome_name, flags, protocol=protocol)
    info: dict[str, Any] = {
        "raw_point_estimate": point,
        "raw_outcome_name": outcome_name,
        "n_in_analysis": n_used,
        "unit_category": kind,
    }
    pop_count, pop_suffix, pop_caveat = _population_scale(
        estimand_klass=estimand_klass,
        n_used=n_used,
        n_treated=n_treated,
        n_control=n_control,
    )
    if kind == "monetary":
        info["per_subject_currency_change"] = round(point, 4)
        info["effect_at_analysis_sample_currency"] = round(point * n_used, 2)
        info["effect_at_analysis_sample_currency_caveat"] = (
            "in-sample effect at the n_used observations; do not "
            "extrapolate without an explicit population scaling factor."
        )
    elif kind == "rate":
        info["percentage_point_change"] = round(point * 100, 2)
        if pop_suffix == "uncertain_population":
            info["expected_count_uncertain_population"] = round(
                point * pop_count, 1
            )
        else:
            info[f"expected_count_{pop_suffix}"] = round(point * pop_count, 1)
    elif kind == "time":
        info["per_subject_time_unit_change"] = round(point, 4)
    elif kind == "count":
        info["per_subject_count_change"] = round(point, 4)
        info["aggregate_count_change_at_analysis_n"] = round(point * n_used, 1)
    elif kind == "concentration":
        info["per_subject_concentration_change"] = round(point, 6)
    else:
        info["per_subject_change"] = round(point, 4)
        info["aggregate_change_at_analysis_n"] = round(point * n_used, 2)

    if pop_caveat is not None:
        info["population_scale_caveat"] = pop_caveat
    return info


# ─────────── LLM-driven synthesis ────────────────────────────────────────


_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a senior domain expert AND a senior causal-inference "
    "statistician — the kind of person a journal editor or a board "
    "would call to review a study. You have just completed a multi-"
    "experiment causal-inference analysis and are writing the "
    "executive synthesis for the audience that owns this domain.\n\n"
    "STEP 1 — Identify the domain. From the dataset description, the "
    "research question, the column names, and the kinds of outcomes "
    "analyzed, infer the most likely domain (business, clinical, "
    "policy, social-science, ecology, physical-science, engineering, "
    "education, marketing, operations, or other). Pick exactly one. "
    "This determines audience, register, units, and the shape of "
    "'next step'.\n\n"
    "STEP 2 — Write in that domain's language. A clinician does not "
    "want 'business impact'. An operator does not want 'mechanistic "
    "interpretation'. A policymaker does not want 'ROI'. Pick the "
    "vocabulary the audience uses every day. NEVER use statistical "
    "jargon ('ATE', 'CATE', 'cross-fitted', 'p-value', 'doubly "
    "robust') in headline / quantified_effect / domain_implication / "
    "suggested_next_step. Those terms can appear only in caveats if "
    "absolutely necessary.\n\n"
    "STEP 3 — Quantify honestly.\n"
    "  - When judging significance, use the adjusted_p_value (which "
    "    reflects the protocol's multiple-testing correction across "
    "    the K experiments), not the raw p-value. The adjusted value "
    "    is what controls family-wise error / FDR across this analysis.\n"
    "  - Use ONLY the magnitudes provided in each experiment's "
    "    'magnitude' block. Do not invent annualization, market "
    "    sizing, lifetime extrapolations, dose-response curves, or "
    "    population scaling we didn't compute.\n"
    "  - Translate the unit_category appropriately:\n"
    "      • monetary → currency phrasing\n"
    "      • rate → percentage points (NOT 'percent')\n"
    "      • time → days / months / hours, as native\n"
    "      • count → events / units / occurrences\n"
    "      • concentration → mg/L, μg/dL, ppm, etc. — keep native units\n"
    "      • continuous → keep the column's natural unit\n"
    "  - If the CI crosses zero, your quantified_effect must reflect "
    "    that — never report a single number as if it were certain.\n\n"
    "STEP 4 — Domain implication.\n"
    "  - Translate the finding into a statement the audience cares "
    "    about. For a clinician: 'patients with X may benefit from Y' "
    "    (never prescriptive). For a policymaker: 'program component X "
    "    appears to drive outcome Y in subgroup Z'. For an engineer: "
    "    'setpoint X corresponds to outcome Y'. For an operator: "
    "    'customers / units exposed to X show Y'.\n\n"
    "STEP 5 — Suggested next step.\n"
    "  - Concrete and falsifiable, in the language of the domain: a "
    "    randomized trial, a pilot rollout, a confirmatory dataset, a "
    "    physical measurement, a policy evaluation, a sensor "
    "    deployment, a focus group, a follow-up cohort, etc.\n"
    "  - Conservative when confidence is low — propose a small "
    "    confirmatory study, not a full rollout / policy change.\n\n"
    "CONFIDENCE FLAGS:\n"
    "  - high: significant effect, green sensitivity, refutations pass.\n"
    "  - medium: significant + yellow sensitivity OR refutations mixed.\n"
    "  - low: CI crosses zero OR red sensitivity OR refutations failed.\n\n"
    "RANKING:\n"
    "  - findings[0] is the single most important thing the audience "
    "    should attend to.\n"
    "  - Rank by domain-relevant impact × confidence, not by p-value. "
    "    A large-magnitude yellow finding outranks a tiny green one.\n\n"
    "HONESTY (non-negotiable):\n"
    "  - If data are observational, say so in overall_caveats and "
    "    name the kind of interventional study that would strengthen "
    "    the conclusion (RCT, manipulative experiment, natural "
    "    experiment, instrumental variable, etc.).\n"
    "  - Per-finding caveats[] should name finding-specific issues "
    "    (small subgroup, support extrapolation, unmeasured "
    "    confounding plausibility, boundary effect).\n\n"
    "Return ONLY a JSON ExecutiveSynthesis."
)


def _build_synthesis_prompt(
    protocol: StudyProtocol, df: pd.DataFrame
) -> str:
    parts: list[str] = []
    parts.append("## Dataset")
    if protocol.dataset:
        parts.append(
            f"  - source: {protocol.dataset.source}\n"
            f"  - n_rows: {protocol.dataset.n_rows}, n_cols: {protocol.dataset.n_cols}"
        )
    if protocol.research_question:
        parts.append(f"  - research question: {protocol.research_question}")
    if protocol.discovery and protocol.discovery.domain_brief:
        parts.append("")
        parts.append("## Domain brief (used earlier in the pipeline)")
        parts.append(protocol.discovery.domain_brief[:1500])

    parts.append("")
    parts.append("## Completed experiments")
    flags = frozenset(protocol.flags)
    for h_id, walk in protocol.roadmap_walks.items():
        if not walk.q7_estimates:
            continue
        est = walk.q7_estimates[-1]
        outcome = walk.q3_estimand.outcome if walk.q3_estimand else ""
        treatment = walk.q3_estimand.treatment if walk.q3_estimand else "?"
        klass = walk.q3_estimand.klass.value if walk.q3_estimand else "?"
        diag = est.diagnostics if isinstance(est.diagnostics, dict) else {}
        n_treated = diag.get("n_treated")
        n_control = diag.get("n_control")
        n_treated_int = int(n_treated) if isinstance(n_treated, (int, float)) else None
        n_control_int = int(n_control) if isinstance(n_control, (int, float)) else None
        magnitude = _magnitude(
            outcome,
            est.point_estimate,
            est.n_used,
            flags,
            protocol=protocol,
            estimand_klass=klass,
            n_treated=n_treated_int,
            n_control=n_control_int,
        )
        ci_str = (
            f"[{est.ci_low:+.4f}, {est.ci_high:+.4f}]"
            if est.ci_low is not None and est.ci_high is not None
            else "—"
        )
        p_str = f"{est.p_value:.4g}" if est.p_value is not None else "NA"
        adj_p = diag.get("adjusted_p_value") if isinstance(diag, dict) else None
        adj_method = diag.get("adjustment_method") if isinstance(diag, dict) else None
        adj_p_str = (
            f"{adj_p:.4g} ({adj_method})"
            if adj_p is not None
            else "NA"
        )
        magnitude["adjusted_p_value"] = adj_p
        magnitude["adjustment_method"] = adj_method
        parts.append(
            f"  - **{h_id}**: {treatment} → {outcome} ({klass}, "
            f"estimator={est.estimator_id})\n"
            f"    - raw point estimate: {est.point_estimate:+.4f}\n"
            f"    - 95% CI: {ci_str}\n"
            f"    - p-value: {p_str}\n"
            f"    - adjusted p-value: {adj_p_str}\n"
            f"    - n: {est.n_used:,}\n"
            f"    - magnitude: {magnitude}\n"
            f"    - sensitivity verdict: {walk.q8_interpretation or '—'}\n"
        )

    parts.append("")
    parts.append(
        "## Task\n"
        "Identify the domain, then write the ExecutiveSynthesis in that "
        "domain's language. Quantify using ONLY the magnitudes above. "
        "Rank findings by domain-relevant impact × confidence."
    )
    return "\n".join(parts)


def _sensitivity_verdict_color(verdict: str | None) -> str | None:
    """Extract 'red' / 'yellow' / 'green' from a free-text verdict, if any."""
    if not verdict:
        return None
    v = verdict.lower()
    for color in ("red", "yellow", "green"):
        if color in v:
            return color
    return None


def _enforce_confidence(
    findings: list[Insight],
    walks_by_id: dict[str, Any],
    warnings_log: list[str],
) -> None:
    """Apply deterministic confidence rules. Mutates ``findings`` in place."""
    rank_high = {"high": 3, "medium": 2, "low": 1}
    rank_low = {3: "high", 2: "medium", 1: "low"}
    for f in findings:
        walk = walks_by_id.get(f.hypothesis_id)
        if walk is None or not walk.q7_estimates:
            continue
        est = walk.q7_estimates[-1]
        forced_low_reason: str | None = None
        cap: str | None = None

        if est.ci_low is not None and est.ci_high is not None:
            if est.ci_low <= 0.0 <= est.ci_high:
                forced_low_reason = "CI crosses zero"
        if _sensitivity_verdict_color(walk.q8_interpretation) == "red":
            forced_low_reason = (
                f"{forced_low_reason}; sensitivity red"
                if forced_low_reason
                else "sensitivity verdict is red"
            )
        if est.n_used < 100:
            cap = "medium"

        original = f.confidence
        new_conf = original
        if forced_low_reason is not None:
            new_conf = "low"
        elif cap is not None and rank_high.get(new_conf, 0) > rank_high[cap]:
            new_conf = cap

        if new_conf != original:
            f.confidence = new_conf  # type: ignore[assignment]
            if forced_low_reason:
                warnings_log.append(
                    f"finding for {f.hypothesis_id}: confidence forced to "
                    f"'low' (was {original!r}) — {forced_low_reason}."
                )
            else:
                warnings_log.append(
                    f"finding for {f.hypothesis_id}: confidence capped at "
                    f"{new_conf!r} (was {original!r}) — n_used={est.n_used} < 100."
                )
        # Suppress unused-name warning
        _ = rank_low


def _validate_against_protocol(
    synthesis: ExecutiveSynthesis,
    protocol: StudyProtocol,
) -> None:
    """Drop findings with fabricated ids; correct fabricated estimator ids.

    Mutates ``synthesis.findings`` and ``synthesis.validation_warnings``
    in place.
    """
    walks = protocol.roadmap_walks
    kept: list[Insight] = []
    for f in synthesis.findings:
        # System-stub findings (synthesis failure) are passed through.
        if f.hypothesis_id == "<system>":
            kept.append(f)
            continue
        walk = walks.get(f.hypothesis_id)
        if walk is None:
            synthesis.validation_warnings.append(
                f"dropped fabricated finding: hypothesis_id "
                f"{f.hypothesis_id!r} not present in roadmap_walks."
            )
            continue
        # Estimator-id check
        actual_estimator: str | None = None
        if walk.q7_estimates:
            actual_estimator = walk.q7_estimates[-1].estimator_id
        if (
            actual_estimator is not None
            and f.estimator_used != actual_estimator
        ):
            synthesis.validation_warnings.append(
                f"finding for {f.hypothesis_id}: estimator_used "
                f"{f.estimator_used!r} did not match the walk's actual "
                f"estimator {actual_estimator!r}; corrected."
            )
            f.estimator_used = actual_estimator
        kept.append(f)
    synthesis.findings = kept

    _enforce_confidence(synthesis.findings, dict(walks), synthesis.validation_warnings)


def _synthesis_failure_stub(
    *,
    err: BaseException,
    n_walks: int,
) -> ExecutiveSynthesis:
    """Build the safe fallback synthesis used when the LLM call fails."""
    return ExecutiveSynthesis(
        inferred_domain="other",
        tldr=f"synthesis failed: {type(err).__name__}: {err}",
        findings=[
            Insight(
                rank=1,
                hypothesis_id="<system>",
                headline="Synthesis layer failed to generate findings",
                quantified_effect="—",
                domain_implication=(
                    f"Pipeline produced {n_walks} experiments but the "
                    "synthesis call errored. Inspect the per-experiment "
                    "Roadmap walks for the raw results."
                ),
                suggested_next_step=(
                    "Re-run synthesis after diagnosing the LLM error "
                    "(see executive_synthesis_error.txt in the project dir)."
                ),
                confidence="low",
                caveats=[
                    "Fallback stub — no LLM-generated findings.",
                    f"Underlying error: {type(err).__name__}: {err}",
                ],
                estimator_used="—",
            )
        ],
        overall_caveats=[
            "Executive synthesis was not produced; only the technical "
            "Roadmap walks are available.",
        ],
        validation_warnings=[
            f"synthesis_failure: {type(err).__name__}: {err}",
        ],
    )


def synthesize_insights(
    *,
    protocol: StudyProtocol,
    df: pd.DataFrame,
    client: OllamaClient,
    error_log_path: Path | None = None,
) -> ExecutiveSynthesis:
    """Produce a domain-aware executive synthesis from a completed protocol.

    On LLM / parse failure, returns a stub :class:`ExecutiveSynthesis`
    instead of propagating. If ``error_log_path`` is provided, the
    traceback is written there for diagnosis.
    """
    n_walks = sum(1 for w in protocol.roadmap_walks.values() if w.q7_estimates)

    # Cross-experiment analysis — surfaces contradictions, reinforcements,
    # and chain narratives BEFORE the synthesis prompt. Failure-safe.
    cross_block = ""
    try:
        from causalrag.reporting.cross_experiment import (
            analyze_cross_experiment,
            cross_experiment_block_for_prompt,
        )

        analysis = analyze_cross_experiment(protocol=protocol, client=client)
        cross_block = cross_experiment_block_for_prompt(analysis)
    except Exception:
        cross_block = ""

    prompt = _build_synthesis_prompt(protocol, df)
    if cross_block:
        prompt = (
            prompt
            + "\n\n## Cross-experiment analysis\n"
            + cross_block
            + "\n\nUse the cross-experiment analysis above to ensure the "
            "synthesis honors contradictions and reinforcements."
        )
    try:
        response = client.parse(
            prompt=prompt,
            schema=ExecutiveSynthesis,
            system=with_honesty(_SYNTHESIS_SYSTEM_PROMPT),
            json_schema=ExecutiveSynthesis.model_json_schema(),
        )
        synthesis = response.parsed
        assert isinstance(synthesis, ExecutiveSynthesis)
    except Exception as e:  # noqa: BLE001 — synthesis is best-effort
        if error_log_path is not None:
            try:
                error_log_path.write_text(
                    "".join(traceback.format_exception(e)),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return _synthesis_failure_stub(err=e, n_walks=n_walks)

    _validate_against_protocol(synthesis, protocol)
    return synthesis


# ─────────── HTML rendering ──────────────────────────────────────────────


_DOMAIN_LABELS: dict[str, str] = {
    "business": "Business",
    "clinical": "Clinical",
    "policy": "Policy",
    "social_science": "Social science",
    "ecology": "Ecology",
    "physical_science": "Physical science",
    "engineering": "Engineering",
    "education": "Education",
    "marketing": "Marketing",
    "operations": "Operations",
    "other": "General",
}


def render_executive_synthesis_html(synth: ExecutiveSynthesis) -> str:
    """Render the synthesis as the top section of the HTML report."""
    import html

    def _e(t: Any) -> str:
        return html.escape("" if t is None else str(t))

    def _conf_color(c: str) -> str:
        return {"high": "#7ed2e6", "medium": "#a3b6da", "low": "#e08877"}.get(
            c.lower(), "#9aa3b5"
        )

    domain_label = _DOMAIN_LABELS.get(synth.inferred_domain, "General")
    parts: list[str] = []
    parts.append(
        f"<div class='eyebrow'>EXECUTIVE SYNTHESIS · {_e(domain_label).upper()}</div>"
    )
    parts.append(f"<p class='lede serif'>{_e(synth.tldr)}</p>")

    for i, f in enumerate(synth.findings, 1):
        color = _conf_color(f.confidence)
        parts.append(
            f"<div class='card' style='border-left: 4px solid {color}; padding-left: 16px;'>"
            f"<div class='eyebrow'>FINDING {i} · {_e(f.confidence.upper())} CONFIDENCE</div>"
            f"<h3 style='margin-top: 4px;'>{_e(f.headline)}</h3>"
            f"<p><strong>Effect:</strong> {_e(f.quantified_effect)}</p>"
            f"<p><strong>Implication:</strong> {_e(f.domain_implication)}</p>"
            f"<p><strong>Suggested next step:</strong> {_e(f.suggested_next_step)}</p>"
        )
        if f.caveats:
            parts.append(
                "<p class='dim'><em>Caveats:</em> "
                + " · ".join(_e(c) for c in f.caveats)
                + "</p>"
            )
        parts.append(
            f"<p class='dim' style='font-size: 11px;'>"
            f"Source: <code>{_e(f.hypothesis_id)}</code> · "
            f"estimator: <code>{_e(f.estimator_used)}</code></p></div>"
        )

    if synth.overall_caveats:
        parts.append("<h3>Overall caveats</h3><ul>")
        for c in synth.overall_caveats:
            parts.append(f"<li>{_e(c)}</li>")
        parts.append("</ul>")

    return "\n".join(parts)


__all__ = [
    "Insight",
    "ExecutiveSynthesis",
    "DomainKind",
    "synthesize_insights",
    "render_executive_synthesis_html",
]
