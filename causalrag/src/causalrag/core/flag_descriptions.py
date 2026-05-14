"""Single source of truth for DataFlag semantics.

Every DataFlag in `core/flags.py` should have a one-line plain-English
description here PLUS a "what this means for the analysis" hint. Used
by:

* Master-loop planner / critic / foundation prompts (so the LLM doesn't
  have to guess what a flag means from its enum name).
* Synthesis prompt (so caveats reference the right causal-inference
  concepts).
* HTML report — rendered as a tooltip / chip on each flag.
* The Sprint 9.5 end-to-end flow audit (`audit_pipeline_flow`) uses
  this as the canonical flag-vocabulary registry.

If a new flag is added to `DataFlag` without a description here, the
flow audit will flag it as undocumented at CI time.
"""

from __future__ import annotations

from dataclasses import dataclass

from causalrag.core.flags import DataFlag


@dataclass(frozen=True)
class FlagDescription:
    summary: str          # 1-sentence plain English
    implication: str      # what this means for the analysis approach
    routes_to: list[str]  # estimator id prefixes / sensitivity tools


_DESCRIPTIONS: dict[DataFlag, FlagDescription] = {
    # ─── Treatment shape ────────────────────────────────────────────
    DataFlag.BINARY_TREATMENT: FlagDescription(
        summary="The treatment column has exactly two values (0/1).",
        implication=(
            "Standard ATE / ATT / RD landscape — DML, matching, weighting, "
            "and meta-learners all admissible."
        ),
        routes_to=["python.dml.linear", "rbridge.matchit", "rbridge.weightit"],
    ),
    DataFlag.CATEGORICAL_TREATMENT: FlagDescription(
        summary="Multi-arm or categorical treatment (>2 levels).",
        implication=(
            "Multi-arm estimators only; pairwise ATE contrasts may need joint "
            "FWER control."
        ),
        routes_to=["rbridge.grf.multi_arm_causal_forest"],
    ),
    DataFlag.CONTINUOUS_TREATMENT: FlagDescription(
        summary="Continuous-dose treatment.",
        implication=(
            "Use dose-response / shift-policy estimators; partial-effect "
            "slopes; positivity needs density-ratio support."
        ),
        routes_to=["rbridge.lmtp.shift", "rbridge.lmtp.sdr", "rbridge.marginaleffects.slopes"],
    ),
    DataFlag.MIXTURE_EXPOSURE: FlagDescription(
        summary="Two or more concurrent treatments / a joint-exposure mixture.",
        implication=(
            "Joint-shift estimators (lmtp.mixture); marginal-of-one-treatment "
            "claims are misleading without joint adjustment."
        ),
        routes_to=["rbridge.lmtp.mixture"],
    ),
    DataFlag.TIME_VARYING_TREATMENT: FlagDescription(
        summary="Treatment level changes within a subject over time.",
        implication=(
            "Use longitudinal-TMLE / g-formula / lmtp longitudinal; standard "
            "DML on stacked rows is biased by time-varying confounding."
        ),
        routes_to=["rbridge.lmtp.shift", "rbridge.lmtp.sdr"],
    ),
    DataFlag.IMBALANCED_TREATMENT: FlagDescription(
        summary="Binary treatment prevalence outside [0.15, 0.85].",
        implication=(
            "X-learner (Künzel et al.) outperforms in this regime; check "
            "overlap diagnostics; consider trimming."
        ),
        routes_to=["python.meta.x_learner"],
    ),

    # ─── Outcome shape ──────────────────────────────────────────────
    DataFlag.BINARY_OUTCOME: FlagDescription(
        summary="Outcome is binary (0/1).",
        implication=(
            "Risk-difference / risk-ratio / odds-ratio estimands; logit-link "
            "where appropriate; E-value on the right scale."
        ),
        routes_to=["python.dr.dr_learner", "rbridge.bartcause"],
    ),
    DataFlag.CONTINUOUS_OUTCOME: FlagDescription(
        summary="Outcome is continuous-valued.",
        implication=(
            "Mean-difference ATE; standardised effect for E-value; DML linear "
            "is the defensible default."
        ),
        routes_to=["python.dml.linear", "python.dml.causal_forest"],
    ),
    DataFlag.COUNT_OUTCOME: FlagDescription(
        summary="Outcome is a non-negative integer count.",
        implication=(
            "Consider Poisson / negative-binomial nuisance; rate ratios over "
            "raw counts; ZIP if zero-inflated."
        ),
        routes_to=["python.dml.linear"],
    ),
    DataFlag.RIGHT_CENSORED_OUTCOME: FlagDescription(
        summary="Time-to-event outcome with right-censoring (event, time).",
        implication=(
            "Causal survival forest (CSF) for CATE; survRM2 / Cox / Royston-"
            "Parmar for ATE / RMST; never bin to binary unless event rare."
        ),
        routes_to=["rbridge.grf.causal_survival_forest", "rbridge.survrm2"],
    ),
    DataFlag.RARE_OUTCOME: FlagDescription(
        summary="Binary outcome prevalence <5% (or sparse-event survival).",
        implication=(
            "DR-learner with stabilised weights; Firth correction; avoid "
            "OLS / raw DML linear; power is the binding constraint."
        ),
        routes_to=["python.dr.dr_learner"],
    ),
    DataFlag.BOUNDED_OUTCOME: FlagDescription(
        summary="Outcome bounded to [0, 1] (proportion / rate).",
        implication=(
            "Logit-link DML; never raw OLS — it silently violates the bound. "
            "Beta-regression for fully Bayesian path."
        ),
        routes_to=["python.dml.linear"],
    ),
    DataFlag.ZERO_INFLATED_OUTCOME: FlagDescription(
        summary="Count outcome with >50% zeros.",
        implication=(
            "Hurdle / ZIP estimators; separate the zero-vs-nonzero question "
            "from the conditional-on-positive intensity."
        ),
        routes_to=["python.dml.linear"],
    ),
    DataFlag.COMPETING_RISKS: FlagDescription(
        summary="Survival outcome with competing-event types.",
        implication=(
            "Fine-Gray subdistribution hazards; separable-effects (Stensrud-"
            "Young-Didelez 2022) — naive cause-specific is biased."
        ),
        routes_to=["rbridge.survrm2"],
    ),
    DataFlag.REPEATED_OUTCOME: FlagDescription(
        summary="Outcome measured at multiple time points per subject.",
        implication=(
            "Use longitudinal estimators; GEE / mixed models for inference; "
            "cluster on subject."
        ),
        routes_to=["python.dml.linear"],
    ),

    # ─── Structure ──────────────────────────────────────────────────
    DataFlag.SMALL_SAMPLE: FlagDescription(
        summary="n < 100 — small-sample regime.",
        implication=(
            "OLS with HC3 robust SE is the honest default; DML / forests "
            "overfit; report wide CIs and avoid heroic CATE."
        ),
        routes_to=["python.linear.ols"],
    ),
    DataFlag.HIGH_DIMENSIONAL: FlagDescription(
        summary="Number of covariates large relative to n (p/n ≥ 0.1).",
        implication=(
            "Sparse-DML or Lasso-final stage; stability subsampling on MB; "
            "expect wider CIs."
        ),
        routes_to=["python.dml.sparse_linear"],
    ),
    DataFlag.POSITIVITY_VIOLATION: FlagDescription(
        summary="Propensity score takes values near 0 or 1 in a non-trivial fraction.",
        implication=(
            "Matching trims to support; weighting suffers tail blowup; "
            "consider Crump trimming or restricted ATT."
        ),
        routes_to=["rbridge.matchit", "rbridge.weightit"],
    ),
    DataFlag.HEAVY_MISSINGNESS: FlagDescription(
        summary="≥20% of rows missing on at least one analysis column.",
        implication=(
            "MICE / IPCW / TMLE-with-process-missing; never complete-case "
            "without checking MAR; OLS dropped from cascade."
        ),
        routes_to=["python.dr.dr_learner", "python.dml.linear"],
    ),
    DataFlag.HEAVY_CENSORING: FlagDescription(
        summary="≥30% of survival observations are right-censored.",
        implication=(
            "IPCW-aware estimators; avoid KM-based RMST without censoring "
            "covariate adjustment."
        ),
        routes_to=["rbridge.grf.causal_survival_forest"],
    ),
    DataFlag.SUSPECTED_INFORMATIVE_CENSORING: FlagDescription(
        summary="Censoring may depend on treatment / outcome (informative).",
        implication=(
            "Survival estimators with separate censoring model required; "
            "naive Kaplan-Meier / survRM2 biased."
        ),
        routes_to=["rbridge.grf.causal_survival_forest"],
    ),
    DataFlag.PANEL_STRUCTURE: FlagDescription(
        summary="Panel data — repeated units across time.",
        implication=(
            "DiD / synthetic control / fixed effects; cluster-CV for "
            "cross-fitting; use modern staggered-DiD methods."
        ),
        routes_to=["python.synth_control.ascm", "rbridge.did.callaway_santanna"],
    ),
    DataFlag.LONGITUDINAL: FlagDescription(
        summary="Subjects observed at multiple time points.",
        implication=(
            "Longitudinal-TMLE / lmtp; account for time-varying confounding; "
            "respect subject clusters in CV."
        ),
        routes_to=["python.dml.linear"],
    ),
    DataFlag.CLUSTERED: FlagDescription(
        summary="Units nested in higher-level clusters (schools, hospitals, …).",
        implication=(
            "Cluster-robust SE; cluster-CV for cross-fitting; consider "
            "multilevel TMLE; treatment-at-cluster requires different identification."
        ),
        routes_to=["python.hierarchical.dml"],
    ),
    DataFlag.NETWORK_INTERFERENCE: FlagDescription(
        summary="Units interact — treatment of one affects outcomes of neighbours.",
        implication=(
            "Aronow-Samii / Hudgens-Halloran / Sävje-Aronow-Hudgens; "
            "SUTVA is violated; report direct + spillover effects."
        ),
        routes_to=["python.interference.aronow_samii", "python.interference.savje"],
    ),
    DataFlag.SINGLE_TREATED_UNIT: FlagDescription(
        summary="Exactly one unit receives the treatment.",
        implication=(
            "Synthetic-control / Augmented SCM / SDiD; placebo + conformal "
            "inference; standard regression is misleading."
        ),
        routes_to=["python.synth_control.scm", "python.synth_control.ascm", "python.synth_control.sdid"],
    ),
    DataFlag.CROSS_SECTIONAL_SLICE: FlagDescription(
        summary="Single time slice with no repeated measurements.",
        implication=(
            "Standard backdoor adjustment landscape; no panel methods; "
            "watch for selection-into-sample bias."
        ),
        routes_to=["python.dml.linear"],
    ),

    # ─── Design hints ────────────────────────────────────────────────
    DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT: FlagDescription(
        summary="A plausible instrumental variable is named in the brief or roles.",
        implication=(
            "IV identification path opens up; check first-stage F (partial F, "
            "Olea-Pflueger); LATE not ATE without strong assumptions."
        ),
        routes_to=["rbridge.grf.instrumental_forest"],
    ),
    DataFlag.MEDIATOR_PROPOSED: FlagDescription(
        summary="A mediator on the T → Y path is named in the brief.",
        implication=(
            "NDE / NIE decomposition becomes available; front-door identification "
            "if the mediator is unconfounded by U."
        ),
        routes_to=["rbridge.mediation", "python.frontdoor"],
    ),
    DataFlag.EFFECT_MODIFICATION_OF_INTEREST: FlagDescription(
        summary="The analyst explicitly wants CATE / HTE — modifiers named.",
        implication=(
            "Causal forest at lower modifier-threshold (≥1); BLP test; "
            "honest pre-specified subgroups beat post-hoc fishing."
        ),
        routes_to=["python.dml.causal_forest", "rbridge.grf.causal_forest"],
    ),
    DataFlag.NEGATIVE_CONTROL_AVAILABLE: FlagDescription(
        summary="Negative-control exposure and/or outcome column is named.",
        implication=(
            "NCO falsification test; proximal CI (Liu-Tchetgen-Tchetgen) "
            "point-identifies under hidden confounding when both NCE and NCO exist."
        ),
        routes_to=["python.proximal.regression"],
    ),
    DataFlag.DIFF_IN_DIFF_CANDIDATE: FlagDescription(
        summary="Panel + pre/post + treated/control structure suitable for DiD.",
        implication=(
            "Modern staggered DiD (Callaway-Sant'Anna / Borusyak-Jaravel-Spiess); "
            "HonestDiD for parallel-trends robustness; never use TWFE blindly."
        ),
        routes_to=["rbridge.did_modern.callaway_santanna"],
    ),
    DataFlag.STAGGERED_ADOPTION: FlagDescription(
        summary="DiD with variable rollout times among treated units.",
        implication=(
            "TWFE produces negative weights — use Callaway-Sant'Anna, BJS "
            "imputation, or de Chaisemartin-D'Haultfoeuille."
        ),
        routes_to=["rbridge.did_modern.callaway_santanna"],
    ),
    DataFlag.IDENTIFICATION_FAILED: FlagDescription(
        summary="DoWhy / ananke reports the effect is not point-identified.",
        implication=(
            "Fall back to partial identification (autobounds), proximal CI "
            "if (NCE, NCO) exist, or refuse to report a number."
        ),
        routes_to=["identify.autobounds"],
    ),
}


def describe(flag: DataFlag) -> FlagDescription:
    """Return the canonical description for ``flag`` (raises KeyError if missing).

    The flow audit relies on this raising on undocumented flags so that
    a new enum member can never ship without a meaning."""
    return _DESCRIPTIONS[flag]


def describe_safe(flag: DataFlag) -> FlagDescription:
    """Like :func:`describe` but returns a placeholder for unknown flags
    so prompt builders never crash mid-render."""
    return _DESCRIPTIONS.get(
        flag,
        FlagDescription(
            summary=f"(no description registered for {flag.value})",
            implication="(consult docs/SPRINT_PLAN_V1.md flag taxonomy)",
            routes_to=[],
        ),
    )


def render_flags_for_prompt(flags: set[DataFlag]) -> str:
    """Render an active flag set as a bullet list with semantic explanations.

    Used by master-loop planner / critic / foundation prompts so the
    LLM sees what each flag MEANS, not just the enum name."""
    if not flags:
        return "(no flags active)"
    parts: list[str] = []
    for f in sorted(flags, key=lambda x: x.value):
        d = describe_safe(f)
        parts.append(f"  - **{f.value}** — {d.summary} _{d.implication}_")
    return "\n".join(parts)


def undocumented_flags() -> set[DataFlag]:
    """Set of enum members not currently described in this module.

    The flow audit (Sprint 9.5.1) calls this and reports red severity
    when non-empty."""
    return {f for f in DataFlag if f not in _DESCRIPTIONS}


__all__ = [
    "FlagDescription",
    "describe",
    "describe_safe",
    "render_flags_for_prompt",
    "undocumented_flags",
]
