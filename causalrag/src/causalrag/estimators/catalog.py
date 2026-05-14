"""Method catalog — a single source of truth for what every registered
estimator is FOR, presented in a form both the auto-selector and the LLM
investigator/expert can read.

The catalog is structured as a flat list of :class:`MethodSpec` rows;
each row names:

- the estimator id (matches the registry)
- the use case in one short sentence
- the flag combination it requires / excludes
- the estimand classes it produces
- the minimum sample size
- domain hints (where it's commonly applied)
- references (the canonical paper / package)

The catalog is exposed to the LLM as a compact markdown table embedded
in the investigator + expert system prompts (so the LLM knows what's
available when proposing the estimand) and consumed by the auto-cascade
selector for routing. See ``estimators.python.select.select_estimator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from causalrag.core.flags import DataFlag


@dataclass(frozen=True)
class MethodSpec:
    estimator_id: str
    backend: Literal["python", "r"]
    use_case: str
    estimands: tuple[str, ...]
    required_flags: tuple[DataFlag, ...]
    excluded_flags: tuple[DataFlag, ...]
    min_n: int
    domain_hint: str
    reference: str


CATALOG: tuple[MethodSpec, ...] = (
    # --- Python ---------------------------------------------------------
    MethodSpec(
        estimator_id="python.linear.ols",
        backend="python",
        use_case="Small-sample OLS with HC3 robust SE. Headline ATE when n < 100.",
        estimands=("ATE",),
        required_flags=(),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME,),
        min_n=5,
        domain_hint="any",
        reference="MacKinnon-White 1985 (HC3 covariance estimator)",
    ),
    MethodSpec(
        estimator_id="python.dml.linear",
        backend="python",
        use_case="DML with linear final stage and SuperLearner-stacked nuisance. Default for n ≥ 100, binary or continuous treatment, continuous outcome.",
        estimands=("ATE", "CATE"),
        required_flags=(),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT, DataFlag.LONGITUDINAL, DataFlag.PANEL_STRUCTURE),
        min_n=100,
        domain_hint="any",
        reference="Chernozhukov et al. 2018 (Double/Debiased ML)",
    ),
    MethodSpec(
        estimator_id="python.dml.causal_forest",
        backend="python",
        use_case="Non-linear CATE with effect-modifier richness. Use when ≥3 modifiers and n ≥ 500.",
        estimands=("ATE", "CATE"),
        required_flags=(),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT, DataFlag.LONGITUDINAL, DataFlag.PANEL_STRUCTURE),
        min_n=200,
        domain_hint="any",
        reference="Athey-Tibshirani-Wager 2019 (EconML implementation)",
    ),
    MethodSpec(
        estimator_id="python.dml.sparse_linear",
        backend="python",
        use_case="High-dimensional adjustment (p > sqrt(n)) with Lasso final stage. Default when HIGH_DIMENSIONAL flag is on.",
        estimands=("ATE", "CATE"),
        required_flags=(),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT, DataFlag.LONGITUDINAL, DataFlag.PANEL_STRUCTURE),
        min_n=100,
        domain_hint="any",
        reference="Belloni-Chernozhukov-Hansen 2014 (post-double-selection)",
    ),
    MethodSpec(
        estimator_id="python.meta.x_learner",
        backend="python",
        use_case="X-learner — best meta-learner for rare-treatment imbalance (prevalence < 15%).",
        estimands=("ATE", "CATE"),
        required_flags=(DataFlag.BINARY_TREATMENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.LONGITUDINAL),
        min_n=200,
        domain_hint="clinical, marketing, social science",
        reference="Künzel et al. 2019",
    ),
    MethodSpec(
        estimator_id="python.dr.dr_learner",
        backend="python",
        use_case="Doubly-robust meta-learner — robust to misspecification in either outcome or propensity model.",
        estimands=("ATE", "CATE"),
        required_flags=(DataFlag.BINARY_TREATMENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.LONGITUDINAL),
        min_n=200,
        domain_hint="any",
        reference="Kennedy 2023 (Towards Optimal Doubly Robust)",
    ),
    MethodSpec(
        estimator_id="python.bart.dml",
        backend="python",
        use_case="Bayesian causal DML with BART nuisance models. Use when calibrated posterior intervals are required.",
        estimands=("ATE", "CATE"),
        required_flags=(DataFlag.BINARY_TREATMENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.LONGITUDINAL),
        min_n=100,
        domain_hint="clinical, social science",
        reference="Chipman-George-McCulloch 2010 (BART)",
    ),
    # --- R-bridged ------------------------------------------------------
    MethodSpec(
        estimator_id="rbridge.grf.causal_forest",
        backend="r",
        use_case="Athey-Wager causal forest (reference implementation). Use for CATE with honest CIs and ≥200 obs.",
        estimands=("ATE", "CATE"),
        required_flags=(),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT),
        min_n=200,
        domain_hint="any",
        reference="grf R package; Athey-Tibshirani-Wager 2019",
    ),
    MethodSpec(
        estimator_id="rbridge.grf.causal_survival_forest",
        backend="r",
        use_case="**Censored-outcome CATE** — the gold-standard method for survival treatment-effect heterogeneity. Use when RIGHT_CENSORED_OUTCOME flag is on.",
        estimands=("RMST_CONTRAST", "ATE"),
        required_flags=(DataFlag.RIGHT_CENSORED_OUTCOME,),
        excluded_flags=(DataFlag.TIME_VARYING_TREATMENT,),
        min_n=200,
        domain_hint="clinical (survival), reliability engineering",
        reference="Cui-Athey-Tibshirani 2023 (Causal Survival Forest)",
    ),
    MethodSpec(
        estimator_id="rbridge.lmtp.shift",
        backend="r",
        use_case="**Continuous-treatment stochastic intervention** — counterfactual mean E[Y(A+δ)] for a constant shift δ. Use for dosage / dose-response analyses on a single continuous treatment.",
        estimands=("MODIFIED_TREATMENT_POLICY", "ATE"),
        required_flags=(DataFlag.CONTINUOUS_TREATMENT,),
        excluded_flags=(),
        min_n=100,
        domain_hint="pharmacology, environmental exposure, agronomy",
        reference="Díaz-Hejazi-Rudolph-van der Laan 2023 (lmtp)",
    ),
    MethodSpec(
        estimator_id="rbridge.lmtp.policy",
        backend="r",
        use_case="**Arbitrary modified treatment policy** — pass any R shift function (e.g., dose escalation schedules, treatment-rule learning). The most general policy-evaluation primitive.",
        estimands=("MODIFIED_TREATMENT_POLICY",),
        required_flags=(),
        excluded_flags=(),
        min_n=100,
        domain_hint="any (pharmacology, marketing, policy)",
        reference="Díaz et al. 2023",
    ),
    MethodSpec(
        estimator_id="rbridge.lmtp.mixture",
        backend="r",
        use_case="**Mixture-exposure intervention** — multiple simultaneous treatments shifted jointly (polypharmacy, chemical mixtures, nutrient bundles). Use when MIXTURE_EXPOSURE flag is on.",
        estimands=("MODIFIED_TREATMENT_POLICY",),
        required_flags=(DataFlag.MIXTURE_EXPOSURE,),
        excluded_flags=(),
        min_n=200,
        domain_hint="environmental, nutrition, pharmacology",
        reference="Díaz et al. 2023; quantile-g-computation literature",
    ),
    MethodSpec(
        estimator_id="rbridge.matchit",
        backend="r",
        use_case="**Propensity score matching** + post-match g-computation via marginaleffects. Preferred when POSITIVITY_VIOLATION is on (matching trims the support) or when the analyst wants transparent ATT.",
        estimands=("ATE", "ATT", "ATC"),
        required_flags=(DataFlag.BINARY_TREATMENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT),
        min_n=50,
        domain_hint="clinical effectiveness, program evaluation",
        reference="Ho-Imai-King-Stuart 2011 (MatchIt); Greifer 2024 (marginaleffects)",
    ),
    MethodSpec(
        estimator_id="rbridge.survrm2",
        backend="r",
        use_case="**Restricted Mean Survival Time contrast** — model-free survival summary when proportional hazards is unreliable. Use for ATE on survival outcomes with binary treatment.",
        estimands=("RMST_CONTRAST",),
        required_flags=(DataFlag.BINARY_TREATMENT, DataFlag.RIGHT_CENSORED_OUTCOME),
        excluded_flags=(DataFlag.TIME_VARYING_TREATMENT,),
        min_n=50,
        domain_hint="clinical (cancer trials, cardiology)",
        reference="Uno et al. 2014 (RMST contrast); survRM2 package",
    ),
    MethodSpec(
        estimator_id="rbridge.mediation",
        backend="r",
        use_case="**Causal mediation analysis** — decompose total effect into Natural Direct (NDE) and Natural Indirect (NIE) through a named mediator. Use when MEDIATOR_PROPOSED is on.",
        estimands=("NDE", "NIE"),
        required_flags=(DataFlag.MEDIATOR_PROPOSED,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME,),
        min_n=100,
        domain_hint="psychology, epidemiology, social science",
        reference="Imai-Keele-Yamamoto 2010 (mediation R package)",
    ),
    MethodSpec(
        estimator_id="rbridge.bartcause",
        backend="r",
        use_case="**Bayesian Causal Forest** (R, canonical bartCause implementation). Calibrated posterior credible intervals, free ITEs, common-support diagnostics. Preferred when Bayesian inference is required.",
        estimands=("ATE", "ATT", "ATC", "CATE"),
        required_flags=(DataFlag.BINARY_TREATMENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.LONGITUDINAL),
        min_n=50,
        domain_hint="clinical effectiveness, social science",
        reference="Hill 2011; Hahn-Murray-Carvalho 2020 (bartCause)",
    ),
    MethodSpec(
        estimator_id="rbridge.weightit",
        backend="r",
        use_case="**Propensity weighting** (GLM/GBM/CBPS/EBAL/BART/SuperLearner). ATE/ATT/ATO/ATM via weighted regression with cobalt balance diagnostics. Use when matching trims too aggressively or when weighting is preferred for transparency.",
        estimands=("ATE", "ATT", "ATC", "ATO", "ATM"),
        required_flags=(DataFlag.BINARY_TREATMENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT),
        min_n=50,
        domain_hint="any (propensity-score family)",
        reference="Greifer 2024 (WeightIt); Imai-Ratkovic 2014 (CBPS); Hainmueller 2012 (EBAL)",
    ),
    MethodSpec(
        estimator_id="rbridge.grf.instrumental_forest",
        backend="r",
        use_case="**Instrumental-variable CATE** via grf::instrumental_forest. Use when INSTRUMENTAL_CANDIDATE_PRESENT is on and an instrument column is named. Tests relevance empirically; logs exclusion as analyst assumption.",
        estimands=("LATE", "CATE"),
        required_flags=(DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT),
        min_n=500,
        domain_hint="econometrics, epidemiology with natural experiments",
        reference="grf instrumental_forest; Athey-Tibshirani-Wager 2019",
    ),
    MethodSpec(
        estimator_id="rbridge.grf.multi_arm_causal_forest",
        backend="r",
        use_case="**Multi-arm treatment** (K ≥ 3 levels). Returns pairwise contrasts vs a chosen baseline. Routes when CATEGORICAL_TREATMENT is on.",
        estimands=("ATE", "CATE"),
        required_flags=(DataFlag.CATEGORICAL_TREATMENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT),
        min_n=500,
        domain_hint="dose-finding, multi-policy comparisons",
        reference="grf multi_arm_causal_forest",
    ),
    MethodSpec(
        estimator_id="rbridge.lmtp.sdr",
        backend="r",
        use_case="**Sequentially Doubly Robust** lmtp variant. Better finite-sample coverage than TMLE when both nuisance models may be mis-specified. Preferred at smaller n.",
        estimands=("MODIFIED_TREATMENT_POLICY", "ATE"),
        required_flags=(DataFlag.CONTINUOUS_TREATMENT,),
        excluded_flags=(),
        min_n=100,
        domain_hint="any (esp. small-n stochastic interventions)",
        reference="Díaz-Hejazi-Rudolph-van der Laan 2023 (lmtp_sdr)",
    ),
    MethodSpec(
        estimator_id="rbridge.lmtp.contrast",
        backend="r",
        use_case="**Dosage contrast** E[Y(A+δ_a)] − E[Y(A+δ_b)] via lmtp_contrast. The semantically correct way to report a dose-shift effect with proper joint SE.",
        estimands=("MODIFIED_TREATMENT_POLICY", "ATE"),
        required_flags=(),
        excluded_flags=(),
        min_n=100,
        domain_hint="pharmacology, environmental",
        reference="Díaz et al. 2023 (lmtp_contrast)",
    ),
    MethodSpec(
        estimator_id="rbridge.marginaleffects.slopes",
        backend="r",
        use_case="**Average marginal slope ∂Y/∂T** for continuous treatment. Cleaner than dose-response when the analyst wants a slope per unit. Supports per-stratum slopes via ``by=``.",
        estimands=("ATE", "CATE"),
        required_flags=(DataFlag.CONTINUOUS_TREATMENT,),
        excluded_flags=(DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT),
        min_n=30,
        domain_hint="any continuous-T setting",
        reference="Arel-Bundock 2024 (marginaleffects)",
    ),
)


def catalog_markdown(backends: tuple[str, ...] = ("python", "r")) -> str:
    """Render the catalog as a markdown table the LLM can read in its
    system prompt. One line per method, with the use case + trigger flags."""
    lines = [
        "| Estimator ID | Backend | When to use it | Estimands | Trigger flags | Min n |",
        "|---|---|---|---|---|---|",
    ]
    for spec in CATALOG:
        if spec.backend not in backends:
            continue
        flags = ", ".join(f.value for f in spec.required_flags) or "—"
        estimands = ", ".join(spec.estimands)
        lines.append(
            f"| `{spec.estimator_id}` | {spec.backend} | {spec.use_case} | {estimands} | {flags} | {spec.min_n} |"
        )
    return "\n".join(lines)


def method_use_case_lookup() -> dict[str, str]:
    """Mapping ``estimator_id → use_case`` for the auto-cascade selector
    and CLI ``causalrag explain --method <id>`` lookup."""
    return {spec.estimator_id: spec.use_case for spec in CATALOG}


__all__ = ["CATALOG", "MethodSpec", "catalog_markdown", "method_use_case_lookup"]
