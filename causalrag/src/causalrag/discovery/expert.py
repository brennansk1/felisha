"""Stage 1e — domain expert brief (PDD §7.5).

Reinvokes the LLM at a higher level of abstraction. Receives the full
investigator report + deterministic profile and produces a Domain Expert Brief
covering plausible treatments, outcomes, confounders, mediators, effect
modifiers, unmeasured-confounder candidates, K candidate DAGs, and domain
identification warnings.

Every confounder claim is statistically validated where possible: a
conditional-independence test on the data flags contradictions for the analyst
without silently overriding the LLM (PDD §7.5: "if the test contradicts the
LLM's claim with high evidence, the contradiction is logged and the analyst is
shown both views").
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.data.profiler import DatasetProfile
from causalrag.discovery.investigator import InvestigatorReport
from causalrag.llm.honesty import with_honesty
from causalrag.llm.ollama_client import LLMResponse, OllamaClient


class TreatmentCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    rationale: str
    suitability: float = Field(..., ge=0.0, le=1.0)
    typical_questions: list[str] = Field(default_factory=list)


class OutcomeCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    rationale: str
    measurement_notes: str | None = None
    censoring_notes: str | None = None


class ConfounderClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    treatment: str
    outcome: str
    confounders: list[str]
    rationale: str | None = None


class UnmeasuredConfounder(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    reason: str
    observed_proxies: list[str] = Field(default_factory=list)


class CandidateDAGSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank: int = Field(..., ge=1)
    rationale: str
    edges: list[tuple[str, str]]
    distinguishing_edges: list[tuple[str, str]] = Field(default_factory=list)


class DomainExpertBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain_summary: str
    treatments: list[TreatmentCandidate]
    outcomes: list[OutcomeCandidate]
    confounders: list[ConfounderClaim] = Field(default_factory=list)
    mediators: list[str] = Field(default_factory=list)
    effect_modifiers: list[str] = Field(default_factory=list)
    unmeasured_confounders: list[UnmeasuredConfounder] = Field(default_factory=list)
    candidate_dags: list[CandidateDAGSpec]
    identification_warnings: list[str] = Field(default_factory=list)


class ConfounderContradiction(BaseModel):
    """Per (treatment, outcome, confounder) row of the validation log."""

    model_config = ConfigDict(extra="forbid")
    treatment: str
    outcome: str
    confounder: str
    p_value_marginal: float
    p_value_conditional_on_others: float | None = None
    verdict: str  # "supported" | "contradicted" | "inconclusive"


_SYSTEM_PROMPT = (
    "You are a senior domain expert and methodologist advising a causal-inference "
    "team. Your output anchors every downstream Petersen-van der Laan Causal "
    "Roadmap step (Step 1 Question, Step 2 Causal Model, Step 5 Identification). "
    "You will be given a deterministic statistical profile, a Stage-1c column-level "
    "investigator report, and (optionally) the user's research question.\n\n"
    "Produce a structured Domain Expert Brief with:\n"
    "- A one-paragraph domain summary identifying the apparent study setting "
    "  AND its standard causal-inference pitfalls (e.g. for clinical claims data: "
    "  physician selection, indication bias, immortal time bias).\n"
    "- A ranked list of plausible **treatment** columns. Rank by causal-effect "
    "  suitability: temporal position (pre-outcome), variability, and ability to "
    "  define a counterfactual.\n"
    "- A ranked list of plausible **outcome** columns. Note measurement quality, "
    "  censoring properties, and whether the outcome is clinically/economically "
    "  meaningful or a proxy.\n"
    "- A **confounders** list per (treatment, outcome) pair. Only variables that "
    "  plausibly affect BOTH treatment AND outcome qualify. Exclude post-treatment "
    "  variables (collider or mediator bias), identifiers, and tautological "
    "  outcome derivatives.\n"
    "- A **mediators** list: variables ON the causal pathway from treatment to "
    "  outcome. Adjusting for these answers a different question (NDE/NIE), so be "
    "  precise.\n"
    "- An **effect_modifiers** list: variables that moderate the treatment effect "
    "  without confounding it (typically pre-treatment, NOT on the path).\n"
    "- **unmeasured_confounders**: variables you would expect to matter but cannot "
    "  see in this dataset. Be specific (e.g. 'baseline disease severity' not "
    "  'unmeasured factors') and name observed proxies that might partially capture "
    "  them.\n"
    "- **candidate_dags**: K plausible DAGs as edge lists, each ranked by your "
    "  confidence. Use only columns from the investigator report. Distinguish the "
    "  DAGs with edges they disagree on (mediator-vs-confounder placements, "
    "  contested directions).\n"
    "- **identification_warnings**: prose alerts about identification hazards "
    "  specific to this domain (e.g. 'in target-trial emulation, treatment-assignment "
    "  bias if grace period is not enforced'; 'in marketing, ad targeting creates "
    "  selection on potential outcomes').\n\n"
    "RULES OF GOOD CAUSAL REASONING:\n"
    "1. Backdoor criterion (Pearl 2009): the adjustment set must block every "
    "   non-causal path from T to Y without blocking causal paths.\n"
    "2. Never adjust for descendants of the treatment (M-bias, collider bias).\n"
    "3. Mediators and effect modifiers are NOT confounders; treat them separately.\n"
    "4. An instrument must satisfy relevance (Z affects T) AND exclusion (Z affects "
    "   Y only through T). Most plausible-sounding 'instruments' violate exclusion.\n"
    "5. When in doubt about a directional edge, surface the ambiguity by including "
    "   both orientations in different candidate DAGs rather than picking one.\n\n"
    "ESTIMATOR CATALOG (the methods the pipeline can route to). Use the "
    "`identification_warnings` field to recommend which one matches the analysis:\n"
    "{CATALOG_TABLE}\n\n"
    "Heuristics for picking a method:\n"
    "- If the outcome is right-censored survival → ``rbridge.grf.causal_survival_forest`` or ``rbridge.survrm2``.\n"
    "- If treatment is continuous AND the question is dose-response → ``rbridge.lmtp.shift`` or ``lmtp.policy``.\n"
    "- If there are MULTIPLE simultaneous treatments (mixture exposure) → ``rbridge.lmtp.mixture``.\n"
    "- If a mediator is named and the user wants direct vs indirect effects → ``rbridge.mediation``.\n"
    "- If POSITIVITY_VIOLATION is likely (treated/untreated arms don't overlap on key covariates) → ``rbridge.matchit`` (matching trims the unsupported region).\n"
    "- High-dimensional adjustment (p > sqrt(n)) → ``python.dml.sparse_linear``.\n"
    "- Rare binary treatment (prevalence < 15%) → ``python.meta.x_learner``.\n"
    "- Calibrated Bayesian credible intervals needed → ``python.bart.dml``.\n\n"
    "Return ONLY a JSON object conforming to the provided schema. Every column "
    "name you reference MUST appear in the investigator report verbatim. Vague "
    "unmeasured-confounder claims ('unobserved variables') will be discarded; be "
    "specific."
)


def _build_prompt(
    profile: DatasetProfile,
    investigator: InvestigatorReport,
    research_question: str | None,
    k: int = 3,
) -> str:
    parts = [
        "## Dataset overview",
        f"{profile.n_rows} rows × {profile.n_cols} columns. "
        f"Domain tag from Stage 1c investigator: {investigator.domain_tag}.",
        "",
        "## Per-column investigator output",
    ]
    for col in investigator.columns:
        parts.append(
            f"- {col.column}: {col.domain_meaning} "
            f"[temporal={col.temporal_position}, role_hint={col.proposed_role}, "
            f"watch={','.join(col.watch_for) or '-'}]"
        )

    # Inject deterministic statistical context that the LLM cannot derive itself
    # but that materially constrains plausible DAGs.
    parts.append("\n## Deterministic statistical findings (Stage 1b — authoritative)")
    parts.append(_temporal_lattice(investigator))
    if profile.column_pairs_high_corr:
        parts.append(
            "\nHigh marginal correlations (|r|≥0.9) — likely shared causes or "
            "deterministic relations; pick ONE per cluster to enter the DAG:"
        )
        for a, b, r in profile.column_pairs_high_corr[:20]:
            parts.append(f"  - {a} ↔ {b}: |r|={r:.3f}")
    if profile.censoring_pairs:
        parts.append(
            "\nSuspected (time, event) censoring pairs — treat as one survival "
            "outcome, not two columns:"
        )
        for t, e in profile.censoring_pairs:
            parts.append(f"  - ({t}, {e})")

    high_missing = [c.name for c in profile.columns if c.missing_rate > 0.20]
    if high_missing:
        parts.append(
            "\nHigh-missingness columns (>20%): "
            + ", ".join(high_missing)
            + ". Avoid relying on these as confounders unless multiple imputation "
            "is feasible."
        )

    constant_cols = [c.name for c in profile.columns if c.constant]
    if constant_cols:
        parts.append("\nConstant columns (zero variance, exclude): " + ", ".join(constant_cols))

    if profile.missingness_clusters:
        parts.append(
            "\nMissingness clusters (columns that go missing together — informative "
            "for MAR/MNAR reasoning, may share an upstream cause):"
        )
        for cluster in profile.missingness_clusters[:5]:
            parts.append("  - {" + ", ".join(cluster) + "}")

    if profile.n_exact_duplicate_rows:
        parts.append(
            f"\n⚠ {profile.n_exact_duplicate_rows} exact-duplicate rows detected "
            "(after dropping identifier columns). Effective sample size may be "
            "smaller than nominal; treat with caution."
        )

    fmt_buckets: dict[str, list[str]] = {}
    for col, fmt in profile.string_formats.items():
        fmt_buckets.setdefault(fmt, []).append(col)
    interesting = [
        ("identifier_like", "Identifier-like text columns (do NOT include as adjustment): "),
        ("date_like", "Date-encoded text columns (recommend parsing to datetime first): "),
        ("free_text", "Free-text columns (require NLP feature extraction before use): "),
        ("email_like", "Email-format columns (PII): "),
        ("zip_like", "ZIP-code columns (geographic confounding candidate): "),
    ]
    for fmt_key, label in interesting:
        if fmt_buckets.get(fmt_key):
            parts.append("\n" + label + ", ".join(fmt_buckets[fmt_key]))

    # Domain-specific causal-inference pitfalls
    domain_hint = _domain_pitfalls(investigator.domain_tag)
    if domain_hint:
        parts.append(f"\n## Domain-specific identification hazards ({investigator.domain_tag})")
        parts.append(domain_hint)

    if research_question:
        parts.append(f"\n## User research question\n{research_question}")

    parts.append(
        f"\n## Task\nReturn a JSON DomainExpertBrief with exactly K={k} candidate "
        "DAGs ranked by plausibility (rank=1 most plausible). Every edge "
        "(source, target) must reference columns that appear in the investigator "
        "report. Use the high-correlation hints and temporal lattice above — "
        "edges that violate temporal order will be rejected automatically."
    )
    return "\n".join(parts)


def _temporal_lattice(investigator: InvestigatorReport) -> str:
    """Emit a compact temporal-position lattice so the LLM can read off
    which edges are temporally permitted (causes precede effects)."""
    buckets: dict[str, list[str]] = {
        "baseline": [],
        "pre_treatment": [],
        "treatment_era": [],
        "post_treatment": [],
        "outcome": [],
        "unknown": [],
    }
    for col in investigator.columns:
        buckets.setdefault(col.temporal_position, []).append(col.column)
    lines = ["Temporal lattice (edges must flow left → right):"]
    for bucket in ("baseline", "pre_treatment", "treatment_era", "post_treatment", "outcome"):
        if buckets[bucket]:
            lines.append(f"  [{bucket}] {', '.join(buckets[bucket])}")
    if buckets["unknown"]:
        lines.append(f"  [unknown — treat conservatively] {', '.join(buckets['unknown'])}")
    return "\n".join(lines)


_DOMAIN_PITFALLS: dict[str, str] = {
    "clinical": (
        "- Physician/prescriber selection: treatment is heavily confounded by "
        "  unmeasured clinical judgment (severity, comorbidity).\n"
        "- Indication bias: the reason for prescribing is itself prognostic.\n"
        "- Immortal time bias: time between cohort entry and treatment start "
        "  cannot be assigned to either arm.\n"
        "- Healthy-adherer effect: treated patients differ in unmeasured "
        "  health-conscious behaviors.\n"
        "- Survival censoring is usually informative when censoring is correlated "
        "  with treatment receipt."
    ),
    "financial": (
        "- Survivor bias in firm/account-level data: failed entities exit the "
        "  panel and are missing not at random.\n"
        "- Look-ahead bias: outcomes that 'predict' but are mechanically derived "
        "  after the predictor.\n"
        "- Self-selection into treatment products (credit, investment products) "
        "  is driven by unobserved risk tolerance."
    ),
    "marketing": (
        "- Ad-targeting creates selection on potential outcomes: the platform "
        "  served the ad precisely to users predicted to respond.\n"
        "- Reverse causality between engagement (treatment) and outcome (revenue) "
        "  when both are measured contemporaneously.\n"
        "- Spillover / network interference if users influence each other."
    ),
    "social_science": (
        "- Self-selection into the treatment program is driven by latent "
        "  motivation, ability, or wealth.\n"
        "- Attrition is non-random: dropouts differ in outcome-relevant ways.\n"
        "- Spillover within social networks violates SUTVA."
    ),
    "education": (
        "- School/teacher selection creates clustered confounding.\n"
        "- Cohort effects make longitudinal comparisons across years risky.\n"
        "- Test-score outcomes have ceiling/floor effects."
    ),
    "web_analytics": (
        "- Bots, repeat sessions, and ad-blockers create non-random missingness "
        "  in instrumentation.\n"
        "- Engagement spirals (treatment depends on prior outcome).\n"
        "- A/A test failure indicates platform-level bias even before treatment."
    ),
    "manufacturing": (
        "- Machine/operator clustering; defects are non-independent.\n"
        "- Process drift means time itself is a confounder.\n"
        "- Outliers from rare-event defects can dominate effect estimates."
    ),
    "environmental": (
        "- Spatial autocorrelation: nearby observations share unmeasured "
        "  weather/geology covariates.\n"
        "- Temporal trends (climate, urbanization) confound long panels.\n"
        "- Measurement instruments drift over time."
    ),
}


def _domain_pitfalls(tag: str) -> str:
    return _DOMAIN_PITFALLS.get(tag, "")


def run_domain_expert(
    *,
    df: pd.DataFrame | None,
    profile: DatasetProfile,
    investigator: InvestigatorReport,
    client: OllamaClient,
    research_question: str | None = None,
    k: int = 3,
) -> tuple[DomainExpertBrief, list[ConfounderContradiction], LLMResponse]:
    prompt = _build_prompt(profile, investigator, research_question, k=k)
    # Inject the live method catalog so the expert knows what's available.
    from causalrag.estimators.catalog import catalog_markdown

    system = _SYSTEM_PROMPT.replace("{CATALOG_TABLE}", catalog_markdown())
    response = client.parse(
        prompt=prompt,
        schema=DomainExpertBrief,
        system=with_honesty(system),
        json_schema=DomainExpertBrief.model_json_schema(),
    )
    brief = response.parsed
    assert isinstance(brief, DomainExpertBrief)

    _validate_brief_columns(brief, profile)
    contradictions = (
        _validate_confounders_statistically(brief, df) if df is not None else []
    )
    return brief, contradictions, response


def _validate_brief_columns(brief: DomainExpertBrief, profile: DatasetProfile) -> list[str]:
    """Semantic Layer 3 check (PDD §16.6): every column the brief names must
    exist in the deterministic profile.

    Unknown column references are silently dropped from the brief and the
    sites are returned for the analyst-decision ledger. The brief is mutated
    in place — bad treatments, outcomes, mediators, effect modifiers, and
    DAG edges are removed; bad confounder names are stripped from each claim.
    This trades strict enforcement for graceful degradation: a single typo in
    one DAG edge name should not crash the entire pipeline.
    """
    known = {c.name for c in profile.columns}
    dropped: list[str] = []

    def _filter(label: str, names: list[str]) -> list[str]:
        keep, drop = [], []
        for n in names:
            (keep if n in known else drop).append(n)
        if drop:
            dropped.append(f"{label}: dropped {drop}")
        return keep

    brief.treatments = [t for t in brief.treatments if t.column in known]
    brief.outcomes = [o for o in brief.outcomes if o.column in known]
    brief.mediators = _filter("mediators", brief.mediators)
    brief.effect_modifiers = _filter("effect_modifiers", brief.effect_modifiers)
    cleaned_claims = []
    for claim in brief.confounders:
        if claim.treatment not in known or claim.outcome not in known:
            dropped.append(
                f"confounder-claim.{claim.treatment}→{claim.outcome}: "
                "dropped (treatment or outcome unknown)"
            )
            continue
        claim.confounders = _filter(
            f"confounders.[{claim.treatment}→{claim.outcome}]", claim.confounders
        )
        cleaned_claims.append(claim)
    brief.confounders = cleaned_claims

    cleaned_dags = []
    for dag in brief.candidate_dags:
        ok_edges = []
        for s, t in dag.edges:
            if s in known and t in known:
                ok_edges.append((s, t))
            else:
                dropped.append(f"dag#{dag.rank}.edge {s}→{t}: dropped (unknown column)")
        dag.edges = ok_edges
        if dag.edges:
            cleaned_dags.append(dag)
    brief.candidate_dags = cleaned_dags
    return dropped


def _validate_confounders_statistically(
    brief: DomainExpertBrief, df: pd.DataFrame
) -> list[ConfounderContradiction]:
    """Statistical Layer 4 check (PDD §7.5): for each claimed confounder, run a
    marginal association test against treatment and outcome. If both arms
    associate with p < 0.05, the confounder is *supported*; if neither
    associates, the confounder is *contradicted*. The test is intentionally
    weak — we surface views rather than overriding the LLM."""
    out: list[ConfounderContradiction] = []
    from scipy.stats import chi2_contingency, kendalltau

    for claim in brief.confounders:
        if claim.treatment not in df.columns or claim.outcome not in df.columns:
            continue
        t_series = df[claim.treatment].dropna()
        y_series = df[claim.outcome].dropna()
        for c in claim.confounders:
            if c not in df.columns:
                continue
            c_series = df[c].dropna()
            joined = df[[claim.treatment, claim.outcome, c]].dropna()
            if len(joined) < 30:
                out.append(
                    ConfounderContradiction(
                        treatment=claim.treatment,
                        outcome=claim.outcome,
                        confounder=c,
                        p_value_marginal=1.0,
                        verdict="inconclusive",
                    )
                )
                continue
            try:
                if joined[claim.treatment].nunique() <= 10 and joined[c].nunique() <= 10:
                    _, p_tc, _, _ = chi2_contingency(
                        pd.crosstab(joined[c], joined[claim.treatment])
                    )
                else:
                    p_tc = float(
                        kendalltau(joined[c], joined[claim.treatment]).pvalue or 1.0
                    )
                if joined[claim.outcome].nunique() <= 10 and joined[c].nunique() <= 10:
                    _, p_yc, _, _ = chi2_contingency(
                        pd.crosstab(joined[c], joined[claim.outcome])
                    )
                else:
                    p_yc = float(
                        kendalltau(joined[c], joined[claim.outcome]).pvalue or 1.0
                    )
            except Exception:
                p_tc = 1.0
                p_yc = 1.0

            verdict = (
                "supported"
                if p_tc < 0.05 and p_yc < 0.05
                else "contradicted"
                if p_tc > 0.5 and p_yc > 0.5
                else "inconclusive"
            )
            out.append(
                ConfounderContradiction(
                    treatment=claim.treatment,
                    outcome=claim.outcome,
                    confounder=c,
                    p_value_marginal=min(p_tc, p_yc),
                    verdict=verdict,
                )
            )
    return out


def flags_from_brief(
    brief: DomainExpertBrief,
    investigator: InvestigatorReport | None = None,
) -> set[DataFlag]:
    """Emit semantic DataFlags from the LLM brief.

    Currently emits:
    - ``INSTRUMENTAL_CANDIDATE_PRESENT`` when the brief lists an unmeasured
      confounder that looks IV-shaped OR the investigator proposed
      ``VariableRole.INSTRUMENT`` for any column.
    - ``MEDIATOR_PROPOSED`` when the brief's ``mediators`` list is non-empty.
    - ``EFFECT_MODIFICATION_OF_INTEREST`` when the brief lists any
      ``effect_modifiers`` (analyst/LLM explicitly wants CATE).
    """
    flags: set[DataFlag] = set()
    if brief.mediators:
        flags.add(DataFlag.MEDIATOR_PROPOSED)
    if brief.effect_modifiers:
        flags.add(DataFlag.EFFECT_MODIFICATION_OF_INTEREST)

    iv_in_brief = any(
        "instrument" in u.reason.lower() or "iv" in u.reason.lower().split()
        for u in brief.unmeasured_confounders
    )
    iv_in_investigator = bool(
        investigator
        and any(c.proposed_role == VariableRole.INSTRUMENT for c in investigator.columns)
    )
    if iv_in_brief or iv_in_investigator:
        flags.add(DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT)
    return flags


def brief_to_candidate_graphs(brief: DomainExpertBrief) -> tuple[CausalGraph, ...]:
    """Project the candidate DAG specs into CausalGraph objects.

    Roles are seeded from the brief's treatment/outcome lists and the
    confounder claims; remaining nodes default to AUXILIARY (Step 2 refines).
    """
    if not brief.candidate_dags:
        return ()
    treatments = {t.column for t in brief.treatments}
    outcomes = {o.column for o in brief.outcomes}
    mediators = set(brief.mediators)
    modifiers = set(brief.effect_modifiers)
    confounders: set[str] = set()
    for claim in brief.confounders:
        confounders.update(claim.confounders)

    graphs: list[CausalGraph] = []
    for spec in sorted(brief.candidate_dags, key=lambda d: d.rank):
        nodes: list[str] = []
        seen: set[str] = set()
        for s, t in spec.edges:
            for n in (s, t):
                if n not in seen:
                    seen.add(n)
                    nodes.append(n)
        roles: dict[str, VariableRole] = {}
        for n in nodes:
            if n in treatments:
                roles[n] = VariableRole.TREATMENT
            elif n in outcomes:
                roles[n] = VariableRole.OUTCOME
            elif n in mediators:
                roles[n] = VariableRole.MEDIATOR
            elif n in modifiers:
                roles[n] = VariableRole.EFFECT_MODIFIER
            elif n in confounders:
                roles[n] = VariableRole.CONFOUNDER
            else:
                roles[n] = VariableRole.AUXILIARY
        graphs.append(
            CausalGraph(
                nodes=tuple(nodes),
                edges=tuple(
                    CausalEdge(source=s, target=t, llm_proposed=True) for s, t in spec.edges
                ),
                roles=roles,
                rank=spec.rank,
            )
        )
    return tuple(graphs)
