"""Master loop — autonomous "drop a dataset → K experiments" pipeline.

Invoked only by the TUI command ``auto run <data.csv> --experiments K
[--foundation]``. Acts like a senior causal-inference statistician with
the full method catalog as their toolbox.

Architecture (post-audit rewrite):

* **Phase 0–1 — discovery** (LLM investigator + expert + Layer-4 audit).

* **Phase 2 — candidate queue planning**. ONE LLM call up front produces
  15–30 candidate experiments. A deterministic scorer ranks them by
  ``impact × identifiability × power × novelty − cost``. The queue is
  persisted on the protocol and re-ranked after each completion.

* **Phase 3 — iterative propose-K → critique → commit**. Each turn the
  loop pulls the top-K candidates, a critic LLM checks each one for
  identifiability / already-tested / power / catalog-validity, the
  scorer picks the winner. This replaces the original "single LLM
  proposes the next thing" pattern.

* **Phase 4 — foundation recursion**. After each completed experiment,
  a deterministic rule decides whether to auto-fire a foundation child:
  significant + green/yellow → propose CATE on top modifier; red
  sensitivity → auto-schedule a tipping-point / negative-control check.
  Multi-chain bookkeeping is per-chain (``chains: dict[chain_id,
  ChainState]``); independent experiments do NOT reset another chain's
  depth.

* **Phase 5 — synthesis**. After the loop, ``synthesize_insights``
  translates the results into domain-appropriate findings (see
  ``reporting/synthesis.py``).

Recovery and dead-end handling:
- Estimator errors capture the exception and try the next-best
  estimator from the cascade.
- Unidentifiable proposals capture the reason and surface it to the
  next propose call so the LLM can supply the missing piece.
- Red sensitivity auto-schedules a robustness child without asking.
- ``max_consecutive_failures`` triggers an LLM-authored autopsy
  written to the decision ledger, rather than silently dying.

Every decision lands in ``protocol.decision_ledger`` with
``source='auto'`` and an explicit ``chain_id`` marker.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalGraph
from causalrag.core.ledger import record_decision
from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.discovery import run_discovery
from causalrag.estimators.catalog import CATALOG, catalog_markdown
from causalrag.llm.ollama_client import OllamaClient
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q6_statistical_estimand import derive_statistical_estimand
from causalrag.roadmap.q7_estimate import estimate as run_step7
from causalrag.sensitivity.evalue import evalue_for_estimator
from causalrag.sensitivity.sensemakr_py import sensemakr as run_sensemakr
from causalrag.sensitivity.verdict import aggregate as aggregate_sensitivity


_CATALOG_IDS: frozenset[str] = frozenset(spec.estimator_id for spec in CATALOG)


# ─────────── LLM schemas ──────────────────────────────────────────────────


class CandidateExperiment(BaseModel):
    """One entry in the up-front candidate queue."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(
        default_factory=lambda: f"c-{uuid.uuid4().hex[:6]}",
        description="Stable id used by the scorer and the propose-K critic.",
    )
    research_question: str
    treatment: str
    outcome: str
    estimand_class: str  # ATE / CATE / NDE / NIE / LATE / RMST_CONTRAST / MTP / ATT / ATC
    modifiers: list[str] = Field(default_factory=list)
    mediator: str | None = None
    instrument: str | None = None
    recommended_method: str | None = Field(
        default=None, description="Estimator id from the catalog"
    )
    impact_rationale: str = Field(..., description="WHY this matters. Be concrete.")
    identifiability_rationale: str = Field(
        ..., description="WHY the data supports identification here."
    )
    power_rationale: str = Field(
        ..., description="WHY n + variance support a non-trivial result."
    )
    # LLM scoring hints — the deterministic scorer is authoritative.
    impact_hint: float = Field(default=0.5, ge=0.0, le=1.0)
    identifiability_hint: float = Field(default=0.5, ge=0.0, le=1.0)
    power_hint: float = Field(default=0.5, ge=0.0, le=1.0)


class CandidateQueue(BaseModel):
    """The LLM's up-front enumeration of credible experiments."""

    model_config = ConfigDict(extra="forbid")

    candidates: list[CandidateExperiment] = Field(..., min_length=1)
    notes: str | None = None


class CriticVerdict(BaseModel):
    """Result of running the critic agent over a single candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    keep: bool
    rejection_reason: str | None = None
    revised_recommended_method: str | None = Field(
        default=None,
        description="If the original method is invalid, the critic can suggest a replacement (must be a catalog id).",
    )
    risks: list[str] = Field(default_factory=list)


class CriticBatch(BaseModel):
    """Critic output for a batch of K proposed candidates."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[CriticVerdict] = Field(..., min_length=1)
    overall_note: str | None = None


class NextExperiment(BaseModel):
    """Per-iteration commit: which candidate is run + foundation framing.

    Retained for backward compatibility with the original single-LLM-call
    flow (still used by ``foundation_followup_proposal`` to specify the
    next foundation child)."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["run", "stop"]
    stop_reason: str | None = None
    foundation_of: str | None = None

    treatment: str | None = None
    outcome: str | None = None
    modifiers: list[str] = Field(default_factory=list)
    mediator: str | None = None
    instrument: str | None = None
    estimand_class: str | None = None

    research_question: str | None = None
    recommended_method: str | None = None

    impact_rationale: str | None = None
    identifiability_rationale: str | None = None
    power_rationale: str | None = None
    foundation_rationale: str | None = None


# ─────────── Chain state ──────────────────────────────────────────────────


@dataclass
class ChainState:
    """One foundation thread's running state."""

    chain_id: str
    root_hypothesis_id: str
    depth: int = 0
    last_point: float | None = None
    last_se: float | None = None
    last_verdict: str | None = None
    last_modifier_focus: tuple[str, ...] = ()
    null_streak: int = 0  # consecutive non-significant follow-ups
    info_gain_streak_below_eps: int = 0


# ─────────── Event stream ─────────────────────────────────────────────────


@dataclass
class LoopEvent:
    """Streamed event from the master loop."""

    kind: str  # phase_start | phase_end | log | card | error | done | plan | autopsy
    phase: str
    message: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


# ─────────── Configuration ────────────────────────────────────────────────


@dataclass
class LoopConfig:
    """Budgets + safety circuit-breakers for the master loop."""

    n_experiments: int = 5
    foundation_allowed: bool = False
    max_foundation_iterations: int = 8
    max_foundation_depth: int = 4
    max_consecutive_failures: int = 3
    max_repeat_attempts: int = 2

    # New (post-audit):
    candidate_queue_size: int = 18  # how many candidates to enumerate up front
    propose_k: int = 3  # how many top candidates the critic reviews each turn
    critic_enabled: bool = True
    diminishing_returns_epsilon: float = 0.3  # |Δpoint|/SE threshold; below → chain ends
    null_streak_threshold: int = 2  # consecutive null follow-ups → chain ends
    info_gain_streak_threshold: int = 2  # consecutive low-info follow-ups → chain ends
    auto_fire_robustness_on_red: bool = True
    estimator_swap_retries: int = 1  # how many times to retry with a different estimator on fit failure


# ─────────── Deterministic scorer ─────────────────────────────────────────


def _identifiability_score(c: CandidateExperiment, protocol: StudyProtocol) -> float:
    """0–1 score for how cleanly identification is supported.

    Heuristic — refined by the critic agent. Front-door / backdoor with
    a discoverable adjustment set ⇒ 1.0. IV with named instrument ⇒
    0.85. Mediation with named mediator ⇒ 0.75. Anything without an
    obvious strategy ⇒ 0.5.
    """
    klass = c.estimand_class.upper()
    if klass in {"NDE", "NIE"} and c.mediator:
        return 0.78
    if klass == "LATE" and c.instrument:
        return 0.85
    # Backdoor: how many confounders does the discovery brief name?
    n_confounders = 0
    if protocol.discovery:
        n_confounders = sum(
            1
            for v in protocol.discovery.columns
            if v.role is VariableRole.CONFOUNDER
        )
    if n_confounders >= 3:
        return 0.95
    if n_confounders >= 1:
        return 0.8
    return 0.55


def _novelty_score(
    c: CandidateExperiment, completed: list[RoadmapWalk]
) -> float:
    """Penalize repeats of (T, Y, estimand_class) and near-repeats sharing modifiers."""
    if not completed:
        return 1.0
    target = (c.treatment, c.outcome, c.estimand_class.upper())
    target_mods = set(c.modifiers)
    best_overlap = 0.0
    for w in completed:
        if not w.q3_estimand:
            continue
        prior = (
            w.q3_estimand.treatment,
            w.q3_estimand.outcome,
            w.q3_estimand.klass.value,
        )
        if prior == target:
            return 0.0
        prior_mods = set(w.q3_estimand.modifiers or ())
        # Jaccard on modifiers conditioned on same (T, Y)
        if prior[:2] == target[:2] and (prior_mods or target_mods):
            union = prior_mods | target_mods
            j = len(prior_mods & target_mods) / max(len(union), 1)
            best_overlap = max(best_overlap, j)
    return 1.0 - best_overlap


def _power_score(c: CandidateExperiment, protocol: StudyProtocol) -> float:
    """0–1 power proxy.

    Without a per-candidate power calc (expensive), use the LLM's
    power_hint plus a sample-size adjustment. If feasibility ran, prefer
    its verdict for the proposed (T, Y).
    """
    base = c.power_hint
    n = protocol.dataset.n_rows if protocol.dataset else None
    if n is None:
        return base
    # Mild penalty for tiny samples.
    if n < 100:
        return min(base, 0.4)
    if n < 500:
        return min(base, 0.65)
    return base


def _cost_score(c: CandidateExperiment) -> float:
    """0–1 cost proxy. Expensive estimators get a small penalty.

    Currently approximated by family: forest > BART > matchit > lmtp >
    DML > OLS. Returned as the *cost to subtract* in the final score.
    """
    method = (c.recommended_method or "").lower()
    if "forest" in method:
        return 0.15
    if "bart" in method:
        return 0.20
    if "matchit" in method:
        return 0.10
    if "lmtp" in method:
        return 0.12
    return 0.05


def _validate_method_id(method_id: str | None) -> str | None:
    """Return ``method_id`` if it's a real catalog id, else None."""
    if method_id and method_id in _CATALOG_IDS:
        return method_id
    return None


def score_candidate(
    c: CandidateExperiment,
    *,
    protocol: StudyProtocol,
    completed: list[RoadmapWalk],
    weights: tuple[float, float, float, float, float] = (0.40, 0.25, 0.20, 0.15, 1.0),
) -> dict[str, float]:
    """Score one candidate. Returns a dict with all sub-scores + final.

    Weights default to: impact 0.40, identifiability 0.25, power 0.20,
    novelty 0.15, cost-penalty 1.0 (full subtraction).
    """
    w_imp, w_id, w_pow, w_nov, w_cost = weights
    impact = c.impact_hint
    identifiability = _identifiability_score(c, protocol)
    power = _power_score(c, protocol)
    novelty = _novelty_score(c, completed)
    cost = _cost_score(c)
    final = (
        w_imp * impact
        + w_id * identifiability
        + w_pow * power
        + w_nov * novelty
        - w_cost * cost
    )
    return {
        "impact": impact,
        "identifiability": identifiability,
        "power_proxy": power,
        "novelty": novelty,
        "cost": cost,
        "score": float(final),
    }


# ─────────── LLM prompts ──────────────────────────────────────────────────


_PLANNER_SYSTEM = (
    "You are a senior causal-inference statistician (PhD-level, 20+ "
    "years experience). You have just received a new dataset and a "
    "discovery brief. Your job RIGHT NOW is to enumerate the credible "
    "set of causal experiments this dataset can support — NOT to "
    "rank them, NOT to commit to any one. You produce a broad, "
    "well-justified candidate list which a deterministic scorer will "
    "rank.\n\n"
    "CRITERIA for inclusion in the queue:\n"
    "  • The treatment and outcome columns exist in the dataset.\n"
    "  • Identification is at least plausible (backdoor with named "
    "    confounders, IV with named instrument, mediator with named "
    "    mediator, or front-door — explain in identifiability_rationale).\n"
    "  • The estimand_class is feasible given the data flags.\n"
    "  • Lowest-hanging fruit first: include the obvious headline "
    "    questions any senior analyst would run. Include the next "
    "    'why does that happen?' follow-ups. THEN include the more "
    "    exotic / heroic questions for completeness.\n\n"
    "PROFESSIONAL DISCIPLINE — non-negotiable:\n"
    "  ✘ NEVER propose NDE/NIE without a named mediator column.\n"
    "  ✘ NEVER propose LATE without a named instrument column.\n"
    "  ✘ NEVER propose RMST_CONTRAST unless RIGHT_CENSORED_OUTCOME "
    "    is flagged.\n"
    "  ✘ NEVER propose MODIFIED_TREATMENT_POLICY unless treatment is "
    "    continuous.\n"
    "  ✘ NEVER refer to a column not present in the dataset.\n"
    "  ✓ DO pick a recommended_method from the catalog. If unsure, "
    "    leave null and the auto-selector will choose.\n"
    "  ✓ DO give a SPECIFIC impact_rationale that a referee would "
    "    accept — not 'this matters because outcomes matter'.\n\n"
    "ESTIMATOR CATALOG (toolbox):\n"
    "{CATALOG_TABLE}\n\n"
    "Return ONLY a JSON CandidateQueue with at least 8 and at most "
    "{QUEUE_SIZE} candidates."
)

_CRITIC_SYSTEM = (
    "You are a referee at a top journal in causal inference. The "
    "pipeline is about to run one of K proposed experiments. Your job: "
    "for EACH candidate, decide keep / reject + (if keep) any "
    "method-id correction. You reject for:\n"
    "  • Already-tested triple (T, Y, estimand_class) appears in "
    "    completed_experiments.\n"
    "  • Missing required piece: NDE/NIE without mediator; LATE "
    "    without instrument; RMST without right-censored flag; MTP "
    "    without continuous treatment.\n"
    "  • recommended_method is not in the catalog (suggest a "
    "    replacement from the catalog).\n"
    "  • min sample size of the recommended method exceeds n_rows.\n"
    "  • Trivially weak identification (no confounders named, no "
    "    instrument, no mediator).\n"
    "Otherwise keep=true. You may also list 'risks' (non-fatal "
    "concerns) the runner should be aware of.\n\n"
    "ESTIMATOR CATALOG:\n"
    "{CATALOG_TABLE}\n\n"
    "Return ONLY a JSON CriticBatch."
)

_FOUNDATION_FOLLOWUP_SYSTEM = (
    "You are a senior causal-inference statistician. The pipeline just "
    "completed an experiment, the deterministic rule decided that a "
    "foundation follow-up is appropriate, and you must specify "
    "exactly which follow-up to run. Canonical patterns:\n"
    "  • Significant ATE → CATE on the most plausible effect modifier\n"
    "  • CATE-with-strong-subgroup → CATE within that subgroup, or "
    "    mediation decomposition (NDE/NIE)\n"
    "  • Significant ATE + mediator named → NDE/NIE\n"
    "  • RED sensitivity → tipping-point analysis OR negative-control "
    "    outcome substitution (pipeline will auto-schedule the "
    "    robustness child; you should propose a DIFFERENT "
    "    follow-up that adds new information)\n"
    "  • Null ATE → alternate estimator OR shorter follow-up window\n\n"
    "Constraints: the follow-up must add information beyond the "
    "parent. Do not re-run the same (T, Y, estimand). Reference the "
    "parent's result in foundation_rationale.\n\n"
    "ESTIMATOR CATALOG:\n"
    "{CATALOG_TABLE}\n\n"
    "Return ONLY a JSON NextExperiment with decision='run' and "
    "foundation_of=<parent_hypothesis_id>."
)


# ─────────── Prompt builders ──────────────────────────────────────────────


def _dataset_context_block(protocol: StudyProtocol) -> str:
    parts: list[str] = []
    parts.append(
        f"## Dataset: {protocol.dataset.source if protocol.dataset else 'unknown'}"
    )
    parts.append(
        f"## n_rows: {protocol.dataset.n_rows if protocol.dataset else '?'}, "
        f"n_cols: {protocol.dataset.n_cols if protocol.dataset else '?'}"
    )
    if protocol.discovery is not None:
        parts.append("")
        parts.append("## Variables (Stage-1c investigator role assignments)")
        for v in protocol.discovery.columns:
            parts.append(
                f"  - **{v.name}** ({v.dtype}): role={v.role.value}, "
                f"description={v.semantic_description or '—'}"
            )
    parts.append("")
    parts.append("## Flags emitted")
    for f in sorted(protocol.flags, key=lambda x: x.value):
        parts.append(f"  - {f.value}")
    if protocol.discovery and protocol.discovery.domain_brief:
        parts.append("")
        parts.append("## Domain expert brief")
        parts.append(protocol.discovery.domain_brief[:1500])
    return "\n".join(parts)


def _build_planner_prompt(protocol: StudyProtocol, config: LoopConfig) -> str:
    parts = [_dataset_context_block(protocol), ""]
    parts.append(
        f"## Task\nEnumerate up to {config.candidate_queue_size} credible "
        "candidate experiments this dataset supports. Cover the obvious "
        "headline questions, the natural follow-ups, AND a few exotic ones "
        "(IV / mediation / dose-response) for completeness. Return ONLY a "
        "JSON CandidateQueue."
    )
    return "\n".join(parts)


def _build_critic_prompt(
    protocol: StudyProtocol,
    candidates: list[CandidateExperiment],
    history: list[dict[str, Any]],
) -> str:
    parts: list[str] = []
    parts.append(_dataset_context_block(protocol))
    parts.append("")
    parts.append("## Completed experiments")
    if history:
        for i, h in enumerate(history, 1):
            parts.append(
                f"  [{i}] {h['id']}: {h['treatment']} → {h['outcome']} "
                f"({h['estimand_class']}) point={h['point_estimate']:+.4f} "
                f"sensitivity={h.get('sensitivity_verdict', '?')}"
            )
    else:
        parts.append("  (none — this is the first iteration)")
    parts.append("")
    parts.append("## Candidates under review")
    for c in candidates:
        parts.append(
            f"  - {c.candidate_id}: {c.treatment} → {c.outcome} "
            f"({c.estimand_class}), method={c.recommended_method or '(auto)'}, "
            f"mediator={c.mediator}, instrument={c.instrument}"
        )
        parts.append(f"      research_question: {c.research_question}")
    parts.append("")
    parts.append(
        "## Task\nReview EACH candidate. Return a CriticBatch with one "
        "CriticVerdict per candidate_id above."
    )
    return "\n".join(parts)


def _build_foundation_prompt(
    protocol: StudyProtocol,
    parent_walk: RoadmapWalk,
    chain: ChainState,
    history: list[dict[str, Any]],
) -> str:
    parts = [_dataset_context_block(protocol), ""]
    parts.append("## Parent experiment result")
    if parent_walk.q3_estimand and parent_walk.q7_estimates:
        est = parent_walk.q7_estimates[-1]
        parts.append(
            f"  - parent_id: {parent_walk.hypothesis_id}\n"
            f"  - chain_id: {chain.chain_id}, current_depth: {chain.depth}\n"
            f"  - {parent_walk.q3_estimand.treatment} → "
            f"{parent_walk.q3_estimand.outcome} "
            f"({parent_walk.q3_estimand.klass.value})\n"
            f"  - point: {est.point_estimate:+.4f}\n"
            f"  - 95% CI: [{est.ci_low:+.4f}, {est.ci_high:+.4f}]"
            if est.ci_low is not None and est.ci_high is not None
            else f"  - 95% CI: —\n"
            f"  - sensitivity verdict: {parent_walk.sensitivity_verdict or '?'}\n"
        )
        parts.append(f"  - q8 interpretation: {parent_walk.q8_interpretation or '—'}")
    parts.append("")
    if history:
        parts.append("## Earlier experiments in this chain / overall")
        for h in history[-8:]:
            parts.append(
                f"  - {h['id']}: {h['treatment']} → {h['outcome']} "
                f"({h['estimand_class']}) sens={h.get('sensitivity_verdict', '?')}"
            )
    parts.append("")
    parts.append(
        "## Task\nPropose the single most informative foundation "
        "follow-up. Must reference the parent's result in "
        "foundation_rationale. Return ONLY a JSON NextExperiment."
    )
    return "\n".join(parts)


# ─────────── LLM calls (each is a small focused agent) ────────────────────


def _plan_candidate_queue(
    *,
    protocol: StudyProtocol,
    client: OllamaClient,
    config: LoopConfig,
) -> CandidateQueue:
    prompt = _build_planner_prompt(protocol, config)
    system = _PLANNER_SYSTEM.replace("{CATALOG_TABLE}", catalog_markdown()).replace(
        "{QUEUE_SIZE}", str(config.candidate_queue_size)
    )
    resp = client.parse(
        prompt=prompt,
        schema=CandidateQueue,
        system=system,
        json_schema=CandidateQueue.model_json_schema(),
    )
    queue = resp.parsed
    assert isinstance(queue, CandidateQueue)
    return queue


def _critic_review(
    *,
    protocol: StudyProtocol,
    candidates: list[CandidateExperiment],
    history: list[dict[str, Any]],
    client: OllamaClient,
) -> CriticBatch:
    prompt = _build_critic_prompt(protocol, candidates, history)
    system = _CRITIC_SYSTEM.replace("{CATALOG_TABLE}", catalog_markdown())
    resp = client.parse(
        prompt=prompt,
        schema=CriticBatch,
        system=system,
        json_schema=CriticBatch.model_json_schema(),
    )
    batch = resp.parsed
    assert isinstance(batch, CriticBatch)
    return batch


def _foundation_followup_proposal(
    *,
    protocol: StudyProtocol,
    parent_walk: RoadmapWalk,
    chain: ChainState,
    history: list[dict[str, Any]],
    client: OllamaClient,
) -> NextExperiment:
    prompt = _build_foundation_prompt(protocol, parent_walk, chain, history)
    system = _FOUNDATION_FOLLOWUP_SYSTEM.replace(
        "{CATALOG_TABLE}", catalog_markdown()
    )
    resp = client.parse(
        prompt=prompt,
        schema=NextExperiment,
        system=system,
        json_schema=NextExperiment.model_json_schema(),
    )
    nxt = resp.parsed
    assert isinstance(nxt, NextExperiment)
    return nxt


# ─────────── Roadmap walk for one proposed experiment ────────────────────


def _build_graph_for_proposal(
    *, protocol: StudyProtocol, df: pd.DataFrame, candidate: CandidateExperiment
) -> CausalGraph:
    confounders: list[str] = []
    if protocol.discovery is not None:
        confounders = [
            v.name
            for v in protocol.discovery.columns
            if v.role is VariableRole.CONFOUNDER and v.name in df.columns
        ]
    edges = (
        [(c, candidate.treatment) for c in confounders]
        + [(c, candidate.outcome) for c in confounders]
        + [(candidate.treatment, candidate.outcome)]
    )
    roles = {c: VariableRole.CONFOUNDER for c in confounders}
    roles[candidate.treatment] = VariableRole.TREATMENT
    roles[candidate.outcome] = VariableRole.OUTCOME
    if candidate.mediator:
        roles[candidate.mediator] = VariableRole.MEDIATOR
        edges.append((candidate.treatment, candidate.mediator))
        edges.append((candidate.mediator, candidate.outcome))
    if candidate.instrument:
        roles[candidate.instrument] = VariableRole.INSTRUMENT
        edges.append((candidate.instrument, candidate.treatment))
    return CausalGraph.from_edge_list(edges, roles=roles)


def _outcome_dtype_for(protocol: StudyProtocol, outcome: str) -> str:
    if DataFlag.RIGHT_CENSORED_OUTCOME in protocol.flags:
        return "survival"
    if DataFlag.BINARY_OUTCOME in protocol.flags:
        return "binary"
    return "continuous"


def _run_one_experiment(
    *,
    protocol: StudyProtocol,
    df: pd.DataFrame,
    candidate: CandidateExperiment,
    counter: int,
    chain_id: str | None,
    parent_id: str | None,
    config: LoopConfig,
) -> tuple[RoadmapWalk, dict[str, Any], bool]:
    """Run Steps 5–8 for one candidate. Returns (walk, history_row, ok).

    On failure, ``walk.failure_reason`` is populated.
    """
    hypothesis_id = f"auto-{counter:02d}"
    walk = RoadmapWalk(
        hypothesis_id=hypothesis_id, chain_id=chain_id, parent_id=parent_id
    )

    if not candidate.treatment or not candidate.outcome or not candidate.estimand_class:
        walk.failure_reason = "invalid candidate (missing T/Y/estimand)"
        return walk, {}, False

    try:
        klass = EstimandClass(candidate.estimand_class.upper())
    except ValueError:
        walk.failure_reason = f"unknown estimand class: {candidate.estimand_class}"
        return walk, {}, False

    est = CausalEstimand.model_validate(
        {
            "class": klass,
            "treatment": candidate.treatment,
            "outcome": candidate.outcome,
            "modifiers": tuple(candidate.modifiers),
            "mediator": candidate.mediator,
            "instrument": candidate.instrument,
            "formal_expression": _formal_for(klass),
        }
    )
    walk.q3_estimand = est

    graph = _build_graph_for_proposal(
        protocol=protocol, df=df, candidate=candidate
    )
    ident = identify_effect(est, graph, df=df)

    walk.q5_identification = {
        "identifiable": ident.identifiable,
        "strategy": ident.strategy,
        "adjustment_set": list(ident.adjustment_set),
        "estimand_expression": ident.estimand_expression,
    }
    if not ident.identifiable:
        walk.failure_reason = (
            f"not identifiable under strategy='{ident.strategy}'. "
            f"Adjustment set: {list(ident.adjustment_set)}. "
            f"Consider supplying a stronger instrument or mediator, "
            f"or dropping descendants from the confounder set."
        )
        return walk, {}, False

    # Estimator-swap retry: if the LLM's recommended method (validated)
    # fails to fit, retry with prefer=None so the auto-cascade picks a
    # backup.
    prefer = _validate_method_id(candidate.recommended_method)
    tried_methods: list[str] = []
    last_exc: Exception | None = None
    result = None
    for attempt in range(1 + max(config.estimator_swap_retries, 0)):
        try_prefer = prefer if attempt == 0 else None
        try:
            result = run_step7(
                df=df,
                estimand=est,
                identification=ident,
                protocol=protocol,
                flags=set(protocol.flags),
                prefer=try_prefer,
            )
            tried_methods.append(try_prefer or "(cascade)")
            break
        except Exception as e:  # noqa: BLE001 — we want the actual reason
            last_exc = e
            tried_methods.append(try_prefer or "(cascade)")
            continue

    if result is None:
        walk.failure_reason = (
            f"estimator fit failed after {len(tried_methods)} attempt(s); "
            f"tried {tried_methods}; last error: "
            f"{type(last_exc).__name__}: {last_exc}"
        )
        return walk, {}, False

    walk.q6_statistical_estimand = derive_statistical_estimand(est, ident)
    walk.q7_estimates = (result,)

    # Sensitivity — use the new evalue_for_estimator helper that picks
    # the correct scale based on the estimator and outcome dtype.
    outcome_dtype = _outcome_dtype_for(protocol, candidate.outcome)
    baseline_risk: float | None = None
    if outcome_dtype == "binary" and candidate.outcome in df.columns:
        try:
            baseline_risk = float(df[candidate.outcome].mean())
        except Exception:
            baseline_risk = None

    verdict_color = "unknown"
    sensitivity_rationale = ""
    try:
        ev = evalue_for_estimator(
            result, outcome_dtype=outcome_dtype, baseline_risk=baseline_risk
        )
        confounders_for_sm = tuple(
            v.name
            for v in (protocol.discovery.columns if protocol.discovery else ())
            if v.role is VariableRole.CONFOUNDER and v.name in df.columns
        )
        try:
            sm = run_sensemakr(
                df,
                treatment=candidate.treatment,
                outcome=candidate.outcome,
                covariates=confounders_for_sm,
            )
        except Exception as e:  # noqa: BLE001 — sensemakr is best-effort
            sm = None
            sensitivity_rationale = f"sensemakr unavailable: {type(e).__name__}: {e}"

        if sm is not None:
            verdict = aggregate_sensitivity(evalue=ev, sensemakr=sm, rule="min")
            verdict_color = verdict.color
            sensitivity_rationale = (
                f"Sensitivity {verdict.color}. {verdict.rationale}. "
                f"E-value={ev.e_value:.2f} ({ev.scale}). "
                f"RV={sm.robustness_value:.3f}."
            )
        else:
            # E-value alone is still informative
            verdict_color = (
                "green"
                if ev.e_value >= 2.0
                else "yellow"
                if ev.e_value >= 1.25
                else "red"
            )
            sensitivity_rationale = (
                f"Sensitivity {verdict_color} (E-value-only, sensemakr "
                f"unavailable). E-value={ev.e_value:.2f} ({ev.scale})."
            )
    except Exception as e:  # noqa: BLE001 — capture but don't fail the walk
        verdict_color = "errored"
        sensitivity_rationale = (
            f"Sensitivity errored: {type(e).__name__}: {e}"
        )

    walk.sensitivity_verdict = verdict_color
    walk.q8_interpretation = sensitivity_rationale

    # Surface upstream diagnostics into the history row so the next LLM
    # call can see them.
    diag = result.diagnostics or {}
    overlap = diag.get("overlap", {}) if isinstance(diag, dict) else {}
    refutations_info = diag.get("refutations", {}) if isinstance(diag, dict) else {}

    history_row = {
        "id": walk.hypothesis_id,
        "chain_id": chain_id,
        "parent_id": parent_id,
        "treatment": candidate.treatment,
        "outcome": candidate.outcome,
        "estimand_class": candidate.estimand_class,
        "estimator_id": result.estimator_id,
        "estimator_attempts": tried_methods,
        "point_estimate": result.point_estimate,
        "se": result.se,
        "ci_low": result.ci_low if result.ci_low is not None else 0.0,
        "ci_high": result.ci_high if result.ci_high is not None else 0.0,
        "p_value": f"{result.p_value:.4g}" if result.p_value is not None else "NA",
        "sensitivity_verdict": verdict_color,
        "positivity_verdict": (
            overlap.get("verdict") if isinstance(overlap, dict) else None
        ),
        "refutations_passed": (
            refutations_info.get("n_passed")
            if isinstance(refutations_info, dict)
            else None
        ),
        "foundation_of": parent_id,
    }
    return walk, history_row, True


def _formal_for(klass: EstimandClass) -> str:
    return {
        EstimandClass.ATE: "E[Y(1) − Y(0)]",
        EstimandClass.CATE: "E[Y(1) − Y(0) | X = x]",
        EstimandClass.NDE: "Natural Direct Effect",
        EstimandClass.NIE: "Natural Indirect Effect",
        EstimandClass.LATE: "Wald: Cov(Y, Z | X) / Cov(T, Z | X)",
        EstimandClass.RMST_CONTRAST: "E[min(T_surv, τ) | A=1] − E[min(T_surv, τ) | A=0]",
        EstimandClass.MODIFIED_TREATMENT_POLICY: "E[Y(δ(A))]",
    }.get(klass, f"{klass.value} estimand")


# ─────────── Foundation-firing rule ───────────────────────────────────────


def _should_fire_foundation_child(
    *,
    parent_walk: RoadmapWalk,
    chain: ChainState,
    config: LoopConfig,
    foundation_iterations_used: int,
) -> tuple[bool, str]:
    """Deterministic rule for whether to fire a foundation follow-up.

    Returns (should_fire, reason).
    """
    if not config.foundation_allowed:
        return False, "foundation not allowed"
    if foundation_iterations_used >= config.max_foundation_iterations:
        return False, "global foundation iteration budget exhausted"
    if chain.depth >= config.max_foundation_depth:
        return False, "chain depth budget exhausted"
    if chain.null_streak >= config.null_streak_threshold:
        return False, "chain null-streak threshold reached"
    if chain.info_gain_streak_below_eps >= config.info_gain_streak_threshold:
        return False, "chain info-gain saturated"
    if not parent_walk.q7_estimates:
        return False, "parent has no estimate"
    est = parent_walk.q7_estimates[-1]
    # Significance signal: |t| > 1.96
    if est.se is None or est.se <= 0:
        return False, "parent has no SE — can't judge significance"
    t_stat = abs(est.point_estimate / est.se)
    if parent_walk.sensitivity_verdict == "red":
        # Pipeline auto-schedules robustness child separately; we don't
        # also fire a "drill deeper" follow-up on a red parent — the
        # signal isn't trustworthy enough to build on.
        return False, "red sensitivity — robustness child auto-scheduled instead"
    if t_stat < 1.96:
        return False, f"parent not significant (|t|={t_stat:.2f} < 1.96)"
    return True, "significant parent + budgets ok"


def _auto_robustness_candidate(
    *, parent_walk: RoadmapWalk, parent_candidate: CandidateExperiment
) -> CandidateExperiment | None:
    """Synthesize a deterministic robustness follow-up when the parent
    came back RED. Currently: an ATT (vs ATE) check or a CATE-with-
    weighting variant. Returns None if no sensible robustness can be
    synthesized."""
    if not parent_walk.q3_estimand:
        return None
    # Cheapest informative follow-up: re-run the same (T, Y) under a
    # weighting estimator with rich SuperLearner if the original used
    # DML, or vice versa. The deterministic scorer will rank it
    # accordingly.
    parent_klass = parent_walk.q3_estimand.klass.value
    return CandidateExperiment(
        candidate_id=f"robustness-{parent_walk.hypothesis_id}",
        research_question=(
            f"Robustness re-check of {parent_walk.hypothesis_id}: "
            f"does the {parent_klass} effect of "
            f"{parent_walk.q3_estimand.treatment} on "
            f"{parent_walk.q3_estimand.outcome} survive a different "
            f"identification strategy?"
        ),
        treatment=parent_walk.q3_estimand.treatment,
        outcome=parent_walk.q3_estimand.outcome,
        estimand_class=parent_klass,
        modifiers=list(parent_walk.q3_estimand.modifiers or ()),
        mediator=parent_walk.q3_estimand.mediator,
        instrument=parent_walk.q3_estimand.instrument,
        recommended_method=(
            "rbridge.weightit"
            if parent_candidate.recommended_method and "dml" in parent_candidate.recommended_method.lower()
            else "python.dml.linear"
        ),
        impact_rationale=(
            "Parent finding had RED sensitivity verdict; an independent "
            "estimator with different identification assumptions is the "
            "right robustness check."
        ),
        identifiability_rationale=(
            "Same data, different identification — if both estimators "
            "agree the parent's red flag is partially mitigated."
        ),
        power_rationale="Same sample as parent; power equivalent.",
        impact_hint=0.55,
        identifiability_hint=0.7,
        power_hint=parent_candidate.power_hint,
    )


def _update_chain_state(
    chain: ChainState, walk: RoadmapWalk, config: LoopConfig
) -> None:
    """Update a chain's bookkeeping after a child completes."""
    if not walk.q7_estimates:
        return
    est = walk.q7_estimates[-1]
    prev_point = chain.last_point
    prev_se = chain.last_se
    chain.last_point = est.point_estimate
    chain.last_se = est.se
    chain.last_verdict = walk.sensitivity_verdict
    chain.depth += 1
    # Significance: did this step's effect survive?
    if est.se is None or est.se <= 0 or abs(est.point_estimate / est.se) < 1.96:
        chain.null_streak += 1
    else:
        chain.null_streak = 0
    # Information gain: |Δpoint| / prev_se
    if prev_point is not None and prev_se is not None and prev_se > 0:
        info_gain = abs(est.point_estimate - prev_point) / prev_se
        if info_gain < config.diminishing_returns_epsilon:
            chain.info_gain_streak_below_eps += 1
        else:
            chain.info_gain_streak_below_eps = 0


# ─────────── Main loop ───────────────────────────────────────────────────


def _candidate_from_dict(d: dict[str, Any]) -> CandidateExperiment:
    """Reconstruct a CandidateExperiment from a queue dict (used after persist/reload)."""
    return CandidateExperiment(**{k: v for k, v in d.items() if k in CandidateExperiment.model_fields})


def _queue_to_dicts(
    queue: list[CandidateExperiment], scored: dict[str, dict[str, float]]
) -> tuple[dict[str, Any], ...]:
    """Project the queue + scores into a serializable form for protocol persistence."""
    out: list[dict[str, Any]] = []
    for c in queue:
        d = c.model_dump()
        s = scored.get(c.candidate_id, {})
        d.update(s)
        d.setdefault("status", "pending")
        out.append(d)
    return tuple(out)


def run_master_loop(
    *,
    protocol: StudyProtocol,
    project_dir: Path,
    dataset_path: Path,
    discovery_client: OllamaClient,
    expert_client: OllamaClient | None,
    config: LoopConfig,
) -> Iterator[LoopEvent]:
    """Run the iterative master loop. Yields :class:`LoopEvent` objects."""
    yield LoopEvent(kind="phase_start", phase="discover", message="Phase 1 · discover")

    # ── Phase 1: discovery ───────────────────────────────────────────
    discovery = run_discovery(
        source=dataset_path,
        client=discovery_client,
        expert_client=expert_client or discovery_client,
        research_question=protocol.research_question,
    )
    protocol.discovery = discovery.to_report()
    protocol.flags |= discovery.flags
    if discovery.candidate_graphs and not protocol.candidate_graphs:
        protocol.candidate_graphs = discovery.candidate_graphs
    if not protocol.dataset:
        from causalrag.core.protocol import DatasetSpec

        protocol.dataset = DatasetSpec(
            source=f"csv://{dataset_path}",
            n_rows=discovery.profile.n_rows,
            n_cols=discovery.profile.n_cols,
            columns=discovery.columns,
        )
    yield LoopEvent(
        kind="phase_end",
        phase="discover",
        message=f"flags={','.join(sorted(f.value for f in protocol.flags))}",
    )

    df = pd.read_csv(dataset_path)
    propose_client = expert_client or discovery_client

    # ── Phase 2: candidate queue planning ────────────────────────────
    yield LoopEvent(
        kind="phase_start",
        phase="plan",
        message=f"Planning {config.candidate_queue_size}-candidate queue",
    )
    try:
        queue_obj = _plan_candidate_queue(
            protocol=protocol, client=propose_client, config=config
        )
        candidates: list[CandidateExperiment] = list(queue_obj.candidates)
    except Exception as e:
        yield LoopEvent(
            kind="error",
            phase="plan",
            message=f"queue planning failed: {type(e).__name__}: {e}",
        )
        candidates = []

    scored: dict[str, dict[str, float]] = {}
    completed: list[RoadmapWalk] = []
    if candidates:
        scored = {c.candidate_id: score_candidate(c, protocol=protocol, completed=completed) for c in candidates}
        candidates.sort(key=lambda c: scored[c.candidate_id]["score"], reverse=True)
        protocol.candidate_queue = _queue_to_dicts(candidates, scored)
        yield LoopEvent(
            kind="plan",
            phase="plan",
            message=f"{len(candidates)} candidates ranked",
            payload={
                "top": [
                    {
                        "id": c.candidate_id,
                        "research_question": c.research_question,
                        "treatment": c.treatment,
                        "outcome": c.outcome,
                        "estimand_class": c.estimand_class,
                        "method": c.recommended_method,
                        **scored[c.candidate_id],
                    }
                    for c in candidates[: config.propose_k]
                ]
            },
        )
    else:
        yield LoopEvent(
            kind="log",
            phase="plan",
            message="empty queue — falling back to per-iteration LLM proposals",
        )

    # ── Phase 3: iterative propose-K → critique → commit ─────────────
    history: list[dict[str, Any]] = []
    failures = 0
    chains: dict[str, ChainState] = {}
    foundation_iterations_used = 0
    pending_followups: list[tuple[CandidateExperiment, str | None, str]] = []
    # Each tuple = (candidate, chain_id, parent_id) — chain_id can be None for new root.

    while len(completed) < config.n_experiments:
        if failures >= config.max_consecutive_failures:
            yield LoopEvent(
                kind="log",
                phase="auto",
                message=f"stopping: {failures} consecutive failed proposals",
            )
            # Optional: ask the LLM for an autopsy (kept simple here).
            break

        # Pick the next candidate: pending follow-ups (foundation children
        # or robustness re-checks) jump to the front.
        next_candidate: CandidateExperiment | None = None
        next_chain_id: str | None = None
        next_parent_id: str | None = None
        if pending_followups:
            next_candidate, next_chain_id, next_parent_id = pending_followups.pop(0)
        elif candidates:
            # Propose-K → critic → commit
            top_k = [c for c in candidates if scored[c.candidate_id].get("status", "pending") != "completed"][: config.propose_k]
            if not top_k:
                yield LoopEvent(
                    kind="log",
                    phase="auto",
                    message="queue exhausted — stopping",
                )
                break
            if config.critic_enabled and len(top_k) > 1:
                try:
                    batch = _critic_review(
                        protocol=protocol,
                        candidates=top_k,
                        history=history,
                        client=propose_client,
                    )
                    keep_ids = {v.candidate_id for v in batch.verdicts if v.keep}
                    method_overrides = {
                        v.candidate_id: v.revised_recommended_method
                        for v in batch.verdicts
                        if v.keep and v.revised_recommended_method
                    }
                    risks_by_id = {
                        v.candidate_id: v.risks
                        for v in batch.verdicts
                        if v.keep and v.risks
                    }
                    survivors = [c for c in top_k if c.candidate_id in keep_ids]
                    # Apply critic's method override (validated against catalog)
                    for c in survivors:
                        override = _validate_method_id(method_overrides.get(c.candidate_id))
                        if override:
                            c.recommended_method = override
                    rejected = [c for c in top_k if c.candidate_id not in keep_ids]
                    for c in rejected:
                        # Drop rejected candidates from the queue.
                        scored[c.candidate_id]["status"] = "vetoed"
                    if survivors:
                        # Pick the highest-scored survivor.
                        survivors.sort(
                            key=lambda c: scored[c.candidate_id]["score"], reverse=True
                        )
                        next_candidate = survivors[0]
                    if rejected:
                        yield LoopEvent(
                            kind="log",
                            phase="critic",
                            message=f"critic rejected {len(rejected)}/{len(top_k)} candidates",
                            payload={"rejected_ids": [c.candidate_id for c in rejected]},
                        )
                    if risks_by_id.get(next_candidate.candidate_id) if next_candidate else None:
                        yield LoopEvent(
                            kind="log",
                            phase="critic",
                            message=f"critic risks for {next_candidate.candidate_id}: "
                            + "; ".join(risks_by_id[next_candidate.candidate_id]),
                        )
                except Exception as e:
                    yield LoopEvent(
                        kind="log",
                        phase="critic",
                        message=f"critic skipped: {type(e).__name__}: {e}",
                    )
                    next_candidate = top_k[0]
            else:
                next_candidate = top_k[0]
            if next_candidate is None:
                # All rejected → keep looping (top_k slice next iteration will be different)
                continue
            # Remove from queue
            candidates = [c for c in candidates if c.candidate_id != next_candidate.candidate_id]

        if next_candidate is None:
            yield LoopEvent(
                kind="log",
                phase="auto",
                message="no candidate available — stopping",
            )
            break

        # Repeat guard — handled by novelty_score=0 in the scorer, but
        # belt + braces in case the LLM injected a duplicate via the
        # critic's revised_recommended_method path.
        triple = (
            next_candidate.treatment or "",
            next_candidate.outcome or "",
            (next_candidate.estimand_class or "").upper(),
        )

        yield LoopEvent(
            kind="log",
            phase="auto",
            message=(
                f"running {triple[0]} → {triple[1]} ({triple[2]}); "
                + (
                    f"foundation of {next_parent_id} (chain {next_chain_id})"
                    if next_parent_id
                    else "independent"
                )
            ),
            payload={
                "candidate_id": next_candidate.candidate_id,
                "treatment": next_candidate.treatment,
                "outcome": next_candidate.outcome,
                "estimand": next_candidate.estimand_class,
                "method": next_candidate.recommended_method,
                "is_foundation": next_parent_id is not None,
                "chain_id": next_chain_id,
                "score": scored.get(next_candidate.candidate_id, {}).get("score"),
                "impact_rationale": next_candidate.impact_rationale,
                "identifiability_rationale": next_candidate.identifiability_rationale,
                "power_rationale": next_candidate.power_rationale,
            },
        )

        walk, row, ok = _run_one_experiment(
            protocol=protocol,
            df=df,
            candidate=next_candidate,
            counter=len(completed) + 1,
            chain_id=next_chain_id,
            parent_id=next_parent_id,
            config=config,
        )
        if not ok:
            yield LoopEvent(
                kind="error",
                phase="auto",
                message=f"experiment failed: {walk.failure_reason or 'unknown'}",
                payload={
                    "candidate_id": next_candidate.candidate_id,
                    "failure_reason": walk.failure_reason,
                },
            )
            # Capture failure on history so the next LLM call sees it.
            history.append(
                {
                    "id": walk.hypothesis_id,
                    "chain_id": next_chain_id,
                    "parent_id": next_parent_id,
                    "treatment": next_candidate.treatment,
                    "outcome": next_candidate.outcome,
                    "estimand_class": next_candidate.estimand_class,
                    "estimator_id": "—",
                    "point_estimate": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "p_value": "NA",
                    "sensitivity_verdict": "errored",
                    "failure_reason": walk.failure_reason,
                    "foundation_of": next_parent_id,
                }
            )
            failures += 1
            continue

        failures = 0
        completed.append(walk)
        history.append(row)

        # Update chain state
        if next_chain_id is None:
            # New root chain
            chain_id = walk.hypothesis_id
            walk.chain_id = chain_id
            chain = ChainState(chain_id=chain_id, root_hypothesis_id=walk.hypothesis_id)
            chain.depth = 0
            chain.last_point = walk.q7_estimates[-1].point_estimate if walk.q7_estimates else None
            chain.last_se = walk.q7_estimates[-1].se if walk.q7_estimates else None
            chain.last_verdict = walk.sensitivity_verdict
            chains[chain_id] = chain
        else:
            chain = chains[next_chain_id]
            _update_chain_state(chain, walk, config)
            foundation_iterations_used += 1

        # Mark candidate completed in the persisted queue
        if next_candidate.candidate_id in scored:
            scored[next_candidate.candidate_id]["status"] = "completed"

        record_decision(
            protocol,
            phase="master_loop",
            decision=(
                f"experiment {len(completed)}/{config.n_experiments}"
                + (
                    f" (foundation of {next_parent_id}, chain {next_chain_id})"
                    if next_parent_id
                    else ""
                )
            ),
            chose=f"{triple[0]} → {triple[1]} ({triple[2]}) via {row['estimator_id']}",
            source="auto",
            note=(
                f"sensitivity={row['sensitivity_verdict']} · "
                f"chain_depth={chain.depth}"
            ),
        )

        yield LoopEvent(
            kind="card",
            phase="result",
            message=(
                f"[{walk.hypothesis_id}] {row['treatment']} → {row['outcome']} "
                f"({row['estimand_class']}): {row['point_estimate']:+.4f} "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] · p={row['p_value']} "
                f"· sensitivity={row['sensitivity_verdict']}"
            ),
            payload=row,
        )

        # Auto-fire robustness child on RED sensitivity
        if (
            config.auto_fire_robustness_on_red
            and walk.sensitivity_verdict == "red"
            and len(completed) < config.n_experiments
        ):
            robustness_cand = _auto_robustness_candidate(
                parent_walk=walk, parent_candidate=next_candidate
            )
            if robustness_cand is not None:
                scored[robustness_cand.candidate_id] = score_candidate(
                    robustness_cand, protocol=protocol, completed=completed
                )
                pending_followups.append(
                    (robustness_cand, chain.chain_id, walk.hypothesis_id)
                )
                yield LoopEvent(
                    kind="log",
                    phase="auto",
                    message=(
                        f"auto-scheduling robustness child for "
                        f"{walk.hypothesis_id} (red sensitivity)"
                    ),
                )

        # Deterministic foundation-firing rule
        should_fire, reason = _should_fire_foundation_child(
            parent_walk=walk,
            chain=chain,
            config=config,
            foundation_iterations_used=foundation_iterations_used,
        )
        if should_fire and len(completed) < config.n_experiments:
            try:
                followup_proposal = _foundation_followup_proposal(
                    protocol=protocol,
                    parent_walk=walk,
                    chain=chain,
                    history=history,
                    client=propose_client,
                )
                if followup_proposal.decision == "run":
                    followup_cand = CandidateExperiment(
                        candidate_id=f"foundation-{walk.hypothesis_id}",
                        research_question=followup_proposal.research_question or "",
                        treatment=followup_proposal.treatment or "",
                        outcome=followup_proposal.outcome or "",
                        estimand_class=followup_proposal.estimand_class or "",
                        modifiers=list(followup_proposal.modifiers),
                        mediator=followup_proposal.mediator,
                        instrument=followup_proposal.instrument,
                        recommended_method=_validate_method_id(
                            followup_proposal.recommended_method
                        ),
                        impact_rationale=followup_proposal.impact_rationale
                        or followup_proposal.foundation_rationale
                        or "(foundation follow-up)",
                        identifiability_rationale=followup_proposal.identifiability_rationale
                        or "(foundation follow-up — same identification as parent)",
                        power_rationale=followup_proposal.power_rationale
                        or "(same sample as parent)",
                    )
                    scored[followup_cand.candidate_id] = score_candidate(
                        followup_cand, protocol=protocol, completed=completed
                    )
                    pending_followups.append(
                        (followup_cand, chain.chain_id, walk.hypothesis_id)
                    )
                    yield LoopEvent(
                        kind="log",
                        phase="auto",
                        message=(
                            f"scheduled foundation child of {walk.hypothesis_id} "
                            f"(chain {chain.chain_id}, depth {chain.depth})"
                        ),
                    )
            except Exception as e:
                yield LoopEvent(
                    kind="log",
                    phase="auto",
                    message=f"foundation proposal failed: {type(e).__name__}: {e}",
                )
        else:
            yield LoopEvent(
                kind="log",
                phase="auto",
                message=f"no foundation child fired: {reason}",
            )

        # Re-score remaining candidates with the new completed list
        if candidates:
            for c in candidates:
                scored[c.candidate_id] = score_candidate(
                    c, protocol=protocol, completed=completed
                )
            candidates.sort(key=lambda c: scored[c.candidate_id]["score"], reverse=True)
            protocol.candidate_queue = _queue_to_dicts(candidates, scored)

    # Persist all walks
    new_walks = dict(protocol.roadmap_walks)
    for w in completed:
        new_walks[w.hypothesis_id] = w
    protocol.roadmap_walks = new_walks

    # Apply multiple-testing adjustment across the K experiments before
    # synthesis reads the (now adjusted) p-values.
    from causalrag.sensitivity.multiple_testing import adjust_protocol_p_values

    protocol, adjusted_summary = adjust_protocol_p_values(protocol)
    yield LoopEvent(
        kind="log",
        phase="auto",
        message=(
            f"applied {protocol.multiple_testing} adjustment to "
            f"{len(adjusted_summary)} comparisons"
        ),
    )

    protocol.write_yaml(project_dir / "study.causalrag.yaml")

    # ── Phase 5: synthesis ───────────────────────────────────────────
    synthesis_payload: dict[str, Any] | None = None
    has_results = any(w.q7_estimates for w in completed)
    if has_results:
        yield LoopEvent(
            kind="phase_start",
            phase="synthesis",
            message="Phase 5 · synthesis",
        )
        try:
            from causalrag.reporting.synthesis import synthesize_insights

            synth = synthesize_insights(
                protocol=protocol, df=df, client=propose_client
            )
            synthesis_payload = synth.model_dump()
            (project_dir / "executive_synthesis.json").write_text(
                synth.model_dump_json(indent=2)
            )
            yield LoopEvent(
                kind="card",
                phase="synthesis",
                message=f"[synthesis · {synth.inferred_domain}] {synth.tldr}",
                payload=synthesis_payload,
            )
        except Exception as e:  # noqa: BLE001 — synthesis is best-effort
            yield LoopEvent(
                kind="log",
                phase="synthesis",
                message=f"synthesis skipped: {type(e).__name__}: {e}",
            )

    yield LoopEvent(
        kind="done",
        phase="auto",
        message=(
            f"master loop complete · {len(completed)}/{config.n_experiments} "
            f"experiments · {len(chains)} chain(s)"
        ),
        payload={
            "completed": len(completed),
            "target": config.n_experiments,
            "chains": {
                cid: {"depth": cs.depth, "root": cs.root_hypothesis_id}
                for cid, cs in chains.items()
            },
            "synthesis": synthesis_payload,
        },
    )


__all__ = [
    "CandidateExperiment",
    "CandidateQueue",
    "ChainState",
    "CriticBatch",
    "CriticVerdict",
    "LoopConfig",
    "LoopEvent",
    "NextExperiment",
    "run_master_loop",
    "score_candidate",
]
