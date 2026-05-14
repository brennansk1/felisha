"""Cross-experiment synthesis pre-pass.

The per-experiment :mod:`causalrag.reporting.synthesis` flow treats each
completed Roadmap walk independently. That works for short studies, but
once the master loop completes a handful of foundation-chained
experiments the executive narrative needs to acknowledge the *joint*
structure of the results: which experiments CONTRADICT each other, which
REINFORCE each other, and what NARRATIVE thread runs along each
foundation chain.

This module computes that joint structure in two steps:

1. **Deterministic bookkeeping** — group walks by ``chain_id``; flag
   pairs sharing (treatment, outcome) whose point estimates disagree in
   sign as candidate contradictions; flag pairs that agree in sign with
   similar magnitude as candidate reinforcements.

2. **LLM filter + narration** — the deterministic candidates and the
   full set of walks are handed to the reasoning LLM, which filters
   apparent contradictions (e.g. different subgroup definitions are
   merely "surface"), promotes genuine ones to "structural", grades
   reinforcement strength, and writes a 2-4 sentence story per chain.
   The LLM may also surface contradictions/reinforcements the
   deterministic pass missed (e.g. a CATE in one walk contradicting an
   ATE in another).

The resulting :class:`CrossExperimentAnalysis` is fed into the existing
``synthesize_insights`` prompt as a context block via
:func:`cross_experiment_block_for_prompt`.

Failure-safe: an LLM error returns the deterministic-only analysis; we
never propagate.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.llm.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


# ─────────── Schema ──────────────────────────────────────────────────────


class Contradiction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exp_a_id: str
    exp_b_id: str
    description: str = Field(
        ...,
        description=(
            "1-2 sentences: 'Experiment A found X; experiment B found "
            "¬X under conditions ...'."
        ),
    )
    severity: Literal["surface", "structural"] = Field(
        ...,
        description=(
            "surface = different subgroups / estimands; structural = "
            "same target, genuinely incompatible conclusion."
        ),
    )


class Reinforcement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exp_ids: list[str] = Field(..., min_length=2)
    description: str = Field(
        ..., description="What the listed experiments jointly establish."
    )
    strength: Literal["weak", "moderate", "strong"]


class ChainNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chain_id: str
    root_hypothesis_id: str
    walk_ids_in_order: list[str]
    story: str = Field(
        ...,
        description="2-4 sentence narrative of what this foundation chain established.",
    )


class CrossExperimentAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contradictions: list[Contradiction] = Field(default_factory=list)
    reinforcements: list[Reinforcement] = Field(default_factory=list)
    chain_narratives: list[ChainNarrative] = Field(default_factory=list)
    overall_theme: str = Field(
        default="",
        description="1-2 sentences capturing the overarching theme across experiments.",
    )


# ─────────── Deterministic pre-pass ──────────────────────────────────────


# Two point estimates are considered "similar in magnitude" (for the
# reinforcement candidate pass) when the smaller is at least this
# fraction of the larger by absolute value.
_MAGNITUDE_SIMILARITY_FLOOR = 0.5


def _completed_walks(protocol: StudyProtocol) -> dict[str, RoadmapWalk]:
    """Return only walks that have at least one EstimationResult."""
    return {
        hid: w
        for hid, w in protocol.roadmap_walks.items()
        if w.q7_estimates
    }


def _walk_summary(walk: RoadmapWalk) -> dict[str, object]:
    """Compact dict used in the LLM prompt — no schema, no Pydantic."""
    est = walk.q7_estimates[-1]
    estimand = walk.q3_estimand
    return {
        "hypothesis_id": walk.hypothesis_id,
        "treatment": estimand.treatment if estimand else None,
        "outcome": estimand.outcome if estimand else None,
        "estimand_class": estimand.klass.value if estimand else None,
        "modifiers": list(estimand.modifiers) if estimand and estimand.modifiers else [],
        "point_estimate": est.point_estimate,
        "ci_low": est.ci_low,
        "ci_high": est.ci_high,
        "n_used": est.n_used,
        "chain_id": walk.chain_id,
        "parent_id": walk.parent_id,
        "sensitivity_verdict": walk.sensitivity_verdict or walk.q8_interpretation,
    }


def _candidate_contradictions(
    walks: dict[str, RoadmapWalk],
) -> list[Contradiction]:
    """Pairs sharing (T, Y) with opposite-sign point estimates."""
    items = list(walks.items())
    out: list[Contradiction] = []
    for i in range(len(items)):
        hid_a, wa = items[i]
        ea = wa.q3_estimand
        if ea is None or not wa.q7_estimates:
            continue
        pa = wa.q7_estimates[-1].point_estimate
        for j in range(i + 1, len(items)):
            hid_b, wb = items[j]
            eb = wb.q3_estimand
            if eb is None or not wb.q7_estimates:
                continue
            if ea.treatment != eb.treatment or ea.outcome != eb.outcome:
                continue
            pb = wb.q7_estimates[-1].point_estimate
            if pa == 0.0 or pb == 0.0:
                continue
            if (pa > 0) == (pb > 0):
                continue
            # Severity heuristic: same estimand class + no modifiers
            # on either side → likely structural; else surface.
            same_target = (
                ea.klass == eb.klass
                and not ea.modifiers
                and not eb.modifiers
            )
            severity = "structural" if same_target else "surface"
            out.append(
                Contradiction(
                    exp_a_id=hid_a,
                    exp_b_id=hid_b,
                    description=(
                        f"{hid_a} estimated {ea.treatment} → {ea.outcome} at "
                        f"{pa:+.4f} ({ea.klass.value}); {hid_b} estimated the "
                        f"same pair at {pb:+.4f} ({eb.klass.value})."
                    ),
                    severity=severity,
                )
            )
    return out


def _candidate_reinforcements(
    walks: dict[str, RoadmapWalk],
) -> list[Reinforcement]:
    """Groups sharing (T, Y) with same-sign, similar-magnitude estimates."""
    # Bucket by (treatment, outcome)
    buckets: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for hid, w in walks.items():
        if w.q3_estimand is None or not w.q7_estimates:
            continue
        key = (w.q3_estimand.treatment, w.q3_estimand.outcome)
        buckets.setdefault(key, []).append(
            (hid, w.q7_estimates[-1].point_estimate)
        )

    out: list[Reinforcement] = []
    for (treatment, outcome), entries in buckets.items():
        if len(entries) < 2:
            continue
        # Split by sign
        pos = [(hid, p) for hid, p in entries if p > 0]
        neg = [(hid, p) for hid, p in entries if p < 0]
        for group in (pos, neg):
            if len(group) < 2:
                continue
            mags = [abs(p) for _, p in group]
            smallest, largest = min(mags), max(mags)
            if largest == 0.0:
                continue
            if smallest / largest < _MAGNITUDE_SIMILARITY_FLOOR:
                # Sign-agreement only — still mark as weak reinforcement.
                strength: Literal["weak", "moderate", "strong"] = "weak"
            elif smallest / largest >= 0.8:
                strength = "strong"
            else:
                strength = "moderate"
            ids = [hid for hid, _ in group]
            direction = "positive" if group is pos else "negative"
            out.append(
                Reinforcement(
                    exp_ids=ids,
                    description=(
                        f"Experiments {', '.join(ids)} agree on a {direction} "
                        f"effect of {treatment} on {outcome} "
                        f"(magnitudes {[round(m, 4) for m in mags]})."
                    ),
                    strength=strength,
                )
            )
    return out


def _candidate_chain_narratives(
    walks: dict[str, RoadmapWalk],
) -> list[ChainNarrative]:
    """One ChainNarrative per chain_id, walks ordered by parent_id depth.

    Walks with no ``chain_id`` are skipped (they're independent roots
    that don't form a chain). Walks whose ``chain_id`` matches their own
    ``hypothesis_id`` are treated as the chain root.
    """
    # Group walks by chain_id
    by_chain: dict[str, list[RoadmapWalk]] = {}
    for w in walks.values():
        if w.chain_id is None:
            continue
        by_chain.setdefault(w.chain_id, []).append(w)

    narratives: list[ChainNarrative] = []
    for chain_id, members in by_chain.items():
        # Topological order by parent_id chain: find root (parent_id None
        # or parent_id not in chain), then BFS.
        member_ids = {w.hypothesis_id for w in members}
        roots = [
            w for w in members
            if w.parent_id is None or w.parent_id not in member_ids
        ]
        if not roots:
            # Cycle or detached — fall back to insertion order.
            ordered_ids = [w.hypothesis_id for w in members]
            root_id = members[0].hypothesis_id
        else:
            root = roots[0]
            root_id = root.hypothesis_id
            # Build parent → children map.
            children: dict[str, list[RoadmapWalk]] = {}
            for w in members:
                if w.parent_id is not None:
                    children.setdefault(w.parent_id, []).append(w)
            ordered_ids = []
            queue: list[RoadmapWalk] = [root]
            seen: set[str] = set()
            while queue:
                head = queue.pop(0)
                if head.hypothesis_id in seen:
                    continue
                seen.add(head.hypothesis_id)
                ordered_ids.append(head.hypothesis_id)
                # Sort children by hypothesis_id for determinism.
                kids = sorted(
                    children.get(head.hypothesis_id, []),
                    key=lambda x: x.hypothesis_id,
                )
                queue.extend(kids)
            # Append any orphans that weren't reached.
            for w in members:
                if w.hypothesis_id not in seen:
                    ordered_ids.append(w.hypothesis_id)

        narratives.append(
            ChainNarrative(
                chain_id=chain_id,
                root_hypothesis_id=root_id,
                walk_ids_in_order=ordered_ids,
                story=(
                    f"Foundation chain rooted at {root_id} produced "
                    f"{len(ordered_ids)} experiment(s): {', '.join(ordered_ids)}."
                ),
            )
        )
    # Stable order by chain_id.
    narratives.sort(key=lambda n: n.chain_id)
    return narratives


def _deterministic_prepass(
    walks: dict[str, RoadmapWalk],
) -> CrossExperimentAnalysis:
    return CrossExperimentAnalysis(
        contradictions=_candidate_contradictions(walks),
        reinforcements=_candidate_reinforcements(walks),
        chain_narratives=_candidate_chain_narratives(walks),
        overall_theme="",
    )


# ─────────── LLM filter + narration ──────────────────────────────────────


_SYSTEM_PROMPT = (
    "You are a senior causal-inference statistician reviewing the joint "
    "structure of a completed multi-experiment study. The deterministic "
    "pre-pass has flagged candidate CONTRADICTIONS between experiments "
    "(opposite-sign point estimates on the same treatment-outcome pair), "
    "candidate REINFORCEMENTS (same-sign, similar-magnitude effects on "
    "the same pair), and candidate CHAIN NARRATIVES (one per master-loop "
    "foundation chain).\n\n"
    "YOUR JOB:\n"
    "  1. Filter the contradictions: which are *surface* (different "
    "     subgroups, different estimand classes, different time windows "
    "     that don't actually disagree) vs *structural* (same target, "
    "     genuinely incompatible)?\n"
    "  2. Grade the reinforcements (weak / moderate / strong) based on "
    "     magnitude alignment AND sensitivity/CI alignment.\n"
    "  3. Write a 2-4 sentence story for each chain narrative — what "
    "     the chain established, where it pivoted, how it ended.\n"
    "  4. Surface contradictions/reinforcements the deterministic pass "
    "     missed — e.g. a CATE in one walk that contradicts an ATE in "
    "     another, or two walks on different but causally adjacent "
    "     outcomes that tell the same story.\n"
    "  5. Write one or two sentences of overall_theme summarizing the "
    "     joint picture.\n\n"
    "Use ONLY the hypothesis_id strings that appear in the input. Never "
    "invent IDs. Return ONLY a JSON CrossExperimentAnalysis."
)


def _build_prompt(
    walks: dict[str, RoadmapWalk],
    prepass: CrossExperimentAnalysis,
) -> str:
    parts: list[str] = []
    parts.append("## All completed experiments")
    for hid, w in walks.items():
        summary = _walk_summary(w)
        parts.append(f"  - {hid}: {summary}")

    parts.append("")
    parts.append("## Deterministic candidate contradictions")
    if prepass.contradictions:
        for c in prepass.contradictions:
            parts.append(
                f"  - {c.exp_a_id} vs {c.exp_b_id} "
                f"[{c.severity}]: {c.description}"
            )
    else:
        parts.append("  - (none flagged)")

    parts.append("")
    parts.append("## Deterministic candidate reinforcements")
    if prepass.reinforcements:
        for r in prepass.reinforcements:
            parts.append(
                f"  - {r.exp_ids} [{r.strength}]: {r.description}"
            )
    else:
        parts.append("  - (none flagged)")

    parts.append("")
    parts.append("## Deterministic chain narratives (raw)")
    if prepass.chain_narratives:
        for cn in prepass.chain_narratives:
            parts.append(
                f"  - chain {cn.chain_id} "
                f"(root={cn.root_hypothesis_id}, "
                f"walks={cn.walk_ids_in_order})"
            )
    else:
        parts.append("  - (no chains)")

    parts.append("")
    parts.append(
        "## Task\n"
        "Filter, grade, narrate, and surface anything the deterministic "
        "pass missed. Return a CrossExperimentAnalysis JSON object."
    )
    return "\n".join(parts)


def _validate_id_references(
    analysis: CrossExperimentAnalysis,
    valid_ids: set[str],
) -> CrossExperimentAnalysis:
    """Drop any contradiction/reinforcement/chain referencing unknown ids."""
    kept_contradictions: list[Contradiction] = []
    for c in analysis.contradictions:
        bad: list[str] = []
        if c.exp_a_id not in valid_ids:
            bad.append(c.exp_a_id)
        if c.exp_b_id not in valid_ids:
            bad.append(c.exp_b_id)
        if bad:
            logger.warning(
                "cross_experiment: dropping contradiction with fabricated id(s): %s",
                bad,
            )
            continue
        kept_contradictions.append(c)

    kept_reinforcements: list[Reinforcement] = []
    for r in analysis.reinforcements:
        bad_ids = [hid for hid in r.exp_ids if hid not in valid_ids]
        if bad_ids:
            logger.warning(
                "cross_experiment: dropping reinforcement with fabricated id(s): %s",
                bad_ids,
            )
            continue
        kept_reinforcements.append(r)

    kept_chains: list[ChainNarrative] = []
    for cn in analysis.chain_narratives:
        bad_ids = [hid for hid in cn.walk_ids_in_order if hid not in valid_ids]
        if cn.root_hypothesis_id not in valid_ids:
            bad_ids.append(cn.root_hypothesis_id)
        if bad_ids:
            logger.warning(
                "cross_experiment: dropping chain narrative with fabricated id(s): %s",
                bad_ids,
            )
            continue
        kept_chains.append(cn)

    return CrossExperimentAnalysis(
        contradictions=kept_contradictions,
        reinforcements=kept_reinforcements,
        chain_narratives=kept_chains,
        overall_theme=analysis.overall_theme,
    )


def analyze_cross_experiment(
    *,
    protocol: StudyProtocol,
    client: OllamaClient,
) -> CrossExperimentAnalysis:
    """Run the deterministic pre-pass, then LLM filter + narration.

    Failure-safe: on any LLM error the deterministic-only analysis is
    returned. Hypothesis-id references the LLM fabricates are dropped
    with a logged warning.
    """
    walks = _completed_walks(protocol)
    prepass = _deterministic_prepass(walks)

    # Nothing to enrich if there are no completed walks.
    if not walks:
        return prepass

    prompt = _build_prompt(walks, prepass)
    try:
        response = client.parse(
            prompt=prompt,
            schema=CrossExperimentAnalysis,
            system=_SYSTEM_PROMPT,
            json_schema=CrossExperimentAnalysis.model_json_schema(),
        )
        analysis = response.parsed
        assert isinstance(analysis, CrossExperimentAnalysis)
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning(
            "cross_experiment: LLM call failed (%s: %s); "
            "returning deterministic pre-pass only.",
            type(e).__name__,
            e,
        )
        return prepass

    valid_ids = set(walks.keys())
    return _validate_id_references(analysis, valid_ids)


# ─────────── Prompt-block formatter for downstream synthesis ─────────────


def cross_experiment_block_for_prompt(
    analysis: CrossExperimentAnalysis,
) -> str:
    """Render the analysis as a markdown block for the synthesis prompt.

    Produces a short, quote-friendly markdown chunk (typically 10–30
    lines) that ``synthesize_insights`` can inject ahead of the per-
    experiment block so the LLM weaves a coherent narrative.
    """
    lines: list[str] = []
    lines.append("## Cross-experiment context")
    if analysis.overall_theme:
        lines.append(f"**Overall theme:** {analysis.overall_theme}")
    else:
        lines.append("**Overall theme:** (none surfaced)")

    lines.append("")
    lines.append("### Contradictions between experiments")
    if analysis.contradictions:
        for c in analysis.contradictions:
            lines.append(
                f"- **{c.exp_a_id} ↔ {c.exp_b_id}** "
                f"({c.severity}): {c.description}"
            )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("### Reinforcements (convergent evidence)")
    if analysis.reinforcements:
        for r in analysis.reinforcements:
            lines.append(
                f"- **{', '.join(r.exp_ids)}** "
                f"({r.strength}): {r.description}"
            )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("### Foundation-chain narratives")
    if analysis.chain_narratives:
        for cn in analysis.chain_narratives:
            walks_str = " → ".join(cn.walk_ids_in_order)
            lines.append(
                f"- **chain {cn.chain_id}** "
                f"(root={cn.root_hypothesis_id}): {walks_str}"
            )
            lines.append(f"  - {cn.story}")
    else:
        lines.append("- (no chains)")

    return "\n".join(lines)


__all__ = [
    "Contradiction",
    "Reinforcement",
    "ChainNarrative",
    "CrossExperimentAnalysis",
    "analyze_cross_experiment",
    "cross_experiment_block_for_prompt",
]
