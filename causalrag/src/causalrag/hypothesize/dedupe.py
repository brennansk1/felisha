"""Hypothesis dedup pass for the candidate-queue planner.

After the up-front planner emits 15-30 candidates, several near-duplicates
typically slip through: same (treatment, outcome, estimand_class) with
slightly different modifier subsets, or candidates whose modifier sets
are strict supersets of one another. Running every one of those wastes
budget and clutters the synthesis.

This module does two passes:

1. A **deterministic pre-pass** that collapses *exact* matches on
   ``(treatment, outcome, estimand_class, frozenset(modifiers))`` —
   keeping the candidate with the highest ``impact_hint`` (ties broken
   by ``candidate_id`` for stable ordering).
2. An **optional LLM refinement** that sees the survivors + the
   deterministic merge proposals and returns additional merges / prunes
   for cases where one candidate is trivially redundant given another
   (e.g. modifiers strictly subsume another's modifiers). The LLM output
   is validated: any referenced ``candidate_id`` that does not exist in
   the input is dropped with a warning, and merges that would empty the
   survivor list are refused. LLM failures fall back to the
   deterministic-only result — this function never re-raises.

Wiring into ``master_loop.py`` is intentionally NOT done here; that's a
separate, narrow change.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # avoid an import cycle at module load
    from causalrag.llm.ollama_client import OllamaClient
    from causalrag.master_loop import CandidateExperiment


logger = logging.getLogger(__name__)


# ─────────── Schemas ──────────────────────────────────────────────────────


class PruneAction(BaseModel):
    """Drop ``candidate_id`` outright (no merge target)."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    reason: str


class MergeAction(BaseModel):
    """Keep ``kept_id``, drop every id in ``dropped_ids``.

    ``merged_rationale`` explains why the dropped candidates are
    redundant given the kept one — surfaced into the decision ledger by
    the caller.
    """

    model_config = ConfigDict(extra="forbid")

    kept_id: str
    dropped_ids: list[str]
    merged_rationale: str


class DedupePlan(BaseModel):
    """Aggregate output of the dedup pass."""

    model_config = ConfigDict(extra="forbid")

    pruned: list[PruneAction] = Field(default_factory=list)
    merged: list[MergeAction] = Field(default_factory=list)
    note: str | None = None


# ─────────── Deterministic pre-pass ───────────────────────────────────────


def _signature(c: "CandidateExperiment") -> tuple[str, str, str, frozenset[str]]:
    """Equivalence key used by the deterministic pre-pass."""
    return (
        (c.treatment or "").strip(),
        (c.outcome or "").strip(),
        (c.estimand_class or "").upper().strip(),
        frozenset(m.strip() for m in (c.modifiers or ())),
    )


def _deterministic_merges(
    candidates: list["CandidateExperiment"],
) -> list[MergeAction]:
    """Group exact-signature duplicates, keep the highest-impact one.

    Stable: ties on ``impact_hint`` are broken by the candidate's
    position in the input list (i.e. the first one wins) — which in turn
    makes ``candidate_id`` the tiebreaker because the planner assigns
    ids deterministically.
    """
    groups: dict[tuple[str, str, str, frozenset[str]], list["CandidateExperiment"]] = {}
    for c in candidates:
        groups.setdefault(_signature(c), []).append(c)

    merges: list[MergeAction] = []
    for sig, members in groups.items():
        if len(members) < 2:
            continue
        # Sort members: highest impact_hint first, original order as tiebreaker.
        order = {c.candidate_id: i for i, c in enumerate(members)}
        members_sorted = sorted(
            members,
            key=lambda c: (-float(c.impact_hint), order[c.candidate_id]),
        )
        kept = members_sorted[0]
        dropped = members_sorted[1:]
        merges.append(
            MergeAction(
                kept_id=kept.candidate_id,
                dropped_ids=[c.candidate_id for c in dropped],
                merged_rationale=(
                    f"Exact (T, Y, estimand, modifiers) match on "
                    f"{sig[0]} → {sig[1]} ({sig[2]}); kept highest-impact "
                    f"candidate ({kept.candidate_id}, impact_hint="
                    f"{kept.impact_hint:.2f})."
                ),
            )
        )
    return merges


# ─────────── LLM prompt ───────────────────────────────────────────────────


_DEDUP_SYSTEM = (
    "You are a senior causal-inference referee performing a dedup pass on a "
    "candidate-experiment queue. The queue already had EXACT duplicates "
    "collapsed deterministically. Your job is the harder case: candidates "
    "whose (treatment, outcome, estimand_class) match and whose modifier "
    "sets are near-redundant (strict subsets, single-element-different "
    "supersets, or semantically equivalent re-statements).\n\n"
    "Decision rules:\n"
    "  • MERGE two candidates A and B if running B after A would add no "
    "    new information a referee would care about — e.g. B's modifiers "
    "    are a strict subset of A's, or A and B differ only in a "
    "    cosmetic re-phrasing of the research question.\n"
    "  • PRUNE a candidate that is dominated on every dimension AND adds "
    "    no novelty (rare; reserve for clearly redundant entries).\n"
    "  • DO NOT merge candidates with different estimand_classes (ATE vs "
    "    CATE is genuinely different even on the same T/Y).\n"
    "  • DO NOT merge candidates with different treatments or outcomes.\n"
    "  • Every candidate_id you reference MUST appear in the input list.\n"
    "  • If nothing to merge or prune beyond the deterministic pre-pass, "
    "    return an empty DedupePlan with a brief explanatory note.\n\n"
    "Return ONLY a JSON DedupePlan."
)


def _build_dedup_prompt(
    candidates: list["CandidateExperiment"],
    deterministic: list[MergeAction],
) -> str:
    parts: list[str] = []
    parts.append("## Candidates (post-deterministic-merge survivors)")
    for c in candidates:
        mods = ", ".join(c.modifiers) if c.modifiers else "—"
        parts.append(
            f"  - {c.candidate_id}: {c.treatment} → {c.outcome} "
            f"({c.estimand_class}); modifiers=[{mods}]; "
            f"mediator={c.mediator}; instrument={c.instrument}; "
            f"impact_hint={c.impact_hint:.2f}"
        )
        parts.append(f"      research_question: {c.research_question}")

    parts.append("")
    parts.append("## Deterministic merges already applied")
    if deterministic:
        for m in deterministic:
            parts.append(
                f"  - kept {m.kept_id}, dropped {m.dropped_ids}: {m.merged_rationale}"
            )
    else:
        parts.append("  (none)")

    parts.append("")
    parts.append(
        "## Task\nIdentify any further near-duplicates that should be "
        "merged or pruned. Return ONLY a JSON DedupePlan."
    )
    return "\n".join(parts)


# ─────────── Validation helpers ───────────────────────────────────────────


def _validate_plan(
    plan: DedupePlan, valid_ids: set[str]
) -> tuple[DedupePlan, list[str]]:
    """Strip references to unknown ids; return cleaned plan + warnings."""
    warnings: list[str] = []
    clean_merges: list[MergeAction] = []
    for m in plan.merged:
        if m.kept_id not in valid_ids:
            warnings.append(
                f"merge dropped: kept_id {m.kept_id!r} not in input"
            )
            continue
        clean_dropped = [d for d in m.dropped_ids if d in valid_ids]
        skipped = [d for d in m.dropped_ids if d not in valid_ids]
        for d in skipped:
            warnings.append(
                f"merge {m.kept_id}: dropped_id {d!r} not in input — ignored"
            )
        if m.kept_id in clean_dropped:
            warnings.append(
                f"merge {m.kept_id}: refused to drop kept_id from its own dropped_ids"
            )
            clean_dropped = [d for d in clean_dropped if d != m.kept_id]
        if not clean_dropped:
            warnings.append(
                f"merge {m.kept_id}: no valid dropped_ids remain — skipping"
            )
            continue
        clean_merges.append(
            MergeAction(
                kept_id=m.kept_id,
                dropped_ids=clean_dropped,
                merged_rationale=m.merged_rationale,
            )
        )

    clean_prunes: list[PruneAction] = []
    for p in plan.pruned:
        if p.candidate_id not in valid_ids:
            warnings.append(
                f"prune dropped: candidate_id {p.candidate_id!r} not in input"
            )
            continue
        clean_prunes.append(p)

    return (
        DedupePlan(pruned=clean_prunes, merged=clean_merges, note=plan.note),
        warnings,
    )


def _apply_plan(
    candidates: list["CandidateExperiment"], plan: DedupePlan
) -> list["CandidateExperiment"]:
    """Remove all dropped/pruned ids; preserve input order."""
    to_drop: set[str] = set()
    for m in plan.merged:
        to_drop.update(m.dropped_ids)
    for p in plan.pruned:
        to_drop.add(p.candidate_id)
    return [c for c in candidates if c.candidate_id not in to_drop]


# ─────────── Public entry point ───────────────────────────────────────────


def dedupe_candidates(
    candidates: list["CandidateExperiment"],
    *,
    client: "OllamaClient | None",
) -> tuple[list["CandidateExperiment"], DedupePlan]:
    """Run the dedup pass over a planner-produced candidate list.

    Always runs the deterministic pre-pass. If ``client`` is provided, an
    LLM refinement runs on the survivors and any additional merges/prunes
    it returns are validated and applied. The LLM is best-effort: any
    exception falls back to the deterministic-only result and is logged.
    """
    if not candidates:
        return [], DedupePlan()

    valid_ids = {c.candidate_id for c in candidates}

    # 1. Deterministic exact-match pre-pass.
    det_merges = _deterministic_merges(candidates)
    det_plan = DedupePlan(merged=det_merges)
    survivors = _apply_plan(candidates, det_plan)

    if client is None or not survivors:
        return survivors, det_plan

    # 2. Optional LLM refinement on the survivors.
    try:
        prompt = _build_dedup_prompt(survivors, det_merges)
        resp = client.parse(
            prompt=prompt,
            schema=DedupePlan,
            system=_DEDUP_SYSTEM,
            json_schema=DedupePlan.model_json_schema(),
        )
        llm_plan = resp.parsed
        assert isinstance(llm_plan, DedupePlan)
    except Exception as e:  # noqa: BLE001 — never raise from a dedup pass
        logger.warning("dedupe LLM pass failed: %s: %s", type(e).__name__, e)
        return survivors, det_plan

    survivor_ids = {c.candidate_id for c in survivors}
    cleaned, warnings = _validate_plan(llm_plan, survivor_ids)
    for w in warnings:
        logger.warning("dedupe: %s", w)

    final = _apply_plan(survivors, cleaned)

    # Merge the two plans for the caller. Note: warnings appended to note.
    combined_note_parts: list[str] = []
    if det_plan.note:
        combined_note_parts.append(det_plan.note)
    if cleaned.note:
        combined_note_parts.append(cleaned.note)
    if warnings:
        combined_note_parts.append("; ".join(warnings))

    combined = DedupePlan(
        pruned=cleaned.pruned,
        merged=det_plan.merged + cleaned.merged,
        note=" | ".join(combined_note_parts) if combined_note_parts else None,
    )
    return final, combined


__all__ = [
    "DedupePlan",
    "MergeAction",
    "PruneAction",
    "dedupe_candidates",
]
