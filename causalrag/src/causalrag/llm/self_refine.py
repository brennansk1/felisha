"""Self-Refine / Reflexion loop for the critic agent (PDD §33 / Sprint 3.3).

When the propose-K critic rejects a candidate, the master loop's planner can
either re-propose the same candidate or simply drop it. Both are wasteful when
the rejection has a structured, fixable cause (e.g., the recommended estimator
needed the ``rbridge`` backend; the candidate duplicated a completed
experiment; the minimum sample size for that estimand class was violated).

This module implements a one-shot Self-Refine round (Madaan et al. 2023) that
chains two small focused LLM calls:

  1. ``CriticReflection`` — given the candidate, the verbatim rejection
     reason, and recent history, the model produces a *structured*
     diagnosis: which failure category fired, a short explanation, the
     proposed revision direction, and a revision-type tag. This is the
     Reflexion-style verbal feedback (Shinn et al. 2023) but constrained
     to a Pydantic schema so the downstream planner can act on it
     mechanically.
  2. ``CriticRevision`` — given the reflection plus the catalog and
     history, the planner emits a delta over the original candidate
     (new treatment / outcome / estimand / method / mediator /
     instrument, or ``abandon=True``). It does not re-emit the whole
     candidate; the master loop merges the delta on top.

The loop is failure-safe: any ``OllamaClient`` exception (cassette miss,
schema-validation failure after retries, network error, JSONDecodeError) is
caught and converted into a ``CriticRevision`` with ``abandon=True`` and the
error captured in ``revision_rationale``. This guarantees the master loop
always gets a well-typed result and can continue.

We intentionally do **not** edit ``master_loop.py``, the existing critic
prompt, or any discovery/synthesis prompts. This module is callable from the
loop but is otherwise self-contained.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from causalrag.llm.honesty import with_honesty

# ─────────── Schemas ──────────────────────────────────────────────────────


PrimaryFailure = Literal[
    "identifiability_weak",
    "min_sample_size_violated",
    "estimand_method_mismatch",
    "duplicate_prior_experiment",
    "missing_required_column",
    "unsupported_data_type",
    "other",
]


RevisionType = Literal[
    "swap_estimator",
    "narrow_estimand",
    "change_estimand_class",
    "add_instrument",
    "add_mediator",
    "abandon_candidate",
]


class CriticReflection(BaseModel):
    """The critic's structured reflection on WHY a candidate was rejected.

    The planner reads this to revise the candidate rather than re-proposing
    the same one.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    primary_failure: PrimaryFailure
    failure_explanation: str  # 1-2 sentence diagnosis
    proposed_revision: str  # e.g., "swap recommended_method to rbridge.weightit"
    revision_type: RevisionType


class CriticRevision(BaseModel):
    """Planner's revised candidate after reading the reflection."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    new_treatment: str | None = None
    new_outcome: str | None = None
    new_estimand_class: str | None = None
    new_recommended_method: str | None = None
    new_mediator: str | None = None
    new_instrument: str | None = None
    abandon: bool = False
    revision_rationale: str


# ─────────── Duck-typed candidate protocol ────────────────────────────────


class _CandidateLike(Protocol):
    """Subset of ``CandidateExperiment`` attributes used by this module.

    Duck-typed on purpose: the real type lives in ``master_loop`` and we
    don't want a cyclic import.
    """

    candidate_id: str
    treatment: str
    outcome: str
    estimand_class: str
    recommended_method: str | None
    mediator: str | None
    instrument: str | None
    research_question: str


# ─────────── System prompts ───────────────────────────────────────────────


_REFLECTION_SYSTEM = (
    "You are the *reflection* half of a Self-Refine loop on a causal-experiment "
    "critic. A previous critic rejected one candidate experiment. Your job is to "
    "diagnose WHY in a structured form so a downstream planner can fix the "
    "candidate.\n\n"
    "Output a single CriticReflection JSON object. Choose `primary_failure` from "
    "the schema's enum, matching the verbatim rejection reason as literally as "
    "possible:\n"
    "  - 'min_sample_size_violated' when the reason mentions 'min sample size', "
    "    'n too small', 'underpowered', 'minimum n', or 'sample size'.\n"
    "  - 'duplicate_prior_experiment' when the same (treatment, outcome, "
    "    estimand_class) triple already appears in completed_history.\n"
    "  - 'estimand_method_mismatch' when the reason mentions estimator/method "
    "    not supporting the estimand class, wrong backend, or invalid catalog id.\n"
    "  - 'identifiability_weak' when the reason mentions confounding, "
    "    overlap/positivity, unmeasured confounders, or weak instrument.\n"
    "  - 'missing_required_column' when the reason mentions a missing column.\n"
    "  - 'unsupported_data_type' when the reason mentions dtype/binary/continuous "
    "    incompatibility.\n"
    "  - 'other' otherwise.\n\n"
    "Keep `failure_explanation` to 1-2 sentences. `proposed_revision` should be "
    "a concrete, actionable string (e.g. 'swap recommended_method to "
    "rbridge.weightit'); `revision_type` must be the matching enum value."
)


_REVISION_SYSTEM = (
    "You are the *revision* half of a Self-Refine loop. Given the original "
    "candidate, the structured reflection from the previous step, the catalog "
    "of available estimators, and the completed-experiment history, emit a "
    "CriticRevision JSON object describing the minimal delta to apply on top "
    "of the original candidate.\n\n"
    "Rules:\n"
    "  - Only set a `new_*` field if you are CHANGING that attribute. Leave "
    "    the rest as null.\n"
    "  - If `revision_type` is 'abandon_candidate' OR the reflection's primary "
    "    failure is 'duplicate_prior_experiment', set `abandon=true`.\n"
    "  - If you set `new_recommended_method`, it MUST be a catalog id quoted "
    "    from the catalog block — never invent estimator ids.\n"
    "  - `revision_rationale` should briefly cite the reflection's "
    "    failure_explanation and the specific change being made.\n\n"
    "Catalog (verbatim — do not fabricate ids that are not in this table):\n"
    "{CATALOG_MARKDOWN}"
)


# ─────────── Prompt builders ──────────────────────────────────────────────


def _format_candidate(c: _CandidateLike) -> str:
    return (
        f"  - candidate_id: {c.candidate_id}\n"
        f"  - research_question: {c.research_question}\n"
        f"  - treatment: {c.treatment}\n"
        f"  - outcome: {c.outcome}\n"
        f"  - estimand_class: {c.estimand_class}\n"
        f"  - recommended_method: {c.recommended_method or '(auto)'}\n"
        f"  - mediator: {c.mediator}\n"
        f"  - instrument: {c.instrument}"
    )


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "  (no completed experiments yet)"
    lines: list[str] = []
    for h in history[-10:]:
        t = h.get("treatment", "?")
        y = h.get("outcome", "?")
        k = h.get("estimand_class", "?")
        hid = h.get("id", h.get("hypothesis_id", "?"))
        lines.append(f"  - {hid}: {t} -> {y} ({k})")
    return "\n".join(lines)


def _build_reflection_prompt(
    candidate: _CandidateLike,
    rejection_reason: str,
    history: list[dict[str, Any]],
) -> str:
    return (
        "## Rejected candidate\n"
        + _format_candidate(candidate)
        + "\n\n## Verbatim rejection reason\n"
        + (rejection_reason or "(none provided)")
        + "\n\n## Completed-experiment history (most recent last)\n"
        + _format_history(history)
        + "\n\n## Task\nReturn a single CriticReflection JSON object diagnosing "
        "the rejection. Match the rejection reason's wording when choosing "
        "primary_failure."
    )


def _build_revision_prompt(
    candidate: _CandidateLike,
    reflection: CriticReflection,
    history: list[dict[str, Any]],
) -> str:
    return (
        "## Original candidate\n"
        + _format_candidate(candidate)
        + "\n\n## Structured reflection (from previous step)\n"
        + json.dumps(reflection.model_dump(), indent=2)
        + "\n\n## Completed-experiment history\n"
        + _format_history(history)
        + "\n\n## Task\nReturn a single CriticRevision JSON object with the "
        "minimal delta to fix the candidate. Set abandon=true only when the "
        "reflection says 'abandon_candidate' or 'duplicate_prior_experiment'."
    )


# ─────────── Heuristic fallback for primary_failure ───────────────────────


_MIN_SAMPLE_TOKENS = (
    "min sample size",
    "minimum sample size",
    "minimum n",
    "n too small",
    "sample size",
    "underpowered",
)

_DUPLICATE_TOKENS = (
    "duplicate",
    "already run",
    "already completed",
    "previously run",
    "prior experiment",
)

_METHOD_MISMATCH_TOKENS = (
    "method does not support",
    "estimator does not support",
    "incompatible estimator",
    "wrong backend",
    "invalid catalog id",
    "method not valid",
    "estimand_method_mismatch",
)

_IDENTIFIABILITY_TOKENS = (
    "unmeasured confound",
    "positivity",
    "overlap",
    "weak instrument",
    "identifiability",
    "confounding",
)

_MISSING_COL_TOKENS = ("missing column", "column not found", "required column")

_UNSUPPORTED_TYPE_TOKENS = ("unsupported data type", "dtype", "binary outcome not", "continuous required")


def _heuristic_primary_failure(
    candidate: _CandidateLike,
    rejection_reason: str,
    history: list[dict[str, Any]],
) -> PrimaryFailure:
    """A small deterministic fallback that the LLM's reflection should agree with.

    Used only when the LLM call fails entirely (so the model never gets a
    chance to vote) and we want to attach a best-effort category to the
    abandon-with-error revision; also used inside tests to assert the
    model's classification aligns with the obvious signal in the prompt.
    """
    r = (rejection_reason or "").lower()
    # Duplicate-history check is structural — beats any string match.
    for h in history:
        if (
            h.get("treatment") == candidate.treatment
            and h.get("outcome") == candidate.outcome
            and h.get("estimand_class") == candidate.estimand_class
        ):
            return "duplicate_prior_experiment"
    for tok in _DUPLICATE_TOKENS:
        if tok in r:
            return "duplicate_prior_experiment"
    for tok in _MIN_SAMPLE_TOKENS:
        if tok in r:
            return "min_sample_size_violated"
    for tok in _METHOD_MISMATCH_TOKENS:
        if tok in r:
            return "estimand_method_mismatch"
    for tok in _MISSING_COL_TOKENS:
        if tok in r:
            return "missing_required_column"
    for tok in _UNSUPPORTED_TYPE_TOKENS:
        if tok in r:
            return "unsupported_data_type"
    for tok in _IDENTIFIABILITY_TOKENS:
        if tok in r:
            return "identifiability_weak"
    return "other"


# ─────────── Main entry point ─────────────────────────────────────────────


def self_refine_critic(
    *,
    candidate: _CandidateLike,
    rejection_reason: str,
    completed_history: list[dict[str, Any]],
    catalog_markdown: str,
    client: Any,  # OllamaClient (duck-typed to keep this module import-light)
    max_rounds: int = 1,
) -> tuple[CriticRevision, list[CriticReflection]]:
    """Run a single Self-Refine round on a rejected candidate.

    Returns the revised candidate spec + the reflection history.
    Failure-safe: any LLM error → returns a CriticRevision with
    ``abandon=True`` and the error in ``revision_rationale``.

    ``max_rounds`` is accepted for forward-compatibility with Reflexion-style
    multi-round refinement (Shinn 2023). Only the first round is wired today.
    """
    reflections: list[CriticReflection] = []

    rounds = max(1, int(max_rounds))
    last_error: str | None = None

    for _ in range(rounds):
        # --- 1) Reflection call -------------------------------------------
        try:
            refl_resp = client.parse(
                prompt=_build_reflection_prompt(candidate, rejection_reason, completed_history),
                schema=CriticReflection,
                system=with_honesty(_REFLECTION_SYSTEM),
                json_schema=CriticReflection.model_json_schema(),
            )
            reflection = refl_resp.parsed
            assert isinstance(reflection, CriticReflection)
            reflections.append(reflection)
        except Exception as exc:  # noqa: BLE001 — failure-safe by design
            last_error = f"reflection failed: {type(exc).__name__}: {exc}"
            continue

        # --- 2) Revision call ---------------------------------------------
        try:
            rev_resp = client.parse(
                prompt=_build_revision_prompt(candidate, reflection, completed_history),
                schema=CriticRevision,
                system=with_honesty(
                    _REVISION_SYSTEM.replace("{CATALOG_MARKDOWN}", catalog_markdown)
                ),
                json_schema=CriticRevision.model_json_schema(),
            )
            revision = rev_resp.parsed
            assert isinstance(revision, CriticRevision)
        except Exception as exc:  # noqa: BLE001
            last_error = f"revision failed: {type(exc).__name__}: {exc}"
            continue

        # Force-abandon if reflection said duplicate — the planner shouldn't
        # be allowed to "fix" a duplicate by tweaking method.
        if reflection.primary_failure == "duplicate_prior_experiment":
            revision = revision.model_copy(
                update={
                    "abandon": True,
                    "revision_rationale": (
                        revision.revision_rationale
                        + " [forced abandon: duplicate_prior_experiment]"
                    ),
                }
            )
        elif reflection.revision_type == "abandon_candidate" and not revision.abandon:
            revision = revision.model_copy(update={"abandon": True})

        return revision, reflections

    # All rounds failed — return a safe-abandon revision.
    return (
        CriticRevision(
            candidate_id=candidate.candidate_id,
            abandon=True,
            revision_rationale=(
                f"self-refine aborted; {last_error or 'unknown error'}"
            ),
        ),
        reflections,
    )


__all__ = [
    "CriticReflection",
    "CriticRevision",
    "PrimaryFailure",
    "RevisionType",
    "self_refine_critic",
]
