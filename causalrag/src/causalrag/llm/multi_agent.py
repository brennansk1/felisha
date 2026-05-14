"""Multi-agent debate for ``--paranoid`` mode (PDD §33 / Sprint 8.4).

Standard propose-K relies on a single planner + a single critic. Under
``--paranoid`` the user wants three independent agents to argue before
the loop commits a candidate to execution:

  * **Planner** — the existing candidate proposer (passed in via
    ``candidate`` plus an optional revision call we may issue here).
  * **Skeptic** — a methodologist looking for risk-of-bias (RoB),
    identifiability gaps, and SUTVA violations.
  * **Statistician** — a senior applied statistician looking at
    estimator choice, cross-fitting, and statistical power.

The three agents reach consensus over up to ``max_rounds`` rounds; the
final outcome is a :class:`DebateConsensus` that the master loop can
merge onto the original candidate. Costs roughly **3× tokens** per
candidate (one call per agent, plus a planner revision call when
either challenger demands changes).

This module is intentionally self-contained — it does **not** import
or modify ``master_loop.py``. Wiring (i.e., deciding when to call
``run_debate``) is a separate concern handled by the loop.

Failure-safe by design: any LLM transport error returns a
``DebateConsensus(keep=True, rationale='debate aborted: <reason>')`` so
the loop never crashes when a challenger model is unavailable.

Consensus rule (deterministic, computed from the three verdicts):

    keep iff at least 2 of 3 agents say "accept" or "revise"

where the three agents are skeptic, statistician, and (implicitly)
the planner — the planner's "vote" is encoded in whether it produced
a non-trivial revision (``accept`` if it didn't have to revise at all,
``revise`` if it did, ``reject`` only if both challengers said reject).
"""

from __future__ import annotations

import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from causalrag.llm.honesty import with_honesty

# ─────────── Schemas ──────────────────────────────────────────────────────


Verdict = Literal["accept", "revise", "reject"]


class SkepticChallenge(BaseModel):
    """The Skeptic agent's structured critique of a candidate.

    Focused on identification (back-door / front-door / IV validity),
    risk of bias (selection, measurement, attrition, info), and SUTVA
    (no interference, no hidden treatment variation)."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    identification_concerns: list[str] = Field(default_factory=list)
    rob_concerns: list[str] = Field(default_factory=list)
    sutva_concerns: list[str] = Field(default_factory=list)
    overall_verdict: Verdict


class StatisticianChallenge(BaseModel):
    """The Statistician agent's structured critique.

    Focused on the estimator choice (does it match the estimand?), the
    cross-fitting / nuisance regime, and whether the design has the
    power to detect a meaningful effect."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    estimator_concerns: list[str] = Field(default_factory=list)
    power_concerns: list[str] = Field(default_factory=list)
    cross_fit_concerns: list[str] = Field(default_factory=list)
    recommended_changes: list[str] = Field(default_factory=list)
    overall_verdict: Verdict


class DebateConsensus(BaseModel):
    """Final, mergeable verdict produced by ``run_debate``.

    ``revised_method`` and ``revised_estimand`` are only populated when
    at least one of the two challengers asked for that specific change.
    ``revised_modifiers`` is the union of any modifier suggestions
    reachable from the statistician's ``recommended_changes``.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    keep: bool
    revised_method: str | None = None
    revised_estimand: str | None = None
    revised_modifiers: list[str] = Field(default_factory=list)
    rationale: str


# ─────────── Duck-typed candidate protocol ────────────────────────────────


class _CandidateLike(Protocol):
    """Subset of ``CandidateExperiment`` attributes we touch.

    Duck-typed on purpose so we don't have to import ``master_loop``."""

    candidate_id: str
    treatment: str
    outcome: str
    estimand_class: str
    recommended_method: str | None
    mediator: str | None
    instrument: str | None
    research_question: str


# ─────────── System prompts ───────────────────────────────────────────────


_SKEPTIC_SYSTEM = (
    "You are the SKEPTIC agent in a 3-agent causal-inference debate. Your job "
    "is to find every plausible reason the proposed experiment will NOT "
    "identify the target estimand, focusing on:\n"
    "  * identification — back-door / front-door / IV validity, unmeasured "
    "    confounding, positivity / overlap, exclusion restriction.\n"
    "  * risk of bias — selection, measurement, attrition, information bias, "
    "    differential measurement across arms.\n"
    "  * SUTVA — interference between units, hidden treatment-variant arms, "
    "    spillovers.\n\n"
    "Return ONE SkepticChallenge JSON object. Be specific — each concern "
    "string should name the actual mechanism (e.g., 'unmeasured U: income "
    "drives both T and Y') not a generic platitude. Set `overall_verdict`:\n"
    "  - 'accept' if you find no blocking concerns;\n"
    "  - 'revise' if the candidate is salvageable with a concrete fix "
    "    (different estimand, add an instrument, restrict the population);\n"
    "  - 'reject' if identification is fundamentally infeasible with this data."
)


_STATISTICIAN_SYSTEM = (
    "You are the STATISTICIAN agent in a 3-agent causal-inference debate. "
    "Your job is to evaluate the *estimation* plan, NOT identification "
    "(that's the Skeptic's job). Focus on:\n"
    "  * estimator choice — does the recommended_method actually target the "
    "    declared estimand_class? Is the family appropriate (DML vs IPW vs "
    "    matching vs RDD vs IV)?\n"
    "  * cross-fitting / nuisance — sample-splitting regime, risk of "
    "    overfitting, double-robustness, tuning of ML nuisances.\n"
    "  * power — given n, base rates, and plausible effect sizes, is the "
    "    design likely to detect a meaningful effect at the standard alpha?\n\n"
    "Return ONE StatisticianChallenge JSON object. Put concrete suggested "
    "estimator IDs (e.g. 'rbridge.weightit', 'econml.linear_dml') in "
    "`recommended_changes` — never invent estimator ids that aren't in the "
    "catalog block. Set `overall_verdict`:\n"
    "  - 'accept' if the estimator + cross-fit + power look adequate;\n"
    "  - 'revise' if a concrete swap or modifier would fix it;\n"
    "  - 'reject' if no estimator in the catalog can reasonably handle the "
    "    design.\n\n"
    "Catalog (verbatim — do not invent ids):\n"
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


def _build_skeptic_prompt(
    candidate: _CandidateLike,
    history: list[dict[str, Any]],
    domain_brief: str | None,
    prior_statistician: StatisticianChallenge | None = None,
) -> str:
    parts = [
        "## Candidate experiment under debate",
        _format_candidate(candidate),
        "",
        "## Domain brief",
        (domain_brief or "(none provided)")[:1500],
        "",
        "## Completed-experiment history",
        _format_history(history),
    ]
    if prior_statistician is not None:
        parts += [
            "",
            "## Statistician's prior-round critique (for context — do NOT just defer to it)",
            json.dumps(prior_statistician.model_dump(), indent=2),
        ]
    parts += [
        "",
        "## Task",
        "Return ONE SkepticChallenge JSON object. List concrete, named "
        "concerns. Choose overall_verdict deliberately.",
    ]
    return "\n".join(parts)


def _build_statistician_prompt(
    candidate: _CandidateLike,
    history: list[dict[str, Any]],
    domain_brief: str | None,
    prior_skeptic: SkepticChallenge | None = None,
) -> str:
    parts = [
        "## Candidate experiment under debate",
        _format_candidate(candidate),
        "",
        "## Domain brief",
        (domain_brief or "(none provided)")[:1500],
        "",
        "## Completed-experiment history",
        _format_history(history),
    ]
    if prior_skeptic is not None:
        parts += [
            "",
            "## Skeptic's prior-round critique (for context — do NOT just defer to it)",
            json.dumps(prior_skeptic.model_dump(), indent=2),
        ]
    parts += [
        "",
        "## Task",
        "Return ONE StatisticianChallenge JSON object. Put concrete catalog "
        "ids in recommended_changes where applicable.",
    ]
    return "\n".join(parts)


# ─────────── Client resolution ────────────────────────────────────────────


def _resolve_clients(client: Any) -> tuple[Any, Any, Any]:
    """``client`` may be a single OllamaClient or a 3-tuple/dict of
    distinct planner/skeptic/statistician clients (e.g., to run each
    agent on a different model). We return ``(planner, skeptic, stat)``
    regardless of the input shape."""
    if isinstance(client, tuple) and len(client) == 3:
        return client[0], client[1], client[2]
    if isinstance(client, dict):
        try:
            return client["planner"], client["skeptic"], client["statistician"]
        except KeyError as exc:  # pragma: no cover — defensive
            raise ValueError(
                "client dict must have keys 'planner', 'skeptic', 'statistician'"
            ) from exc
    return client, client, client


# ─────────── Consensus rule ───────────────────────────────────────────────


def _planner_self_verdict(
    skeptic: SkepticChallenge | None,
    statistician: StatisticianChallenge | None,
) -> Verdict:
    """The planner's implicit vote.

    We don't issue a separate planner-vote LLM call; instead we infer
    it from what the challengers said. The planner's stance is:

      * 'accept' if neither challenger demanded a change;
      * 'revise' if exactly one challenger demanded a revision;
      * 'reject' only when both challengers reject.

    This keeps the debate to 2 LLM calls per round (skeptic + stat)
    plus the planner's existing proposal call, totalling ~3× a normal
    single-LLM iteration."""
    s_v = skeptic.overall_verdict if skeptic else "accept"
    t_v = statistician.overall_verdict if statistician else "accept"
    if s_v == "reject" and t_v == "reject":
        return "reject"
    if s_v == "accept" and t_v == "accept":
        return "accept"
    return "revise"


def _consensus_keep(
    skeptic: SkepticChallenge | None,
    statistician: StatisticianChallenge | None,
) -> tuple[bool, Verdict, Verdict, Verdict]:
    """Compute ``keep`` and return the three verdicts.

    Rule: keep iff at least 2 of 3 verdicts are in {accept, revise}."""
    s_v: Verdict = skeptic.overall_verdict if skeptic else "accept"
    t_v: Verdict = statistician.overall_verdict if statistician else "accept"
    p_v = _planner_self_verdict(skeptic, statistician)
    keep_count = sum(1 for v in (s_v, t_v, p_v) if v != "reject")
    keep = keep_count >= 2
    return keep, s_v, t_v, p_v


# ─────────── Revision extraction ──────────────────────────────────────────


def _extract_revised_method(
    candidate: _CandidateLike,
    statistician: StatisticianChallenge | None,
    catalog_markdown: str,
) -> str | None:
    """Pull a catalog-id from statistician.recommended_changes if any.

    We only set a revised method when the statistician explicitly
    requested one (matching a token in the catalog block). The skeptic
    is allowed to *influence* the choice (via its identification
    concerns) but cannot directly dictate an estimator id."""
    if statistician is None:
        return None
    if statistician.overall_verdict == "accept":
        return None
    # Build the set of known catalog ids from the markdown table.
    candidate_ids = _parse_catalog_ids(catalog_markdown)
    for change in statistician.recommended_changes:
        for tok in candidate_ids:
            if tok and tok in change and tok != candidate.recommended_method:
                return tok
    return None


def _extract_revised_estimand(
    candidate: _CandidateLike,
    skeptic: SkepticChallenge | None,
    statistician: StatisticianChallenge | None,
) -> str | None:
    """Pull a revised estimand from challengers' concern text.

    A revision only fires when at least one challenger said 'revise'
    or 'reject' AND a known estimand token appears in one of their
    concern fields that differs from the candidate's current
    estimand_class."""
    if skeptic is None and statistician is None:
        return None
    s_wants_change = skeptic is not None and skeptic.overall_verdict in ("revise", "reject")
    t_wants_change = statistician is not None and statistician.overall_verdict in ("revise", "reject")
    if not (s_wants_change or t_wants_change):
        return None

    haystack_parts: list[str] = []
    if skeptic is not None:
        haystack_parts += skeptic.identification_concerns
        haystack_parts += skeptic.rob_concerns
        haystack_parts += skeptic.sutva_concerns
    if statistician is not None:
        haystack_parts += statistician.recommended_changes
        haystack_parts += statistician.estimator_concerns
    haystack = " ".join(haystack_parts).upper()

    known = ("ATE", "ATT", "ATC", "CATE", "NDE", "NIE", "LATE", "RMST_CONTRAST", "MTP")
    current = (candidate.estimand_class or "").upper()
    for est in known:
        if est == current:
            continue
        # Bracket the token with word boundaries to avoid e.g. ATE matching CATE.
        if f" {est} " in f" {haystack} " or f" {est}," in haystack or f" {est}." in haystack:
            return est
    return None


def _extract_revised_modifiers(
    statistician: StatisticianChallenge | None,
) -> list[str]:
    if statistician is None:
        return []
    if statistician.overall_verdict == "accept":
        return []
    mods: list[str] = []
    for change in statistician.recommended_changes:
        low = change.lower()
        if "modifier" in low or "subgroup" in low or "stratif" in low:
            mods.append(change)
    return mods


def _parse_catalog_ids(catalog_markdown: str) -> list[str]:
    """Best-effort extraction of catalog estimator ids from a markdown
    table. We look for tokens that look like ``family.something`` since
    that's the convention used elsewhere in the codebase (e.g.,
    ``rbridge.weightit``, ``econml.linear_dml``). Plain words are
    skipped to avoid false positives."""
    ids: list[str] = []
    for line in catalog_markdown.splitlines():
        for cell in line.split("|"):
            tok = cell.strip()
            if "." in tok and " " not in tok and len(tok) > 3:
                ids.append(tok)
    # Preserve ordering, de-dup.
    seen: set[str] = set()
    out: list[str] = []
    for t in ids:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ─────────── Main entry point ─────────────────────────────────────────────


def run_debate(
    *,
    candidate: _CandidateLike,
    completed_history: list[dict[str, Any]],
    catalog_markdown: str,
    domain_brief: str | None,
    client: Any,
    max_rounds: int = 2,
) -> DebateConsensus:
    """Run the 3-agent debate over a single candidate.

    Issues 2 LLM calls per round (skeptic + statistician); the planner's
    vote is computed deterministically from the challengers' verdicts.
    Returns a :class:`DebateConsensus` that the master loop can merge
    onto the original candidate.

    Failure-safe: any exception thrown by ``client.parse`` returns
    ``DebateConsensus(keep=True, rationale='debate aborted: <reason>')``
    so the loop never crashes when a debate model is unavailable.
    """
    rounds = max(1, int(max_rounds))
    planner_client, skeptic_client, stat_client = _resolve_clients(client)

    skeptic: SkepticChallenge | None = None
    statistician: StatisticianChallenge | None = None

    for round_idx in range(rounds):
        # --- Skeptic call -------------------------------------------------
        try:
            sresp = skeptic_client.parse(
                prompt=_build_skeptic_prompt(
                    candidate, completed_history, domain_brief, statistician
                ),
                schema=SkepticChallenge,
                system=with_honesty(_SKEPTIC_SYSTEM),
                json_schema=SkepticChallenge.model_json_schema(),
            )
            skeptic = sresp.parsed
            assert isinstance(skeptic, SkepticChallenge)
        except Exception as exc:  # noqa: BLE001 — failure-safe by design
            return DebateConsensus(
                candidate_id=candidate.candidate_id,
                keep=True,
                rationale=f"debate aborted: skeptic failed: {type(exc).__name__}: {exc}",
            )

        # --- Statistician call -------------------------------------------
        try:
            tresp = stat_client.parse(
                prompt=_build_statistician_prompt(
                    candidate, completed_history, domain_brief, skeptic
                ),
                schema=StatisticianChallenge,
                system=with_honesty(
                    _STATISTICIAN_SYSTEM.replace("{CATALOG_MARKDOWN}", catalog_markdown)
                ),
                json_schema=StatisticianChallenge.model_json_schema(),
            )
            statistician = tresp.parsed
            assert isinstance(statistician, StatisticianChallenge)
        except Exception as exc:  # noqa: BLE001
            return DebateConsensus(
                candidate_id=candidate.candidate_id,
                keep=True,
                rationale=f"debate aborted: statistician failed: {type(exc).__name__}: {exc}",
            )

        # Early-exit: if both challengers accept, no further rounds needed.
        if skeptic.overall_verdict == "accept" and statistician.overall_verdict == "accept":
            break

    # planner_client is intentionally not invoked here — its candidate is
    # already on the table and its 'vote' is inferred from the challengers
    # per the consensus rule. Holding the reference lets future versions
    # add an optional planner-revision call without changing the API.
    _ = planner_client

    keep, s_v, t_v, p_v = _consensus_keep(skeptic, statistician)
    revised_method = _extract_revised_method(candidate, statistician, catalog_markdown)
    revised_estimand = _extract_revised_estimand(candidate, skeptic, statistician)
    revised_modifiers = _extract_revised_modifiers(statistician)

    rationale = (
        f"debate verdicts: skeptic={s_v}, statistician={t_v}, planner={p_v}; "
        f"keep={keep}."
    )
    if revised_method:
        rationale += f" revised_method={revised_method}."
    if revised_estimand:
        rationale += f" revised_estimand={revised_estimand}."

    return DebateConsensus(
        candidate_id=candidate.candidate_id,
        keep=keep,
        revised_method=revised_method,
        revised_estimand=revised_estimand,
        revised_modifiers=revised_modifiers,
        rationale=rationale,
    )


__all__ = [
    "DebateConsensus",
    "SkepticChallenge",
    "StatisticianChallenge",
    "Verdict",
    "run_debate",
]
