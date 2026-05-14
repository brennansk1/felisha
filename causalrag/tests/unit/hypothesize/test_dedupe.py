"""Unit tests for ``causalrag.hypothesize.dedupe``.

Covers four cases:

1. Deterministic exact-duplicate collapse — two candidates with the
   same (T, Y, estimand, modifiers) leave one survivor.
2. LLM refinement — three near-duplicates with overlapping modifiers
   are merged by a stub LLM into a single MergeAction.
3. LLM raises — the function falls back to the deterministic-only
   result and never re-raises.
4. LLM references a nonexistent id — that reference is dropped, a
   warning is logged, and the rest of the plan is honored.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from causalrag.hypothesize.dedupe import (
    DedupePlan,
    MergeAction,
    PruneAction,
    dedupe_candidates,
)
from causalrag.llm.ollama_client import FakeOllamaTransport, OllamaClient
from causalrag.master_loop import CandidateExperiment


# ─────────── Helpers ─────────────────────────────────────────────────────


def _cand(
    cid: str,
    *,
    treatment: str = "T",
    outcome: str = "Y",
    estimand: str = "ATE",
    modifiers: list[str] | None = None,
    impact_hint: float = 0.5,
) -> CandidateExperiment:
    return CandidateExperiment(
        candidate_id=cid,
        research_question=f"does {treatment} affect {outcome}?",
        treatment=treatment,
        outcome=outcome,
        estimand_class=estimand,
        modifiers=modifiers or [],
        impact_rationale="x",
        identifiability_rationale="x",
        power_rationale="x",
        impact_hint=impact_hint,
    )


def _client(response: Any) -> OllamaClient:
    """Build an OllamaClient backed by a FakeOllamaTransport that always
    returns ``response`` (a JSON-serializable DedupePlan dict)."""
    transport = FakeOllamaTransport({"": response})
    return OllamaClient(
        model="test-model",
        transport=transport,
        allow_live=True,
        cassette_dir=None,
    )


# ─────────── Test 1 — deterministic exact-duplicate collapse ─────────────


def test_deterministic_exact_duplicates_collapse() -> None:
    a = _cand("c1", impact_hint=0.7)
    b = _cand("c2", impact_hint=0.4)  # exact (T,Y,ATE,{}) match — should drop
    c = _cand("c3", treatment="T2", impact_hint=0.5)  # different T → keep

    survivors, plan = dedupe_candidates([a, b, c], client=None)

    survivor_ids = {s.candidate_id for s in survivors}
    assert survivor_ids == {"c1", "c3"}
    assert len(plan.merged) == 1
    merge = plan.merged[0]
    # Higher impact wins.
    assert merge.kept_id == "c1"
    assert merge.dropped_ids == ["c2"]


# ─────────── Test 2 — LLM merges three near-duplicates ───────────────────


def test_llm_merges_overlapping_modifiers() -> None:
    a = _cand("c1", modifiers=["age", "sex"], impact_hint=0.8)
    b = _cand("c2", modifiers=["age"], impact_hint=0.5)
    c = _cand("c3", modifiers=["sex"], impact_hint=0.5)
    independent = _cand("c4", outcome="Y2")

    llm_response = {
        "pruned": [],
        "merged": [
            {
                "kept_id": "c1",
                "dropped_ids": ["c2", "c3"],
                "merged_rationale": (
                    "c2 and c3 are strict subsets of c1's modifier set; "
                    "the CATE on (age, sex) subsumes both single-modifier views."
                ),
            }
        ],
        "note": "consolidated CATE family on T→Y",
    }
    client = _client(llm_response)

    survivors, plan = dedupe_candidates([a, b, c, independent], client=client)

    survivor_ids = {s.candidate_id for s in survivors}
    assert survivor_ids == {"c1", "c4"}
    # The combined plan should contain the LLM-emitted merge.
    merges = [m for m in plan.merged if m.kept_id == "c1"]
    assert len(merges) == 1
    assert set(merges[0].dropped_ids) == {"c2", "c3"}


# ─────────── Test 3 — LLM raises → deterministic-only fallback ───────────


def test_llm_failure_falls_back_to_deterministic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two exact duplicates so the deterministic pass has something to do.
    a = _cand("c1", impact_hint=0.9)
    b = _cand("c2", impact_hint=0.3)
    c = _cand("c3", treatment="T2")

    # Return invalid JSON so parse() raises SchemaValidationFailed.
    transport = FakeOllamaTransport({"": "not-json"})
    client = OllamaClient(
        model="test-model",
        transport=transport,
        allow_live=True,
        cassette_dir=None,
        max_retries=0,
    )

    with caplog.at_level(logging.WARNING, logger="causalrag.hypothesize.dedupe"):
        survivors, plan = dedupe_candidates([a, b, c], client=client)

    survivor_ids = {s.candidate_id for s in survivors}
    assert survivor_ids == {"c1", "c3"}
    # Only the deterministic merge survives.
    assert len(plan.merged) == 1
    assert plan.merged[0].kept_id == "c1"
    assert plan.merged[0].dropped_ids == ["c2"]
    # The failure was logged, not raised.
    assert any("dedupe LLM pass failed" in r.message for r in caplog.records)


# ─────────── Test 4 — LLM references nonexistent id → dropped + warned ───


def test_llm_unknown_id_is_dropped_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    a = _cand("c1", modifiers=["age"], impact_hint=0.8)
    b = _cand("c2", modifiers=["sex"], impact_hint=0.5)

    llm_response = {
        "pruned": [
            {"candidate_id": "ghost-prune", "reason": "fabricated"},
        ],
        "merged": [
            {
                # kept_id valid; one dropped_id valid, one fabricated.
                "kept_id": "c1",
                "dropped_ids": ["c2", "does-not-exist"],
                "merged_rationale": "subset",
            },
            {
                # Entire merge has an invalid kept_id — should be skipped.
                "kept_id": "ghost-keep",
                "dropped_ids": ["c1"],
                "merged_rationale": "bogus",
            },
        ],
        "note": None,
    }
    client = _client(llm_response)

    with caplog.at_level(logging.WARNING, logger="causalrag.hypothesize.dedupe"):
        survivors, plan = dedupe_candidates([a, b], client=client)

    survivor_ids = {s.candidate_id for s in survivors}
    # c1 kept (still valid), c2 dropped by the validated merge,
    # the fabricated kept-id merge is rejected so c1 is NOT dropped.
    assert survivor_ids == {"c1"}

    # The applied merge should only reference real ids.
    applied = [m for m in plan.merged if m.kept_id == "c1"]
    assert len(applied) == 1
    assert applied[0].dropped_ids == ["c2"]

    # Warnings were emitted for each fabricated reference.
    messages = " ".join(r.message for r in caplog.records)
    assert "does-not-exist" in messages
    assert "ghost-keep" in messages
    assert "ghost-prune" in messages
