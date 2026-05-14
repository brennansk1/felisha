"""Unit tests for ``causalrag.llm.self_refine`` (Sprint 3.3).

The tests stub the OllamaClient at the ``.parse(...)`` boundary — we don't
exercise the cassette / retry plumbing here (that lives in
``test_ollama_client.py``). What we DO verify:

  * happy path: both LLM calls succeed -> a revision is returned with
    ``abandon=False`` and the reflection is recorded.
  * the reflection picks ``min_sample_size_violated`` when the rejection
    reason mentions "min sample size".
  * the reflection picks ``duplicate_prior_experiment`` when the history
    already contains the same (T, Y, estimand) — and the helper forces
    ``abandon=True`` even if the stubbed revision tried to say otherwise.
  * LLM exception -> ``abandon=True`` with the error class name in
    ``revision_rationale``.
  * a transient bad-JSON failure that exhausts the client's retry budget
    still yields ``abandon=True`` (retry is the client's responsibility;
    we just confirm we surface the final failure correctly).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from causalrag.llm.self_refine import (
    CriticReflection,
    CriticRevision,
    self_refine_critic,
)


# ─────────── Helpers ──────────────────────────────────────────────────────


@dataclass
class _FakeCandidate:
    """Minimal duck-typed stand-in for ``CandidateExperiment``."""

    candidate_id: str = "c-abc123"
    research_question: str = "Does T cause Y?"
    treatment: str = "T_drug"
    outcome: str = "Y_recovery"
    estimand_class: str = "ATE"
    recommended_method: str | None = "dowhy.linear_regression"
    mediator: str | None = None
    instrument: str | None = None


@dataclass
class _StubResp:
    parsed: BaseModel


class _StubClient:
    """A minimal stand-in for OllamaClient: records calls and returns scripted
    parsed objects in order. Raises ``RuntimeError`` if it runs out of
    scripted responses."""

    def __init__(self, responses: list[BaseModel | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def parse(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
        json_schema: dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> _StubResp:
        self.calls.append(
            {"prompt": prompt, "schema": schema, "system": system, "json_schema": json_schema}
        )
        if not self._responses:
            raise RuntimeError("no scripted response left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        # Sanity: the scripted object must match the requested schema so we
        # don't accidentally pass a Revision where a Reflection was expected.
        assert isinstance(item, schema), (
            f"scripted response {type(item).__name__} does not match requested "
            f"schema {schema.__name__}"
        )
        return _StubResp(parsed=item)


# ─────────── Tests ────────────────────────────────────────────────────────


def test_happy_path_returns_revision_without_abandon() -> None:
    cand = _FakeCandidate()
    reflection = CriticReflection(
        candidate_id=cand.candidate_id,
        primary_failure="estimand_method_mismatch",
        failure_explanation="dowhy.linear_regression does not support the chosen estimand class.",
        proposed_revision="swap recommended_method to rbridge.weightit",
        revision_type="swap_estimator",
    )
    revision = CriticRevision(
        candidate_id=cand.candidate_id,
        new_recommended_method="rbridge.weightit",
        abandon=False,
        revision_rationale="estimator swap per reflection",
    )
    client = _StubClient([reflection, revision])

    out_rev, hist = self_refine_critic(
        candidate=cand,
        rejection_reason="method does not support this estimand class",
        completed_history=[],
        catalog_markdown="| id | family |\n|---|---|\n| rbridge.weightit | weighting |",
        client=client,
    )

    assert out_rev.abandon is False
    assert out_rev.new_recommended_method == "rbridge.weightit"
    assert len(hist) == 1
    assert hist[0].primary_failure == "estimand_method_mismatch"
    # The honesty preamble should be wired in via with_honesty().
    assert "Honesty rules" in client.calls[0]["system"]
    # The catalog markdown should have been substituted into the revision
    # system prompt (and NOT the reflection one).
    assert "rbridge.weightit" in client.calls[1]["system"]
    assert "{CATALOG_MARKDOWN}" not in client.calls[1]["system"]


def test_reflection_picks_min_sample_size_failure() -> None:
    """When the rejection reason mentions 'min sample size', the prompt
    should reach the LLM with that text verbatim and the test asserts the
    scripted-LLM answer is honored end-to-end."""
    cand = _FakeCandidate()
    reflection = CriticReflection(
        candidate_id=cand.candidate_id,
        primary_failure="min_sample_size_violated",
        failure_explanation="n=18 is below the 40-row floor for this estimator.",
        proposed_revision="narrow estimand to a higher-prevalence subgroup",
        revision_type="narrow_estimand",
    )
    revision = CriticRevision(
        candidate_id=cand.candidate_id,
        new_estimand_class="CATE",
        revision_rationale="narrow estimand per reflection",
    )
    client = _StubClient([reflection, revision])

    out_rev, hist = self_refine_critic(
        candidate=cand,
        rejection_reason="min sample size violated: only 18 rows for treatment arm",
        completed_history=[],
        catalog_markdown="(catalog)",
        client=client,
    )

    assert hist[0].primary_failure == "min_sample_size_violated"
    assert out_rev.new_estimand_class == "CATE"
    assert out_rev.abandon is False
    # The verbatim rejection reason must reach the reflection prompt so the
    # LLM has the signal it needs to make this classification.
    assert "min sample size" in client.calls[0]["prompt"].lower()


def test_reflection_picks_duplicate_when_history_matches_and_forces_abandon() -> None:
    """If the LLM correctly flags duplicate_prior_experiment, the helper
    forces abandon=True even if the scripted revision wanted to tweak the
    method — duplicates can't be 'fixed', only abandoned."""
    cand = _FakeCandidate(treatment="T_x", outcome="Y_y", estimand_class="ATE")
    history = [
        {"id": "h-001", "treatment": "T_x", "outcome": "Y_y", "estimand_class": "ATE"},
    ]
    reflection = CriticReflection(
        candidate_id=cand.candidate_id,
        primary_failure="duplicate_prior_experiment",
        failure_explanation="(T_x, Y_y, ATE) was already executed as h-001.",
        proposed_revision="abandon",
        revision_type="abandon_candidate",
    )
    # Adversarial: the scripted revision tries to NOT abandon. The helper
    # must override this for duplicates.
    revision = CriticRevision(
        candidate_id=cand.candidate_id,
        new_recommended_method="rbridge.weightit",
        abandon=False,
        revision_rationale="try a different method",
    )
    client = _StubClient([reflection, revision])

    out_rev, hist = self_refine_critic(
        candidate=cand,
        rejection_reason="this candidate duplicates a prior experiment",
        completed_history=history,
        catalog_markdown="(catalog)",
        client=client,
    )

    assert hist[0].primary_failure == "duplicate_prior_experiment"
    assert out_rev.abandon is True
    assert "duplicate_prior_experiment" in out_rev.revision_rationale
    # History rendering should include the duplicate so the LLM can see it.
    assert "h-001" in client.calls[0]["prompt"]


def test_llm_exception_returns_safe_abandon() -> None:
    cand = _FakeCandidate()
    client = _StubClient([RuntimeError("ollama down")])

    out_rev, hist = self_refine_critic(
        candidate=cand,
        rejection_reason="something",
        completed_history=[],
        catalog_markdown="(catalog)",
        client=client,
    )

    assert out_rev.abandon is True
    assert "RuntimeError" in out_rev.revision_rationale
    assert "ollama down" in out_rev.revision_rationale
    # Reflection failed before being parsed -> empty history.
    assert hist == []
    assert out_rev.candidate_id == cand.candidate_id


def test_revision_call_failure_after_successful_reflection_abandons() -> None:
    """If the reflection succeeds but the revision LLM call blows up
    (e.g., SchemaValidationFailed after retries), the helper still returns
    a safe-abandon revision and preserves the reflection in history."""
    cand = _FakeCandidate()
    reflection = CriticReflection(
        candidate_id=cand.candidate_id,
        primary_failure="other",
        failure_explanation="unclassified rejection",
        proposed_revision="abandon",
        revision_type="abandon_candidate",
    )
    client = _StubClient([reflection, ValueError("bad JSON after retries")])

    out_rev, hist = self_refine_critic(
        candidate=cand,
        rejection_reason="???",
        completed_history=[],
        catalog_markdown="(catalog)",
        client=client,
    )

    assert out_rev.abandon is True
    assert "ValueError" in out_rev.revision_rationale
    assert len(hist) == 1
    assert hist[0].primary_failure == "other"


def test_revision_type_abandon_candidate_forces_abandon_flag() -> None:
    """When reflection.revision_type == 'abandon_candidate' but the scripted
    revision forgot to set abandon=True, the helper should set it."""
    cand = _FakeCandidate()
    reflection = CriticReflection(
        candidate_id=cand.candidate_id,
        primary_failure="identifiability_weak",
        failure_explanation="No valid instrument, T-Y confounding is unmeasured.",
        proposed_revision="abandon",
        revision_type="abandon_candidate",
    )
    revision = CriticRevision(
        candidate_id=cand.candidate_id,
        abandon=False,
        revision_rationale="(forgot to abandon)",
    )
    client = _StubClient([reflection, revision])

    out_rev, _ = self_refine_critic(
        candidate=cand,
        rejection_reason="unmeasured confounding; weak identifiability",
        completed_history=[],
        catalog_markdown="(catalog)",
        client=client,
    )

    assert out_rev.abandon is True


def test_two_llm_calls_per_round() -> None:
    cand = _FakeCandidate()
    reflection = CriticReflection(
        candidate_id=cand.candidate_id,
        primary_failure="other",
        failure_explanation="x",
        proposed_revision="y",
        revision_type="swap_estimator",
    )
    revision = CriticRevision(
        candidate_id=cand.candidate_id,
        new_recommended_method="rbridge.weightit",
        revision_rationale="ok",
    )
    client = _StubClient([reflection, revision])

    self_refine_critic(
        candidate=cand,
        rejection_reason="?",
        completed_history=[],
        catalog_markdown="(catalog)",
        client=client,
    )

    assert len(client.calls) == 2
    assert client.calls[0]["schema"] is CriticReflection
    assert client.calls[1]["schema"] is CriticRevision
    # Reflection prompt should embed the candidate; revision prompt should
    # embed the reflection JSON.
    assert cand.candidate_id in client.calls[0]["prompt"]
    refl_json = json.loads(
        # Pull the JSON-ish block out of the revision prompt to confirm it
        # carries the reflection structure.
        client.calls[1]["prompt"]
        .split("## Structured reflection")[1]
        .split("## Completed-experiment history")[0]
        .strip()
        .lstrip("(from previous step)")
        .strip()
    )
    assert refl_json["candidate_id"] == cand.candidate_id


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("min sample size violated", "min_sample_size_violated"),
        ("n too small for the estimator", "min_sample_size_violated"),
        ("unmeasured confounding precludes ATE", "identifiability_weak"),
        ("the recommended estimator does not support NDE", "estimand_method_mismatch"),
        ("missing required column 'income'", "missing_required_column"),
    ],
)
def test_heuristic_primary_failure_classifier(reason: str, expected: str) -> None:
    """The internal heuristic mirrors the LLM's expected classification.
    Useful as a stand-alone sanity check + as the fallback used when the
    LLM call fails (we don't surface it directly today, but exercising
    it pins down the intended mapping)."""
    from causalrag.llm.self_refine import _heuristic_primary_failure

    cand = _FakeCandidate()
    assert _heuristic_primary_failure(cand, reason, []) == expected


def test_heuristic_detects_duplicate_from_history_even_without_reason_text() -> None:
    from causalrag.llm.self_refine import _heuristic_primary_failure

    cand = _FakeCandidate(treatment="T", outcome="Y", estimand_class="ATE")
    history = [{"id": "h0", "treatment": "T", "outcome": "Y", "estimand_class": "ATE"}]
    assert _heuristic_primary_failure(cand, "something else", history) == "duplicate_prior_experiment"
