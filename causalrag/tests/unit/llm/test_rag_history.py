"""Tests for the local history RAG (Sprint 3.5)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from causalrag.llm.rag_history import (
    HistoryCase,
    HistoryRAG,
    default_cache_path,
)


# ─────────── fixtures ─────────────────────────────────────────────────────


def _case(
    case_id: str,
    *,
    flags: list[str],
    treatment: str,
    outcome: str,
    estimand_class: str = "ATE",
    estimator_id: str = "python.dml.linear",
    verdict: str = "green",
    failure_reason: str | None = None,
    extra_text: str = "",
) -> HistoryCase:
    flag_str = ", ".join(sorted(flags)) if flags else "no special data flags"
    text = (
        f"Treatment: {treatment}. Outcome: {outcome}. "
        f"Estimand class: {estimand_class}. Estimator: {estimator_id}. "
        f"Sensitivity verdict: {verdict}. Data flags: {flag_str}. {extra_text}"
    ).strip()
    return HistoryCase(
        case_id=case_id,
        flags=sorted(flags),
        treatment=treatment,
        outcome=outcome,
        estimand_class=estimand_class,
        estimator_id=estimator_id,
        sensitivity_verdict=verdict,
        failure_reason=failure_reason,
        text=text,
        metadata={},
    )


@pytest.fixture
def five_cases() -> list[HistoryCase]:
    return [
        _case(
            "c-binY-binT",
            flags=["binary_treatment", "binary_outcome"],
            treatment="aspirin",
            outcome="mortality_30d",
            estimand_class="ATE",
            estimator_id="python.iptw",
            verdict="green",
        ),
        _case(
            "c-binT-contY",
            flags=["binary_treatment", "continuous_outcome"],
            treatment="statins",
            outcome="ldl_change",
            estimand_class="ATE",
            estimator_id="python.dml.linear",
            verdict="green",
        ),
        _case(
            "c-iv",
            flags=["binary_treatment", "binary_outcome", "instrumental_candidate_present"],
            treatment="college_admit",
            outcome="income_10y",
            estimand_class="LATE",
            estimator_id="python.iv.2sls",
            verdict="yellow",
        ),
        _case(
            "c-surv",
            flags=["binary_treatment", "right_censored_outcome"],
            treatment="chemo_regimen",
            outcome="time_to_relapse",
            estimand_class="RMST_CONTRAST",
            estimator_id="r.survival.rmst",
            verdict="red",
            failure_reason=None,
        ),
        _case(
            "c-mediation",
            flags=["binary_treatment", "continuous_outcome", "mediator_proposed"],
            treatment="exercise_program",
            outcome="hba1c_change",
            estimand_class="NIE",
            estimator_id="python.mediation",
            verdict="green",
        ),
    ]


# ─────────── tests ────────────────────────────────────────────────────────


def test_default_cache_path_under_home() -> None:
    p = default_cache_path()
    assert p.name == "cases.jsonl"
    assert ".causalrag" in p.parts
    assert "history" in p.parts


def test_search_returns_most_similar(five_cases: list[HistoryCase]) -> None:
    rag = HistoryRAG(five_cases, backend="tfidf")
    assert rag.backend == "tfidf"
    assert len(rag) == 5

    hits = rag.search(
        "Treatment statins lowering LDL continuous outcome ATE",
        top_k=2,
    )
    assert len(hits) == 2
    # The statin / continuous-outcome case should be the top hit.
    assert hits[0].case_id == "c-binT-contY"


def test_search_matches_flag_profile(five_cases: list[HistoryCase]) -> None:
    rag = HistoryRAG(five_cases, backend="tfidf")
    hits = rag.search(
        "binary_treatment continuous_outcome ATE",
        top_k=3,
    )
    assert hits, "expected at least one hit"
    top_ids = {h.case_id for h in hits}
    # Both binary-T + continuous-Y cases should rank in the top 3.
    assert "c-binT-contY" in top_ids


def test_empty_index_returns_empty_list() -> None:
    rag = HistoryRAG(backend="tfidf")
    assert len(rag) == 0
    assert rag.search("anything", top_k=5) == []
    assert rag.few_shot_examples(current_flags=set(), top_k=3) == ""


def test_save_load_roundtrips_jsonl(
    tmp_path: Path, five_cases: list[HistoryCase]
) -> None:
    rag = HistoryRAG(five_cases, backend="tfidf")
    out = tmp_path / "history" / "cases.jsonl"
    rag.save(out)
    assert out.exists()
    # JSONL → one line per case.
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 5

    fresh = HistoryRAG(backend="tfidf")
    fresh.load(out)
    assert len(fresh) == 5
    loaded_ids = {c.case_id for c in fresh.cases}
    assert loaded_ids == {c.case_id for c in five_cases}

    # Searches still work after a round-trip.
    hits = fresh.search("statins continuous outcome", top_k=1)
    assert hits and hits[0].case_id == "c-binT-contY"


def test_load_missing_file_yields_empty(tmp_path: Path) -> None:
    rag = HistoryRAG(backend="tfidf")
    rag.load(tmp_path / "nope.jsonl")
    assert len(rag) == 0


def test_few_shot_examples_non_empty_markdown(
    five_cases: list[HistoryCase],
) -> None:
    rag = HistoryRAG(five_cases, backend="tfidf")
    md = rag.few_shot_examples(
        current_flags={"binary_treatment", "continuous_outcome"},
        current_treatment_hint="statins",
        current_outcome_hint="ldl_change",
        top_k=2,
    )
    assert md
    assert "Prior-run case bank" in md
    assert "### Case 1" in md
    # The block should reference at least one estimator id from our bank.
    assert "python." in md or "r." in md


def test_few_shot_examples_handles_enum_flags(
    five_cases: list[HistoryCase],
) -> None:
    """Flags arriving as enum-like objects (with .value) must be handled."""

    class _F:
        def __init__(self, v: str) -> None:
            self.value = v

    rag = HistoryRAG(five_cases, backend="tfidf")
    md = rag.few_shot_examples(
        current_flags={_F("binary_treatment"), _F("continuous_outcome")},
        top_k=1,
    )
    assert "### Case 1" in md


def test_add_case_extends_index(five_cases: list[HistoryCase]) -> None:
    rag = HistoryRAG(five_cases[:3], backend="tfidf")
    assert len(rag) == 3
    rag.add_case(five_cases[3])
    rag.add_case(five_cases[4])
    assert len(rag) == 5
    hits = rag.search("mediator NIE", top_k=1)
    assert hits and hits[0].case_id == "c-mediation"


def test_add_from_walk_builds_case() -> None:
    """add_from_walk pulls fields off a duck-typed walk-like object."""

    walk = types.SimpleNamespace(
        hypothesis_id="h-001",
        q3_estimand=types.SimpleNamespace(estimand_class="ATE"),
        q4_observed_data_spec={"treatment": "drug_X", "outcome": "bp_change"},
        q6_statistical_estimand=None,
        q7_estimates=(
            types.SimpleNamespace(
                estimator_id="python.dml.linear",
                point_estimate=0.42,
                estimand_class="ATE",
            ),
        ),
        sensitivity_verdict="green",
        failure_reason=None,
        chain_id="h-001",
        parent_id=None,
    )

    rag = HistoryRAG(backend="tfidf")
    case = rag.add_from_walk(
        walk,
        flags={"binary_treatment", "continuous_outcome"},
        dataset_label="synthetic_v1",
    )
    assert case.treatment == "drug_X"
    assert case.outcome == "bp_change"
    assert case.estimator_id == "python.dml.linear"
    assert case.estimand_class == "ATE"
    assert case.sensitivity_verdict == "green"
    assert "binary_treatment" in case.flags
    assert case.metadata["dataset_label"] == "synthetic_v1"
    assert case.metadata["point_estimate"] == pytest.approx(0.42)
    assert len(rag) == 1


def test_tfidf_fallback_when_sentence_transformers_missing(
    monkeypatch: pytest.MonkeyPatch, five_cases: list[HistoryCase]
) -> None:
    """If sentence-transformers can't import, backend silently falls back."""

    # Ensure any cached module entry is gone, then block re-import.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    rag = HistoryRAG(five_cases, backend="sentence-transformers")
    assert rag.backend == "tfidf"
    hits = rag.search("LATE instrumental", top_k=1)
    assert hits and hits[0].case_id == "c-iv"


def test_history_case_from_dict_normalises_flags() -> None:
    raw = {
        "case_id": "x",
        "flags": ["binary_outcome", "binary_treatment"],
        "treatment": "t",
        "outcome": "y",
        "estimand_class": "ATE",
        "estimator_id": "e",
        "sensitivity_verdict": "green",
        "failure_reason": None,
        "text": "Treatment: t. Outcome: y.",
        "metadata": {"k": "v"},
        # Stray key gets pushed into metadata rather than crashing.
        "extra_field": 123,
    }
    c = HistoryCase.from_dict(raw)
    assert c.flags == ["binary_outcome", "binary_treatment"]
    assert c.metadata["k"] == "v"
    assert c.metadata["extra_field"] == 123
