"""Unit tests for the master_loop helper functions added in round 3.

Covers:
- _compress_history (working-memory compressor)
- _render_history_block
- _render_chain_forest
- _is_duplicate_followup (dedup guard for pending follow-ups)
- _classify_estimator_family
- _pick_robustness_method
- score_candidate (deterministic scorer)
"""

from __future__ import annotations

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.result import EstimationResult
from causalrag.master_loop import (
    CandidateExperiment,
    _classify_estimator_family,
    _compress_history,
    _is_duplicate_followup,
    _pick_robustness_method,
    _render_chain_forest,
    _render_history_block,
    score_candidate,
)


def _make_row(
    id_: str,
    t: str = "treat",
    y: str = "y",
    klass: str = "ATE",
    point: float = 0.5,
    verdict: str = "green",
    p_value: str = "0.01",
    chain_id: str | None = None,
    parent_id: str | None = None,
    failure_reason: str | None = None,
) -> dict:
    return {
        "id": id_,
        "treatment": t,
        "outcome": y,
        "estimand_class": klass,
        "point_estimate": point,
        "ci_low": point - 0.2,
        "ci_high": point + 0.2,
        "p_value": p_value,
        "sensitivity_verdict": verdict,
        "chain_id": chain_id,
        "parent_id": parent_id,
        "failure_reason": failure_reason,
    }


# ─────────── _compress_history ───────────────────────────────────────────


def test_compress_history_passthrough_when_short() -> None:
    history = [_make_row(f"h{i:02d}") for i in range(3)]
    recent, summary = _compress_history(history, keep_verbatim=5)
    assert recent == history
    assert summary is None


def test_compress_history_splits_when_long() -> None:
    history = [_make_row(f"h{i:02d}", p_value=str(0.01 if i < 3 else 0.5)) for i in range(12)]
    recent, summary = _compress_history(history, keep_verbatim=5)
    assert len(recent) == 5
    assert summary is not None
    assert summary["n_older"] == 7
    assert "ATE" in summary["estimand_classes_run"]
    assert summary["estimand_classes_run"]["ATE"] == 7
    # Three significant findings (p<0.05) recorded
    assert len(summary["significant_findings"]) == 3


def test_compress_history_captures_dead_ends() -> None:
    history = [
        _make_row(f"h{i:02d}", failure_reason=f"reason {i}" if i % 2 else None)
        for i in range(10)
    ]
    _, summary = _compress_history(history, keep_verbatim=3)
    assert summary is not None
    assert summary["dead_ends"]  # at least one
    # Don't claim it caught every dead end — just that the field is populated
    assert all("reason" in d for d in summary["dead_ends"])


# ─────────── _render_history_block ───────────────────────────────────────


def test_render_history_block_first_iteration() -> None:
    parts = _render_history_block([])
    text = "\n".join(parts)
    assert "Recent experiments (last 0)" in text
    assert "(none — this is the first iteration)" in text


def test_render_history_block_shows_chain_marker() -> None:
    history = [_make_row("h01", chain_id="root-a")]
    parts = _render_history_block(history)
    text = "\n".join(parts)
    assert "CHAIN=root-a" in text


def test_render_history_block_shows_failure_marker() -> None:
    history = [_make_row("h01", failure_reason="not identifiable")]
    parts = _render_history_block(history)
    text = "\n".join(parts)
    assert "❌" in text
    assert "not identifiable" in text


# ─────────── _render_chain_forest ────────────────────────────────────────


def test_render_chain_forest_empty() -> None:
    assert _render_chain_forest([]) == []


def test_render_chain_forest_renders_root_and_child() -> None:
    history = [
        _make_row("h01", chain_id="h01"),
        _make_row("h02", chain_id="h01", parent_id="h01"),
        _make_row("h03", chain_id="h01", parent_id="h02"),
    ]
    lines = _render_chain_forest(history)
    text = "\n".join(lines)
    assert "[h01]" in text
    assert "[h02]" in text
    assert "[h03]" in text
    # Child should be more indented than root
    h01_indent = next(l for l in lines if "[h01]" in l).index("[")
    h02_indent = next(l for l in lines if "[h02]" in l).index("[")
    assert h02_indent > h01_indent


# ─────────── _classify_estimator_family ──────────────────────────────────


def test_classify_estimator_family_known_ids() -> None:
    assert _classify_estimator_family("python.dml.linear") == "dml"
    assert _classify_estimator_family("python.dml.causal_forest") in {"forest", "dml"}
    assert _classify_estimator_family("rbridge.weightit") == "weight"
    assert _classify_estimator_family("rbridge.matchit") == "match"
    assert _classify_estimator_family("rbridge.bartcause") == "bart"
    assert _classify_estimator_family("rbridge.lmtp.shift") == "lmtp"
    assert _classify_estimator_family("python.linear.ols") == "ols"
    assert _classify_estimator_family(None) == "_default"
    assert _classify_estimator_family("totally-unknown-id") == "_default"


# ─────────── _pick_robustness_method ─────────────────────────────────────


def test_pick_robustness_method_returns_different_family() -> None:
    # Importing causalrag.estimators registers the catalog
    import causalrag.estimators  # noqa: F401

    parent_id = "python.dml.linear"
    swap = _pick_robustness_method(parent_id, "ATE")
    if swap is not None:
        assert swap != parent_id
        assert _classify_estimator_family(swap) != _classify_estimator_family(parent_id)


def test_pick_robustness_method_returns_none_when_unsupported_estimand() -> None:
    import causalrag.estimators  # noqa: F401

    swap = _pick_robustness_method("python.dml.linear", "TOTALLY_FAKE_ESTIMAND")
    assert swap is None


# ─────────── _is_duplicate_followup ──────────────────────────────────────


def _make_walk(
    hid: str, t: str = "treat", y: str = "y", klass: EstimandClass = EstimandClass.ATE,
    estimator_id: str = "python.dml.linear",
) -> RoadmapWalk:
    est = CausalEstimand.model_validate(
        {
            "class": klass,
            "treatment": t,
            "outcome": y,
            "modifiers": (),
            "formal_expression": "E[Y(1) - Y(0)]",
        }
    )
    walk = RoadmapWalk(hypothesis_id=hid)
    walk.q3_estimand = est
    walk.q7_estimates = (
        EstimationResult(
            estimator_id=estimator_id,
            estimand_class="ATE",
            point_estimate=1.0,
            se=0.2,
            n_used=200,
        ),
    )
    return walk


def test_is_duplicate_followup_detects_same_family_same_TY() -> None:
    completed = [_make_walk("h01", "treat", "y")]
    candidate = CandidateExperiment(
        candidate_id="dup",
        research_question="?",
        treatment="treat",
        outcome="y",
        estimand_class="ATE",
        recommended_method="python.dml.linear",
        impact_rationale="?",
        identifiability_rationale="?",
        power_rationale="?",
    )
    assert _is_duplicate_followup(
        candidate, completed=completed, pending=[], parent_estimator_id=None
    )


def test_is_duplicate_followup_allows_different_family() -> None:
    completed = [_make_walk("h01", "treat", "y", estimator_id="python.dml.linear")]
    candidate = CandidateExperiment(
        candidate_id="rb",
        research_question="?",
        treatment="treat",
        outcome="y",
        estimand_class="ATE",
        recommended_method="rbridge.weightit",  # different family
        impact_rationale="?",
        identifiability_rationale="?",
        power_rationale="?",
    )
    assert not _is_duplicate_followup(
        candidate, completed=completed, pending=[], parent_estimator_id=None
    )


def test_is_duplicate_followup_allows_different_estimand() -> None:
    completed = [_make_walk("h01", "treat", "y", klass=EstimandClass.ATE)]
    candidate = CandidateExperiment(
        candidate_id="cate",
        research_question="?",
        treatment="treat",
        outcome="y",
        estimand_class="CATE",  # different estimand
        recommended_method="python.dml.linear",
        impact_rationale="?",
        identifiability_rationale="?",
        power_rationale="?",
    )
    assert not _is_duplicate_followup(
        candidate, completed=completed, pending=[], parent_estimator_id=None
    )


def test_is_duplicate_followup_detects_pending_match() -> None:
    cand1 = CandidateExperiment(
        candidate_id="p1",
        research_question="?",
        treatment="treat",
        outcome="y",
        estimand_class="ATE",
        recommended_method="python.dml.linear",
        impact_rationale="?",
        identifiability_rationale="?",
        power_rationale="?",
    )
    cand2 = CandidateExperiment(
        candidate_id="p2",
        research_question="?",
        treatment="treat",
        outcome="y",
        estimand_class="ATE",
        recommended_method="python.dml.linear",
        impact_rationale="?",
        identifiability_rationale="?",
        power_rationale="?",
    )
    pending = [(cand1, "chain-a", "parent-a")]
    assert _is_duplicate_followup(
        cand2, completed=[], pending=pending, parent_estimator_id=None
    )


# ─────────── score_candidate ─────────────────────────────────────────────


def _make_candidate(cid: str = "c1", mods: list[str] | None = None) -> CandidateExperiment:
    return CandidateExperiment(
        candidate_id=cid,
        research_question="rq",
        treatment="treat",
        outcome="y",
        estimand_class="ATE",
        modifiers=mods or [],
        recommended_method="python.dml.linear",
        impact_rationale="rationale",
        identifiability_rationale="rationale",
        power_rationale="rationale",
        impact_hint=0.8,
        identifiability_hint=0.7,
        power_hint=0.6,
    )


def test_score_candidate_returns_all_components() -> None:
    protocol = StudyProtocol(name="t")
    cand = _make_candidate()
    scored = score_candidate(cand, protocol=protocol, completed=[])
    for k in ("impact", "identifiability", "power_proxy", "novelty", "cost", "score"):
        assert k in scored
    assert 0.0 <= scored["impact"] <= 1.0


def test_score_candidate_novelty_zero_on_repeat() -> None:
    protocol = StudyProtocol(name="t")
    walk = _make_walk("h01", t="treat", y="y", klass=EstimandClass.ATE)
    cand = _make_candidate()
    scored = score_candidate(cand, protocol=protocol, completed=[walk])
    assert scored["novelty"] == 0.0
