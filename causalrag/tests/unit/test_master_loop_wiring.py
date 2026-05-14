"""Integration tests for round-4 wiring: dedupe, missingness, narration,
anomaly audit, sensitivity interpretation, cross-experiment.

These don't run a full /auto loop (that needs Ollama). They verify the
components are importable, callable, and degrade gracefully when their
LLM client is None.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from causalrag.core.estimand import EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.result import EstimationResult
from causalrag.hypothesize.dedupe import dedupe_candidates
from causalrag.master_loop import CandidateExperiment
from causalrag.reporting.cross_experiment import analyze_cross_experiment


def _walk_with_estimate(
    hid: str, t: str, y: str, point: float = 1.0, p: float = 0.01,
    chain_id: str | None = None, parent_id: str | None = None,
) -> RoadmapWalk:
    from causalrag.core.estimand import CausalEstimand

    est = CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": t,
            "outcome": y,
            "modifiers": (),
            "formal_expression": "E[Y(1)-Y(0)]",
        }
    )
    walk = RoadmapWalk(
        hypothesis_id=hid, chain_id=chain_id, parent_id=parent_id
    )
    walk.q3_estimand = est
    walk.q7_estimates = (
        EstimationResult(
            estimator_id="python.dml.linear",
            estimand_class="ATE",
            point_estimate=point,
            se=0.2,
            ci_low=point - 0.4,
            ci_high=point + 0.4,
            p_value=p,
            n_used=300,
        ),
    )
    walk.sensitivity_verdict = "green" if abs(point) > 0.5 else "yellow"
    return walk


def _cand(cid: str, t: str = "treat", y: str = "y", klass: str = "ATE") -> CandidateExperiment:
    return CandidateExperiment(
        candidate_id=cid,
        research_question="rq",
        treatment=t,
        outcome=y,
        estimand_class=klass,
        recommended_method="python.dml.linear",
        impact_rationale="r",
        identifiability_rationale="r",
        power_rationale="r",
    )


# ─────────── dedupe ──────────────────────────────────────────────────────


def test_dedupe_no_client_collapses_exact_duplicates() -> None:
    cands = [
        _cand("c1", "treat", "y", "ATE"),
        _cand("c2", "treat", "y", "ATE"),  # exact dup
        _cand("c3", "treat", "y", "CATE"),
    ]
    survivors, plan = dedupe_candidates(cands, client=None)
    ids = {s.candidate_id for s in survivors}
    # Two distinct (T,Y,estimand) tuples; the duplicate is dropped.
    assert len(survivors) == 2
    # Either c1 or c2 survives plus c3
    assert "c3" in ids


def test_dedupe_preserves_distinct_candidates() -> None:
    cands = [
        _cand("c1", "treat", "y", "ATE"),
        _cand("c2", "treat", "y", "CATE"),
        _cand("c3", "treat", "y2", "ATE"),
    ]
    survivors, _ = dedupe_candidates(cands, client=None)
    assert len(survivors) == 3


# ─────────── cross_experiment ────────────────────────────────────────────


def test_cross_experiment_detects_chain_narrative() -> None:
    protocol = StudyProtocol(name="t")
    walks = {
        "auto-01": _walk_with_estimate("auto-01", "t", "y", point=1.5, chain_id="auto-01"),
        "auto-02": _walk_with_estimate(
            "auto-02", "t", "y", point=1.4, chain_id="auto-01", parent_id="auto-01"
        ),
        "auto-03": _walk_with_estimate(
            "auto-03", "t", "y", point=1.3, chain_id="auto-01", parent_id="auto-02"
        ),
    }
    protocol.roadmap_walks = walks
    analysis = analyze_cross_experiment(protocol=protocol, client=None)
    # Without an LLM client we get the deterministic-only pre-pass
    assert len(analysis.chain_narratives) == 1
    chain = analysis.chain_narratives[0]
    assert chain.chain_id == "auto-01"
    assert chain.walk_ids_in_order == ["auto-01", "auto-02", "auto-03"]


def test_cross_experiment_detects_contradiction() -> None:
    protocol = StudyProtocol(name="t")
    protocol.roadmap_walks = {
        "auto-01": _walk_with_estimate("auto-01", "t", "y", point=+1.5),
        "auto-02": _walk_with_estimate("auto-02", "t", "y", point=-1.5),
    }
    analysis = analyze_cross_experiment(protocol=protocol, client=None)
    assert len(analysis.contradictions) >= 1


def test_cross_experiment_empty_protocol() -> None:
    protocol = StudyProtocol(name="t")
    analysis = analyze_cross_experiment(protocol=protocol, client=None)
    assert analysis.contradictions == []
    assert analysis.reinforcements == []
    assert analysis.chain_narratives == []


# ─────────── missingness ─────────────────────────────────────────────────


def test_missingness_detects_heavy() -> None:
    from causalrag.data.missingness import diagnose_missingness

    df = pd.DataFrame(
        {
            "treat": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 10,
            "y": [1.0, 2.0, 1.5, 2.5] * 25,
        }
    )
    # Inject ~30% missingness on one column
    df.loc[: int(len(df) * 0.30), "y"] = None
    report = diagnose_missingness(df, treatment="treat", outcome="y")
    assert max(report.per_column_rate.values()) > 0.20


def test_missingness_clean_data_proceeds() -> None:
    from causalrag.data.missingness import diagnose_missingness

    df = pd.DataFrame({"treat": [0, 1] * 50, "y": [1.0, 2.0] * 50})
    report = diagnose_missingness(df, treatment="treat", outcome="y")
    assert report.recommendation == "proceed_complete_case"


# ─────────── anomaly audit (no client) ──────────────────────────────────


def test_anomaly_audit_deterministic_only() -> None:
    from causalrag.sensitivity.anomaly_audit import audit_for_anomalies

    walk = _walk_with_estimate("auto-01", "treat", "y")
    bad_result = EstimationResult(
        estimator_id="python.dml.linear",
        estimand_class="ATE",
        point_estimate=0.05,
        se=0.01,
        ci_low=-1.0,
        ci_high=1.0,
        p_value=0.5,
        n_used=5,  # near zero
    )
    audit = audit_for_anomalies(
        result=bad_result,
        walk=walk,
        treatment="treat",
        outcome="y",
        client=None,
    )
    # NEAR_ZERO_N_USED should fire, recommendation disqualify
    assert audit.recommendation == "disqualify"
