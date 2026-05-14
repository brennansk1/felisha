"""Tests for the LLM-assisted anomaly / sanity-check audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from causalrag.core.protocol import RoadmapWalk
from causalrag.core.result import EstimationResult
from causalrag.llm.ollama_client import FakeOllamaTransport, OllamaClient
from causalrag.sensitivity.anomaly_audit import (
    AnomalyAudit,
    AnomalyAuditFlag,
    audit_for_anomalies,
)


# --- helpers ----------------------------------------------------------------


def _walk(hyp_id: str = "h1") -> RoadmapWalk:
    return RoadmapWalk(hypothesis_id=hyp_id, q1_question="Does T affect Y?")


def _result(
    *,
    point: float = 0.2,
    se: float | None = 0.05,
    ci_low: float | None = 0.10,
    ci_high: float | None = 0.30,
    p_value: float | None = 0.01,
    n_used: int = 1000,
    diagnostics: dict[str, Any] | None = None,
    refutations: dict[str, Any] | None = None,
) -> EstimationResult:
    return EstimationResult(
        estimator_id="python.dml.linear",
        estimand_class="ATE",
        point_estimate=point,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        n_used=n_used,
        diagnostics=diagnostics or {},
        refutations=refutations or {},
    )


def _llm_client(payload: dict[str, Any], tmp_path: Path) -> OllamaClient:
    transport = FakeOllamaTransport({"": json.dumps(payload)})
    return OllamaClient(
        model="qwen3:14b-q4_K_M",
        seed=0,
        cassette_dir=tmp_path,
        transport=transport,
        allow_live=True,
    )


# --- deterministic-only tests -----------------------------------------------


def test_near_zero_n_used_flagged_and_disqualifies() -> None:
    res = _result(n_used=5, point=0.2, ci_low=0.1, ci_high=0.3, p_value=0.01)
    audit = audit_for_anomalies(
        result=res,
        walk=_walk(),
        treatment="T",
        outcome="Y",
    )
    assert AnomalyAuditFlag.NEAR_ZERO_N_USED in audit.flags
    assert audit.recommendation == "disqualify"


def test_ci_too_wide_flagged() -> None:
    # point=0.05, CI width = 2 → 40x point, way over 10x threshold.
    res = _result(point=0.05, ci_low=-1.0, ci_high=1.0, p_value=0.9, se=0.5, n_used=500)
    audit = audit_for_anomalies(
        result=res,
        walk=_walk(),
        treatment="T",
        outcome="Y",
    )
    assert AnomalyAuditFlag.CI_TOO_WIDE in audit.flags
    # Disqualify floor doesn't fire; no implausible-magnitude or sign-flip → accept.
    assert audit.recommendation == "accept"


def test_saturated_propensity_flagged() -> None:
    res = _result(
        diagnostics={"overlap": {"p_min": 0.05, "p_max": 0.999}},
    )
    audit = audit_for_anomalies(
        result=res,
        walk=_walk(),
        treatment="T",
        outcome="Y",
    )
    assert AnomalyAuditFlag.SATURATED_PROPENSITY in audit.flags


def test_p_value_inconsistent_with_ci_flagged() -> None:
    # CI excludes 0 but p is large.
    res = _result(point=0.2, ci_low=0.1, ci_high=0.3, p_value=0.6, n_used=200)
    audit = audit_for_anomalies(
        result=res, walk=_walk(), treatment="T", outcome="Y"
    )
    assert AnomalyAuditFlag.P_VALUE_INCONSISTENT_WITH_CI in audit.flags


def test_sign_flip_vs_naive_deterministic() -> None:
    res = _result(point=-0.2, ci_low=-0.3, ci_high=-0.1, p_value=0.01, n_used=400)
    audit = audit_for_anomalies(
        result=res,
        walk=_walk(),
        treatment="T",
        outcome="Y",
        naive_estimate=+0.3,
    )
    assert AnomalyAuditFlag.SIGN_FLIP_VS_NAIVE in audit.flags
    assert audit.recommendation == "rerun_with_different_estimator"


def test_refutation_divergence_flagged() -> None:
    refs = {
        "tests": [
            {"name": "placebo_treatment", "delta_in_se_units": 5.2},
            {"name": "subset", "delta_in_se_units": 0.4},
        ]
    }
    res = _result(refutations=refs)
    audit = audit_for_anomalies(
        result=res, walk=_walk(), treatment="T", outcome="Y"
    )
    assert AnomalyAuditFlag.REFUTATION_DIVERGENCE in audit.flags


def test_client_none_returns_deterministic_only() -> None:
    # Provide an input that triggers two deterministic flags.
    res = _result(
        n_used=5,
        diagnostics={"overlap": {"p_min": 0.001, "p_max": 0.5}},
    )
    audit = audit_for_anomalies(
        result=res, walk=_walk(), treatment="T", outcome="Y", client=None
    )
    assert set(audit.flags) == {
        AnomalyAuditFlag.NEAR_ZERO_N_USED,
        AnomalyAuditFlag.SATURATED_PROPENSITY,
    }
    assert audit.recommendation == "disqualify"
    assert "Deterministic" in audit.overall_note


def test_clean_result_no_flags_accepts() -> None:
    res = _result()
    audit = audit_for_anomalies(
        result=res, walk=_walk(), treatment="T", outcome="Y"
    )
    assert audit.flags == []
    assert audit.recommendation == "accept"
    assert isinstance(audit, AnomalyAudit)


# --- LLM-integrated tests ---------------------------------------------------


def test_llm_implausible_magnitude_forces_rerun(tmp_path: Path) -> None:
    res = _result(point=0.85, ci_low=0.80, ci_high=0.90, p_value=1e-9, n_used=2000)
    payload = {
        "flags": [AnomalyAuditFlag.IMPLAUSIBLE_MAGNITUDE.value],
        "rationale_per_flag": {
            AnomalyAuditFlag.IMPLAUSIBLE_MAGNITUDE.value: (
                "An 85-percentage-point absolute risk reduction is "
                "implausible for any pharmaceutical intervention."
            )
        },
        "recommendation": "accept",  # LLM's recommendation, will be overridden
        "overall_note": "Magnitude is implausible.",
    }
    client = _llm_client(payload, tmp_path)

    audit = audit_for_anomalies(
        result=res,
        walk=_walk(),
        treatment="drug_X",
        outcome="five_year_survival",
        domain_brief="Oncology trial; typical effect sizes are 5-15 pp.",
        client=client,
    )
    assert AnomalyAuditFlag.IMPLAUSIBLE_MAGNITUDE in audit.flags
    # Deterministic override forces rerun_with_different_estimator.
    assert audit.recommendation == "rerun_with_different_estimator"
    # Preserve the LLM's rationale.
    assert (
        AnomalyAuditFlag.IMPLAUSIBLE_MAGNITUDE.value
        in audit.rationale_per_flag
    )


def test_llm_flags_merged_with_deterministic(tmp_path: Path) -> None:
    # Deterministic: NEAR_ZERO_N_USED + CI_TOO_WIDE
    # LLM: OVERFIT_RISK
    res = _result(
        point=0.05,
        ci_low=-1.0,
        ci_high=1.0,
        p_value=0.9,
        n_used=8,  # <10 → disqualify
    )
    payload = {
        "flags": [AnomalyAuditFlag.OVERFIT_RISK.value],
        "rationale_per_flag": {
            AnomalyAuditFlag.OVERFIT_RISK.value: "Per-row CATE forest is suspect."
        },
        "recommendation": "accept",
        "overall_note": "Overfit risk noted.",
    }
    client = _llm_client(payload, tmp_path)
    audit = audit_for_anomalies(
        result=res, walk=_walk(), treatment="T", outcome="Y", client=client
    )
    # All three present, dedupe holds order: deterministic first.
    assert AnomalyAuditFlag.NEAR_ZERO_N_USED in audit.flags
    assert AnomalyAuditFlag.CI_TOO_WIDE in audit.flags
    assert AnomalyAuditFlag.OVERFIT_RISK in audit.flags
    # n_used < 10 forces disqualify regardless of LLM.
    assert audit.recommendation == "disqualify"


def test_llm_recommendation_wins_when_no_severe_pre_screen(tmp_path: Path) -> None:
    res = _result()  # clean
    payload = {
        "flags": [AnomalyAuditFlag.OVERFIT_RISK.value],
        "rationale_per_flag": {
            AnomalyAuditFlag.OVERFIT_RISK.value: "Heuristic concern only."
        },
        "recommendation": "rerun_with_different_estimator",
        "overall_note": "Re-run advised.",
    }
    client = _llm_client(payload, tmp_path)
    audit = audit_for_anomalies(
        result=res, walk=_walk(), treatment="T", outcome="Y", client=client
    )
    assert audit.recommendation == "rerun_with_different_estimator"


def test_llm_failure_falls_back_to_deterministic(tmp_path: Path) -> None:
    # Transport returns garbage that fails JSON / schema validation,
    # so client.parse raises SchemaValidationFailed inside audit. We
    # must NOT raise out; we must return the deterministic-only audit.
    transport = FakeOllamaTransport({"": "not-valid-json"})
    client = OllamaClient(
        model="m",
        seed=0,
        cassette_dir=tmp_path,
        transport=transport,
        allow_live=True,
        max_retries=0,
    )
    res = _result(n_used=5)
    audit = audit_for_anomalies(
        result=res, walk=_walk(), treatment="T", outcome="Y", client=client
    )
    assert AnomalyAuditFlag.NEAR_ZERO_N_USED in audit.flags
    assert audit.recommendation == "disqualify"
    assert "LLM audit failed" in audit.overall_note


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])
