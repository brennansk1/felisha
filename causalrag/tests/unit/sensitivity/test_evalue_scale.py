"""Scale-routing tests for the E-value sensitivity helper.

These tests target the bug where ``scale='standardized'`` silently clamped
implausible magnitudes to |d|=10, producing e_value ≈ 9000 and a falsely
"robust" verdict. They also exercise the new ``evalue_for_estimator``
dispatcher.
"""

from __future__ import annotations

import math

import pytest

from causalrag.core.result import EstimationResult
from causalrag.sensitivity.evalue import (
    EValueResult,
    evalue,
    evalue_for_estimator,
)


# ---------------------------------------------------------------------------
# Standardized scale: implausible magnitudes are refused, not clamped
# ---------------------------------------------------------------------------


def test_standardized_extreme_magnitude_returns_unknown_not_clamped() -> None:
    # Pre-fix, this passed (point_estimate=50) was clamped to d=10 and gave
    # e_value ≈ e^9.1 ≈ 9000 — a falsely robust verdict. Post-fix the
    # function must refuse.
    result = evalue(50.0, scale="standardized")

    assert result.reason is not None, "should have refused to compute"
    assert "standardized" in result.reason.lower()
    # Should NOT have produced the buggy ~9000 e-value.
    assert result.e_value < 100.0
    assert "unknown" in result.verdict.lower()


def test_standardized_negative_extreme_also_unknown() -> None:
    result = evalue(-12.0, scale="standardized")
    assert result.reason is not None
    assert "unknown" in result.verdict.lower()


def test_standardized_reasonable_magnitude_still_works() -> None:
    # Cohen's d = 0.5 is a moderate effect; we expect the old behavior.
    result = evalue(0.5, scale="standardized")
    assert result.reason is None
    # exp(0.91*0.5) ≈ 1.578, e-value(1.578) ≈ 2.53
    assert result.e_value == pytest.approx(2.53, abs=0.05)


def test_standardized_boundary_just_above_is_unknown() -> None:
    # |d| > 5 is the cutoff; 5.01 should be unknown.
    result = evalue(5.01, scale="standardized")
    assert result.reason is not None


# ---------------------------------------------------------------------------
# Risk difference scale
# ---------------------------------------------------------------------------


def test_risk_difference_with_baseline_risk_converts_to_rr() -> None:
    # baseline 20%, RD = +10pp → p1 = 30%, RR = 1.5
    # E-value(1.5) = 1.5 + sqrt(1.5*0.5) ≈ 2.366
    result = evalue(0.10, scale="risk_difference", baseline_risk=0.20)
    assert result.reason is None
    assert result.e_value == pytest.approx(2.366, abs=0.05)


def test_risk_difference_without_baseline_risk_is_unknown() -> None:
    result = evalue(0.10, scale="risk_difference")
    assert result.reason is not None
    assert "baseline_risk" in result.reason
    assert "unknown" in result.verdict.lower()


def test_risk_difference_negative_rd_works() -> None:
    # Protective RD: baseline 30%, RD = -10pp → p1 = 20%, RR ≈ 0.667
    # Mirrored to 1.5 → e-value ≈ 2.366
    result = evalue(-0.10, scale="risk_difference", baseline_risk=0.30)
    assert result.reason is None
    assert result.e_value == pytest.approx(2.366, abs=0.05)


def test_risk_difference_inconsistent_with_baseline_is_unknown() -> None:
    # baseline 5% with RD = +99pp implies p1 = 1.04 — impossible
    result = evalue(0.99, scale="risk_difference", baseline_risk=0.05)
    assert result.reason is not None


def test_risk_difference_propagates_ci_bounds() -> None:
    result = evalue(
        0.10,
        scale="risk_difference",
        baseline_risk=0.20,
        ci_low=0.05,
        ci_high=0.15,
    )
    assert result.reason is None
    assert result.e_value_ci is not None
    # The CI does not cross the null on the RR scale (both bounds > 0),
    # so e_value_ci should be a genuine RR-based e-value, not the 1.0
    # "fragile" sentinel.
    assert result.e_value_ci > 1.0


# ---------------------------------------------------------------------------
# evalue_for_estimator dispatcher
# ---------------------------------------------------------------------------


def _result(estimator_id: str, point: float, **kw: object) -> EstimationResult:
    return EstimationResult(
        estimator_id=estimator_id,
        estimand_class=kw.get("estimand_class", "ATE"),  # type: ignore[arg-type]
        point_estimate=point,
        n_used=kw.get("n_used", 500),  # type: ignore[arg-type]
        ci_low=kw.get("ci_low"),  # type: ignore[arg-type]
        ci_high=kw.get("ci_high"),  # type: ignore[arg-type]
    )


def test_dispatcher_picks_risk_difference_for_lineardml_on_binary() -> None:
    res = _result("python.dml.linear", 0.08)
    out = evalue_for_estimator(res, outcome_dtype="binary", baseline_risk=0.2)
    assert out.scale == "risk_difference"
    assert out.reason is None


def test_dispatcher_lineardml_binary_missing_baseline_is_unknown() -> None:
    res = _result("python.dml.linear", 0.08)
    out = evalue_for_estimator(res, outcome_dtype="binary")
    assert out.scale == "risk_difference"
    assert out.reason is not None


def test_dispatcher_picks_hazard_ratio_for_survival_forest() -> None:
    # log-hazard = log(2) so the converted HR is 2 → e-value(2) ≈ 3.41
    res = _result("rbridge.grf.causal_survival_forest", math.log(2.0))
    out = evalue_for_estimator(res, outcome_dtype="survival")
    assert out.scale == "hazard_ratio"
    assert out.reason is None
    assert out.e_value == pytest.approx(3.41, abs=0.05)


def test_dispatcher_picks_hazard_ratio_for_survrm2() -> None:
    res = _result("rbridge.survrm2.rmst_ratio", math.log(1.5))
    out = evalue_for_estimator(res, outcome_dtype="survival")
    assert out.scale == "hazard_ratio"
    assert out.reason is None


def test_dispatcher_picks_odds_ratio_for_bart_on_binary() -> None:
    # log-odds = log(4) → OR=4 → odds_ratio branch uses sqrt → RR=2 → e≈3.41
    res = _result("python.bart", math.log(4.0))
    out = evalue_for_estimator(res, outcome_dtype="binary")
    assert out.scale == "odds_ratio"
    assert out.reason is None
    assert out.e_value == pytest.approx(3.41, abs=0.1)


def test_dispatcher_continuous_outcome_uses_standardized() -> None:
    res = _result("python.dml.linear", 0.4)
    out = evalue_for_estimator(res, outcome_dtype="continuous")
    assert out.scale == "standardized"
    assert out.reason is None


def test_dispatcher_continuous_extreme_is_unknown() -> None:
    # If the caller forgets to pre-standardize and hands us a raw effect of
    # 50, the underlying evalue() must refuse rather than clamp.
    res = _result("python.dml.linear", 50.0)
    out = evalue_for_estimator(res, outcome_dtype="continuous")
    assert out.reason is not None


def test_dispatcher_unknown_estimator_id_on_binary_is_unknown() -> None:
    res = _result("python.someoneelses.thing", 0.1)
    out = evalue_for_estimator(res, outcome_dtype="binary", baseline_risk=0.2)
    assert out.reason is not None


# ---------------------------------------------------------------------------
# Backward-compat smoke test (the public evalue() signature must not break)
# ---------------------------------------------------------------------------


def test_backward_compat_risk_ratio_unchanged() -> None:
    result = evalue(3.9, scale="risk_ratio")
    assert isinstance(result, EValueResult)
    assert result.reason is None
    assert result.e_value == pytest.approx(7.26, abs=0.1)
