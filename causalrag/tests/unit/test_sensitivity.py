from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.sensitivity.evalue import evalue
from causalrag.sensitivity.sensemakr_py import sensemakr
from causalrag.sensitivity.verdict import aggregate


def test_evalue_risk_ratio_canonical() -> None:
    # VanderWeele-Ding 2017 example: RR = 3.9 should give E-value ≈ 7.26
    result = evalue(3.9, scale="risk_ratio")
    assert result.e_value == pytest.approx(7.26, abs=0.1)


def test_evalue_mirrors_protective_effects() -> None:
    result = evalue(0.5, scale="risk_ratio")
    # RR=0.5 mirrors to RR=2 → e-value = 3.41
    assert result.e_value == pytest.approx(3.41, abs=0.05)


def test_evalue_odds_ratio_rare_outcome_uses_sqrt() -> None:
    # Rare outcome (no prevalence supplied) → sqrt approximation
    result = evalue(4.0, scale="odds_ratio")
    # E-value of RR=2 is 3.41
    assert result.e_value == pytest.approx(3.41, abs=0.05)


def test_evalue_standardized_effect() -> None:
    result = evalue(0.5, scale="standardized")
    # exp(0.91*0.5) = 1.578; e-value(1.578) = ~2.53
    assert result.e_value == pytest.approx(2.53, abs=0.05)


def test_evalue_ci_crosses_null_is_fragile() -> None:
    result = evalue(1.5, scale="risk_ratio", ci_low=0.9, ci_high=2.5)
    assert result.e_value_ci == 1.0
    assert "fragile" in result.verdict.lower()


def test_evalue_robust_finding() -> None:
    result = evalue(8.0, scale="risk_ratio", ci_low=4.0, ci_high=16.0)
    assert result.e_value > 5
    assert "robust" in result.verdict.lower() or "moderately" in result.verdict.lower()


def test_sensemakr_fallback_when_pkg_missing() -> None:
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=n)
    t = (rng.normal(size=n) + 0.5 * x > 0).astype(float)
    y = 2.0 * t + 1.5 * x + rng.normal(size=n)
    df = pd.DataFrame({"T": t, "Y": y, "X": x})
    result = sensemakr(df, treatment="T", outcome="Y", covariates=("X",))
    assert result.estimate == pytest.approx(2.0, abs=0.4)
    assert result.robustness_value >= 0.0
    # If sensemakr is not installed, we should see the fallback note.
    if result.backend == "fallback":
        assert result.notes


def test_verdict_min_rule_picks_worst() -> None:
    from causalrag.sensitivity.evalue import EValueResult
    from causalrag.sensitivity.sensemakr_py import SensemakrResult

    e = EValueResult(
        scale="risk_ratio",
        point_estimate=2.0,
        e_value=3.5,
        verdict="robust",
    )
    s = SensemakrResult(
        treatment="T",
        outcome="Y",
        estimate=1.0,
        se=1.0,
        t_value=1.0,
        robustness_value=0.03,
        robustness_value_q=0.03,
    )
    v = aggregate(evalue=e, sensemakr=s)
    assert v.color == "red"  # sensemakr says red, min rule picks red


def test_verdict_handles_no_inputs() -> None:
    v = aggregate()
    assert v.color == "yellow"
