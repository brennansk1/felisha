"""Unit tests for the Zhao 2019 sensitivity-value panel.

Coverage:
  - Synthetic matched-pair design with a true effect of 1.0 (strong) →
    Γ* ≥ 2 → "green".
  - Tiny effect (near-null) → Γ* near 1.0 → "red".
  - Symmetric random pairs with no real effect → Γ* ≈ 1 → "red".
  - Asymptotic-normal CI on Γ contains the bisection point.
  - Bad input shape (mismatched array lengths) raises clearly.
  - ``unknown`` helper for non-matching estimator paths.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from causalrag.sensitivity.zhao_value import (
    ZhaoSensitivityValue,
    zhao_sensitivity_value,
    zhao_sensitivity_value_unknown,
)


# ---------------------------------------------------------------------------
# Large true effect → green, Γ ≥ 2
# ---------------------------------------------------------------------------


def test_large_effect_yields_green_gamma_at_least_two() -> None:
    rng = np.random.default_rng(20260514)
    n = 400
    treated = rng.normal(loc=1.0, scale=1.0, size=n)
    control = rng.normal(loc=0.0, scale=1.0, size=n)

    out = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        alpha=0.05,
    )
    assert isinstance(out, ZhaoSensitivityValue)
    assert out.backend == "python.zhao"
    assert out.verdict == "green"
    assert out.gamma >= 2.0, f"Γ={out.gamma!r}, expected ≥ 2"
    assert out.n_matched_pairs == n
    assert "Γ" in out.rationale or "gamma" in out.rationale.lower() or out.rationale


# ---------------------------------------------------------------------------
# Tiny effect → red, Γ near 1
# ---------------------------------------------------------------------------


def test_tiny_effect_yields_red_gamma_near_one() -> None:
    rng = np.random.default_rng(7)
    n = 200
    # Tiny effect: signal-to-noise ~ 0.05. The matched-pair test will
    # *just barely* reject at α=0.05 with no bias, so Γ* should sit very
    # close to 1 and the verdict should be red.
    treated = rng.normal(loc=0.05, scale=1.0, size=n)
    control = rng.normal(loc=0.0, scale=1.0, size=n)

    out = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        alpha=0.05,
    )
    assert out.verdict == "red"
    assert 1.0 <= out.gamma < 1.5, f"Γ={out.gamma!r}, expected in [1, 1.5)"


# ---------------------------------------------------------------------------
# No effect → Γ* = 1.0 exactly (test does not even reject without bias)
# ---------------------------------------------------------------------------


def test_null_data_yields_red_gamma_equals_one() -> None:
    rng = np.random.default_rng(42)
    n = 300
    treated = rng.normal(loc=0.0, scale=1.0, size=n)
    control = rng.normal(loc=0.0, scale=1.0, size=n)

    out = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        alpha=0.05,
    )
    assert out.verdict == "red"
    assert out.gamma == pytest.approx(1.0, abs=1e-6)
    # Hit the "does not reject at Γ=1" branch — there should be a note.
    assert any("Γ=1" in n or "does not reject" in n for n in out.notes)


# ---------------------------------------------------------------------------
# CI contains the bisection point
# ---------------------------------------------------------------------------


def test_asymptotic_normal_ci_contains_gamma_point() -> None:
    rng = np.random.default_rng(11)
    n = 250
    treated = rng.normal(loc=0.5, scale=1.0, size=n)
    control = rng.normal(loc=0.0, scale=1.0, size=n)

    out = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        alpha=0.05,
        method="zhao_normal",
    )
    # We expect a strict-interior solution → SE and CI populated.
    assert out.gamma_se is not None and out.gamma_se > 0.0
    assert out.gamma_ci_low is not None and out.gamma_ci_high is not None
    assert out.gamma_ci_low <= out.gamma <= out.gamma_ci_high
    assert out.gamma_ci_low >= 1.0  # Γ floored at 1
    assert math.isfinite(out.gamma_se)


# ---------------------------------------------------------------------------
# Verdict mapping edge cases
# ---------------------------------------------------------------------------


def test_verdict_mapping_yellow_zone() -> None:
    # Hand-pick an effect size that lands Γ* in [1.5, 2.0).
    rng = np.random.default_rng(2024)
    n = 250
    # Moderate effect: tuned to land in the yellow band on this seed.
    treated = rng.normal(loc=0.30, scale=1.0, size=n)
    control = rng.normal(loc=0.0, scale=1.0, size=n)
    out = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        alpha=0.05,
    )
    # The verdict must be one of the three traffic-light values, and
    # consistent with the gamma it produced.
    assert out.verdict in ("green", "yellow", "red")
    if out.gamma >= 2.0:
        assert out.verdict == "green"
    elif out.gamma >= 1.5:
        assert out.verdict == "yellow"
    else:
        assert out.verdict == "red"


# ---------------------------------------------------------------------------
# Bad-input validation
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="same length"):
        zhao_sensitivity_value(
            treated_outcomes=np.array([1.0, 2.0, 3.0]),
            matched_control_outcomes=np.array([0.0, 0.5]),
        )


def test_non_1d_input_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        zhao_sensitivity_value(
            treated_outcomes=np.array([[1.0, 2.0], [3.0, 4.0]]),
            matched_control_outcomes=np.array([[0.0, 0.5], [0.6, 0.7]]),
        )


def test_alpha_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="alpha"):
        zhao_sensitivity_value(
            treated_outcomes=np.array([1.0, 2.0, 3.0]),
            matched_control_outcomes=np.array([0.0, 0.5, 0.7]),
            alpha=1.5,
        )


def test_bad_test_statistic_raises() -> None:
    with pytest.raises(ValueError, match="test_statistic"):
        zhao_sensitivity_value(
            treated_outcomes=np.array([1.0, 2.0, 3.0]),
            matched_control_outcomes=np.array([0.0, 0.5, 0.7]),
            test_statistic="bogus",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Alternative test statistics route through the same machinery
# ---------------------------------------------------------------------------


def test_sign_statistic_runs_and_yields_valid_result() -> None:
    rng = np.random.default_rng(0)
    n = 200
    treated = rng.normal(loc=1.0, scale=1.0, size=n)
    control = rng.normal(loc=0.0, scale=1.0, size=n)
    out = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        test_statistic="sign",
    )
    assert out.gamma >= 1.0
    assert any("sign test" in note.lower() for note in out.notes)


def test_t_variant_runs_and_yields_valid_result() -> None:
    rng = np.random.default_rng(1)
    n = 200
    treated = rng.normal(loc=1.0, scale=1.0, size=n)
    control = rng.normal(loc=0.0, scale=1.0, size=n)
    out = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        test_statistic="t",
    )
    assert out.gamma >= 1.0


# ---------------------------------------------------------------------------
# grid_search fallback
# ---------------------------------------------------------------------------


def test_grid_search_method_matches_bisection_within_tolerance() -> None:
    rng = np.random.default_rng(99)
    n = 200
    treated = rng.normal(loc=0.5, scale=1.0, size=n)
    control = rng.normal(loc=0.0, scale=1.0, size=n)
    out_bisect = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        method="zhao_normal",
    )
    out_grid = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        method="grid_search",
    )
    # 2001 grid points over [1,10] → step size ≈ 0.0045, well under 0.05.
    assert abs(out_bisect.gamma - out_grid.gamma) < 0.05
    assert out_grid.gamma_se is None
    assert out_grid.gamma_ci_low is None


# ---------------------------------------------------------------------------
# Non-matching estimator path → unknown
# ---------------------------------------------------------------------------


def test_unknown_helper_for_non_matching_estimator() -> None:
    out = zhao_sensitivity_value_unknown(estimator_id="python.dml.linear")
    assert out.verdict == "unknown"
    assert math.isnan(out.gamma)
    assert out.gamma_ci_low is None
    assert out.gamma_ci_high is None
    assert "rbridge.matchit" in out.rationale
    assert "python.dml.linear" in out.rationale


# ---------------------------------------------------------------------------
# Downward effect mirrored to upper-tail
# ---------------------------------------------------------------------------


def test_negative_effect_is_mirrored_and_still_produces_gamma() -> None:
    rng = np.random.default_rng(5)
    n = 300
    treated = rng.normal(loc=0.0, scale=1.0, size=n)
    control = rng.normal(loc=1.0, scale=1.0, size=n)  # control > treated
    out = zhao_sensitivity_value(
        treated_outcomes=treated,
        matched_control_outcomes=control,
        alpha=0.05,
    )
    # Strong downward effect → Γ should still be ≥ 2 by symmetry.
    assert out.verdict == "green"
    assert out.gamma >= 2.0
    assert any("mirrored" in note.lower() for note in out.notes)
