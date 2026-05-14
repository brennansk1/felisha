"""Unit tests for Sprint 3.1 (EIG) + Sprint 3.6 (saturation) scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from causalrag.loop_scoring import (
    expected_information_gain,
    saturation_probability,
    should_continue_chain_eig,
)


@dataclass
class _FakeChain:
    """Stand-in for :class:`master_loop.ChainState` — only the two
    attributes the scoring API touches."""

    last_point: float | None = None
    last_se: float | None = None


# ─────────── expected_information_gain ───────────────────────────────────


class TestExpectedInformationGain:
    def test_large_when_current_se_far_exceeds_anticipated(self) -> None:
        """A precise next step (small s) on a fuzzy posterior (large σ)
        should yield a big EIG."""
        eig = expected_information_gain(
            current_point=0.5,
            current_se=1.0,
            anticipated_se=0.01,
        )
        # σ/s = 100 ⇒ EIG = 0.5 log(1 + 10_000) ≈ 4.61 nats
        assert eig > 4.0

    def test_approaches_zero_as_anticipated_se_grows(self) -> None:
        """An uninformative next step (s → ∞) carries no EIG."""
        eig_small = expected_information_gain(
            current_point=0.5,
            current_se=0.2,
            anticipated_se=1e6,
        )
        assert eig_small == pytest.approx(0.0, abs=1e-10)

    def test_monotone_in_sigma_over_s(self) -> None:
        """For fixed s, EIG must be monotone non-decreasing in σ."""
        s = 0.3
        sigmas = [0.05, 0.1, 0.2, 0.5, 1.0, 5.0]
        eigs = [
            expected_information_gain(
                current_point=0.0, current_se=sigma, anticipated_se=s
            )
            for sigma in sigmas
        ]
        assert eigs == sorted(eigs)
        # Strictly increasing — no plateaus on this scale.
        for a, b in zip(eigs, eigs[1:], strict=False):
            assert b > a

    def test_matches_closed_form(self) -> None:
        """Spot-check the closed form ½·log(1 + σ²/s²)."""
        sigma, s = 0.4, 0.2
        expected = 0.5 * math.log1p((sigma / s) ** 2)
        got = expected_information_gain(
            current_point=1.0, current_se=sigma, anticipated_se=s
        )
        assert got == pytest.approx(expected)

    def test_zero_when_current_se_is_zero(self) -> None:
        assert (
            expected_information_gain(
                current_point=0.0, current_se=0.0, anticipated_se=0.5
            )
            == 0.0
        )

    def test_zero_when_current_se_is_negative(self) -> None:
        assert (
            expected_information_gain(
                current_point=0.0, current_se=-1.0, anticipated_se=0.5
            )
            == 0.0
        )

    def test_zero_when_inputs_are_nan(self) -> None:
        assert (
            expected_information_gain(
                current_point=0.0,
                current_se=float("nan"),
                anticipated_se=0.5,
            )
            == 0.0
        )
        assert (
            expected_information_gain(
                current_point=0.0,
                current_se=0.5,
                anticipated_se=float("nan"),
            )
            == 0.0
        )

    def test_infinite_when_anticipated_se_is_zero(self) -> None:
        """A degenerate "perfect" next step is signalled by +inf —
        downstream comparisons must remain well-defined."""
        eig = expected_information_gain(
            current_point=0.0, current_se=0.5, anticipated_se=0.0
        )
        assert eig == float("inf")


# ─────────── saturation_probability ──────────────────────────────────────


class TestSaturationProbability:
    def test_deterministic_for_fixed_seed(self) -> None:
        p1 = saturation_probability(
            current_point=0.5,
            current_se=0.2,
            epsilon_ci_width=0.1,
            seed=123,
        )
        p2 = saturation_probability(
            current_point=0.5,
            current_se=0.2,
            epsilon_ci_width=0.1,
            seed=123,
        )
        assert p1 == p2

    def test_in_unit_interval(self) -> None:
        p = saturation_probability(
            current_point=0.0,
            current_se=0.3,
            epsilon_ci_width=0.05,
            n_simulations=2000,
            seed=7,
        )
        assert 0.0 <= p <= 1.0

    def test_precise_estimate_saturates(self) -> None:
        """When the user demands large shrinkage (ε=0.9), a half-normal
        next step rarely delivers — saturation probability must be
        high. (The interpretation: a precise / already-narrow
        estimate is unlikely to halve again.)"""
        p = saturation_probability(
            current_point=0.5,
            current_se=0.05,
            epsilon_ci_width=0.9,
            n_simulations=5000,
            seed=0,
        )
        assert p > 0.85

    def test_lax_epsilon_lowers_saturation(self) -> None:
        """Demanding only a 1 % shrinkage admits many next steps;
        saturation probability must drop below the strict case."""
        p_strict = saturation_probability(
            current_point=0.5,
            current_se=0.2,
            epsilon_ci_width=0.9,
            n_simulations=5000,
            seed=11,
        )
        p_lax = saturation_probability(
            current_point=0.5,
            current_se=0.2,
            epsilon_ci_width=0.01,
            n_simulations=5000,
            seed=11,
        )
        assert p_lax < p_strict

    def test_zero_se_returns_one(self) -> None:
        """Degenerate posterior — nothing to shrink."""
        assert (
            saturation_probability(
                current_point=0.0, current_se=0.0, epsilon_ci_width=0.1
            )
            == 1.0
        )

    def test_negative_se_returns_one(self) -> None:
        assert (
            saturation_probability(
                current_point=0.0, current_se=-0.2, epsilon_ci_width=0.1
            )
            == 1.0
        )

    def test_zero_simulations_returns_one(self) -> None:
        assert (
            saturation_probability(
                current_point=0.0,
                current_se=0.2,
                epsilon_ci_width=0.1,
                n_simulations=0,
            )
            == 1.0
        )


# ─────────── should_continue_chain_eig ───────────────────────────────────


class TestShouldContinueChainEig:
    def test_continues_when_chain_has_no_estimate(self) -> None:
        chain = _FakeChain(last_point=None, last_se=None)
        cont, reason = should_continue_chain_eig(chain_state=chain)
        assert cont is True
        assert "first step" in reason

    def test_continues_when_se_is_zero(self) -> None:
        chain = _FakeChain(last_point=0.5, last_se=0.0)
        cont, reason = should_continue_chain_eig(chain_state=chain)
        assert cont is True
        assert "non-positive" in reason

    def test_continues_when_se_is_negative(self) -> None:
        chain = _FakeChain(last_point=0.5, last_se=-0.1)
        cont, reason = should_continue_chain_eig(chain_state=chain)
        assert cont is True

    def test_stops_a_saturated_chain(self) -> None:
        """Default conservative anchor (s = σ) yields EIG = ½·log 2 ≈
        0.347 nats. Demanding a *huge* shrinkage (ε_ci=0.95) means
        almost every half-normal draw on the next SE fails to beat
        it, so saturation probability runs ≈1. We raise the EIG floor
        above 0.347 so both legs of the AND fire."""
        chain = _FakeChain(last_point=0.2, last_se=0.001)
        cont, reason = should_continue_chain_eig(
            chain_state=chain,
            epsilon_eig=1.0,  # demand >1 nat; default anchor gives ≈0.347
            epsilon_ci=0.95,  # demand 95% CI shrinkage — rarely happens
            saturation_threshold=0.5,
        )
        assert cont is False
        assert "saturated" in reason

    def test_continues_under_explored_chain(self) -> None:
        """Anticipated SE much smaller than current SE ⇒ huge EIG ⇒
        chain should keep going even if saturation looks high."""
        chain = _FakeChain(last_point=0.4, last_se=0.5)
        cont, reason = should_continue_chain_eig(
            chain_state=chain,
            epsilon_eig=0.05,
            epsilon_ci=0.1,
            saturation_threshold=0.9,
            anticipated_se=0.01,  # very informative next step
        )
        assert cont is True
        assert "EIG=" in reason

    def test_requires_both_conditions_to_stop(self) -> None:
        """EIG below threshold alone must not stop the chain."""
        chain = _FakeChain(last_point=0.0, last_se=0.1)
        # Make saturation impossibly hard to trigger (threshold ≈ 1.0)
        # while EIG sits at the default ½·log 2 ≈ 0.347 — above 0.05
        # so the EIG-floor isn't tripped either.
        cont, _ = should_continue_chain_eig(
            chain_state=chain,
            epsilon_eig=0.05,
            saturation_threshold=0.999,
        )
        assert cont is True

    def test_reason_string_is_informative(self) -> None:
        chain = _FakeChain(last_point=0.3, last_se=0.2)
        _, reason = should_continue_chain_eig(chain_state=chain)
        # Reason should mention EIG and saturation so observers can
        # see *why* the chain continued or stopped.
        assert "EIG" in reason
        assert "saturation" in reason or "saturated" in reason
