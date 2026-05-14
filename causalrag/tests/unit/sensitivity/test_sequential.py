"""Tests for the always-valid (anytime-valid) confidence-sequence helpers.

Sprint 9.4 acceptance criteria:

1.  ``always_valid_ci`` covers the true mean of synthetic IF values at
    *every* sample size we peek at (anytime validity).
2.  The CI shrinks as ``n`` grows.
3.  The anytime-valid CI is *strictly wider* than the fixed-n Wald CI
    at the same nominal coverage -- this is the price of optional
    stopping.
4.  ``update_anytime_ci`` is consistent with reconstructing from
    scratch on the concatenated vector.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from causalrag.sensitivity.sequential import (
    AnytimeValidCI,
    always_valid_ci,
    update_anytime_ci,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _wald_half_width(phi: np.ndarray, alpha: float) -> float:
    """Plain fixed-n Wald half-width (z_{1-alpha/2} * sigma_hat / sqrt(n))."""
    n = phi.size
    # For alpha=0.05 the exact constant is 1.959963984540054.
    z = math.sqrt(2.0) * _inv_erf(1.0 - alpha)
    return z * float(phi.std(ddof=1)) / math.sqrt(n)


def _inv_erf(x: float) -> float:
    """Winitzki 2008 approximation to erf^{-1}, accurate enough for tests."""
    a = 0.147
    ln = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(first * first - ln / a) - first), x)


# --------------------------------------------------------------------------- #
# Basic shape / contract
# --------------------------------------------------------------------------- #


class TestContract:
    """Surface-area sanity checks."""

    def test_returns_anytime_valid_ci_dataclass(self) -> None:
        rng = np.random.default_rng(0)
        phi = rng.normal(0.5, 1.0, size=200)
        ci = always_valid_ci(phi, alpha=0.05)
        assert isinstance(ci, AnytimeValidCI)
        assert ci.method == "betting"
        assert ci.coverage == pytest.approx(0.95)
        assert ci.n_at_check == 200
        assert ci.lower < ci.point < ci.upper
        assert ci.rationale  # non-empty string

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            always_valid_ci(np.array([]))

    def test_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            always_valid_ci(np.array([1.0, 2.0]), alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            always_valid_ci(np.array([1.0, 2.0]), alpha=1.0)

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="unknown method"):
            always_valid_ci(np.array([1.0, 2.0]), method="nonsense")  # type: ignore[arg-type]

    def test_asymptotic_cs_branch_runs(self) -> None:
        rng = np.random.default_rng(1)
        phi = rng.normal(0.0, 1.0, size=500)
        ci = always_valid_ci(phi, alpha=0.05, method="asymptotic-cs")
        assert ci.method == "asymptotic-cs"
        assert ci.lower < ci.upper
        assert "Howard" in ci.rationale or "asymptotic-cs" in ci.rationale

    def test_hoeffding_cs_alias_runs(self) -> None:
        phi = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ci = always_valid_ci(phi, alpha=0.1, method="hoeffding-cs")
        assert ci.method == "hoeffding-cs"
        assert "hoeffding-cs" in ci.rationale


# --------------------------------------------------------------------------- #
# Anytime coverage at every peek
# --------------------------------------------------------------------------- #


class TestAnytimeCoverage:
    """The defining property of a confidence sequence: cover *at every n*."""

    @pytest.mark.parametrize("method", ["betting", "asymptotic-cs"])
    def test_covers_truth_at_every_check(self, method: str) -> None:
        """Walk through n in [50, 1000] step 50; truth must always lie inside.

        This is one realization with a friendly seed; we are not claiming
        the worst-case ``1 - alpha`` rate here (that needs Monte Carlo).
        We are checking that on a *single typical path* the CI never
        excludes the truth -- which it should not, since the WSR/HRMS
        CIs are very conservative on Gaussian data.
        """
        rng = np.random.default_rng(42)
        truth = 0.7
        phi_full = rng.normal(truth, 1.5, size=1000)
        for n in range(50, 1001, 50):
            ci = always_valid_ci(phi_full[:n], alpha=0.05, method=method)  # type: ignore[arg-type]
            assert ci.lower <= truth <= ci.upper, (
                f"method={method} n={n}: CI [{ci.lower:.3f}, {ci.upper:.3f}] "
                f"excludes truth={truth}"
            )

    def test_high_coverage_under_monte_carlo(self) -> None:
        """Empirical coverage at the planned n should be >= 1 - alpha.

        We run many independent draws and confirm the betting CI covers
        the truth in at least 95% of them. This is the type-I rate
        guarantee, not the anytime one (which is strictly stronger).
        """
        rng = np.random.default_rng(7)
        truth = -0.3
        n_reps = 400
        hits = 0
        for _ in range(n_reps):
            phi = rng.normal(truth, 1.0, size=300)
            ci = always_valid_ci(phi, alpha=0.05, method="betting")
            if ci.lower <= truth <= ci.upper:
                hits += 1
        # Confidence-sequence CIs are conservative -- expect well above 95%.
        assert hits / n_reps >= 0.95


# --------------------------------------------------------------------------- #
# Width shrinks with n
# --------------------------------------------------------------------------- #


class TestShrinkage:
    """As more data arrives, the CI half-width should monotonically shrink."""

    @pytest.mark.parametrize("method", ["betting", "asymptotic-cs"])
    def test_width_shrinks_in_n(self, method: str) -> None:
        rng = np.random.default_rng(11)
        phi = rng.normal(0.0, 1.0, size=2000)
        widths = []
        for n in (100, 500, 1000, 2000):
            ci = always_valid_ci(phi[:n], alpha=0.05, method=method)  # type: ignore[arg-type]
            widths.append(ci.upper - ci.lower)
        # Strictly decreasing -- confidence sequences can wobble but on
        # a single homogeneous sample the trend is monotone in n.
        for prev, nxt in pairwise(widths):
            assert nxt < prev, f"widths did not shrink: {widths}"

    def test_width_scales_roughly_as_one_over_sqrt_n(self) -> None:
        """Sanity: doubling n cuts the width by less than sqrt(2) and more than 1."""
        rng = np.random.default_rng(13)
        phi = rng.normal(0.0, 1.0, size=4000)
        ci_small = always_valid_ci(phi[:1000], alpha=0.05)
        ci_big = always_valid_ci(phi[:4000], alpha=0.05)
        w_small = ci_small.upper - ci_small.lower
        w_big = ci_big.upper - ci_big.lower
        ratio = w_small / w_big
        # 4x the data: sqrt(4) = 2.0 is the parametric bound; CS pays
        # log iteration so the actual ratio is somewhat less. Allow a band.
        assert 1.5 < ratio < 2.5, f"unexpected shrinkage ratio {ratio:.3f}"


# --------------------------------------------------------------------------- #
# Comparison to fixed-n Wald
# --------------------------------------------------------------------------- #


class TestVsWald:
    """The anytime-valid CI is the price of peeking -- it MUST be wider."""

    @pytest.mark.parametrize("method", ["betting", "asymptotic-cs"])
    @pytest.mark.parametrize("n", [200, 1000, 5000])
    def test_wider_than_fixed_n_wald(self, method: str, n: int) -> None:
        rng = np.random.default_rng(17 + n)
        phi = rng.normal(0.0, 1.0, size=n)
        ci = always_valid_ci(phi, alpha=0.05, method=method)  # type: ignore[arg-type]
        cs_half = (ci.upper - ci.lower) / 2.0
        wald_half = _wald_half_width(phi, alpha=0.05)
        # Strictly wider, by a non-negligible margin at small n,
        # narrowing toward (but never crossing) the fixed-n width.
        assert cs_half > wald_half, (
            f"{method} n={n}: cs_half={cs_half:.4f} not > wald_half={wald_half:.4f}"
        )


# --------------------------------------------------------------------------- #
# Online update consistency
# --------------------------------------------------------------------------- #


class TestOnlineUpdate:
    @pytest.mark.parametrize("method", ["betting", "asymptotic-cs"])
    def test_update_matches_reconstruction(self, method: str) -> None:
        """Update on a batch == reconstruct from the concatenated vector."""
        rng = np.random.default_rng(23)
        phi1 = rng.normal(0.2, 1.0, size=300)
        phi2 = rng.normal(0.2, 1.0, size=200)

        ci1 = always_valid_ci(phi1, alpha=0.05, method=method)  # type: ignore[arg-type]
        ci_online = update_anytime_ci(ci1, phi2)
        ci_full = always_valid_ci(
            np.concatenate([phi1, phi2]), alpha=0.05, method=method  # type: ignore[arg-type]
        )

        assert ci_online.n_at_check == ci_full.n_at_check == 500
        assert ci_online.point == pytest.approx(ci_full.point, rel=1e-10)
        assert ci_online.lower == pytest.approx(ci_full.lower, rel=1e-8, abs=1e-10)
        assert ci_online.upper == pytest.approx(ci_full.upper, rel=1e-8, abs=1e-10)
        assert ci_online.method == ci_full.method == method

    def test_update_with_empty_batch_is_idempotent(self) -> None:
        rng = np.random.default_rng(29)
        phi = rng.normal(0.0, 1.0, size=100)
        ci = always_valid_ci(phi, alpha=0.05)
        ci2 = update_anytime_ci(ci, np.array([]))
        assert ci2.point == pytest.approx(ci.point)
        assert ci2.lower == pytest.approx(ci.lower)
        assert ci2.upper == pytest.approx(ci.upper)
        assert ci2.n_at_check == ci.n_at_check

    def test_update_rejects_state_less_ci(self) -> None:
        """A hand-built CI without _state cannot be updated."""
        bare = AnytimeValidCI(
            point=0.0,
            lower=-1.0,
            upper=1.0,
            coverage=0.95,
            n_at_check=10,
            method="betting",
            rationale="hand-built",
        )
        with pytest.raises(ValueError, match="no carried state"):
            update_anytime_ci(bare, np.array([0.1, 0.2]))

    def test_repeated_online_updates_match_full(self) -> None:
        """Five back-to-back updates == one reconstruction at the end.

        Models the auto-mode loop: peek, peek, peek, peek, peek.
        """
        rng = np.random.default_rng(31)
        chunks = [rng.normal(-0.1, 0.7, size=120) for _ in range(5)]
        ci = always_valid_ci(chunks[0], alpha=0.05, method="betting")
        for c in chunks[1:]:
            ci = update_anytime_ci(ci, c)
        ci_full = always_valid_ci(
            np.concatenate(chunks), alpha=0.05, method="betting"
        )
        assert ci.n_at_check == ci_full.n_at_check
        assert ci.point == pytest.approx(ci_full.point, rel=1e-10)
        assert ci.lower == pytest.approx(ci_full.lower, rel=1e-8, abs=1e-10)
        assert ci.upper == pytest.approx(ci_full.upper, rel=1e-8, abs=1e-10)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


class TestEdges:
    def test_degenerate_constant_if_collapses(self) -> None:
        """If all IF values are identical, sigma_hat^2 = 0 and the CI collapses."""
        phi = np.full(50, 0.42)
        ci = always_valid_ci(phi, alpha=0.05, method="asymptotic-cs")
        assert ci.lower == pytest.approx(0.42)
        assert ci.upper == pytest.approx(0.42)
        assert ci.point == pytest.approx(0.42)

    def test_single_observation_does_not_crash(self) -> None:
        ci = always_valid_ci(np.array([1.7]), alpha=0.05, method="asymptotic-cs")
        assert ci.n_at_check == 1
        # With n=1 the unbiased sigma_hat^2 is 0 by convention -- degenerate.
        assert ci.lower == pytest.approx(1.7)
        assert ci.upper == pytest.approx(1.7)

    def test_alpha_tighter_means_wider_ci(self) -> None:
        """Smaller alpha (tighter coverage) -> wider CI."""
        rng = np.random.default_rng(41)
        phi = rng.normal(0.0, 1.0, size=500)
        ci_loose = always_valid_ci(phi, alpha=0.10)
        ci_tight = always_valid_ci(phi, alpha=0.01)
        assert (ci_tight.upper - ci_tight.lower) > (ci_loose.upper - ci_loose.lower)
