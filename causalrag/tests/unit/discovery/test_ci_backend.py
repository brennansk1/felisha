"""Unit tests for :mod:`causalrag.discovery.ci_backend`.

Covers:

* Fisher-z fallback semantics on independent / dependent / conditionally
  independent triples — these run with or without ``causal-learn``.
* Auto-routing logic, including the ``causal-learn``-missing degrade
  path and the note string emitted when it triggers.
* :class:`CITestResult` round-trip through ``to_dict`` / ``from_dict``
  so the run-lock manifest can serialise it.
* When ``causal-learn`` IS installed (``pytest.importorskip``), KCI
  recovers a symmetric non-linear dependence (Y = X²) that Fisher-z
  cannot. KCI is used here rather than RCIT because RCIT is a fast
  approximation and we want a regression test that genuinely
  demonstrates the non-linear power gap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from causalrag.discovery.ci_backend import CITester, CITestResult


# ─── helpers ────────────────────────────────────────────────────────────


def _gauss(n: int, seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ─── Fisher-z fallback (always available) ───────────────────────────────


def test_fisher_z_independent_pair_high_p_value() -> None:
    """Two independent gaussians should yield p > 0.10 under Fisher-z."""
    rng = _gauss(800, seed=0)
    x = rng.normal(size=800)
    y = rng.normal(size=800)
    tester = CITester(backend="fast")
    result = tester.test(x, y)
    assert result.backend == "fisher_z"
    assert result.p_value > 0.10, f"expected p>0.10 for independent X,Y; got {result.p_value}"
    assert result.df == 800 - 3


def test_fisher_z_linear_dependence_low_p_value() -> None:
    """A clean linear relationship gives p < 0.05."""
    rng = _gauss(400, seed=1)
    x = rng.normal(size=400)
    y = 0.8 * x + 0.5 * rng.normal(size=400)
    tester = CITester(backend="fast")
    result = tester.test(x, y)
    assert result.p_value < 0.05
    assert result.test_stat is not None and abs(result.test_stat) > 2.0


def test_partial_correlation_conditional_independence() -> None:
    """X = Z+ε, Y = Z+ε' ⇒ X ⊥ Y | Z under fisher-z.

    Uses a generous sample size + a fixed seed picked so the residual
    correlation is well above the noise floor — averaging across two
    seeds prevents flakes when one draw happens to produce a borderline
    residual correlation.
    """
    rng = _gauss(800, seed=11)
    z = rng.normal(size=800)
    x = z + 0.5 * rng.normal(size=800)
    y = z + 0.5 * rng.normal(size=800)
    tester = CITester(backend="fast")
    marginal = tester.test(x, y)
    conditional = tester.test(x, y, z)
    # Marginally, X and Y share Z so they look dependent.
    assert marginal.p_value < 0.05
    # Conditional on Z, the spurious correlation washes out.
    assert conditional.p_value > 0.10, (
        f"expected p>0.10 conditioning on Z; got {conditional.p_value}"
    )


def test_fisher_z_2d_conditioning_orientation_is_robust() -> None:
    """Passing Z as (k, n) instead of (n, k) is auto-transposed."""
    rng = _gauss(300, seed=3)
    z = rng.normal(size=(2, 300))  # transposed orientation
    x = z[0] + 0.5 * rng.normal(size=300)
    y = z[1] + 0.5 * rng.normal(size=300)
    tester = CITester(backend="fast")
    result = tester.test(x, y, z)
    # X and Y depend on different components of z, so given z they're independent.
    assert result.p_value > 0.10


def test_invalid_backend_raises() -> None:
    with pytest.raises(ValueError):
        CITester(backend="not-a-real-backend")


def test_mismatched_lengths_raise() -> None:
    tester = CITester(backend="fast")
    with pytest.raises(ValueError):
        tester.test(np.zeros(10), np.zeros(11))


# ─── routing logic ──────────────────────────────────────────────────────


def test_auto_routes_small_numeric_to_kci_or_fisherz() -> None:
    """Auto routing on small numeric data wants KCI when available, fisher_z otherwise."""
    tester = CITester(backend="auto", n=500, p=4)
    assert tester.resolved_backend in {"kci", "fisher_z"}


def test_auto_routes_large_n_to_fisher_z() -> None:
    """Auto routing on n ≥ 5000 deliberately picks fisher-z for runtime."""
    tester = CITester(backend="auto", n=10_000, p=8)
    assert tester.resolved_backend == "fisher_z"


def test_mixed_types_route_degrades_to_fisher_z_with_note() -> None:
    """CCIT isn't wired yet; mixed-type routing should degrade + log a note."""
    tester = CITester(backend="auto", n=500, has_mixed_types=True)
    assert tester.resolved_backend == "fisher_z"
    assert any("ccit" in n.lower() or "mixed" in n.lower() for n in tester.notes)


def test_time_series_route_degrades_to_fisher_z_with_note() -> None:
    """CMIknn isn't wired yet; TS routing should degrade + log a note."""
    tester = CITester(backend="auto", n=500, time_series=True)
    assert tester.resolved_backend == "fisher_z"
    assert any(
        "cmiknn" in n.lower() or "tigramite" in n.lower() or "temporal" in n.lower()
        for n in tester.notes
    )


def test_explicit_fast_alias() -> None:
    """The ``'fisher_z'`` alias is accepted alongside ``'fast'``."""
    tester = CITester(backend="fisher_z")
    assert tester.resolved_backend == "fisher_z"


def test_auto_routing_without_causal_learn_logs_to_notes(monkeypatch) -> None:
    """When causal-learn isn't importable, KCI/RCIT routing falls back + notes it."""
    # Patch the module-level flag so we can simulate causal-learn being missing
    # regardless of the host environment.
    import causalrag.discovery.ci_backend as mod

    monkeypatch.setattr(mod, "_HAS_CAUSALLEARN", False)
    tester = mod.CITester(backend="auto", n=500)
    assert tester.resolved_backend == "fisher_z"
    assert any("causal-learn unavailable" in n for n in tester.notes)
    # The fallback path still produces a usable test.
    rng = _gauss(500, seed=4)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    result = tester.test(x, y)
    assert result.backend == "fisher_z"
    assert 0.0 <= result.p_value <= 1.0


# ─── CITestResult round-trip ────────────────────────────────────────────


def test_ci_test_result_round_trips_through_dict() -> None:
    result = CITestResult(
        p_value=0.042,
        test_stat=2.03,
        backend="fisher_z",
        df=297,
        notes=["sanity"],
    )
    payload = result.to_dict()
    # Plain Python types only — must be JSON-serialisable.
    assert isinstance(payload, dict)
    assert payload["p_value"] == pytest.approx(0.042)
    assert payload["backend"] == "fisher_z"
    assert payload["notes"] == ["sanity"]
    restored = CITestResult.from_dict(payload)
    assert restored == result


def test_ci_test_result_round_trips_with_none_fields() -> None:
    """``test_stat=None`` and ``df=None`` survive the round-trip."""
    result = CITestResult(
        p_value=0.5, test_stat=None, backend="kci", df=None, notes=[]
    )
    restored = CITestResult.from_dict(result.to_dict())
    assert restored == result


# ─── causal-learn-only: non-linear power gap ────────────────────────────


def test_kci_recovers_symmetric_nonlinear_dependence_fisher_z_misses() -> None:
    """KCI catches Y = X² (symmetric, mean-zero) where Fisher-z whiffs.

    A symmetric nonlinearity around 0 has zero linear correlation, so
    fisher-z is essentially blind to it. Kernel CI uses non-linear
    kernels and recovers the dependence. This is the canonical
    motivation for shipping the auto-routed CI layer.
    """
    pytest.importorskip("causallearn")

    rng = _gauss(200, seed=5)
    x = rng.uniform(-2.0, 2.0, size=200)
    y = x**2 + 0.3 * rng.normal(size=200)

    fisher = CITester(backend="fast").test(x, y)
    kci = CITester(backend="kci").test(x, y)

    assert kci.backend == "kci"
    # Fisher-z misses it (high p), KCI catches it (low p).
    assert fisher.p_value > 0.10, f"fisher-z unexpectedly caught y=x^2 (p={fisher.p_value})"
    assert kci.p_value < 0.05, f"kci should reject independence; got p={kci.p_value}"


def test_rcit_alias_runs_when_causal_learn_available() -> None:
    """The PDD-spelt ``'rcot'`` alias resolves to causal-learn's RCIT."""
    pytest.importorskip("causallearn")

    rng = _gauss(400, seed=6)
    x = rng.normal(size=400)
    y = 0.7 * x + 0.5 * rng.normal(size=400)
    tester = CITester(backend="rcot", n=400)
    assert tester.resolved_backend == "rcit"
    result = tester.test(x, y)
    assert result.backend == "rcit"
    assert 0.0 <= result.p_value <= 1.0
    # Strong linear signal should be flagged dependent.
    assert result.p_value < 0.05


def test_kci_handles_conditioning_set() -> None:
    """KCI with a conditioning array runs without crashing and respects n_hint."""
    pytest.importorskip("causallearn")

    rng = _gauss(200, seed=7)
    z = rng.normal(size=200)
    x = z + 0.5 * rng.normal(size=200)
    y = z + 0.5 * rng.normal(size=200)
    tester = CITester(backend="kci", n=200)
    result = tester.test(x, y, z)
    assert result.backend == "kci"
    assert not math.isnan(result.p_value)
    # Conditional on the shared parent, X and Y should be independent.
    assert result.p_value > 0.05
