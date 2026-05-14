"""Tests for the Calonico-Cattaneo-Titiunik RD bridge (``rbridge.rd``).

The full R-bridged tests are gated on rpy2 + the ``rdrobust`` /
``rddensity`` R packages. A pure-Python diagnostics-shape test runs
without R by stubbing the ``r_session`` module-level helpers so the
suite has coverage even on CI machines that don't ship R.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalrag.estimators.rbridge import rd as rd_mod
from causalrag.estimators.rbridge.rd import RDRobustEstimator


# ---------------------------------------------------------------------------
# Helpers — gating + synthetic DGPs.
# ---------------------------------------------------------------------------


def _have_r_rdrobust() -> bool:
    """rpy2 installed AND the R ``rdrobust`` package available."""
    try:
        import rpy2  # noqa: F401
        import rpy2.robjects as ro
    except Exception:
        return False
    try:
        return bool(
            list(ro.r('requireNamespace("rdrobust", quietly = TRUE)'))[0]
        )
    except Exception:
        return False


def _have_r_rddensity() -> bool:
    try:
        import rpy2  # noqa: F401
        import rpy2.robjects as ro
    except Exception:
        return False
    try:
        return bool(
            list(ro.r('requireNamespace("rddensity", quietly = TRUE)'))[0]
        )
    except Exception:
        return False


def _sharp_rd_df(n: int = 2000, tau: float = 2.0, seed: int = 0) -> pd.DataFrame:
    """Y = alpha + tau * 1{X >= 0} + beta * X + eps, X ~ Uniform[-1, 1]."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=n)
    eps = rng.normal(0.0, 1.0, size=n)
    Y = 0.5 + tau * (X >= 0.0).astype(float) + 0.6 * X + eps
    return pd.DataFrame({"y": Y, "x": X})


def _fuzzy_rd_df(
    n: int = 2500, tau: float = 1.5, seed: int = 0
) -> pd.DataFrame:
    """Fuzzy RD: T = 1 with prob 0.8 if X >= 0, prob 0.2 if X < 0;
    Y = alpha + tau * T + 0.5 * X + eps."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=n)
    above = X >= 0.0
    p_t = np.where(above, 0.8, 0.2)
    T = (rng.uniform(0.0, 1.0, size=n) < p_t).astype(float)
    eps = rng.normal(0.0, 1.0, size=n)
    Y = 0.5 + tau * T + 0.5 * X + eps
    return pd.DataFrame({"y": Y, "x": X, "t": T})


# ---------------------------------------------------------------------------
# Pure-Python shape tests via a stubbed r_session.
# ---------------------------------------------------------------------------


class _FakeFloatVector(list):
    pass


class _StubR:
    """A tiny stand-in for ``rpy2.robjects`` that records globalenv writes
    and dispatches a fixed set of R(...) expressions to canned values."""

    def __init__(self) -> None:
        self.globalenv: dict[str, Any] = {}
        # Conventional / bias-corrected / robust triple
        self._coef = [2.05, 2.08, 2.10]
        self._se = [0.12, 0.13, 0.14]
        self._pv = [1e-6, 1e-5, 1e-5]
        self._ci_lo = [1.80, 1.81, 1.82]
        self._ci_hi = [2.30, 2.32, 2.38]
        self._density_p = 0.42

    class _R:
        def __init__(self, outer: "_StubR") -> None:
            self._outer = outer

        def __call__(self, expr: str) -> Any:
            outer = self._outer
            # rdrobust run — record the call but no return needed.
            if expr.startswith("rd_ <- rdrobust::rdrobust"):
                return None
            if expr.startswith("rdd_ <- rddensity::rddensity"):
                return None
            if expr == "sum(rd_$N_h)":
                return [1234]
            if expr == "as.numeric(rd_$coef)":
                return list(outer._coef)
            if expr == "as.numeric(rd_$se)":
                return list(outer._se)
            if expr == "as.numeric(rd_$pv)":
                return list(outer._pv)
            if expr == "as.numeric(rd_$ci[,1])":
                return list(outer._ci_lo)
            if expr == "as.numeric(rd_$ci[,2])":
                return list(outer._ci_hi)
            if expr == "as.numeric(rd_$bws[1, 1])":
                return [0.30]
            if expr == "as.numeric(rd_$bws[2, 1])":
                return [0.45]
            if expr == "rdd_$test$p_jk":
                return [outer._density_p]
            # R.version.string / R.home / packageVersion - all string returns
            return ["?"]

    @property
    def r(self) -> "_StubR._R":
        return _StubR._R(self)

    # rpy2 vector constructors used by the wrapper
    def FloatVector(self, arr: Any) -> _FakeFloatVector:  # noqa: N802
        return _FakeFloatVector(arr)


@pytest.fixture
def stub_r(monkeypatch: pytest.MonkeyPatch) -> _StubR:
    """Replace the R-bridge helpers used by ``rd.py`` with no-op shims +
    a tiny stub ``robjects`` module so the wrapper can be exercised end-
    to-end without rpy2."""
    stub = _StubR()

    monkeypatch.setattr(rd_mod, "require", lambda pkg: None)
    monkeypatch.setattr(rd_mod, "r_session", lambda: stub)

    # converter() is used as a contextmanager — return a no-op CM.
    class _NullCM:
        def __enter__(self) -> None:  # noqa: D401
            return None

        def __exit__(self, *a: Any) -> bool:
            return False

    monkeypatch.setattr(rd_mod, "converter", lambda: _NullCM())
    monkeypatch.setattr(
        rd_mod,
        "r_session_metadata",
        lambda: {"r_version": "stub", "packages": {"rdrobust": "stub"}},
    )
    return stub


def test_diagnostics_shape_with_stubbed_r(stub_r: _StubR) -> None:
    """Without any real R install, the wrapper still populates the full
    diagnostics dict the downstream report expects."""
    df = _sharp_rd_df(n=400, tau=2.0, seed=0)
    est = RDRobustEstimator(
        running_variable="x",
        cutoff=0.0,
        outcome="y",
        bandwidth_method="mserd",
    )
    est.min_sample_size = 100  # type: ignore[misc]
    est.fit(df)
    res = est.estimate()

    # Headline numbers come from the conventional + robust slots.
    assert res.point_estimate == pytest.approx(2.05)
    assert res.se == pytest.approx(0.12)
    assert res.ci_low == pytest.approx(1.82)
    assert res.ci_high == pytest.approx(2.38)
    assert res.n_used == 1234

    diag = res.diagnostics
    for key in (
        "bandwidth_h",
        "bandwidth_b",
        "bandwidth_method",
        "rd_design",
        "manipulation_test_pvalue",
        "conventional_point",
        "conventional_se",
        "conventional_ci",
        "bias_corrected_point",
        "bias_corrected_se",
        "robust_point",
        "robust_se",
        "robust_ci",
    ):
        assert key in diag, f"diagnostics missing {key!r}"

    assert diag["rd_design"] == "sharp"
    assert diag["bandwidth_method"] == "mserd"
    assert diag["bandwidth_h"] == pytest.approx(0.30)
    assert diag["bandwidth_b"] == pytest.approx(0.45)
    assert diag["conventional_ci"] == [pytest.approx(1.80), pytest.approx(2.30)]
    assert diag["robust_ci"] == [pytest.approx(1.82), pytest.approx(2.38)]
    assert diag["manipulation_test_pvalue"] == pytest.approx(0.42)


def test_fuzzy_rd_design_flag_in_diagnostics(stub_r: _StubR) -> None:
    """Passing ``fuzzy_treatment`` flips diagnostics['rd_design'] to fuzzy."""
    df = _fuzzy_rd_df(n=400, tau=1.5, seed=0)
    est = RDRobustEstimator(
        running_variable="x",
        cutoff=0.0,
        outcome="y",
        fuzzy_treatment="t",
    )
    est.min_sample_size = 100  # type: ignore[misc]
    est.fit(df)
    res = est.estimate()
    assert res.diagnostics["rd_design"] == "fuzzy"


def test_invalid_bandwidth_method_rejected() -> None:
    with pytest.raises(ValueError, match="bandwidth_method"):
        RDRobustEstimator(
            running_variable="x",
            cutoff=0.0,
            outcome="y",
            bandwidth_method="bogus",  # type: ignore[arg-type]
        )


def test_registered_in_catalog() -> None:
    from causalrag.core.registry import get_registry

    entry = get_registry().get("rbridge.rd.rdrobust")
    assert entry.backend == "r"
    assert "LATE_AT_CUTOFF" in entry.supported_estimands


# ---------------------------------------------------------------------------
# End-to-end tests against real R. Skipped on machines without rdrobust.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_r_rdrobust(), reason="rpy2 and/or R 'rdrobust' not available"
)
def test_sharp_rd_recovers_tau_within_1p5_se() -> None:
    pytest.importorskip("rpy2")
    tau_true = 2.0
    df = _sharp_rd_df(n=2000, tau=tau_true, seed=11)
    est = RDRobustEstimator(running_variable="x", cutoff=0.0, outcome="y")
    est.fit(df).estimate()
    res = est.estimate()
    assert res.se is not None and res.se > 0
    assert abs(res.point_estimate - tau_true) <= 1.5 * res.se, (
        f"point {res.point_estimate} not within 1.5 SE ({res.se}) of tau={tau_true}"
    )
    assert res.diagnostics["rd_design"] == "sharp"


@pytest.mark.skipif(
    not _have_r_rdrobust(), reason="rpy2 and/or R 'rdrobust' not available"
)
def test_fuzzy_rd_recovers_late_within_2_se() -> None:
    pytest.importorskip("rpy2")
    tau_true = 1.5
    df = _fuzzy_rd_df(n=3000, tau=tau_true, seed=13)
    est = RDRobustEstimator(
        running_variable="x", cutoff=0.0, outcome="y", fuzzy_treatment="t"
    )
    est.fit(df)
    res = est.estimate()
    assert res.se is not None and res.se > 0
    assert abs(res.point_estimate - tau_true) <= 2.0 * res.se, (
        f"fuzzy LATE {res.point_estimate} not within 2 SE ({res.se}) of tau={tau_true}"
    )
    assert res.diagnostics["rd_design"] == "fuzzy"


@pytest.mark.skipif(
    not (_have_r_rdrobust() and _have_r_rddensity()),
    reason="rpy2 and/or R 'rdrobust'/'rddensity' not available",
)
def test_manipulation_test_clean_dgp_has_high_pvalue() -> None:
    """No sorting around the cutoff -> McCrary-style p > 0.10."""
    pytest.importorskip("rpy2")
    df = _sharp_rd_df(n=2000, tau=1.0, seed=21)
    est = RDRobustEstimator(running_variable="x", cutoff=0.0, outcome="y")
    est.fit(df)
    res = est.estimate()
    p = res.diagnostics["manipulation_test_pvalue"]
    assert p is not None
    assert p > 0.1, f"clean DGP should have high manipulation-test p; got {p}"


@pytest.mark.skipif(
    not (_have_r_rdrobust() and _have_r_rddensity()),
    reason="rpy2 and/or R 'rdrobust'/'rddensity' not available",
)
def test_manipulation_test_detects_sorting() -> None:
    """Inject excess mass just above the cutoff -> p < 0.10."""
    pytest.importorskip("rpy2")
    rng = np.random.default_rng(31)
    n = 2000
    X = rng.uniform(-1.0, 1.0, size=n)
    # Move ~25% of just-below observations to just-above the cutoff.
    just_below = (X > -0.2) & (X < 0.0)
    idx = np.where(just_below)[0]
    move = rng.choice(idx, size=int(0.6 * len(idx)), replace=False)
    X[move] = rng.uniform(0.0, 0.2, size=len(move))
    eps = rng.normal(0.0, 1.0, size=n)
    Y = 0.5 + 1.5 * (X >= 0.0).astype(float) + 0.4 * X + eps
    df = pd.DataFrame({"y": Y, "x": X})

    est = RDRobustEstimator(running_variable="x", cutoff=0.0, outcome="y")
    est.fit(df)
    res = est.estimate()
    p = res.diagnostics["manipulation_test_pvalue"]
    assert p is not None
    assert p < 0.1, f"manipulated DGP should be flagged; got p={p}"
