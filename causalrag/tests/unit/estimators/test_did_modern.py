"""Tests for the modern DiD bridge (``rbridge.did_modern``).

Two layers:

1. *Stub* tests that exercise the diagnostics-dict shape WITHOUT R,
   by monkey-patching ``r_session`` / ``require`` / ``converter`` /
   ``r_session_metadata``. These run on any CI machine.
2. *R-dependent* tests that ``importorskip`` rpy2 and the relevant R
   package, run on a synthetic staggered panel (n=200 units, T=10
   periods, 3 treatment cohorts, true ATT = 2.0), and require the
   point estimate to land near the truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalrag.estimators.rbridge import did_modern as did_mod
from causalrag.estimators.rbridge.did_modern import (
    BJSImputationDiDEstimator,
    CallawaySantAnnaDiDEstimator,
    DIDMultipleGTEstimator,
    HonestDiDSensitivity,
)


# ---------------------------------------------------------------------------
# Gating helpers
# ---------------------------------------------------------------------------


def _have_r_pkg(pkg: str) -> bool:
    try:
        import rpy2  # noqa: F401
        import rpy2.robjects as ro
    except Exception:
        return False
    try:
        return bool(
            list(ro.r(f'requireNamespace("{pkg}", quietly = TRUE)'))[0]
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Synthetic staggered-adoption panel
# ---------------------------------------------------------------------------


def _staggered_panel(
    n_units: int = 200,
    n_periods: int = 10,
    tau_true: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Three treatment cohorts (g in {4, 6, 8}) + a never-treated cohort.

    Y_it = alpha_i + gamma_t + tau * 1{t >= g_i} + eps,
    g_i in {4, 6, 8, NA} with roughly equal share.
    """
    rng = np.random.default_rng(seed)
    # Quarter the units into 4 cohorts.
    cohorts = rng.choice([4, 6, 8, np.nan], size=n_units, p=[0.25, 0.25, 0.25, 0.25])
    alpha = rng.normal(0.0, 1.0, size=n_units)
    gamma = rng.normal(0.0, 0.5, size=n_periods)

    rows = []
    for i in range(n_units):
        g_i = cohorts[i]
        for t in range(1, n_periods + 1):
            treated = (not np.isnan(g_i)) and (t >= g_i)
            eps = rng.normal(0.0, 1.0)
            y = alpha[i] + gamma[t - 1] + tau_true * float(treated) + eps
            rows.append(
                {
                    "id": i,
                    "t": t,
                    "g": g_i if not np.isnan(g_i) else np.nan,
                    "y": y,
                    "d": float(treated),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tiny stubbed R for shape tests
# ---------------------------------------------------------------------------


class _StubR:
    """A minimal stand-in for ``rpy2.robjects`` used to drive the
    wrappers through ``fit`` + ``estimate`` without R.

    Records every ``r(expr)`` invocation and dispatches a fixed table
    of canned responses keyed on the expression substring.
    """

    def __init__(self, canned: dict[str, Any] | None = None) -> None:
        self.globalenv: dict[str, Any] = {}
        self.calls: list[str] = []
        # default canned responses — populated per-test
        self.canned = canned or {}

    class _Conversion:
        @staticmethod
        def py2rpy(df: pd.DataFrame) -> pd.DataFrame:
            return df  # identity is fine — we never read it back

    @property
    def conversion(self) -> "_StubR._Conversion":
        return _StubR._Conversion()

    class _R:
        def __init__(self, outer: "_StubR") -> None:
            self._outer = outer

        def __call__(self, expr: str) -> Any:
            self._outer.calls.append(expr)
            # Exact-match canned responses first.
            if expr in self._outer.canned:
                return self._outer.canned[expr]
            # Substring-match fallbacks for fit() assignment lines.
            for prefix in (
                "att_ <- did::att_gt",
                "agg_dyn_ <- did::aggte",
                "agg_simple_ <- did::aggte",
                "bjs_ <- didimputation",
                "dch_ <- DIDmultiplegt",
                "hd_ <- HonestDiD",
            ):
                if expr.startswith(prefix):
                    return None
            return ["?"]

    @property
    def r(self) -> "_StubR._R":
        return _StubR._R(self)

    # Vector constructors used by the wrappers.
    def FloatVector(self, arr: Any) -> list:  # noqa: N802
        return list(arr)


@pytest.fixture
def stub_r(monkeypatch: pytest.MonkeyPatch):
    """Replace the R-bridge module-level hooks used by ``did_modern``."""
    stub = _StubR()
    monkeypatch.setattr(did_mod, "require", lambda pkg: None)
    monkeypatch.setattr(did_mod, "r_session", lambda: stub)

    class _NullCM:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *a: Any) -> bool:
            return False

    monkeypatch.setattr(did_mod, "converter", lambda: _NullCM())
    monkeypatch.setattr(
        did_mod,
        "r_session_metadata",
        lambda: {"r_version": "stub", "packages": {"did": "stub"}},
    )
    return stub


# ---------------------------------------------------------------------------
# 1. Callaway-Sant'Anna — shape test
# ---------------------------------------------------------------------------


def test_callaway_santanna_diagnostics_shape(stub_r: _StubR) -> None:
    stub_r.canned = {
        "as.numeric(agg_simple_$overall.att)": [2.04],
        "as.numeric(agg_simple_$overall.se)": [0.10],
        "as.numeric(att_$group)": [4.0, 4.0, 6.0, 6.0, 8.0],
        "as.numeric(att_$t)": [4.0, 5.0, 6.0, 7.0, 8.0],
        "as.numeric(att_$att)": [2.10, 2.05, 1.98, 2.02, 2.00],
        "as.numeric(att_$se)": [0.20, 0.21, 0.18, 0.19, 0.22],
        "as.numeric(agg_dyn_$egt)": [-2.0, -1.0, 0.0, 1.0, 2.0],
        "as.numeric(agg_dyn_$att.egt)": [0.05, -0.02, 2.01, 2.07, 2.04],
        "as.numeric(agg_dyn_$se.egt)": [0.10, 0.10, 0.20, 0.21, 0.22],
        "as.numeric(att_$Wpval)": [0.81],
    }
    panel = _staggered_panel(n_units=120, n_periods=8, seed=0)
    est = CallawaySantAnnaDiDEstimator(
        subject_id="id",
        time="t",
        treatment_onset_time="g",
        outcome="y",
    )
    est.fit(panel)
    res = est.estimate()
    assert res.point_estimate == pytest.approx(2.04)
    assert res.se == pytest.approx(0.10)
    d = res.diagnostics
    for key in (
        "did_design",
        "group_time_atts",
        "event_study_dynamic_effects",
        "parallel_trends_pretest_pvalue",
        "pretest_caveat",
    ):
        assert key in d
    assert d["did_design"] == "staggered_never_treated"
    assert d["parallel_trends_pretest_pvalue"] == pytest.approx(0.81)
    assert "HonestDiD" in d["pretest_caveat"]
    assert len(d["group_time_atts"]) == 5
    assert d["group_time_atts"][0]["group"] == 4.0
    assert len(d["event_study_dynamic_effects"]) == 5


def test_callaway_santanna_rejects_bad_control_group() -> None:
    with pytest.raises(ValueError, match="control_group"):
        CallawaySantAnnaDiDEstimator(
            subject_id="id",
            time="t",
            treatment_onset_time="g",
            outcome="y",
            control_group="bogus",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 2. BJS imputation — shape test
# ---------------------------------------------------------------------------


def test_bjs_imputation_diagnostics_shape(stub_r: _StubR) -> None:
    stub_r.canned = {
        "as.character(bjs_$term)": ["-2", "-1", "0", "1", "2"],
        "as.numeric(bjs_$estimate)": [0.03, -0.01, 2.05, 2.10, 1.96],
        "as.numeric(bjs_$std.error)": [0.10, 0.10, 0.20, 0.21, 0.22],
    }
    panel = _staggered_panel(n_units=120, n_periods=8, seed=1)
    est = BJSImputationDiDEstimator(
        subject_id="id", time="t", treatment_onset_time="g", outcome="y"
    )
    est.fit(panel)
    res = est.estimate()
    # Headline = mean of post horizons (>=0).
    expected = float(np.mean([2.05, 2.10, 1.96]))
    assert res.point_estimate == pytest.approx(expected)
    d = res.diagnostics
    assert d["did_design"] == "staggered_never_treated"
    assert "event_study_dynamic_effects" in d
    assert d["parallel_trends_pretest_pvalue"] is not None
    assert "HonestDiD" in d["pretest_caveat"]


def test_bjs_imputation_rejects_panel_with_no_never_treated(stub_r: _StubR) -> None:
    panel = _staggered_panel(n_units=100, n_periods=6, seed=2)
    # Fill all NAs to drop the never-treated.
    panel["g"] = panel["g"].fillna(20)
    est = BJSImputationDiDEstimator(
        subject_id="id", time="t", treatment_onset_time="g", outcome="y"
    )
    with pytest.raises(ValueError, match="never-treated"):
        est.fit(panel)


# ---------------------------------------------------------------------------
# 3. DIDmultiplegt — shape test, with negative-weight share
# ---------------------------------------------------------------------------


def test_dch_multiplegt_surfaces_negative_weight_share(stub_r: _StubR) -> None:
    stub_r.canned = {
        "as.numeric(dch_$results$ATE$Estimate)": [2.03],
        "as.numeric(dch_$results$ATE$SE)": [0.15],
        "as.numeric(dch_$weights$neg_share)": [0.27],
        "as.numeric(dch_$results$placebo_pval)": [0.65],
        "as.numeric(dch_$results$Effects$Estimate)": [2.01, 2.05, 2.10],
        "as.numeric(dch_$results$Effects$SE)": [0.20, 0.21, 0.22],
    }
    panel = _staggered_panel(n_units=120, n_periods=8, seed=3)
    est = DIDMultipleGTEstimator(
        subject_id="id", time="t", treatment="d", outcome="y"
    )
    est.fit(panel)
    res = est.estimate()
    assert res.point_estimate == pytest.approx(2.03)
    d = res.diagnostics
    assert d["did_design"] == "continuous_staggered"
    assert d["negative_weight_share"] == pytest.approx(0.27)
    assert d["parallel_trends_pretest_pvalue"] == pytest.approx(0.65)
    assert len(d["event_study_dynamic_effects"]) == 3


# ---------------------------------------------------------------------------
# 4. HonestDiD — sensitivity shape test
# ---------------------------------------------------------------------------


def test_honest_did_sensitivity_shape_and_breakdown(stub_r: _StubR) -> None:
    # Pretend a fitted CSA exists.
    stub_r.canned = {
        "as.numeric(agg_simple_$overall.att)": [2.00],
        "as.numeric(agg_simple_$overall.se)": [0.10],
        "as.numeric(att_$group)": [4.0],
        "as.numeric(att_$t)": [4.0],
        "as.numeric(att_$att)": [2.00],
        "as.numeric(att_$se)": [0.10],
        "as.numeric(agg_dyn_$egt)": [-1.0, 0.0],
        "as.numeric(agg_dyn_$att.egt)": [0.0, 2.0],
        "as.numeric(agg_dyn_$se.egt)": [0.1, 0.2],
        "as.numeric(att_$Wpval)": [0.9],
        # Sensitivity grid: tighter CIs at low M-bar, widens to cover 0 at M=1.5.
        "as.numeric(hd_$Mbar)": [0.0, 0.5, 1.0, 1.5, 2.0],
        "as.numeric(hd_$lb)": [1.60, 1.10, 0.60, -0.10, -0.50],
        "as.numeric(hd_$ub)": [2.40, 2.90, 3.40, 3.90, 4.50],
    }
    panel = _staggered_panel(n_units=120, n_periods=6, seed=4)
    csa = CallawaySantAnnaDiDEstimator(
        subject_id="id", time="t", treatment_onset_time="g", outcome="y"
    )
    csa.fit(panel)
    hd = HonestDiDSensitivity(csa, m_bar_grid=(0.0, 0.5, 1.0, 1.5, 2.0))
    hd.fit()
    res = hd.estimate()

    d = res.diagnostics
    assert "honest_did_sensitivity_grid" in d
    assert "honest_did_breakdown_M_bar" in d
    # Breakdown is the smallest M-bar whose CI covers 0 -> 1.5.
    assert d["honest_did_breakdown_M_bar"] == pytest.approx(1.5)
    # Tightest (M=0) robust CI is what we report on EstimationResult.
    assert res.ci_low == pytest.approx(1.60)
    assert res.ci_high == pytest.approx(2.40)
    assert len(d["honest_did_sensitivity_grid"]) == 5


def test_honest_did_requires_fitted_csa(stub_r: _StubR) -> None:
    csa = CallawaySantAnnaDiDEstimator(
        subject_id="id", time="t", treatment_onset_time="g", outcome="y"
    )
    hd = HonestDiDSensitivity(csa)
    with pytest.raises(RuntimeError, match="fitted"):
        hd.fit()


# ---------------------------------------------------------------------------
# Registry / flag-routing sanity
# ---------------------------------------------------------------------------


def test_all_four_registered_with_correct_flags() -> None:
    from causalrag.core.flags import DataFlag
    from causalrag.core.registry import get_registry

    reg = get_registry()
    for est_id in (
        "rbridge.did.callaway_santanna",
        "rbridge.did.bjs_imputation",
        "rbridge.did.dch_multiplegt",
        "rbridge.did.honest_did",
    ):
        entry = reg.get(est_id)
        assert entry.backend == "r"
        assert "ATT" in entry.supported_estimands
        if hasattr(DataFlag, "STAGGERED_ADOPTION"):
            assert DataFlag.STAGGERED_ADOPTION in entry.required_flags
        if hasattr(DataFlag, "SINGLE_TREATED_UNIT"):
            assert DataFlag.SINGLE_TREATED_UNIT in entry.excluded_flags


def test_excluded_flag_excludes_from_candidates() -> None:
    from causalrag.core.flags import DataFlag
    from causalrag.core.registry import get_registry

    if not (hasattr(DataFlag, "STAGGERED_ADOPTION") and hasattr(DataFlag, "SINGLE_TREATED_UNIT")):
        pytest.skip("required flags not in this build")

    reg = get_registry()
    # When the situation flags say STAGGERED_ADOPTION + SINGLE_TREATED_UNIT,
    # the DiD estimators should NOT appear (single-treated-unit excluded).
    cands = reg.candidates_for(
        "ATT",
        required={DataFlag.STAGGERED_ADOPTION, DataFlag.SINGLE_TREATED_UNIT},
    )
    ids = {c.id for c in cands}
    assert "rbridge.did.callaway_santanna" not in ids
    assert "rbridge.did.bjs_imputation" not in ids
    assert "rbridge.did.dch_multiplegt" not in ids


# ---------------------------------------------------------------------------
# End-to-end R tests on synthetic staggered panel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_r_pkg("did"), reason="rpy2 and/or R 'did' not available"
)
def test_csa_recovers_tau_on_staggered_panel() -> None:
    pytest.importorskip("rpy2")
    tau_true = 2.0
    panel = _staggered_panel(n_units=200, n_periods=10, tau_true=tau_true, seed=11)
    est = CallawaySantAnnaDiDEstimator(
        subject_id="id", time="t", treatment_onset_time="g", outcome="y"
    )
    est.fit(panel)
    res = est.estimate()
    assert res.se is not None and res.se > 0
    assert abs(res.point_estimate - tau_true) <= 3.0 * res.se, (
        f"CSA point {res.point_estimate} not within 3 SE ({res.se}) of tau={tau_true}"
    )
    assert res.diagnostics["did_design"] == "staggered_never_treated"
    assert len(res.diagnostics["group_time_atts"]) > 0


@pytest.mark.skipif(
    not _have_r_pkg("didimputation"),
    reason="rpy2 and/or R 'didimputation' not available",
)
def test_bjs_recovers_tau_on_staggered_panel() -> None:
    pytest.importorskip("rpy2")
    tau_true = 2.0
    panel = _staggered_panel(n_units=200, n_periods=10, tau_true=tau_true, seed=13)
    est = BJSImputationDiDEstimator(
        subject_id="id", time="t", treatment_onset_time="g", outcome="y"
    )
    est.fit(panel)
    res = est.estimate()
    assert res.se is not None and res.se > 0
    assert abs(res.point_estimate - tau_true) <= 3.0 * res.se


@pytest.mark.skipif(
    not _have_r_pkg("DIDmultiplegt"),
    reason="rpy2 and/or R 'DIDmultiplegt' not available",
)
def test_dch_runs_on_staggered_panel() -> None:
    pytest.importorskip("rpy2")
    panel = _staggered_panel(n_units=200, n_periods=10, tau_true=2.0, seed=17)
    est = DIDMultipleGTEstimator(
        subject_id="id", time="t", treatment="d", outcome="y", effects=3, placebo=1
    )
    est.fit(panel)
    res = est.estimate()
    assert res.n_used > 0
    assert "negative_weight_share" in res.diagnostics


@pytest.mark.skipif(
    not (_have_r_pkg("did") and _have_r_pkg("HonestDiD")),
    reason="rpy2 and/or R 'did'/'HonestDiD' not available",
)
def test_honest_did_runs_on_staggered_panel() -> None:
    pytest.importorskip("rpy2")
    panel = _staggered_panel(n_units=200, n_periods=10, tau_true=2.0, seed=19)
    csa = CallawaySantAnnaDiDEstimator(
        subject_id="id", time="t", treatment_onset_time="g", outcome="y"
    )
    csa.fit(panel)
    hd = HonestDiDSensitivity(csa, m_bar_grid=(0.0, 0.5, 1.0))
    hd.fit()
    res = hd.estimate()
    assert "honest_did_sensitivity_grid" in res.diagnostics
    assert len(res.diagnostics["honest_did_sensitivity_grid"]) > 0
