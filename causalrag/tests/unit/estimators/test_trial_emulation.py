"""Tests for the TrialEmulation R bridge (``rbridge.trial_emulation``).

Two layers, mirroring the ``test_did_modern.py`` pattern:

1. *Stub* tests that exercise the diagnostics-dict shape WITHOUT R, by
   monkey-patching ``r_session`` / ``require`` / ``converter`` /
   ``r_session_metadata``. These run anywhere.
2. *R-dependent* gating tests that ``importorskip('rpy2')`` and check
   for the ``TrialEmulation`` package before running.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from causalrag.estimators.rbridge import trial_emulation as te_mod
from causalrag.estimators.rbridge.trial_emulation import TrialEmulationEstimator


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
# Synthetic long-format target-trial panel
# ---------------------------------------------------------------------------


def _trial_panel(
    n_subjects: int = 600,
    n_periods: int = 6,
    seed: int = 0,
) -> pd.DataFrame:
    """Long-format panel matching the TrialEmulation input contract.

    Each subject has rows ``(subject_id, period)`` with:
      - ``eligible`` = 1 at period 0, 0 otherwise.
      - ``treatment_strategy_id`` in {0, 1}; assigned at baseline.
      - ``treatment_received_this_period`` in {0, 1}; may deviate.
      - ``outcome`` (rare event indicator).
      - ``censoring_indicator`` (administrative censor at horizon).
      - one time-varying confounder ``x_tv`` and one baseline ``x_base``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for sid in range(n_subjects):
        strat = int(rng.integers(0, 2))
        x_base = float(rng.normal(0.0, 1.0))
        deviated = False
        for t in range(n_periods):
            x_tv = float(rng.normal(0.0, 1.0))
            # 10% per-period chance of deviating from assigned strategy.
            if not deviated and rng.random() < 0.10:
                deviated = True
            trt = strat if not deviated else 1 - strat
            outcome = int(rng.random() < 0.03)
            censored = int(t == n_periods - 1)
            rows.append(
                {
                    "subject_id": sid,
                    "period": t,
                    "eligible": 1 if t == 0 else 0,
                    "treatment_strategy_id": strat,
                    "treatment_received_this_period": trt,
                    "outcome": outcome,
                    "censoring_indicator": censored,
                    "x_tv": x_tv,
                    "x_base": x_base,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tiny stubbed R
# ---------------------------------------------------------------------------


class _StubR:
    """A minimal stand-in for ``rpy2.robjects``. Records every ``r(expr)``
    invocation and dispatches a canned-response table keyed on substring.
    """

    def __init__(self, canned: dict[str, Any] | None = None) -> None:
        self.globalenv: dict[str, Any] = {}
        self.calls: list[str] = []
        self.canned = canned or {}

    class _Conversion:
        @staticmethod
        def py2rpy(df: pd.DataFrame) -> pd.DataFrame:
            return df

    @property
    def conversion(self) -> "_StubR._Conversion":
        return _StubR._Conversion()

    class _R:
        def __init__(self, outer: "_StubR") -> None:
            self._outer = outer

        def __call__(self, expr: str) -> Any:
            self._outer.calls.append(expr)
            # Exact match first.
            if expr in self._outer.canned:
                return self._outer.canned[expr]
            # Assignment statements (data_preparation, trial_msm) return None.
            for prefix in (
                "te_prep_pp_ <-",
                "te_pp_ <-",
                "te_prep_itt_ <-",
                "te_itt_ <-",
                "te_prep_at_ <-",
                "te_at_ <-",
            ):
                if expr.startswith(prefix):
                    return None
            return []

    @property
    def r(self) -> "_StubR._R":
        return _StubR._R(self)


@pytest.fixture
def stub_r(monkeypatch: pytest.MonkeyPatch) -> _StubR:
    """Replace the R-bridge module-level hooks used by ``trial_emulation``."""
    stub = _StubR()
    monkeypatch.setattr(te_mod, "require", lambda pkg: None)
    monkeypatch.setattr(te_mod, "r_session", lambda: stub)

    class _NullCM:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *a: Any) -> bool:
            return False

    monkeypatch.setattr(te_mod, "converter", lambda: _NullCM())
    monkeypatch.setattr(
        te_mod,
        "r_session_metadata",
        lambda: {"r_version": "stub", "packages": {"TrialEmulation": "stub"}},
    )
    return stub


def _make_estimator() -> TrialEmulationEstimator:
    return TrialEmulationEstimator(
        subject_id="subject_id",
        period="period",
        eligible="eligible",
        treatment_strategy_id="treatment_strategy_id",
        treatment_received_this_period="treatment_received_this_period",
        outcome="outcome",
        censoring_indicator="censoring_indicator",
        time_varying_confounders=["x_tv"],
        baseline_covariates=["x_base"],
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_unknown_estimand() -> None:
    with pytest.raises(ValueError, match="estimand"):
        TrialEmulationEstimator(
            subject_id="subject_id",
            period="period",
            eligible="eligible",
            treatment_strategy_id="treatment_strategy_id",
            treatment_received_this_period="treatment_received_this_period",
            outcome="outcome",
            censoring_indicator="censoring_indicator",
            estimand="median_survival",  # type: ignore[arg-type]
        )


def test_estimate_before_fit_raises() -> None:
    est = _make_estimator()
    with pytest.raises(RuntimeError, match="fit"):
        est.estimate()


def test_fit_rejects_missing_columns(stub_r: _StubR) -> None:
    est = _make_estimator()
    panel = _trial_panel(n_subjects=200, n_periods=4).drop(columns=["x_tv"])
    with pytest.raises(ValueError, match="missing"):
        est.fit(panel)


def test_fit_rejects_undersized_panel(stub_r: _StubR) -> None:
    est = _make_estimator()
    # 50 subjects * 4 periods = 200 rows, below min_sample_size = 500.
    panel = _trial_panel(n_subjects=50, n_periods=4)
    with pytest.raises(ValueError, match=">= 500"):
        est.fit(panel)


# ---------------------------------------------------------------------------
# Shape test — three-row diagnostics table
# ---------------------------------------------------------------------------


def test_triple_row_diagnostics_shape(stub_r: _StubR) -> None:
    # Canned MSM coefficient pulls for each of the three sister analyses.
    # The wrapper queries `<obj>$robust$summary$estimate[...]` first.
    trt_col = "treatment_received_this_period"

    def _est_expr(obj: str) -> str:
        return (
            f'as.numeric({obj}$robust$summary$estimate'
            f'[{obj}$robust$summary$term == "{trt_col}"])'
        )

    def _se_expr(obj: str) -> str:
        return (
            f'as.numeric({obj}$robust$summary$robust_se'
            f'[{obj}$robust$summary$term == "{trt_col}"])'
        )

    stub_r.canned = {
        # ITT: small effect, log HR ~ -0.10.
        _est_expr("te_itt_"): [-0.10],
        _se_expr("te_itt_"): [0.05],
        # PP: stronger protective effect under sustained adherence.
        _est_expr("te_pp_"): [-0.40],
        _se_expr("te_pp_"): [0.08],
        # As-treated: noisy, intermediate.
        _est_expr("te_at_"): [-0.25],
        _se_expr("te_at_"): [0.07],
    }

    panel = _trial_panel(n_subjects=200, n_periods=6, seed=0)
    est = _make_estimator()
    est.fit(panel)
    res = est.estimate()

    # Headline = PP log-hazard contrast.
    assert res.point_estimate == pytest.approx(-0.40)
    assert res.se == pytest.approx(0.08)
    assert res.ci_low == pytest.approx(-0.40 - 1.96 * 0.08)
    assert res.ci_high == pytest.approx(-0.40 + 1.96 * 0.08)
    assert res.p_value is not None and 0.0 < res.p_value < 1.0
    assert res.estimator_id == "rbridge.trial_emulation"

    d = res.diagnostics
    for key in (
        "analyses",
        "headline_analysis",
        "n_subjects",
        "n_subject_periods",
        "time_varying_confounders",
        "baseline_covariates",
        "weighting_scheme",
    ):
        assert key in d
    assert d["headline_analysis"] == "PP"
    assert d["time_varying_confounders"] == ["x_tv"]
    assert d["baseline_covariates"] == ["x_base"]

    analyses = d["analyses"]
    assert len(analyses) == 3
    labels = [a["analysis"] for a in analyses]
    assert labels == ["ITT", "PP", "as_treated"]

    # Triple-row table carries log-HR, HR, SE, p-value for each.
    by_label = {a["analysis"]: a for a in analyses}
    assert by_label["ITT"]["log_hazard_ratio"] == pytest.approx(-0.10)
    assert by_label["PP"]["log_hazard_ratio"] == pytest.approx(-0.40)
    assert by_label["as_treated"]["log_hazard_ratio"] == pytest.approx(-0.25)
    assert by_label["PP"]["hazard_ratio"] == pytest.approx(math.exp(-0.40))
    assert by_label["ITT"]["hazard_ratio"] == pytest.approx(math.exp(-0.10))
    assert by_label["PP"]["p_value"] is not None
    assert 0.0 < by_label["PP"]["p_value"] < 1.0


def test_diagnose_reflects_fit_state(stub_r: _StubR) -> None:
    est = _make_estimator()
    d0 = est.diagnose()
    assert d0["fitted"] is False
    assert d0["n_used"] == 0

    stub_r.canned = {
        # Provide one minimal estimate so the pulls succeed.
        'as.numeric(te_pp_$robust$summary$estimate[te_pp_$robust$summary$term == "treatment_received_this_period"])': [-0.3],
        'as.numeric(te_pp_$robust$summary$robust_se[te_pp_$robust$summary$term == "treatment_received_this_period"])': [0.1],
    }
    panel = _trial_panel(n_subjects=200, n_periods=6, seed=1)
    est.fit(panel)
    d1 = est.diagnose()
    assert d1["fitted"] is True
    assert d1["n_used"] > 0
    assert d1["n_subjects"] == 200
    assert d1["headline_analysis"] == "PP"


def test_estimate_handles_missing_se_gracefully(stub_r: _StubR) -> None:
    """If R returns no SE (e.g. singular MSM), the wrapper degrades cleanly."""
    # Only an ITT estimate; PP returns nothing -> headline is NaN, CI is NaN,
    # but the diagnostics row for PP still exists with None fields.
    stub_r.canned = {
        'as.numeric(te_itt_$robust$summary$estimate[te_itt_$robust$summary$term == "treatment_received_this_period"])': [-0.1],
        'as.numeric(te_itt_$robust$summary$robust_se[te_itt_$robust$summary$term == "treatment_received_this_period"])': [0.05],
    }
    panel = _trial_panel(n_subjects=200, n_periods=6, seed=2)
    est = _make_estimator()
    est.fit(panel)
    res = est.estimate()

    # PP unavailable -> headline NaN, p_value None.
    assert math.isnan(res.point_estimate)
    assert res.p_value is None

    by_label = {a["analysis"]: a for a in res.diagnostics["analyses"]}
    assert by_label["PP"]["log_hazard_ratio"] is None
    assert by_label["PP"]["se"] is None
    assert by_label["PP"]["hazard_ratio"] is None
    # ITT row still populated.
    assert by_label["ITT"]["log_hazard_ratio"] == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# Registry / flag-routing
# ---------------------------------------------------------------------------


def test_registered_with_panel_structure_flag() -> None:
    from causalrag.core.flags import DataFlag
    from causalrag.core.registry import get_registry

    reg = get_registry()
    entry = reg.get("rbridge.trial_emulation")
    assert entry.backend == "r"
    assert "ATE" in entry.supported_estimands
    assert "RMST_CONTRAST" in entry.supported_estimands
    assert "ATT" in entry.supported_estimands
    assert entry.min_sample_size == 500
    assert entry.propensity_required is True
    # PANEL_STRUCTURE is in current builds; assert if present.
    if hasattr(DataFlag, "PANEL_STRUCTURE"):
        assert DataFlag.PANEL_STRUCTURE in entry.required_flags


def test_fit_calls_data_preparation_and_trial_msm_three_times(stub_r: _StubR) -> None:
    stub_r.canned = {}
    panel = _trial_panel(n_subjects=200, n_periods=6, seed=3)
    est = _make_estimator()
    est.fit(panel)

    # Three estimand pipelines, each calling data_preparation + trial_msm.
    dp_calls = [c for c in stub_r.calls if c.startswith("te_prep_")]
    msm_calls = [
        c
        for c in stub_r.calls
        if c.startswith("te_pp_ <-")
        or c.startswith("te_itt_ <-")
        or c.startswith("te_at_ <-")
    ]
    assert len(dp_calls) == 3
    assert len(msm_calls) == 3

    # Each estimand_type appears in the data_preparation calls.
    dp_joined = " ".join(dp_calls)
    assert '"PP"' in dp_joined
    assert '"ITT"' in dp_joined
    assert '"As-Treated"' in dp_joined


# ---------------------------------------------------------------------------
# R-dependent gating
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _have_r_pkg("TrialEmulation"),
    reason="rpy2 and/or R 'TrialEmulation' not available",
)
def test_trial_emulation_runs_on_synthetic_panel() -> None:
    pytest.importorskip("rpy2")
    panel = _trial_panel(n_subjects=600, n_periods=6, seed=42)
    est = _make_estimator()
    est.fit(panel)
    res = est.estimate()
    assert res.n_used > 0
    assert "analyses" in res.diagnostics
    assert len(res.diagnostics["analyses"]) == 3
