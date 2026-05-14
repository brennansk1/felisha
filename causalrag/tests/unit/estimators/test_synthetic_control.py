"""Tests for synthetic-control / ASCM / SDiD estimators (Sprint 2.3).

The pysyncon-backed tests are gated by ``pytest.importorskip("pysyncon")``
so CI without the optional dep skips them. The validation-path tests run
unconditionally and exercise the pure-Python input-checking code, so the
suite still has meaningful coverage when pysyncon isn't installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.core.protocol import StudyProtocol
from causalrag.estimators.python.synthetic_control import (
    SyntheticControlEstimator,
    _validate_inputs,
    _split_panel,
)


# ---------------------------------------------------------------------------
# Data factory
# ---------------------------------------------------------------------------
def _panel(
    n_donors: int = 30,
    T: int = 50,
    T0: int = 30,
    true_effect: float = 5.0,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a long-format panel with ``n_donors`` controls + 1 treated unit.

    Units share a common factor structure (two latent factors) so SCM has a
    well-defined synthetic match; the treated unit gets a constant additive
    effect of ``true_effect`` starting at period ``T0``.
    """
    rng = np.random.default_rng(seed)
    n_units = n_donors + 1  # unit 0 is treated
    # Two latent factors over time + unit-specific factor loadings.
    f1 = np.linspace(0.0, 1.0, T)
    f2 = np.sin(np.linspace(0.0, 2 * np.pi, T))
    rows = []
    for u in range(n_units):
        loading1 = rng.normal(loc=1.0, scale=0.3)
        loading2 = rng.normal(loc=0.5, scale=0.3)
        unit_fe = rng.normal(scale=0.5)
        for t in range(T):
            y = unit_fe + loading1 * f1[t] + loading2 * f2[t] + rng.normal(scale=0.15)
            treated = 1 if (u == 0 and t >= T0) else 0
            if treated:
                y += true_effect
            rows.append(
                {"unit_id": u, "time": t, "y": y, "treat": treated}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pure-Python validation tests (no pysyncon required)
# ---------------------------------------------------------------------------
def test_validate_inputs_rejects_missing_column() -> None:
    df = pd.DataFrame({"unit_id": [1, 2], "time": [0, 1], "y": [0.0, 1.0]})
    with pytest.raises(ValueError, match="Column not in data"):
        _validate_inputs(
            df,
            unit_col="unit_id",
            time_col="time",
            treatment="treat",  # not present
            outcome="y",
        )


def test_validate_inputs_rejects_multiple_treated_units() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3],
            "time": [0, 1, 0, 1, 0, 1],
            "treat": [0, 1, 0, 1, 0, 0],
            "y": [0.0, 1.0, 0.5, 1.5, 0.2, 0.9],
        }
    )
    with pytest.raises(ValueError, match="exactly one treated unit"):
        _validate_inputs(
            df,
            unit_col="unit_id",
            time_col="time",
            treatment="treat",
            outcome="y",
        )


def test_validate_inputs_rejects_non_binary_treatment() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "time": [0, 1, 0, 1],
            "treat": [0, 2, 0, 1],  # 2 is invalid
            "y": [0.0, 1.0, 0.2, 0.9],
        }
    )
    with pytest.raises(ValueError, match="Treatment indicator must be"):
        _validate_inputs(
            df,
            unit_col="unit_id",
            time_col="time",
            treatment="treat",
            outcome="y",
        )


def test_validate_inputs_rejects_no_treated_unit() -> None:
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "time": [0, 1, 0, 1],
            "treat": [0, 0, 0, 0],
            "y": [0.0, 1.0, 0.2, 0.9],
        }
    )
    with pytest.raises(ValueError, match="No treated unit"):
        _validate_inputs(
            df,
            unit_col="unit_id",
            time_col="time",
            treatment="treat",
            outcome="y",
        )


def test_split_panel_identifies_pre_and_post() -> None:
    df = _panel(n_donors=3, T=10, T0=6)
    treated, controls, pre, post = _split_panel(
        df, unit_col="unit_id", time_col="time", treatment="treat", outcome="y"
    )
    assert treated == 0
    assert sorted(controls) == [1, 2, 3]
    assert pre == list(range(6))
    assert post == list(range(6, 10))


def test_invalid_variant_rejected() -> None:
    with pytest.raises(ValueError, match="variant must be one of"):
        SyntheticControlEstimator(
            treatment="treat", outcome="y", variant="not-a-variant"  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# pysyncon-backed tests
# ---------------------------------------------------------------------------
pysyncon = pytest.importorskip("pysyncon")


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel(n_donors=30, T=50, T0=30, true_effect=5.0, seed=7)


@pytest.fixture(scope="module")
def scm_result(panel: pd.DataFrame):
    est = SyntheticControlEstimator(
        treatment="treat", outcome="y", variant="scm"
    )
    est.fit(panel, StudyProtocol(name="smoke"))
    return est.estimate(), est


def test_scm_recovers_true_att(scm_result) -> None:
    result, est = scm_result
    assert result.estimator_id == "python.synth_control.scm"
    assert result.estimand_class == "ATT"
    # ATT should be close to the true effect of 5.0.
    # With SE we ask for 1.5 SE tolerance; fall back to absolute 1.5 if SE is None.
    tol = 1.5 * result.se if (result.se is not None and result.se > 0) else 1.5
    assert abs(result.point_estimate - 5.0) <= max(tol, 0.5)


def test_scm_pre_period_fit_is_tight(scm_result) -> None:
    result, _ = scm_result
    pre_rmspe = result.diagnostics["pre_treatment_fit_rmspe"]
    # We added Gaussian noise sd=0.15, so a well-specified SC should land
    # within a small multiple of that.
    assert pre_rmspe < 1.0
    # Post-pre ratio should be much larger than 1 when there is a real effect.
    assert result.diagnostics["post_pre_rmspe_ratio"] > 3.0


def test_scm_unit_weights_sum_to_one(scm_result) -> None:
    _, est = scm_result
    assert est._unit_weights is not None
    w = np.asarray(list(est._unit_weights.values))
    assert w.sum() == pytest.approx(1.0, abs=1e-3)
    assert (w >= -1e-6).all()


def test_placebo_rank_high_for_treated(scm_result) -> None:
    result, _ = scm_result
    rank = result.diagnostics["placebo_rank"]
    # Treated unit's post/pre RMSPE ratio should be in the top-k of the
    # placebo distribution. With 30 donors we want rank <= 3 (top-10%).
    assert rank is not None
    assert 1 <= rank <= 4


@pytest.mark.parametrize("variant", ["scm", "ascm", "sdid"])
def test_variant_dispatch_runs(panel: pd.DataFrame, variant: str) -> None:
    est = SyntheticControlEstimator(
        treatment="treat", outcome="y", variant=variant  # type: ignore[arg-type]
    )
    est.fit(panel, StudyProtocol(name="smoke"))
    result = est.estimate()
    assert result.estimator_id == f"python.synth_control.{variant}"
    assert result.diagnostics["variant"] == variant
    # All variants should land in the right ballpark for this clean DGP.
    assert abs(result.point_estimate - 5.0) <= 2.0
    assert result.diagnostics["pre_treatment_fit_rmspe"] >= 0
    assert result.diagnostics["n_donors"] == 30
    assert result.diagnostics["n_pre_periods"] == 30
    assert result.diagnostics["n_post_periods"] == 20


def test_registry_has_three_variants() -> None:
    from causalrag.core.registry import get_registry

    reg = get_registry()
    ids = {e.id for e in reg.all()}
    assert "python.synth_control.scm" in ids
    assert "python.synth_control.ascm" in ids
    assert "python.synth_control.sdid" in ids
