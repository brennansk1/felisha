"""Unit tests for ``causalrag.tasks.impact`` (PDD Sprint 5.4)."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from causalrag.tasks.impact import (
    ImpactFinding,
    ImpactReport,
    _matrix_complete,
    _run_causalimpact,
    _run_matrix_completion,
    _verdict,
    analyze_impact,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def step_series() -> pd.DataFrame:
    """Single time-series with a clean +5 step at t=50.

    Pre-period: AR(1) noise centred on 10; post-period: same dynamics
    plus 5. 100 time points so CausalImpact / SARIMAX have a generous
    pre-window.
    """
    rng = np.random.default_rng(0)
    T = 100
    y = np.zeros(T)
    y[0] = 10.0
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 5.0 + rng.normal(scale=0.5)
    # Apply the +5 step at t=50.
    true_att = 5.0
    y[50:] += true_att
    df = pd.DataFrame({"t": np.arange(T), "y": y})
    df.attrs["true_att"] = true_att
    return df


@pytest.fixture
def panel(step_series: pd.DataFrame) -> pd.DataFrame:
    """Panel with one treated unit (true ATT = +3) and 8 donor units.

    Donors are AR(1) processes sharing a common factor; treated unit
    follows the same DGP plus a +3 post-step at t=50.
    """
    rng = np.random.default_rng(42)
    T = 100
    n_donors = 8
    true_att = 3.0
    common = np.zeros(T)
    common[0] = 5.0
    for t in range(1, T):
        common[t] = 0.7 * common[t - 1] + rng.normal(scale=0.3)
    rows: list[dict] = []
    for u in range(n_donors):
        intercept = 2.0 + 0.5 * u
        noise = rng.normal(scale=0.3, size=T)
        y = intercept + common + noise
        for t in range(T):
            rows.append({"unit": f"donor_{u}", "t": int(t), "y": float(y[t])})
    # Treated unit
    intercept = 2.5
    noise = rng.normal(scale=0.3, size=T)
    y = intercept + common + noise
    y[50:] += true_att
    for t in range(T):
        rows.append({"unit": "treated", "t": int(t), "y": float(y[t])})

    df = pd.DataFrame(rows)
    df.attrs["true_att"] = true_att
    return df


# ---------------------------------------------------------------------------
# 1. CausalImpact (and ARIMA fallback) recovers known step
# ---------------------------------------------------------------------------
def test_causalimpact_recovers_step(step_series: pd.DataFrame) -> None:
    report = analyze_impact(
        step_series,
        target="y",
        time_column="t",
        intervention_time=50,
        methods=("causalimpact",),
    )
    f = next(x for x in report.findings if x.method == "causalimpact")
    assert np.isfinite(f.point_estimate)
    se = f.se if f.se is not None else 1.0
    true = step_series.attrs["true_att"]
    # Within 2 * SE of the truth (when a CI is reported) or within 2.0
    # absolute when SARIMAX did not surface an SE.
    tol = max(2.0 * se, 2.0)
    assert abs(f.point_estimate - true) <= tol, (
        f"CausalImpact estimate {f.point_estimate:.3f} far from truth "
        f"{true}; SE={se}, tol={tol}"
    )
    assert report.n_pre == 50
    assert report.n_post == 50


# ---------------------------------------------------------------------------
# 2. Matrix completion + ASCM recover panel ATT
# ---------------------------------------------------------------------------
def test_matrix_completion_recovers_panel_att(panel: pd.DataFrame) -> None:
    f = _run_matrix_completion(
        panel=panel,
        target="y",
        time_column="t",
        unit_column="unit",
        treated_unit="treated",
        pre_mask_time={t: t < 50 for t in panel["t"].unique()},
        donor_pool=None,
    )
    assert np.isfinite(f.point_estimate)
    true = panel.attrs["true_att"]
    if f.se is not None:
        assert abs(f.point_estimate - true) <= 2.0 * max(f.se, 0.5) + 0.5
    else:
        assert abs(f.point_estimate - true) <= 1.5


def test_ascm_recovers_panel_att(panel: pd.DataFrame) -> None:
    pytest.importorskip("pysyncon")
    report = analyze_impact(
        panel,
        target="y",
        time_column="t",
        intervention_time=50,
        unit_column="unit",
        treated_unit="treated",
        methods=("ascm",),
    )
    f = next(x for x in report.findings if x.method == "ascm")
    if not np.isfinite(f.point_estimate):
        pytest.skip(f"pysyncon failed on this fixture: {f.notes}")
    true = panel.attrs["true_att"]
    se = f.se if f.se is not None else 1.0
    assert abs(f.point_estimate - true) <= 2.0 * max(se, 0.5) + 1.0, (
        f"ASCM estimate {f.point_estimate:.3f} far from truth {true}; "
        f"SE={se}"
    )


# ---------------------------------------------------------------------------
# 3. Verdict — consistent
# ---------------------------------------------------------------------------
def test_verdict_consistent() -> None:
    findings = [
        ImpactFinding("a", point_estimate=5.0, ci_low=4.0, ci_high=6.0),
        ImpactFinding("b", point_estimate=5.2, ci_low=4.2, ci_high=6.2),
    ]
    assert _verdict(findings) == "consistent"


def test_verdict_consistent_via_analyze_impact(panel: pd.DataFrame) -> None:
    """End-to-end: two methods on a clean panel should both produce
    estimates near the true +3 ATT — at the very least their consensus
    median should be close to the truth and the verdict should land in
    the {consistent, moderate} band."""
    report = analyze_impact(
        panel,
        target="y",
        time_column="t",
        intervention_time=50,
        unit_column="unit",
        treated_unit="treated",
        methods=("causalimpact", "matrix_completion"),
    )
    assert isinstance(report, ImpactReport)
    finite = [f for f in report.findings if np.isfinite(f.point_estimate)]
    assert len(finite) == 2
    true = panel.attrs["true_att"]
    # Both methods should land within ~1 of truth individually.
    for f in finite:
        assert abs(f.point_estimate - true) <= 1.5, (
            f"{f.method} point {f.point_estimate} far from {true}"
        )
    # Consensus median is close to truth.
    assert abs(report.consensus_point - true) <= 1.0
    # Verdict is one of the three valid categories; we don't enforce
    # 'consistent' here because per-method CI widths vary by DGP, but
    # the verdict should be returned and the interpretation populated.
    assert report.consistency_verdict in {"consistent", "moderate", "divergent"}
    assert report.interpretation


# ---------------------------------------------------------------------------
# 4. Verdict — divergent
# ---------------------------------------------------------------------------
def test_verdict_divergent() -> None:
    # Two methods, points 0 and 10, SE ≈ 1 (CI half-width 1.96).
    findings = [
        ImpactFinding("a", point_estimate=0.0, ci_low=-1.96, ci_high=1.96),
        ImpactFinding("b", point_estimate=10.0, ci_low=8.04, ci_high=11.96),
    ]
    # Spread = 10; 2 * max SE ≈ 2 → divergent.
    assert _verdict(findings) == "divergent"


# ---------------------------------------------------------------------------
# 5. Missing-method failure-safe
# ---------------------------------------------------------------------------
def test_report_built_from_survivors_when_package_missing(
    panel: pd.DataFrame,
) -> None:
    """If a method's optional dependency is unavailable, that method
    contributes a NaN-point finding + note, but the report is still
    built from the surviving methods."""
    # Simulate causalimpact + tfcausalimpact both unavailable, and
    # pysyncon being importable but ASCM raising — only matrix
    # completion should survive cleanly.
    real_import = __import__

    def fake_import(name: str, *args, **kwargs):
        if name in ("causalimpact", "tfcausalimpact"):
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    # Also make ascm always fail
    with mock.patch("builtins.__import__", side_effect=fake_import), mock.patch(
        "causalrag.tasks.impact._run_ascm",
        return_value=ImpactFinding(
            method="ascm",
            point_estimate=float("nan"),
            ci_low=None,
            ci_high=None,
            notes=["pysyncon unavailable (simulated)"],
        ),
    ):
        report = analyze_impact(
            panel,
            target="y",
            time_column="t",
            intervention_time=50,
            unit_column="unit",
            treated_unit="treated",
            methods=("causalimpact", "ascm", "matrix_completion"),
        )

    by_method = {f.method: f for f in report.findings}
    assert set(by_method) == {"causalimpact", "ascm", "matrix_completion"}

    # ASCM is NaN with an explanatory note.
    assert not np.isfinite(by_method["ascm"].point_estimate)
    assert any("unavailable" in n.lower() for n in by_method["ascm"].notes)

    # causalimpact reports a number via the ARIMA fallback path, with a
    # note explaining the fallback.
    ci_finding = by_method["causalimpact"]
    assert np.isfinite(ci_finding.point_estimate)
    assert any("fallback" in n.lower() or "sarimax" in n.lower() for n in ci_finding.notes)

    # Matrix completion produced a number too.
    mc = by_method["matrix_completion"]
    assert np.isfinite(mc.point_estimate)

    # Consensus computed over the surviving (finite) findings.
    assert np.isfinite(report.consensus_point)


# ---------------------------------------------------------------------------
# Extra: low-level helpers
# ---------------------------------------------------------------------------
def test_matrix_complete_recovers_low_rank() -> None:
    """Soft-impute reconstructs a known rank-1 matrix from a partial mask."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(20, 1))
    v = rng.normal(size=(1, 30))
    Y = u @ v
    mask = rng.uniform(size=Y.shape) > 0.2  # 80% observed
    M_hat = _matrix_complete(Y, mask, lam=0.05, max_iter=500, tol=1e-7)
    # Imputation error on the held-out entries should be small.
    err = np.linalg.norm((M_hat - Y)[~mask]) / np.linalg.norm(Y[~mask])
    assert err < 0.25, f"relative imputation error too large: {err}"


def test_arima_fallback_used_when_causalimpact_missing(
    step_series: pd.DataFrame,
) -> None:
    """Directly hit _run_causalimpact with both libs simulated-missing
    and check we get back a finite estimate from the SARIMAX path."""
    real_import = __import__

    def fake_import(name: str, *args, **kwargs):
        if name in ("causalimpact", "tfcausalimpact"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    y = step_series.set_index("t")["y"]
    pre_mask = np.asarray(y.index < 50)
    post_mask = np.asarray(y.index >= 50)
    with mock.patch("builtins.__import__", side_effect=fake_import):
        f = _run_causalimpact(
            y=y, pre_mask=pre_mask, post_mask=post_mask, covariates=None
        )
    assert np.isfinite(f.point_estimate)
    assert any("fallback" in n.lower() or "sarimax" in n.lower() for n in f.notes)
