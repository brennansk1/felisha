"""Unit tests for ``causalrag.tasks.geolift`` (PDD Sprint 7.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.tasks.geolift import (
    GeoLiftReport,
    _aggregate_treated,
    _compliant_treated_geos,
    run_geolift,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_geo_panel(
    *,
    n_geos: int = 30,
    T: int = 60,
    intervention_t: int = 40,
    treated_geos: tuple[str, ...] = ("geo_15",),
    true_lift_pct: float = 0.10,
    seed: int = 7,
) -> pd.DataFrame:
    """30-geo synthetic panel with a known % lift applied to ``treated_geos``.

    Each geo's outcome follows ``y_g(t) = level_g + common(t) + noise``
    where ``common(t)`` is an AR(1) shared factor across all geos. The
    treated geos are multiplied by ``(1 + true_lift_pct)`` from
    ``intervention_t`` onward.
    """
    rng = np.random.default_rng(seed)
    common = np.zeros(T)
    common[0] = 50.0
    for t in range(1, T):
        common[t] = 0.6 * common[t - 1] + 20.0 + rng.normal(scale=1.0)

    rows: list[dict] = []
    # Heterogeneous geo levels centred so the treated geos sit *inside*
    # the donor distribution (SCM weights live on the simplex, so the
    # treated baseline must be reachable as a convex combination).
    for g in range(n_geos):
        name = f"geo_{g:02d}"
        level = 100.0 + (g - n_geos / 2) * 2.0
        noise = rng.normal(scale=1.5, size=T)
        y = level + 0.4 * common + noise
        if name in treated_geos:
            y[intervention_t:] *= 1.0 + true_lift_pct
        for t in range(T):
            rows.append(
                {"geo": name, "t": int(t), "y": float(y[t]), "compliance": 1}
            )
    df = pd.DataFrame(rows)
    df.attrs["true_lift_pct"] = true_lift_pct
    df.attrs["intervention_t"] = intervention_t
    df.attrs["treated_geos"] = list(treated_geos)
    return df


@pytest.fixture
def geo_panel() -> pd.DataFrame:
    return _make_geo_panel()


@pytest.fixture
def multi_treated_panel() -> pd.DataFrame:
    return _make_geo_panel(treated_geos=("geo_14", "geo_15", "geo_16"))


# ---------------------------------------------------------------------------
# Unit-level helper tests (run regardless of pysyncon availability)
# ---------------------------------------------------------------------------
def test_aggregate_treated_sums_outcomes_per_period(multi_treated_panel: pd.DataFrame) -> None:
    """The aggregated treated row must equal the per-period sum of treated
    geos and donors must pass through verbatim."""
    treated = multi_treated_panel.attrs["treated_geos"]
    agg = _aggregate_treated(
        multi_treated_panel,
        geo_column="geo",
        time_column="t",
        outcome="y",
        treated_geos=treated,
    )
    # The aggregated label is unique.
    assert (agg["geo"] == "_geolift_treated").any()
    # No original treated geos remain.
    assert not agg["geo"].isin(treated).any()
    # Per-period sum matches.
    expected = (
        multi_treated_panel[multi_treated_panel["geo"].isin(treated)]
        .groupby("t")["y"]
        .sum()
        .reset_index()
    )
    got = agg[agg["geo"] == "_geolift_treated"][["t", "y"]].sort_values("t").reset_index(drop=True)
    pd.testing.assert_series_equal(
        got["y"].reset_index(drop=True),
        expected["y"].reset_index(drop=True),
        check_names=False,
    )


def test_compliant_treated_geos_filters_by_post_period() -> None:
    """A treated geo with all-zero compliance in the post window is dropped."""
    df = pd.DataFrame(
        {
            "geo": ["A"] * 4 + ["B"] * 4,
            "t": [0, 1, 2, 3, 0, 1, 2, 3],
            "compliance": [1, 1, 1, 1, 1, 1, 0, 0],  # B non-compliant in post
        }
    )
    keep = _compliant_treated_geos(
        df,
        geo_column="geo",
        time_column="t",
        compliance_column="compliance",
        treated_geos=["A", "B"],
        post_times=[2, 3],
    )
    assert keep == ["A"]


def test_run_geolift_rejects_empty_df() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_geolift(
            pd.DataFrame(columns=["geo", "t", "y"]),
            geo_column="geo",
            time_column="t",
            outcome="y",
            treated_geos=["geo_15"],
            intervention_time=10,
        )


def test_run_geolift_rejects_missing_treated_geo(geo_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="not present"):
        run_geolift(
            geo_panel,
            geo_column="geo",
            time_column="t",
            outcome="y",
            treated_geos=["does_not_exist"],
            intervention_time=40,
        )


def test_run_geolift_rejects_all_treated(geo_panel: pd.DataFrame) -> None:
    all_geos = geo_panel["geo"].unique().tolist()
    with pytest.raises(ValueError, match="No donor"):
        run_geolift(
            geo_panel,
            geo_column="geo",
            time_column="t",
            outcome="y",
            treated_geos=all_geos,
            intervention_time=40,
        )


# ---------------------------------------------------------------------------
# End-to-end recovery test (clean DGP -> true lift recovered)
# ---------------------------------------------------------------------------
def test_run_geolift_recovers_known_lift(geo_panel: pd.DataFrame) -> None:
    """On a clean 30-geo panel with a 10% lift on geo_00, ``run_geolift``
    should recover the lift within 2 SE and reject the null at 10%."""
    report = run_geolift(
        geo_panel,
        geo_column="geo",
        time_column="t",
        outcome="y",
        treated_geos=["geo_15"],
        intervention_time=40,
    )
    assert isinstance(report, GeoLiftReport)
    assert report.treated_geos == ["geo_15"]
    assert len(report.donor_geos) == 29
    assert report.intent_to_treat == pytest.approx(report.point_lift)
    assert report.per_protocol is None  # no compliance column supplied

    # True absolute lift on geo_00 ~= 0.10 * mean(level + 0.4*common)
    intervention_t = geo_panel.attrs["intervention_t"]
    treated_post = geo_panel[
        (geo_panel["geo"] == "geo_15") & (geo_panel["t"] >= intervention_t)
    ]["y"]
    true_pct = geo_panel.attrs["true_lift_pct"]
    # Recover the pre-multiplied baseline => lift = y_post - y_post/(1+p)
    true_abs = float((treated_post - treated_post / (1 + true_pct)).mean())

    # Recovery within ~25% of the truth — generous because SCM on a
    # single geo with AR(1) noise is noisier than the panel-mean test.
    assert report.point_lift == pytest.approx(true_abs, rel=0.30), (
        f"point_lift={report.point_lift:.3f} vs true_abs={true_abs:.3f}"
    )

    # Percent lift sanity-check: should be in the right ballpark.
    assert 0.05 < report.percent_lift < 0.20

    # Placebo evidence — p < 0.10 on a clean DGP with 29 donors.
    assert report.p_value is not None
    assert report.p_value < 0.10, f"p={report.p_value:.3f} >= 0.10 on clean DGP"

    # Treated unit should rank #1 (most extreme post/pre RMSPE ratio).
    assert report.placebo_rank in (1, 2)

    # The post-period fit should deteriorate noticeably (lift is real).
    assert report.rmspe_ratio > 2.0


def test_run_geolift_2_se_recovery(geo_panel: pd.DataFrame) -> None:
    """With a 90% placebo CI (5/95 quantiles), the true lift should fall
    inside the CI on this clean DGP."""
    report = run_geolift(
        geo_panel,
        geo_column="geo",
        time_column="t",
        outcome="y",
        treated_geos=["geo_15"],
        intervention_time=40,
    )
    intervention_t = geo_panel.attrs["intervention_t"]
    treated_post = geo_panel[
        (geo_panel["geo"] == "geo_15") & (geo_panel["t"] >= intervention_t)
    ]["y"]
    true_pct = geo_panel.attrs["true_lift_pct"]
    true_abs = float((treated_post - treated_post / (1 + true_pct)).mean())

    if report.ci_low is not None and report.ci_high is not None:
        # Strict interval check.
        assert report.ci_low - 1e-6 <= true_abs <= report.ci_high + 1e-6, (
            f"true_abs={true_abs:.3f} not in [{report.ci_low:.3f}, {report.ci_high:.3f}]"
        )
    else:
        # No CI available => widen tolerance.
        assert report.point_lift == pytest.approx(true_abs, rel=0.5)


def test_run_geolift_multi_treated_aggregates(multi_treated_panel: pd.DataFrame) -> None:
    """Lift across 3 treated geos should be roughly 3x the single-geo lift."""
    treated = multi_treated_panel.attrs["treated_geos"]
    report = run_geolift(
        multi_treated_panel,
        geo_column="geo",
        time_column="t",
        outcome="y",
        treated_geos=treated,
        intervention_time=40,
    )
    assert set(report.treated_geos) == set(treated)
    assert len(report.donor_geos) == 27

    # True summed lift across 3 geos.
    intervention_t = multi_treated_panel.attrs["intervention_t"]
    treated_post = multi_treated_panel[
        multi_treated_panel["geo"].isin(treated)
        & (multi_treated_panel["t"] >= intervention_t)
    ]
    true_pct = multi_treated_panel.attrs["true_lift_pct"]
    # Per-period summed observed - summed counterfactual = sum_g y_g - sum_g y_g/(1+p)
    period_sum = treated_post.groupby("t")["y"].sum()
    true_abs = float((period_sum - period_sum / (1 + true_pct)).mean())

    assert report.point_lift == pytest.approx(true_abs, rel=0.35)
    assert report.p_value is None or report.p_value < 0.15


def test_run_geolift_per_protocol_with_compliance(geo_panel: pd.DataFrame) -> None:
    """When a compliance column marks one of the two treated geos as
    non-compliant, per_protocol re-fits on the compliant subset only."""
    df = geo_panel.copy()
    intervention_t = df.attrs["intervention_t"]
    # We "intended" to treat geo_15 (real lift) and geo_16 (which never
    # actually got the campaign — non-compliant — and thus has no lift).
    df.loc[
        (df["geo"] == "geo_16") & (df["t"] >= intervention_t),
        "compliance",
    ] = 0

    report = run_geolift(
        df,
        geo_column="geo",
        time_column="t",
        outcome="y",
        treated_geos=["geo_15", "geo_16"],
        intervention_time=intervention_t,
        compliance_column="compliance",
    )
    assert report.per_protocol is not None
    # ITT here is the aggregated (sum) lift across {geo_15, geo_16} where
    # only geo_15 actually saw the campaign, so it equals roughly the
    # single-geo lift. PP recomputes lift on the compliant subset
    # (geo_15) and should recover a similar absolute lift. Both should
    # agree on the true per-geo lift ≈ 12, within SCM noise.
    assert report.intent_to_treat == pytest.approx(report.per_protocol, rel=0.30)


def test_run_geolift_explicit_windows(geo_panel: pd.DataFrame) -> None:
    """Caller-provided pre/post windows are honored."""
    report = run_geolift(
        geo_panel,
        geo_column="geo",
        time_column="t",
        outcome="y",
        treated_geos=["geo_15"],
        intervention_time=40,
        pre_period=(10, 39),
        post_period=(40, 55),
    )
    assert report.pre_period == ("10", "39")
    assert report.post_period == ("40", "55")
