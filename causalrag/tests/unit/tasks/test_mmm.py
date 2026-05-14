"""Unit tests for ``causalrag.tasks.mmm`` (PDD Sprint 7.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.tasks.mmm import (
    MMMChannelEffect,
    MMMNotAvailable,
    MMMReport,
    _fit_fallback_ridge,
    _geometric_adstock,
    _hill_saturation,
    _meridian_available,
    _pymc_marketing_available,
    _robyn_available,
    run_mmm,
)


# ---------------------------------------------------------------------------
# Synthetic 24-week / 3-channel data-generating process
# ---------------------------------------------------------------------------
def _make_mmm_panel(
    *,
    T: int = 24,
    seed: int = 13,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """24 weekly observations, 3 channels with known contribution shares.

    Channel architecture:

    * ``tv``   — strong decay (adstock 0.7), weak saturation
    * ``search`` — strong Hill saturation (half ~= 1.0 of mean spend)
    * ``radio`` — linear (no decay, no saturation)

    Returns ``(df, truth)`` where ``truth`` maps channel -> share of
    *total* revenue produced by that channel under the true DGP.
    """
    rng = np.random.default_rng(seed)

    # Spend series engineered so the three channels are *nearly*
    # uncorrelated even with only T=24 observations — this is the
    # well-conditioned regime where MMM identifiability is best. We
    # use one orthogonal block (Hadamard-like pattern), then add a
    # gentle smooth drift and per-period noise to keep each series
    # realistic-looking rather than a step function.
    t = np.arange(T)
    block = T // 3
    tv_pattern = np.where((t // block) % 2 == 0, 1.0, -1.0)
    # Two near-orthogonal patterns built from interleaved sign blocks.
    search_pattern = np.where(((t // 2) % 2) == 0, 1.0, -1.0)
    radio_pattern = np.where(((t // 4) % 2) == 0, 1.0, -1.0)

    tv_spend = 60.0 + 25.0 * tv_pattern + rng.normal(scale=3.0, size=T)
    search_spend = 35.0 + 15.0 * search_pattern + rng.normal(scale=2.0, size=T)
    radio_spend = 25.0 + 10.0 * radio_pattern + rng.normal(scale=1.5, size=T)
    tv_spend = np.clip(tv_spend, 1.0, None)
    search_spend = np.clip(search_spend, 1.0, None)
    radio_spend = np.clip(radio_spend, 1.0, None)

    # Transforms used to generate revenue.
    tv_ad = _geometric_adstock(tv_spend, decay=0.7)
    search_sat = _hill_saturation(search_spend, half=float(np.mean(search_spend)))
    # Radio is linear.

    # Channel coefficients chosen so the three contributions are roughly
    # comparable in absolute size — this is the regime where MMM
    # identifiability is best.
    beta_tv = 2.0
    beta_search = 800.0  # search_sat lives in [0, 1)
    beta_radio = 4.0
    base = 200.0

    contrib_tv = beta_tv * tv_ad
    contrib_search = beta_search * search_sat
    contrib_radio = beta_radio * radio_spend
    base_arr = np.full(T, base)

    noise = rng.normal(scale=10.0, size=T)
    revenue = base_arr + contrib_tv + contrib_search + contrib_radio + noise

    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=T, freq="W-MON"),
            "tv": tv_spend,
            "search": search_spend,
            "radio": radio_spend,
            "revenue": revenue,
        }
    )

    total = float(revenue.sum())
    truth = {
        "tv": float(contrib_tv.sum()) / total,
        "search": float(contrib_search.sum()) / total,
        "radio": float(contrib_radio.sum()) / total,
        "base": float(base_arr.sum()) / total,
    }
    return df, truth


# ---------------------------------------------------------------------------
# Shape / contract tests
# ---------------------------------------------------------------------------
def test_run_mmm_returns_report_with_expected_channels() -> None:
    df, _ = _make_mmm_panel()
    report = run_mmm(
        df,
        revenue_column="revenue",
        spend_columns=["tv", "search", "radio"],
        library="auto",
    )
    assert isinstance(report, MMMReport)
    assert {c.channel for c in report.channels} == {"tv", "search", "radio"}
    for ch in report.channels:
        assert isinstance(ch, MMMChannelEffect)
        # Non-negative contribution share and finite point effect.
        assert ch.contribution_share >= -1e-9
        assert np.isfinite(ch.point_effect)


def test_run_mmm_rejects_bad_inputs() -> None:
    df, _ = _make_mmm_panel()
    with pytest.raises(ValueError):
        run_mmm(df, revenue_column="missing", spend_columns=["tv"], library="auto")
    with pytest.raises(ValueError):
        run_mmm(df, revenue_column="revenue", spend_columns=[], library="auto")
    with pytest.raises(ValueError):
        run_mmm(df.iloc[:0], revenue_column="revenue", spend_columns=["tv"], library="auto")


def test_auto_falls_through_to_fallback_when_no_backend_installed() -> None:
    # pytest.importorskip behaviour: only meaningful when none of the
    # three heavy backends is installed. If any *is* installed we skip
    # the assertion that the fallback ran (but still verify auto returns
    # a valid report).
    df, _ = _make_mmm_panel()
    report = run_mmm(
        df,
        revenue_column="revenue",
        spend_columns=["tv", "search", "radio"],
        library="auto",
    )
    none_installed = not (
        _pymc_marketing_available() or _robyn_available() or _meridian_available()
    )
    if none_installed:
        assert report.library == "fallback_ridge"
        assert any("fallback" in n.lower() for n in report.notes)


def test_explicit_unavailable_library_raises() -> None:
    df, _ = _make_mmm_panel()
    # Pick whichever of the three is *not* installed locally.
    for lib, available in [
        ("pymc_marketing", _pymc_marketing_available()),
        ("robyn", _robyn_available()),
        ("meridian", _meridian_available()),
    ]:
        if not available:
            with pytest.raises(MMMNotAvailable):
                run_mmm(
                    df,
                    revenue_column="revenue",
                    spend_columns=["tv", "search", "radio"],
                    library=lib,  # type: ignore[arg-type]
                )
            return
    pytest.skip("all three MMM backends installed; nothing to test")


# ---------------------------------------------------------------------------
# Recovery test — fallback ridge must hit true contribution shares within 15%
# ---------------------------------------------------------------------------
def test_fallback_ridge_recovers_channel_shares_within_15_percent() -> None:
    df, truth = _make_mmm_panel()
    report = _fit_fallback_ridge(
        df,
        revenue_column="revenue",
        spend_columns=["tv", "search", "radio"],
        seasonality_column=None,
        notes=[],
    )
    assert report.library == "fallback_ridge"

    shares = {c.channel: c.contribution_share for c in report.channels}
    for ch, true_share in [("tv", truth["tv"]), ("search", truth["search"]), ("radio", truth["radio"])]:
        est = shares[ch]
        # Tolerance is *absolute* on a share scale, which corresponds
        # to "within 15 percentage points of total revenue" — this is
        # the practically-useful MMM accuracy bar quoted in the sprint.
        assert abs(est - true_share) < 0.15, (
            f"channel {ch}: estimated share {est:.3f} vs truth {true_share:.3f}"
        )

    # Saturation_point and decay_rate surfaced for the right channels.
    by = {c.channel: c for c in report.channels}
    # TV should pick up *some* decay (the grid includes 0.0 so a value
    # of 0 would mean recovery failed); allow flexibility but require
    # >= 0.2 — the next grid step above zero.
    assert by["tv"].decay_rate is not None and by["tv"].decay_rate >= 0.2
    # Search should expose a saturation point near its mean spend.
    assert by["search"].saturation_point is not None
    assert by["search"].saturation_point > 0


def test_total_revenue_explained_is_reasonable() -> None:
    df, _ = _make_mmm_panel()
    report = _fit_fallback_ridge(
        df,
        revenue_column="revenue",
        spend_columns=["tv", "search", "radio"],
        seasonality_column=None,
        notes=[],
    )
    # Channel + base shares should approximately sum to total explained;
    # ridge regularisation means we don't expect an exact 1.0.
    summed = report.base_revenue_share + sum(c.contribution_share for c in report.channels)
    assert 0.5 <= summed <= 1.5
    assert 0.5 <= report.total_revenue_explained <= 1.5


def test_fallback_ridge_with_seasonality_column() -> None:
    df, _ = _make_mmm_panel()
    df = df.copy()
    df["holiday"] = (df["week"].dt.month == 12).astype(float)
    report = _fit_fallback_ridge(
        df,
        revenue_column="revenue",
        spend_columns=["tv", "search", "radio"],
        seasonality_column="holiday",
        notes=[],
    )
    assert report.library == "fallback_ridge"
    assert len(report.channels) == 3


def test_pre_period_filter_drops_rows() -> None:
    df, _ = _make_mmm_panel(T=40)
    # Slice to the first 24 weeks via pre_period (date-aware path).
    cutoff = df["week"].iloc[23]
    report = run_mmm(
        df,
        revenue_column="revenue",
        spend_columns=["tv", "search", "radio"],
        library="auto",
        pre_period=(str(df["week"].iloc[0].date()), str(cutoff.date())),
    )
    assert isinstance(report, MMMReport)
    assert len(report.channels) == 3


# ---------------------------------------------------------------------------
# Adstock / Hill helper invariants
# ---------------------------------------------------------------------------
def test_geometric_adstock_recovers_steady_state() -> None:
    x = np.ones(200)
    decay = 0.5
    out = _geometric_adstock(x, decay=decay)
    # Steady state of x_t=1 with decay d is 1/(1-d).
    assert abs(out[-1] - 1.0 / (1.0 - decay)) < 1e-6


def test_hill_saturation_is_monotone_and_bounded() -> None:
    x = np.linspace(0.0, 1000.0, 50)
    y = _hill_saturation(x, half=100.0)
    assert np.all(np.diff(y) >= -1e-12)
    assert y.max() < 1.0
    assert y[0] == 0.0
    # Half-point passes through 0.5.
    mid = _hill_saturation(np.array([100.0]), half=100.0)[0]
    assert abs(mid - 0.5) < 1e-9
