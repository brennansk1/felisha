"""GeoLift-style geo-experiment incrementality (PDD Sprint 7.4).

Answers the marketing-measurement question

    "When we turned on a campaign in geos {A, B, C, ...} on date T,
     what incremental lift did it produce in metric Y, versus a
     counterfactual built from the geos we did *not* touch?"

The methodology mirrors Meta's open-source GeoLift R package
(Arias-Castro & Saghafian 2022 / GeoLift docs) but without taking that
heavy R dependency:

1. **Treated-side aggregation** — the outcome ``Y`` is summed across the
   list of treated geos to form a single aggregated "treated" series.
   This is the GeoLift convention and lets us reuse a single-treated-
   unit synthetic-control backbone.
2. **Synthetic-control counterfactual** — we re-use
   :class:`causalrag.estimators.python.synthetic_control.SyntheticControlEstimator`
   (variant ``"scm"``) on the untreated geos as the donor pool. The
   counterfactual is the SC-predicted aggregated series in the post
   period; lift is (observed − counterfactual) summed/averaged in the
   post window.
3. **Inference via in-space placebo** — every donor geo is, in turn,
   treated as a placebo "treated" unit and the same SCM fit produces a
   gap distribution. The treated unit's rank inside that distribution
   gives a Fisher-exact-style p-value (Abadie-Diamond-Hainmueller 2010,
   §III.D), and the 5-95 % quantiles of placebo post-period gaps give a
   placebo-CI on the absolute lift. The estimator's existing
   ``_run_placebo`` does this work for us.
4. **Intent-to-treat vs per-protocol** — when a ``compliance_column``
   is supplied (1 = geo actually activated, 0 = scheduled but missed),
   we recompute lift restricting the treated-side to the compliant geos
   only ("per-protocol"). The full treated-list result is the intent-
   to-treat estimate.

If ``pysyncon`` is not installed we fall back to an unconstrained OLS
weighting of donor geos on the pre-period -- the estimator surface stays
the same, only the CI/p-value channels are filled with ``None`` /
``nan`` and a note explains the degradation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_TREATED_AGG_LABEL = "_geolift_treated"
_TREATMENT_COL = "_geolift_treatment"


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class GeoLiftReport:
    """Result of a geo-experiment incrementality analysis.

    Attributes
    ----------
    treated_geos:
        Geos receiving the intervention.
    donor_geos:
        Untreated geos used to build the counterfactual.
    intervention_time:
        First time step of the post-intervention window.
    pre_period, post_period:
        Inclusive ``(start, end)`` windows as ISO-ish strings of the
        underlying time index (whatever its dtype).
    point_lift:
        Absolute post-period lift averaged per time step (so it has the
        same units as ``Y`` itself, *not* a window-sum).
    percent_lift:
        ``point_lift / mean(counterfactual_post)``, expressed as a
        fraction (0.10 = 10% lift). ``nan`` if the counterfactual mean
        is zero.
    ci_low, ci_high:
        Placebo-derived 90% CI on ``point_lift``; ``None`` when fewer
        than two donor placebos succeeded.
    p_value:
        Two-sided in-space placebo p-value (rank of the treated unit's
        post/pre RMSPE ratio within the donor distribution).
    intent_to_treat:
        Lift computed with the *full* treated-geo list (always
        populated; equals ``point_lift`` when ``compliance_column`` is
        not supplied).
    per_protocol:
        Lift recomputed with treated-side restricted to compliant geos.
        ``None`` when no ``compliance_column`` was given.
    placebo_rank:
        Rank of the treated unit's RMSPE ratio in the donor
        distribution (1 = most extreme = strongest evidence).
    rmspe_ratio:
        Post-period / pre-period RMSPE for the treated unit. Values
        >> 1 indicate the SC fit deteriorates post-intervention, which
        is what we expect under a real lift.
    notes:
        Free-form diagnostic messages, e.g. fallback paths used.
    """

    treated_geos: list[str]
    donor_geos: list[str]
    intervention_time: pd.Timestamp
    pre_period: tuple[str, str]
    post_period: tuple[str, str]
    point_lift: float
    percent_lift: float
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    intent_to_treat: float
    per_protocol: float | None
    placebo_rank: int
    rmspe_ratio: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _coerce_timestamp(value: Any) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (int, np.integer)):
        return pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(value))
    return pd.to_datetime(value)


def _is_numeric_time(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _to_comparable_time(value: Any, *, numeric: bool) -> Any:
    if numeric:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return float(value)  # type: ignore[arg-type]
    return pd.to_datetime(value)


def _aggregate_treated(
    df: pd.DataFrame,
    *,
    geo_column: str,
    time_column: str,
    outcome: str,
    treated_geos: list[str],
    scale_for_scm: bool = False,
) -> pd.DataFrame:
    """Sum the treated-geo outcomes per time step into one aggregated row.

    Returns a long-format frame containing the donor geos verbatim plus
    one new geo named :data:`_TREATED_AGG_LABEL`.

    When ``scale_for_scm`` is ``True`` we divide the summed treated
    series by the number of treated geos. SCM unit weights live on the
    simplex, so the synthetic counterfactual is bounded by the donor
    convex hull; if many treated geos are summed, their baseline can
    exceed every donor's level and SCM becomes infeasible. Working in
    the *average* per treated geo keeps the treated level comparable to
    donors. Callers must multiply the resulting lift back by the same
    factor.
    """
    treated_mask = df[geo_column].isin(treated_geos)
    donors = df.loc[~treated_mask, [geo_column, time_column, outcome]].copy()
    treated = (
        df.loc[treated_mask, [time_column, outcome]]
        .groupby(time_column, as_index=False)[outcome]
        .sum()
    )
    if scale_for_scm and len(treated_geos) > 1:
        treated[outcome] = treated[outcome] / len(treated_geos)
    treated[geo_column] = _TREATED_AGG_LABEL
    treated = treated[[geo_column, time_column, outcome]]
    return pd.concat([donors, treated], ignore_index=True)


def _build_panel_with_treatment(
    agg: pd.DataFrame,
    *,
    geo_column: str,
    time_column: str,
    int_time: Any,
    pre_start: Any,
    pre_end: Any,
    post_start: Any,
    post_end: Any,
    pre_period: bool,
    post_period: bool,
) -> pd.DataFrame:
    """Add the treatment indicator column expected by ``SyntheticControlEstimator``."""
    panel = agg.copy()
    is_treated_unit = panel[geo_column] == _TREATED_AGG_LABEL
    t = panel[time_column]
    if post_period:
        in_post = (t >= post_start) & (t <= post_end)
    else:
        in_post = t >= int_time
    panel[_TREATMENT_COL] = (is_treated_unit & in_post).astype(int)
    # Drop any rows that fall outside the union of pre+post (e.g. when the
    # user passes a narrower pre_period than the full available window).
    if pre_period:
        in_pre = (t >= pre_start) & (t <= pre_end)
    else:
        in_pre = t < int_time
    keep = in_pre | in_post
    return panel.loc[keep].copy()


def _ols_fallback_counterfactual(
    panel: pd.DataFrame,
    *,
    geo_column: str,
    time_column: str,
    outcome: str,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Donor-OLS fallback when ``pysyncon`` is not available.

    Fits y_treated ~ donors on the pre-period and predicts the
    counterfactual on the full window. Returns ``(y_obs, y_hat,
    placebo_post_gaps)`` aligned to the time axis.

    ``placebo_post_gaps`` maps donor name -> post-period gap series
    (each donor in turn is held out as the "placebo treated").
    """
    wide = panel.pivot_table(
        index=time_column, columns=geo_column, values=outcome, aggfunc="mean"
    )
    wide = wide.sort_index()
    y_tr = wide[_TREATED_AGG_LABEL].to_numpy(dtype=float)
    donor_cols = [c for c in wide.columns if c != _TREATED_AGG_LABEL]
    X = wide[donor_cols].to_numpy(dtype=float)

    # Solve y_tr_pre = X_pre @ beta in least-squares.
    X_pre = X[pre_mask]
    y_pre = y_tr[pre_mask]
    X_pre_aug = np.column_stack([np.ones(X_pre.shape[0]), X_pre])
    beta, *_ = np.linalg.lstsq(X_pre_aug, y_pre, rcond=None)
    X_full_aug = np.column_stack([np.ones(X.shape[0]), X])
    y_hat = X_full_aug @ beta

    # Donor-as-placebo: drop each donor, refit, predict.
    placebos: dict[str, np.ndarray] = {}
    for j, name in enumerate(donor_cols):
        y_placebo_full = X[:, j]
        others = [k for k in range(len(donor_cols)) if k != j]
        X_oth = X[:, others]
        X_oth_pre = X_oth[pre_mask]
        X_oth_pre_aug = np.column_stack([np.ones(X_oth_pre.shape[0]), X_oth_pre])
        beta_p, *_ = np.linalg.lstsq(X_oth_pre_aug, y_placebo_full[pre_mask], rcond=None)
        X_oth_full_aug = np.column_stack([np.ones(X_oth.shape[0]), X_oth])
        y_hat_p = X_oth_full_aug @ beta_p
        placebos[name] = (y_placebo_full - y_hat_p)[post_mask]
    return y_tr, y_hat, placebos


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_geolift(
    df: pd.DataFrame,
    *,
    geo_column: str,
    time_column: str,
    outcome: str,
    treated_geos: list[str],
    intervention_time: Any,
    pre_period: tuple[Any, Any] | None = None,
    post_period: tuple[Any, Any] | None = None,
    compliance_column: str | None = None,
) -> GeoLiftReport:
    """Estimate incremental lift from a geo-experiment.

    Parameters
    ----------
    df:
        Long-format panel with one row per ``(geo, time)`` carrying the
        outcome metric. Donor geos = every geo not in ``treated_geos``.
    geo_column, time_column, outcome:
        Required column names.
    treated_geos:
        Geos receiving the intervention. Their outcomes are summed each
        period to form the aggregated treated series.
    intervention_time:
        First time step of the post-intervention window. ``pre`` is
        ``t < intervention_time``; ``post`` is ``t >= intervention_time``
        unless overridden by explicit windows.
    pre_period, post_period:
        Optional inclusive ``(start, end)`` overrides.
    compliance_column:
        Optional binary column (1=compliant, 0=non-compliant) at the
        geo-period level. When supplied, we additionally compute a
        per-protocol lift restricting the treated-side to geos with at
        least one compliant post-period observation.

    Returns
    -------
    GeoLiftReport
        Always returned, even on a system without ``pysyncon`` (CIs and
        p-values are filled with ``None`` / ``nan`` in that case and a
        note explains the fallback).
    """
    # --- input validation ----------------------------------------------
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    for col in (geo_column, time_column, outcome):
        if col not in df.columns:
            raise ValueError(f"Required column {col!r} not in df.")
    if not treated_geos:
        raise ValueError("treated_geos must contain at least one geo.")
    geos_in_df = set(df[geo_column].unique())
    missing = [g for g in treated_geos if g not in geos_in_df]
    if missing:
        raise ValueError(f"treated_geos not present in df: {missing}")
    donors_full = sorted(geos_in_df - set(treated_geos))
    if not donors_full:
        raise ValueError("No donor geos available (all geos are treated).")
    if compliance_column is not None and compliance_column not in df.columns:
        raise ValueError(f"compliance_column {compliance_column!r} not in df.")

    notes: list[str] = []

    # --- time handling --------------------------------------------------
    df = df.copy()
    time_series = df[time_column]
    numeric_time = _is_numeric_time(time_series)
    if not numeric_time:
        df[time_column] = pd.to_datetime(df[time_column])
        time_series = df[time_column]
    int_time = _to_comparable_time(intervention_time, numeric=numeric_time)

    pre_explicit = pre_period is not None
    post_explicit = post_period is not None
    pre_start = (
        _to_comparable_time(pre_period[0], numeric=numeric_time)
        if pre_explicit
        else time_series.min()
    )
    pre_end = (
        _to_comparable_time(pre_period[1], numeric=numeric_time)
        if pre_explicit
        else int_time
    )
    post_start = (
        _to_comparable_time(post_period[0], numeric=numeric_time)
        if post_explicit
        else int_time
    )
    post_end = (
        _to_comparable_time(post_period[1], numeric=numeric_time)
        if post_explicit
        else time_series.max()
    )

    # --- aggregate treated side & build panel with treatment indicator -
    # We aggregate as the *mean* per treated geo (not the sum) before
    # feeding SCM, then scale the resulting lift back up by n_treated so
    # the report's ``point_lift`` reflects the absolute aggregated lift
    # the user expects from "sum across treated geos". This keeps the
    # treated baseline inside the donor convex hull (see
    # ``_aggregate_treated`` docstring).
    n_treated = len(treated_geos)
    agg = _aggregate_treated(
        df,
        geo_column=geo_column,
        time_column=time_column,
        outcome=outcome,
        treated_geos=list(treated_geos),
        scale_for_scm=True,
    )
    panel = _build_panel_with_treatment(
        agg,
        geo_column=geo_column,
        time_column=time_column,
        int_time=int_time,
        pre_start=pre_start,
        pre_end=pre_end,
        post_start=post_start,
        post_end=post_end,
        pre_period=pre_explicit,
        post_period=post_explicit,
    )

    # Pre/post time index for both backends.
    times_sorted = sorted(panel[time_column].unique())
    if pre_explicit:
        pre_times = [t for t in times_sorted if pre_start <= t <= pre_end]
    else:
        pre_times = [t for t in times_sorted if t < int_time]
    if post_explicit:
        post_times = [t for t in times_sorted if post_start <= t <= post_end]
    else:
        post_times = [t for t in times_sorted if t >= int_time]
    if not pre_times:
        raise ValueError("No pre-intervention rows found.")
    if not post_times:
        raise ValueError("No post-intervention rows found.")

    # --- run SCM via the Sprint 2.3 estimator ---------------------------
    point_lift_avg, ci_low, ci_high, p_value, placebo_rank, rmspe_ratio, backend_notes = (
        _fit_scm_and_summarize(
            panel,
            geo_column=geo_column,
            time_column=time_column,
            outcome=outcome,
            pre_times=pre_times,
            post_times=post_times,
        )
    )
    notes.extend(backend_notes)

    # Rescale lift back to the SUM aggregation the user asked for.
    point_lift = point_lift_avg * n_treated
    if ci_low is not None:
        ci_low = ci_low * n_treated
    if ci_high is not None:
        ci_high = ci_high * n_treated

    # Percent lift relative to the counterfactual mean. Use the unscaled
    # SUMmed observed post mean (re-derive from the raw frame) so the
    # ratio is interpretable as "% over what would have happened".
    treated_obs_post_sum = float(
        df[
            df[geo_column].isin(treated_geos) & df[time_column].isin(post_times)
        ]
        .groupby(time_column)[outcome]
        .sum()
        .mean()
    )
    cf_mean = treated_obs_post_sum - point_lift
    percent_lift = float(point_lift / cf_mean) if abs(cf_mean) > 1e-12 else float("nan")

    intent_to_treat = float(point_lift)

    # --- per-protocol re-fit (optional) ---------------------------------
    per_protocol: float | None = None
    if compliance_column is not None:
        compliant_geos = _compliant_treated_geos(
            df,
            geo_column=geo_column,
            time_column=time_column,
            compliance_column=compliance_column,
            treated_geos=list(treated_geos),
            post_times=post_times,
        )
        if not compliant_geos:
            notes.append(
                "compliance_column supplied but no treated geo was compliant; "
                "per_protocol omitted"
            )
        elif set(compliant_geos) == set(treated_geos):
            per_protocol = intent_to_treat
            notes.append("all treated geos compliant; per_protocol == intent_to_treat")
        else:
            agg_pp = _aggregate_treated(
                df,
                geo_column=geo_column,
                time_column=time_column,
                outcome=outcome,
                treated_geos=compliant_geos,
                scale_for_scm=True,
            )
            panel_pp = _build_panel_with_treatment(
                agg_pp,
                geo_column=geo_column,
                time_column=time_column,
                int_time=int_time,
                pre_start=pre_start,
                pre_end=pre_end,
                post_start=post_start,
                post_end=post_end,
                pre_period=pre_explicit,
                post_period=post_explicit,
            )
            pp_lift_avg, *_rest = _fit_scm_and_summarize(
                panel_pp,
                geo_column=geo_column,
                time_column=time_column,
                outcome=outcome,
                pre_times=pre_times,
                post_times=post_times,
            )
            per_protocol = float(pp_lift_avg * len(compliant_geos))

    return GeoLiftReport(
        treated_geos=list(treated_geos),
        donor_geos=donors_full,
        intervention_time=_coerce_timestamp(int_time),
        pre_period=(str(pre_times[0]), str(pre_times[-1])),
        post_period=(str(post_times[0]), str(post_times[-1])),
        point_lift=float(point_lift),
        percent_lift=percent_lift,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        intent_to_treat=intent_to_treat,
        per_protocol=per_protocol,
        placebo_rank=int(placebo_rank),
        rmspe_ratio=float(rmspe_ratio),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------
def _fit_scm_and_summarize(
    panel: pd.DataFrame,
    *,
    geo_column: str,
    time_column: str,
    outcome: str,
    pre_times: list[Any],
    post_times: list[Any],
) -> tuple[float, float | None, float | None, float | None, int, float, list[str]]:
    """Return ``(point_lift, ci_low, ci_high, p_value, placebo_rank,
    rmspe_ratio, notes)``.

    Tries the ``pysyncon``-backed estimator first; falls back to an
    OLS-on-donors counterfactual when the optional dep is missing.
    """
    notes: list[str] = []
    try:
        from causalrag.estimators.python.synthetic_control import (
            SyntheticControlEstimator,
            _pysyncon_available,
        )
    except Exception as exc:  # pragma: no cover - wiring failure
        notes.append(f"could not import SyntheticControlEstimator: {exc}")
        return _ols_summarize(
            panel,
            geo_column=geo_column,
            time_column=time_column,
            outcome=outcome,
            pre_times=pre_times,
            post_times=post_times,
            notes=notes,
        )

    if not _pysyncon_available():
        notes.append("pysyncon unavailable; using OLS-on-donors fallback")
        return _ols_summarize(
            panel,
            geo_column=geo_column,
            time_column=time_column,
            outcome=outcome,
            pre_times=pre_times,
            post_times=post_times,
            notes=notes,
        )

    try:
        est = SyntheticControlEstimator(
            treatment=_TREATMENT_COL,
            outcome=outcome,
            variant="scm",
            unit_col=geo_column,
            time_col=time_column,
        )
        est.fit(panel, protocol=None)  # type: ignore[arg-type]
        result = est.estimate()
    except Exception as exc:
        notes.append(
            f"pysyncon SCM raised {type(exc).__name__}: {exc}; falling back to OLS"
        )
        return _ols_summarize(
            panel,
            geo_column=geo_column,
            time_column=time_column,
            outcome=outcome,
            pre_times=pre_times,
            post_times=post_times,
            notes=notes,
        )

    point_lift = float(result.point_estimate)
    ci_low = float(result.ci_low) if result.ci_low is not None else None
    ci_high = float(result.ci_high) if result.ci_high is not None else None
    p_value = float(result.p_value) if result.p_value is not None else None
    diag = result.diagnostics or {}
    placebo_rank = int(diag.get("placebo_rank") or -1)
    rmspe_ratio = float(diag.get("post_pre_rmspe_ratio") or float("nan"))
    notes.append("pysyncon SCM backend")
    return point_lift, ci_low, ci_high, p_value, placebo_rank, rmspe_ratio, notes


def _ols_summarize(
    panel: pd.DataFrame,
    *,
    geo_column: str,
    time_column: str,
    outcome: str,
    pre_times: list[Any],
    post_times: list[Any],
    notes: list[str],
) -> tuple[float, float | None, float | None, float | None, int, float, list[str]]:
    """OLS-on-donors counterfactual used when ``pysyncon`` is unavailable."""
    wide = panel.pivot_table(
        index=time_column, columns=geo_column, values=outcome, aggfunc="mean"
    ).sort_index()
    times = list(wide.index)
    pre_mask = np.array([t in set(pre_times) for t in times])
    post_mask = np.array([t in set(post_times) for t in times])
    y_obs, y_hat, placebos = _ols_fallback_counterfactual(
        panel,
        geo_column=geo_column,
        time_column=time_column,
        outcome=outcome,
        pre_mask=pre_mask,
        post_mask=post_mask,
    )
    gap_post = y_obs[post_mask] - y_hat[post_mask]
    gap_pre = y_obs[pre_mask] - y_hat[pre_mask]
    point_lift = float(np.mean(gap_post))

    # Placebo distribution: mean post-period gap of each held-out donor.
    placebo_means = np.array([float(g.mean()) for g in placebos.values()])
    if placebo_means.size >= 2:
        ci_low = float(np.quantile(placebo_means, 0.05))
        ci_high = float(np.quantile(placebo_means, 0.95))
        # Two-sided rank p-value on |gap|.
        abs_treated = abs(point_lift)
        abs_placebo = np.abs(placebo_means)
        # +1 numerator counts the treated unit itself (Fisher-exact style)
        p_value = float((np.sum(abs_placebo >= abs_treated) + 1) / (placebo_means.size + 1))
        # Rank: 1 = most extreme.
        all_abs = np.concatenate([[abs_treated], abs_placebo])
        order = np.argsort(-all_abs)
        placebo_rank = int(np.where(order == 0)[0][0]) + 1
    else:
        ci_low = ci_high = None
        p_value = None
        placebo_rank = -1

    pre_rmspe = float(np.sqrt(np.mean(gap_pre ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap_post ** 2)))
    rmspe_ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("nan")
    return point_lift, ci_low, ci_high, p_value, placebo_rank, rmspe_ratio, notes


def _compliant_treated_geos(
    df: pd.DataFrame,
    *,
    geo_column: str,
    time_column: str,
    compliance_column: str,
    treated_geos: list[str],
    post_times: list[Any],
) -> list[str]:
    """Return treated geos that have at least one compliant (1) post row."""
    post_set = set(post_times)
    out: list[str] = []
    for g in treated_geos:
        sub = df[(df[geo_column] == g) & (df[time_column].isin(post_set))]
        if sub.empty:
            continue
        if (sub[compliance_column].astype(float) > 0).any():
            out.append(g)
    return out
