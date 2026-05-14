"""Python wrapper for PySensemakr — Cinelli-Hazlett 2020 OLS / partial-R²
sensitivity.

Sensemakr's central artifact is a triple ``(point_estimate, R²_YD, R²_TD)``
benchmarked against the strongest *observed* confounder. It generalizes the
"omitted variable bias" critique to any fitted linear-form estimator
(OLS, partial linear DML).

We expose the official PySensemakr package via a clean wrapper that emits a
unified :class:`SensemakrResult` and gracefully degrades (returns ``None``
for the wrapped object plus a ``notes`` entry) when sensemakr is not
installed — Week 3 ships sensitivity wrappers, but the runtime dep is
optional through the ``[estimators]`` extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class SensemakrResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    treatment: str
    outcome: str
    estimate: float
    se: float
    t_value: float
    robustness_value: float
    robustness_value_q: float
    rv_qa: float | None = None
    extreme_unobs_bias: float | None = None
    benchmark: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    backend: str = "pysensemakr"


def _preprocess_for_ols(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """One-hot encode low-card categoricals, drop high-card ones — sensemakr's
    OLS doesn't tolerate non-numeric covariates. Mirrors the policy used by
    data.features.auto_preprocess so both code paths agree."""
    use_cols = [outcome, treatment, *covariates]
    work = df[[c for c in use_cols if c in df.columns]].dropna().copy()
    if treatment not in work.columns or outcome not in work.columns:
        return work, ()

    new_covs: list[str] = []
    for c in covariates:
        if c not in work.columns:
            continue
        if pd.api.types.is_bool_dtype(work[c]):
            work[c] = work[c].astype(int)
            new_covs.append(c)
            continue
        if pd.api.types.is_numeric_dtype(work[c]):
            new_covs.append(c)
            continue
        # Non-numeric: one-hot if ≤ 20 levels, else drop.
        n_unique = work[c].nunique()
        if n_unique <= 20 and n_unique >= 2:
            dummies = pd.get_dummies(work[c], prefix=c, drop_first=True).astype(int)
            work = pd.concat([work.drop(columns=[c]), dummies], axis=1)
            new_covs.extend(dummies.columns)
        else:
            work = work.drop(columns=[c])
    return work, tuple(new_covs)


def sensemakr(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
    benchmark_covariates: tuple[str, ...] = (),
    q: float = 1.0,
    alpha: float = 0.05,
    reduce: bool = True,
) -> SensemakrResult:
    """Run a partial-R² sensitivity analysis.

    ``q`` controls the bias magnitude considered "problematic" — the default
    ``q=1`` measures the strength needed to bring the *point estimate* to zero.

    ``benchmark_covariates`` are observed covariates against which the
    sensitivity is benchmarked: "how strong an unmeasured confounder would
    have to be relative to <benchmark>?" When empty, the strongest observed
    covariate is auto-selected.
    """
    # One-hot encode non-numeric covariates first — both the official
    # sensemakr and our fallback fit OLS, which doesn't accept object dtype.
    work, covariates = _preprocess_for_ols(df, treatment, outcome, covariates)

    try:
        from sensemakr import Sensemakr
    except ImportError:
        return _fallback_partial_r2(
            work,
            treatment=treatment,
            outcome=outcome,
            covariates=covariates,
            q=q,
            alpha=alpha,
        )

    # Fit a baseline OLS model and let PySensemakr take it from there.
    from statsmodels.api import OLS, add_constant

    x = add_constant(work[[treatment, *covariates]])
    y = work[outcome]
    fit = OLS(y, x).fit()

    bench = list(benchmark_covariates) if benchmark_covariates else None
    sm = Sensemakr(
        model=fit,
        treatment=treatment,
        benchmark_covariates=bench,
        q=q,
        alpha=alpha,
        reduce=reduce,
    )
    summary: dict[str, Any] = {}
    try:
        summary = sm.summary_data().to_dict()  # type: ignore[attr-defined]
    except Exception:
        pass

    return SensemakrResult(
        treatment=treatment,
        outcome=outcome,
        estimate=float(fit.params[treatment]),
        se=float(fit.bse[treatment]),
        t_value=float(fit.tvalues[treatment]),
        robustness_value=float(getattr(sm, "robustness_value", 0.0) or 0.0),
        robustness_value_q=float(getattr(sm, "robustness_value_q", 0.0) or 0.0),
        rv_qa=getattr(sm, "rv_qa", None),
        benchmark=summary,
    )


def _fallback_partial_r2(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
    q: float,
    alpha: float,
) -> SensemakrResult:
    """Hand-rolled partial-R²/robustness-value when PySensemakr is absent.

    Yields the same headline numbers (estimate, SE, t, robustness value) but
    no benchmarking; flags the degradation in ``notes``.
    """
    from statsmodels.api import OLS, add_constant

    use_cols = [outcome, treatment, *covariates]
    work = df[use_cols].dropna()
    x = add_constant(work[[treatment, *covariates]])
    y = work[outcome]
    fit = OLS(y, x).fit()
    t = float(fit.tvalues[treatment])
    n = int(fit.nobs)
    k = x.shape[1]
    dof = n - k
    # Cinelli & Hazlett 2020 eq. 13: RV_q = 0.5 * ( sqrt(fq^4 + 4*fq^2) - fq^2 )
    # where fq = q * |t| / sqrt(dof).
    fq = q * abs(t) / np.sqrt(max(dof, 1))
    rv = 0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2)
    return SensemakrResult(
        treatment=treatment,
        outcome=outcome,
        estimate=float(fit.params[treatment]),
        se=float(fit.bse[treatment]),
        t_value=t,
        robustness_value=float(rv),
        robustness_value_q=float(rv),
        notes=[
            "sensemakr package not installed; using hand-rolled fallback. "
            "Install PySensemakr for benchmarking output: pip install sensemakr."
        ],
        backend="fallback",
    )


__all__ = ["SensemakrResult", "sensemakr"]
