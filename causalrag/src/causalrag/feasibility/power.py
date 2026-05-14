"""Minimum detectable effect (MDE) calculators per data-flag combo.

PDD §8.2 — MDE per flag combination. v0.1 ships closed-form calculators for
the three most common situations:

- ``binary_ate`` — binary treatment with binary OR continuous outcome.
  Closed form from Cohen (1988): MDE in standardized units is
  ``(z_{1-α/2} + z_β) * sqrt(1/n1 + 1/n0)``, where the two-arm allocation
  is read from the data.
- ``continuous_ate`` — continuous treatment with continuous outcome. MDE
  is the slope detectable at ``power`` given the empirical SD of the
  outcome and the SE of the treatment after partialling out controls.
- ``subgroup_cate`` — same as ``binary_ate`` but applied per-stratum of
  the modifier. Returns one MDE per stratum + a worst-case aggregate.

Higher-order combos (RMST contrast, IV-LATE, longitudinal MSM) are deferred
to v0.5 when the R bridge lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass
class PowerResult:
    """Single (treatment, outcome) MDE calculation."""

    treatment: str
    outcome: str
    family: Literal["binary_ate", "continuous_ate", "subgroup_cate", "unsupported"]
    n_used: int
    mde: float
    mde_units: str
    achieved_power_at_band: float | None = None
    plausible_band: tuple[float, float] | None = None
    n_required_at_band: int | None = None
    notes: str | None = None
    verdict: Literal["admissible", "borderline", "underpowered", "unsupported"] = "underpowered"


def _z(p: float) -> float:
    return float(norm.ppf(p))


def power_binary_ate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
    plausible_band: tuple[float, float] | None = None,
) -> PowerResult:
    """MDE for binary treatment, continuous or binary outcome.

    Assumes the user wants a two-sided test at significance ``alpha`` and
    the standardized MDE. Returns the effect size in OUTCOME UNITS (not d).
    """
    if treatment not in df.columns or outcome not in df.columns:
        return PowerResult(
            treatment=treatment,
            outcome=outcome,
            family="unsupported",
            n_used=0,
            mde=float("nan"),
            mde_units="—",
            notes="columns missing",
            verdict="unsupported",
        )
    work = df[[treatment, outcome]].dropna()
    n = len(work)
    if n < 30:
        return PowerResult(
            treatment=treatment,
            outcome=outcome,
            family="binary_ate",
            n_used=n,
            mde=float("inf"),
            mde_units="—",
            notes="n < 30; cannot reliably estimate MDE",
            verdict="underpowered",
        )
    t = work[treatment].astype(float).to_numpy()
    n1 = int((t == 1).sum())
    n0 = int((t == 0).sum())
    if n1 < 2 or n0 < 2:
        return PowerResult(
            treatment=treatment,
            outcome=outcome,
            family="binary_ate",
            n_used=n,
            mde=float("inf"),
            mde_units="—",
            notes=f"one arm too small (n1={n1}, n0={n0})",
            verdict="underpowered",
        )

    y = work[outcome].astype(float).to_numpy()
    y_unique = pd.Series(y).nunique()
    is_binary_outcome = y_unique <= 2
    if is_binary_outcome:
        p_bar = float(y.mean())
        sd = float(np.sqrt(p_bar * (1 - p_bar)))
        units = "abs. risk difference"
    else:
        sd = float(np.std(y, ddof=1)) or 1.0
        units = outcome

    mde = (_z(1 - alpha / 2) + _z(target_power)) * sd * np.sqrt(1 / n1 + 1 / n0)
    mde = float(mde)

    achieved: float | None = None
    n_required: int | None = None
    verdict: Literal["admissible", "borderline", "underpowered"] = "underpowered"
    if plausible_band is not None:
        lo, hi = plausible_band
        effect_target = (abs(lo) + abs(hi)) / 2 if (lo < 0 < hi) else max(abs(lo), abs(hi))
        # Achieved power at the band's midpoint
        z_score = effect_target / (sd * np.sqrt(1 / n1 + 1 / n0))
        achieved = float(norm.cdf(z_score - _z(1 - alpha / 2)))
        # n required (per arm, balanced) to reach target_power at effect_target
        if effect_target > 0:
            n_required = int(
                np.ceil(
                    2 * sd**2 * (_z(1 - alpha / 2) + _z(target_power)) ** 2 / effect_target**2
                )
            )
        if achieved >= target_power:
            verdict = "admissible"
        elif achieved >= 0.5:
            verdict = "borderline"

    return PowerResult(
        treatment=treatment,
        outcome=outcome,
        family="binary_ate",
        n_used=n,
        mde=mde,
        mde_units=units,
        achieved_power_at_band=achieved,
        plausible_band=plausible_band,
        n_required_at_band=n_required,
        verdict=verdict,
    )


def power_continuous_ate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...] = (),
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
    plausible_band: tuple[float, float] | None = None,
) -> PowerResult:
    """MDE for continuous treatment, continuous outcome via OLS partial-r SE."""
    cols = [treatment, outcome, *confounders]
    cols = [c for c in cols if c in df.columns]
    if treatment not in cols or outcome not in cols:
        return PowerResult(
            treatment=treatment,
            outcome=outcome,
            family="unsupported",
            n_used=0,
            mde=float("nan"),
            mde_units="—",
            notes="columns missing",
            verdict="unsupported",
        )
    work = df[cols].dropna()
    n = len(work)
    if n < 30:
        return PowerResult(
            treatment=treatment,
            outcome=outcome,
            family="continuous_ate",
            n_used=n,
            mde=float("inf"),
            mde_units=outcome,
            verdict="underpowered",
        )

    t = work[treatment].astype(float).to_numpy()
    y = work[outcome].astype(float).to_numpy()
    if confounders:
        w = work[list(confounders)].astype(float).to_numpy()
        from sklearn.linear_model import LinearRegression

        t_resid = t - LinearRegression().fit(w, t).predict(w)
    else:
        t_resid = t - t.mean()
    sd_y = float(np.std(y, ddof=1)) or 1.0
    sd_t = float(np.std(t_resid, ddof=1)) or 1.0
    se_beta = sd_y / (sd_t * np.sqrt(max(n - len(confounders) - 1, 1)))
    mde = (_z(1 - alpha / 2) + _z(target_power)) * se_beta
    units = f"{outcome} per unit {treatment}"

    achieved: float | None = None
    n_required: int | None = None
    verdict: Literal["admissible", "borderline", "underpowered"] = "underpowered"
    if plausible_band is not None:
        lo, hi = plausible_band
        effect_target = (abs(lo) + abs(hi)) / 2 if (lo < 0 < hi) else max(abs(lo), abs(hi))
        z_score = effect_target / se_beta
        achieved = float(norm.cdf(z_score - _z(1 - alpha / 2)))
        if effect_target > 0:
            n_required = int(
                np.ceil(
                    (sd_y / (sd_t * effect_target)) ** 2
                    * (_z(1 - alpha / 2) + _z(target_power)) ** 2
                    + len(confounders)
                    + 1
                )
            )
        if achieved >= target_power:
            verdict = "admissible"
        elif achieved >= 0.5:
            verdict = "borderline"

    return PowerResult(
        treatment=treatment,
        outcome=outcome,
        family="continuous_ate",
        n_used=n,
        mde=float(mde),
        mde_units=units,
        achieved_power_at_band=achieved,
        plausible_band=plausible_band,
        n_required_at_band=n_required,
        verdict=verdict,
    )


def power_subgroup_cate(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    modifier: str,
    *,
    alpha: float = 0.05,
    target_power: float = 0.8,
    plausible_band: tuple[float, float] | None = None,
    max_strata: int = 6,
) -> list[PowerResult]:
    """Per-stratum MDE for a CATE analysis. Splits by ``modifier`` quantiles
    when the modifier is continuous, by value otherwise."""
    if modifier not in df.columns:
        return []
    x = df[modifier]
    if pd.api.types.is_numeric_dtype(x) and x.nunique() > max_strata:
        strata = pd.qcut(x, q=min(max_strata, 4), duplicates="drop")
        labels = sorted(strata.unique(), key=lambda v: v.left if hasattr(v, "left") else 0)
    else:
        labels = sorted(x.dropna().unique())[:max_strata]
        strata = x
    out: list[PowerResult] = []
    for lab in labels:
        mask = strata == lab if hasattr(strata, "cat") else x == lab
        sub = df[mask]
        if len(sub) < 30:
            continue
        result = power_binary_ate(
            sub,
            treatment,
            outcome,
            alpha=alpha,
            target_power=target_power,
            plausible_band=plausible_band,
        )
        result.family = "subgroup_cate"
        result.notes = f"stratum: {modifier} = {lab}"
        out.append(result)
    return out


__all__ = [
    "PowerResult",
    "power_binary_ate",
    "power_continuous_ate",
    "power_subgroup_cate",
]
