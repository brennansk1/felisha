"""Missing-data diagnostic stage (PDD §7.x).

Optional pipeline stage. Callers invoke ``diagnose_missingness`` on the
analytic frame and decide — based on the returned ``MissingnessReport`` — how
to proceed: complete-case, multiple imputation (MICE), inverse-probability-of-
censoring weighting (IPCW), or refusing to estimate at all.

The diagnostic is intentionally *advisory*: it does not mutate the frame and
nothing in the master loop calls it implicitly. This keeps the missing-data
choice a deliberate decision rather than a silent default.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MissingnessReport", "diagnose_missingness"]


Recommendation = Literal[
    "proceed_complete_case",
    "consider_mice",
    "consider_ipcw",
    "refuse",
]


class MissingnessReport(BaseModel):
    """Advisory diagnostic report for missing-data handling."""

    model_config = ConfigDict(extra="forbid")

    per_column_rate: dict[str, float] = Field(default_factory=dict)
    mcar_test_p_value: float | None = None
    complete_case_bias_score: float = 0.0
    recommendation: Recommendation = "proceed_complete_case"
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Little's MCAR test (heuristic approximation)
# ---------------------------------------------------------------------------


def _littles_mcar_test(df: pd.DataFrame) -> float | None:
    """Heuristic approximation of Little's (1988) MCAR test.

    statsmodels does not ship Little's test, so we approximate it with the
    following procedure: for each *missingness pattern* (a distinct row of the
    ``df.isna()`` matrix) compute the standardized squared distance of that
    pattern's column means from the global mean, sum across patterns, and
    refer the statistic to a chi-square distribution. This is the textbook
    decomposition Little uses; we drop the small-sample correction and just
    return the asymptotic p-value.

    Returns ``None`` if the test is not computable (no missingness, no numeric
    columns, single pattern, or any internal numerical failure).
    """
    try:
        from scipy import stats  # local import keeps module import light
    except Exception:
        return None

    try:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0 or numeric.shape[0] < 2:
            return None
        miss = numeric.isna()
        if not miss.values.any():
            return None

        # Group rows by missingness pattern.
        pattern_key = miss.astype(int).astype(str).agg("".join, axis=1)
        patterns = pattern_key.unique()
        if len(patterns) < 2:
            return None

        # Global means/variances computed on observed values per column.
        global_mean = numeric.mean(skipna=True)
        global_var = numeric.var(skipna=True, ddof=1).replace(0.0, np.nan)

        d2_total = 0.0
        dof_total = 0
        for pat in patterns:
            idx = pattern_key == pat
            n_j = int(idx.sum())
            if n_j < 2:
                continue
            sub = numeric.loc[idx]
            observed_cols = [c for c in numeric.columns if not miss.loc[idx, c].any()]
            if not observed_cols:
                continue
            diff = sub[observed_cols].mean() - global_mean[observed_cols]
            var = global_var[observed_cols]
            term = (diff ** 2) / var
            term = term.replace([np.inf, -np.inf], np.nan).dropna()
            if term.empty:
                continue
            d2_total += float(n_j * term.sum())
            dof_total += int(len(term))

        if dof_total <= 0 or not np.isfinite(d2_total):
            return None

        p = float(stats.chi2.sf(d2_total, df=dof_total))
        if not np.isfinite(p):
            return None
        return p
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Complete-case bias proxy
# ---------------------------------------------------------------------------


def _complete_case_bias(df: pd.DataFrame, treatment: str) -> float:
    """Standardized gap in E[T] between fully-observed rows and rows with any
    missingness elsewhere.

    Score = |E[T|complete] − E[T|any-missing]| / SE(T). Returns 0.0 if not
    computable (e.g., no missing rows, no complete rows, zero variance).
    """
    if treatment not in df.columns:
        return 0.0
    other_cols = [c for c in df.columns if c != treatment]
    if not other_cols:
        return 0.0
    any_missing = df[other_cols].isna().any(axis=1)
    if not any_missing.any() or any_missing.all():
        return 0.0
    t_full = pd.to_numeric(df.loc[~any_missing, treatment], errors="coerce").dropna()
    t_miss = pd.to_numeric(df.loc[any_missing, treatment], errors="coerce").dropna()
    if len(t_full) == 0 or len(t_miss) == 0:
        return 0.0
    t_all = pd.to_numeric(df[treatment], errors="coerce").dropna()
    se = float(t_all.std(ddof=1))
    if not np.isfinite(se) or se == 0.0:
        return 0.0
    return float(abs(t_full.mean() - t_miss.mean()) / se)


# ---------------------------------------------------------------------------
# Censoring detection
# ---------------------------------------------------------------------------


def _has_right_censoring_signal(df: pd.DataFrame, outcome: str | None) -> bool:
    """Detect a right-censoring indicator paired with the outcome.

    We treat a 0/1 column whose name suggests an event/status/censoring flag
    as evidence the outcome is a time-to-event endpoint requiring IPCW rather
    than imputation.
    """
    if outcome is None or outcome not in df.columns:
        return False
    import re

    pat = re.compile(r"(event|status|censor|indicator)", re.I)
    for col in df.columns:
        if col == outcome:
            continue
        if not pat.search(col):
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        try:
            uniq = set(pd.to_numeric(s, errors="coerce").dropna().unique().tolist())
        except Exception:
            continue
        if uniq and uniq.issubset({0, 1, 0.0, 1.0}):
            return True
    return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def diagnose_missingness(
    df: pd.DataFrame,
    treatment: str | None = None,
    outcome: str | None = None,
    threshold_pct: float = 0.05,
) -> MissingnessReport:
    """Produce a ``MissingnessReport`` for the analytic frame.

    Parameters
    ----------
    df:
        Analytic frame post-Stage-1 preprocessing. Mutated columns/typing
        should already match what the estimator will see.
    treatment:
        Column name for the treatment T. Required for the bias proxy.
    outcome:
        Column name for the outcome Y. Used only to detect a paired
        censoring indicator.
    threshold_pct:
        Per-column rate above which a "non-trivial missingness" note is
        emitted (default 5%). Does *not* drive the recommendation by itself.
    """
    notes: list[str] = []

    if df.shape[1] == 0:
        return MissingnessReport(notes=["empty frame"])

    per_column_rate = {c: float(df[c].isna().mean()) for c in df.columns}
    max_rate = max(per_column_rate.values()) if per_column_rate else 0.0
    any_missing = max_rate > 0.0

    flagged = [c for c, r in per_column_rate.items() if r > threshold_pct]
    if flagged:
        notes.append(
            f"{len(flagged)} column(s) exceed {threshold_pct:.0%} missingness: "
            f"{', '.join(sorted(flagged)[:6])}"
        )

    mcar_p = _littles_mcar_test(df) if any_missing else None
    if mcar_p is None and any_missing:
        notes.append("Little's MCAR approximation could not be computed")
    elif mcar_p is not None:
        notes.append(f"Little's MCAR approx p={mcar_p:.4f}")

    if treatment is not None and any_missing:
        bias = _complete_case_bias(df, treatment)
    else:
        bias = 0.0
    if bias > 0.5:
        notes.append(
            f"complete-case bias proxy = {bias:.2f} (>0.5σ shift in E[T])"
        )

    censored = _has_right_censoring_signal(df, outcome)

    # ── Recommendation logic ────────────────────────────────────────────────
    recommendation: Recommendation
    if max_rate > 0.50:
        recommendation = "refuse"
        notes.append(
            f"max per-column missingness {max_rate:.0%} exceeds 50%; "
            "estimation is not advisable"
        )
    elif censored:
        # Right-censoring takes precedence over MICE: imputing a censored
        # survival time is rarely the right move.
        recommendation = "consider_ipcw"
        notes.append("right-censoring indicator detected for outcome; prefer IPCW")
    elif max_rate > 0.20 or (
        mcar_p is not None and mcar_p < 0.05 and bias > 0.5
    ):
        recommendation = "consider_mice"
    else:
        recommendation = "proceed_complete_case"

    return MissingnessReport(
        per_column_rate=per_column_rate,
        mcar_test_p_value=mcar_p,
        complete_case_bias_score=bias,
        recommendation=recommendation,
        notes=notes,
    )
