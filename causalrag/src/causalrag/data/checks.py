"""Data-side identifiability diagnostics (PDD §13 ``data/checks.py``).

These run as part of Step 5 (identifiability) and Step 7 (estimation) to
surface assumption violations *before* the analyst trusts an estimate:

- :func:`propensity_overlap` — fits a propensity model and reports the
  distribution of ``ê(X)`` per treatment arm. Returns a verdict
  (`green/yellow/red`) based on the worst-case violation of the strict
  positivity assumption ``0 < e(X) < 1`` (Rosenbaum & Rubin 1983).
- :func:`balance_diagnostic` — pre-/post-adjustment standardized mean
  difference per covariate. A standardized difference > 0.1 after weighting
  signals residual imbalance (Austin 2009).
- :func:`overlap_summary` — convenience wrapper that runs both and rolls them
  into a single diagnostics dict suitable for ``EstimationResult.diagnostics``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Verdict = Literal["green", "yellow", "red"]


@dataclass
class PositivityResult:
    propensity_min: float
    propensity_max: float
    propensity_p01: float
    propensity_p99: float
    n_extreme: int
    pct_extreme: float
    verdict: Verdict
    note: str
    threshold_low: float = 0.05
    threshold_high: float = 0.95


@dataclass
class BalanceRow:
    covariate: str
    std_diff_unweighted: float
    std_diff_weighted: float | None = None
    imbalanced: bool = False


@dataclass
class OverlapDiagnostics:
    positivity: PositivityResult
    balance: list[BalanceRow] = field(default_factory=list)
    worst_imbalance: float | None = None

    def to_dict(self) -> dict:
        return {
            "positivity": {
                "verdict": self.positivity.verdict,
                "note": self.positivity.note,
                "propensity_min": self.positivity.propensity_min,
                "propensity_max": self.positivity.propensity_max,
                "propensity_p01": self.positivity.propensity_p01,
                "propensity_p99": self.positivity.propensity_p99,
                "n_extreme": self.positivity.n_extreme,
                "pct_extreme": self.positivity.pct_extreme,
            },
            "balance": [
                {
                    "covariate": r.covariate,
                    "std_diff_unweighted": r.std_diff_unweighted,
                    "std_diff_weighted": r.std_diff_weighted,
                    "imbalanced": r.imbalanced,
                }
                for r in self.balance
            ],
            "worst_imbalance": self.worst_imbalance,
        }


def propensity_overlap(
    df: pd.DataFrame,
    treatment: str,
    confounders: tuple[str, ...],
    *,
    threshold_low: float = 0.05,
    threshold_high: float = 0.95,
    random_state: int = 42,
) -> PositivityResult:
    """Fit a propensity model and report positivity diagnostics.

    Uses the project's SuperLearner-stacked classifier when n is large enough,
    otherwise a tuned GradientBoosting. Returns a verdict plus the empirical
    propensity-distribution summary.
    """
    from causalrag.estimators.python.nuisance import super_learner_classifier

    cols = [treatment, *confounders]
    work = df[cols].dropna()
    n = len(work)
    if n < 50:
        return PositivityResult(
            propensity_min=float("nan"),
            propensity_max=float("nan"),
            propensity_p01=float("nan"),
            propensity_p99=float("nan"),
            n_extreme=0,
            pct_extreme=0.0,
            verdict="yellow",
            note="Sample too small for positivity diagnosis (n<50).",
        )

    t = work[treatment].to_numpy().astype(int)
    x = work[list(confounders)].to_numpy().astype(float)
    clf = super_learner_classifier(random_state, library="auto", n=n)
    clf.fit(x, t)
    e_hat = clf.predict_proba(x)[:, 1]

    extreme = ((e_hat < threshold_low) | (e_hat > threshold_high)).sum()
    pct = float(extreme / n)

    p_min = float(e_hat.min())
    p_max = float(e_hat.max())
    p01, p99 = float(np.quantile(e_hat, 0.01)), float(np.quantile(e_hat, 0.99))

    if p_min < 0.01 or p_max > 0.99 or pct > 0.10:
        verdict: Verdict = "red"
        note = (
            f"Positivity strongly violated: {pct:.0%} of observations have "
            f"propensity ∉ [{threshold_low}, {threshold_high}]; range "
            f"[{p_min:.3f}, {p_max:.3f}]. IPW/PSM unreliable; prefer DML with "
            f"trimming or doubly-robust estimators."
        )
    elif p_min < threshold_low or p_max > threshold_high or pct > 0.05:
        verdict = "yellow"
        note = (
            f"Mild positivity concern: {pct:.0%} of observations near the boundary "
            f"(range [{p_min:.3f}, {p_max:.3f}]). Consider trimming or "
            f"propensity-stabilized weights."
        )
    else:
        verdict = "green"
        note = (
            f"Positivity OK: propensity in [{p_min:.3f}, {p_max:.3f}] "
            f"with no extreme tail mass."
        )

    return PositivityResult(
        propensity_min=p_min,
        propensity_max=p_max,
        propensity_p01=p01,
        propensity_p99=p99,
        n_extreme=int(extreme),
        pct_extreme=pct,
        verdict=verdict,
        note=note,
        threshold_low=threshold_low,
        threshold_high=threshold_high,
    )


def balance_diagnostic(
    df: pd.DataFrame,
    treatment: str,
    confounders: tuple[str, ...],
    *,
    propensity: np.ndarray | None = None,
    imbalance_threshold: float = 0.10,
) -> list[BalanceRow]:
    """Standardized mean difference per covariate, unweighted and (if a
    propensity vector is supplied) IPW-weighted (Austin 2009)."""
    cols = [treatment, *confounders]
    work = df[cols].dropna()
    if propensity is not None and len(propensity) != len(work):
        propensity = None

    t = work[treatment].to_numpy().astype(int)
    rows: list[BalanceRow] = []
    for c in confounders:
        x = work[c].to_numpy().astype(float)
        x1 = x[t == 1]
        x0 = x[t == 0]
        if len(x1) < 2 or len(x0) < 2:
            continue
        s1 = x1.std(ddof=1)
        s0 = x0.std(ddof=1)
        pooled = float(np.sqrt((s1**2 + s0**2) / 2))
        if pooled == 0:
            continue
        smd = float((x1.mean() - x0.mean()) / pooled)

        smd_w: float | None = None
        if propensity is not None:
            w = np.where(t == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))
            num = (w * t * x).sum() / max((w * t).sum(), 1e-9) - (w * (1 - t) * x).sum() / max(
                (w * (1 - t)).sum(), 1e-9
            )
            smd_w = float(num / pooled)
        rows.append(
            BalanceRow(
                covariate=c,
                std_diff_unweighted=smd,
                std_diff_weighted=smd_w,
                imbalanced=abs(smd_w if smd_w is not None else smd) > imbalance_threshold,
            )
        )
    return rows


def overlap_summary(
    df: pd.DataFrame,
    treatment: str,
    confounders: tuple[str, ...],
) -> OverlapDiagnostics:
    """One-call wrapper used by ``q5_identify`` and Step 7."""
    pos = propensity_overlap(df, treatment, confounders)
    bal = balance_diagnostic(df, treatment, confounders)
    worst = max((abs(r.std_diff_weighted or r.std_diff_unweighted) for r in bal), default=None)
    return OverlapDiagnostics(positivity=pos, balance=bal, worst_imbalance=worst)


__all__ = [
    "PositivityResult",
    "BalanceRow",
    "OverlapDiagnostics",
    "propensity_overlap",
    "balance_diagnostic",
    "overlap_summary",
]
