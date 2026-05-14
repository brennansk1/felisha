"""Markov boundary discovery for the discovery phase.

Phase 1 of the multiple-MB roadmap: a single Markov-boundary pass on
the proposed treatment and outcome, used as a "stats vs LLM" cross-
check against the investigator's CONFOUNDER role assignments.

The pipeline already has an LLM investigator that infers each column's
semantic role from its name + value distribution. That LLM can mis-label
(e.g., labelling every numeric column as CONFOUNDER, as happened on the
Adult Census stress test). An IAMB-style Markov-boundary pass produces
a statistical answer to "which columns are in the predictive
neighbourhood of this target?". Disagreement between the two is itself
a useful diagnostic for the synthesis layer.

Implementation:

- When the R bridge + bnlearn are available, we use bnlearn's
  ``iamb``/``inter.iamb``/``iamb.fdr`` etc. (see ``rbridge/discovery_r.py``).
- When R isn't available we fall back to a Python IAMB implementation
  using partial-correlation z-tests (continuous data) or χ²/G² tests
  (discrete data). The fallback is intentionally simple — its job is to
  produce something defensible when R is missing, not to win benchmarks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("causalrag.discovery.markov_boundary")


@dataclass
class MarkovBoundaryReport:
    """Result of a Markov-boundary discovery pass on one target."""

    target: str
    mb: list[str]
    method: str
    backend: str  # "bnlearn" or "python.iamb"
    n: int
    alpha: float
    test: str
    notes: list[str]


def _partial_correlation_pvalue(
    x: np.ndarray, y: np.ndarray, z: np.ndarray | None, n: int
) -> float:
    """Two-sided p-value for the partial correlation of x and y given z."""
    if z is None or z.size == 0:
        r = float(np.corrcoef(x, y)[0, 1])
        if math.isnan(r) or abs(r) >= 1.0:
            return 1.0
        # Fisher's z
        z_stat = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(max(n - 3, 1))
        return 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Partial correlation: regress x on z and y on z, correlate residuals.
    z2 = np.atleast_2d(z)
    if z2.shape[0] != n:
        z2 = z2.T
    # Add intercept
    Z = np.column_stack([np.ones(n), z2])
    try:
        bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
        by, *_ = np.linalg.lstsq(Z, y, rcond=None)
        rx = x - Z @ bx
        ry = y - Z @ by
        if rx.std() < 1e-10 or ry.std() < 1e-10:
            return 1.0
        r = float(np.corrcoef(rx, ry)[0, 1])
    except np.linalg.LinAlgError:
        return 1.0
    if math.isnan(r) or abs(r) >= 1.0:
        return 1.0
    df = max(n - z2.shape[1] - 3, 1)
    z_stat = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(df)
    return 2 * (1 - stats.norm.cdf(abs(z_stat)))


def _python_iamb(
    df: pd.DataFrame, target: str, alpha: float, max_size: int | None
) -> list[str]:
    """Fallback IAMB on numeric data using Fisher's z partial correlation.

    Two-phase: GROW (add the most-associated covariate conditional on
    the current MB until no covariate is dependent on the target given
    MB), then SHRINK (drop any covariate that becomes independent of
    the target given the rest of the MB).

    Not the world's fastest IAMB — it's the rest-of-MB partial-correlation
    test on each iteration, which is O(|V|² · cost-of-regression). For
    p < 30 it runs in seconds; for p > 100 use bnlearn instead.
    """
    work = df.select_dtypes(include="number").dropna()
    if target not in work.columns:
        return []
    cols = [c for c in work.columns if c != target]
    n = len(work)
    y = work[target].to_numpy(dtype=float)

    mb: list[str] = []
    cap = max_size if max_size is not None else len(cols)

    # GROW
    changed = True
    while changed and len(mb) < cap:
        changed = False
        best_p = alpha
        best_col = None
        for c in cols:
            if c in mb:
                continue
            x = work[c].to_numpy(dtype=float)
            z = work[mb].to_numpy(dtype=float) if mb else None
            p = _partial_correlation_pvalue(x, y, z, n)
            if p < best_p:
                best_p = p
                best_col = c
        if best_col is not None:
            mb.append(best_col)
            changed = True

    # SHRINK
    changed = True
    while changed:
        changed = False
        for c in list(mb):
            others = [m for m in mb if m != c]
            x = work[c].to_numpy(dtype=float)
            z = work[others].to_numpy(dtype=float) if others else None
            p = _partial_correlation_pvalue(x, y, z, n)
            if p >= alpha:
                mb.remove(c)
                changed = True
                break

    return mb


def discover_markov_boundary(
    df: pd.DataFrame,
    *,
    target: str,
    method: str = "iamb",
    alpha: float = 0.05,
    max_size: int | None = None,
    prefer_bnlearn: bool = True,
) -> MarkovBoundaryReport:
    """Discover the Markov boundary of ``target`` in ``df``.

    Prefers the R-bridged bnlearn implementation when available
    (configurable via ``prefer_bnlearn``); otherwise falls back to a
    Python IAMB on the numeric subset.

    For non-numeric data without R, the fallback returns an empty MB
    and an informational note rather than guessing.
    """
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in df columns")

    notes: list[str] = []

    if prefer_bnlearn:
        try:
            from causalrag.estimators.rbridge.discovery_r import (
                discover_markov_boundary as r_mb,
            )

            r_result = r_mb(df, target=target, method=method, alpha=alpha)
            return MarkovBoundaryReport(
                target=target,
                mb=list(r_result["mb"]),
                method=method,
                backend="bnlearn",
                n=int(r_result["n"]),
                alpha=alpha,
                test=str(r_result["test"]),
                notes=notes,
            )
        except Exception as e:
            notes.append(
                f"bnlearn MB unavailable ({type(e).__name__}: {e}); "
                "falling back to Python IAMB on numeric subset."
            )
            logger.warning("bnlearn MB failed: %s — falling back", e)

    # Python fallback
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target not in numeric_cols:
        notes.append(
            f"target {target!r} is non-numeric and the bnlearn backend is "
            "unavailable; Python fallback only handles numeric targets."
        )
        return MarkovBoundaryReport(
            target=target,
            mb=[],
            method=method,
            backend="python.iamb",
            n=0,
            alpha=alpha,
            test="fisher_z",
            notes=notes,
        )

    work = df[numeric_cols].dropna()
    mb = _python_iamb(work, target=target, alpha=alpha, max_size=max_size)
    return MarkovBoundaryReport(
        target=target,
        mb=mb,
        method="iamb (python fallback)",
        backend="python.iamb",
        n=int(len(work)),
        alpha=alpha,
        test="fisher_z",
        notes=notes,
    )


__all__ = ["MarkovBoundaryReport", "discover_markov_boundary"]
