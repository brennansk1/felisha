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
    # Phase 2 / 3 — optional fields
    alternative_mbs: list[list[str]] | None = None  # k-1 distinct alternative MBs
    stability_scores: dict[str, float] | None = None  # per-variable selection freq
    bootstrap_iterations: int | None = None  # B from stability subsampling


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


# ─── Phase 2 — multiple-MB triangulation ──────────────────────────────────


def _verify_mb_on_original(
    df: pd.DataFrame, target: str, mb: list[str], alpha: float
) -> bool:
    """Check that ``mb`` satisfies the MB definition on the original data.

    For every variable v ∉ MB ∪ {target}, test whether v ⊥ target | MB.
    If any v fails (small p), this MB is not valid on the original
    distribution and should be rejected. Used as the TIE*-style verify
    step on candidate alternative MBs.
    """
    work = df.select_dtypes(include="number").dropna()
    if target not in work.columns:
        return False
    n = len(work)
    y = work[target].to_numpy(dtype=float)
    z = work[mb].to_numpy(dtype=float) if mb else None
    others = [c for c in work.columns if c not in set(mb) and c != target]
    # Bonferroni-correct so multiple-testing across the held-out set doesn't
    # spuriously reject too aggressively.
    bonf_alpha = alpha / max(len(others), 1)
    for v in others:
        x = work[v].to_numpy(dtype=float)
        p = _partial_correlation_pvalue(x, y, z, n)
        if p < bonf_alpha:
            return False
    return True


def _kiamb_one_run(
    df: pd.DataFrame,
    target: str,
    alpha: float,
    max_size: int | None,
    rng: np.random.Generator,
    randomness: float = 0.5,
) -> list[str]:
    """Stochastic IAMB (KIAMB-style).

    During the grow phase, instead of always picking the most-associated
    covariate, sample from the top-``ceil(randomness × |remaining|)``
    candidates. ``randomness=0`` → deterministic IAMB (fastest single MB);
    ``randomness=1`` → uniform random over surviving candidates (most
    diverse). Returns one (possibly different) MB per call.
    """
    work = df.select_dtypes(include="number").dropna()
    if target not in work.columns:
        return []
    cols = [c for c in work.columns if c != target]
    n = len(work)
    y = work[target].to_numpy(dtype=float)
    mb: list[str] = []
    cap = max_size if max_size is not None else len(cols)

    changed = True
    while changed and len(mb) < cap:
        changed = False
        # Score every remaining covariate
        candidates: list[tuple[float, str]] = []
        for c in cols:
            if c in mb:
                continue
            x = work[c].to_numpy(dtype=float)
            z = work[mb].to_numpy(dtype=float) if mb else None
            p = _partial_correlation_pvalue(x, y, z, n)
            if p < alpha:
                candidates.append((p, c))
        if not candidates:
            break
        # Sort ascending by p (most-associated first)
        candidates.sort(key=lambda t: t[0])
        # KIAMB: sample uniformly from the top-K
        top_k = max(1, int(math.ceil(randomness * len(candidates))))
        idx = int(rng.integers(0, top_k))
        mb.append(candidates[idx][1])
        changed = True

    # SHRINK (same as deterministic IAMB)
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


def discover_multiple_mbs(
    df: pd.DataFrame,
    *,
    target: str,
    k: int = 3,
    alpha: float = 0.05,
    randomness: float = 0.5,
    seed: int = 42,
    verify_on_original: bool = True,
    max_size: int | None = None,
) -> MarkovBoundaryReport:
    """Discover up to ``k`` distinct, verified Markov boundaries of ``target``.

    Uses stochastic IAMB (KIAMB) with multiple random restarts.  Each
    candidate MB is verified against the original distribution (the
    TIE* validation step) — candidates that look like an MB on a
    permuted scan but fail the conditional-independence definition on
    the original data get rejected.

    Returns one :class:`MarkovBoundaryReport` whose ``mb`` is the
    primary (deterministic) MB and whose ``alternative_mbs`` lists
    the additional verified MBs found.  When only one MB survives,
    ``alternative_mbs`` is empty.

    This is Phase 2 of the multi-MB roadmap — it does NOT pretend to
    be a full TIE* re-implementation. It surfaces information-equivalent
    MBs when they exist (faithfulness violations from multicollinear
    pathways) but won't find every alternative in pathological
    distributions. For high-dim genomics (n ≪ p) the caller should set
    ``verify_on_original=True`` and combine with stability subsampling
    via :func:`discover_stable_mb`.
    """
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in df columns")
    work = df.select_dtypes(include="number").dropna()
    if target not in work.columns:
        return MarkovBoundaryReport(
            target=target,
            mb=[],
            method="kiamb (python)",
            backend="python.iamb",
            n=0,
            alpha=alpha,
            test="fisher_z",
            notes=[
                f"target {target!r} non-numeric; multi-MB requires bnlearn "
                "for non-numeric targets."
            ],
            alternative_mbs=[],
        )

    rng = np.random.default_rng(seed)

    # Run 1 — deterministic (randomness=0) to anchor the primary MB
    primary_mb = _python_iamb(work, target=target, alpha=alpha, max_size=max_size)
    found: list[list[str]] = [primary_mb] if primary_mb else []

    # Run 2..k — stochastic restarts
    attempts = 0
    max_attempts = max(k * 4, 12)
    while len(found) < k and attempts < max_attempts:
        attempts += 1
        candidate = _kiamb_one_run(
            work,
            target=target,
            alpha=alpha,
            max_size=max_size,
            rng=rng,
            randomness=randomness,
        )
        if not candidate:
            continue
        # Skip if equivalent to a previously-found MB (set equality)
        if any(set(candidate) == set(prev) for prev in found):
            continue
        # Verify on original distribution
        if verify_on_original and not _verify_mb_on_original(
            work, target=target, mb=candidate, alpha=alpha
        ):
            continue
        found.append(candidate)

    return MarkovBoundaryReport(
        target=target,
        mb=found[0] if found else [],
        method="kiamb (python)",
        backend="python.iamb",
        n=int(len(work)),
        alpha=alpha,
        test="fisher_z",
        notes=[
            f"requested k={k}, found {len(found)} distinct MBs in {attempts} attempts"
        ],
        alternative_mbs=[mb for mb in found[1:]],
    )


# ─── Phase 3 — stability subsampling + FDR ────────────────────────────────


def discover_stable_mb(
    df: pd.DataFrame,
    *,
    target: str,
    bootstrap_iterations: int = 20,
    subsample_fraction: float = 0.8,
    stability_threshold: float = 0.6,
    alpha: float = 0.05,
    seed: int = 42,
    max_size: int | None = None,
    method: str = "iamb.fdr",
    prefer_bnlearn: bool = True,
) -> MarkovBoundaryReport:
    """Stability-selected Markov boundary via bootstrap subsampling.

    Runs ``bootstrap_iterations`` IAMB passes on subsamples (size
    ``subsample_fraction × n``, drawn with replacement). Counts how
    often each candidate variable appears in the resulting MBs. Keeps
    variables whose selection frequency is at least
    ``stability_threshold``. Returns the stability-selected MB plus
    per-variable selection frequencies.

    Designed for n ≪ p / high-dim regimes (oncology genomics, brain
    imaging, financial feature sets) where a single IAMB pass is too
    fragile. When ``prefer_bnlearn=True`` and ``method='iamb.fdr'`` the
    underlying CI test is FDR-controlled (Pena 2008), which further
    reins in false discoveries under multiple testing.

    Cost is roughly ``bootstrap_iterations × cost(single IAMB)``.
    """
    if target not in df.columns:
        raise ValueError(f"target {target!r} not in df columns")
    work = df.select_dtypes(include="number").dropna()
    if target not in work.columns:
        return MarkovBoundaryReport(
            target=target,
            mb=[],
            method=f"stability+{method}",
            backend="none",
            n=0,
            alpha=alpha,
            test="n/a",
            notes=[f"target {target!r} is non-numeric — stability MB skipped"],
            stability_scores={},
            bootstrap_iterations=0,
        )

    rng = np.random.default_rng(seed)
    n = len(work)
    sub_n = max(int(subsample_fraction * n), 30)
    counts: dict[str, int] = {}
    successes = 0
    used_backend = "python.iamb"
    used_test = "fisher_z"

    for b in range(bootstrap_iterations):
        idx = rng.choice(n, size=sub_n, replace=True)
        sub = work.iloc[idx].reset_index(drop=True)
        try:
            if prefer_bnlearn:
                from causalrag.estimators.rbridge.discovery_r import (
                    discover_markov_boundary as r_mb,
                )

                r_result = r_mb(sub, target=target, method=method, alpha=alpha)
                mb_b = list(r_result["mb"])
                used_backend = "bnlearn"
                used_test = str(r_result.get("test", "cor"))
            else:
                mb_b = _python_iamb(
                    sub, target=target, alpha=alpha, max_size=max_size
                )
        except Exception as e:
            logger.warning(
                "bootstrap iter %d failed: %s — falling back to Python", b, e
            )
            try:
                mb_b = _python_iamb(
                    sub, target=target, alpha=alpha, max_size=max_size
                )
            except Exception:
                continue
        for c in mb_b:
            counts[c] = counts.get(c, 0) + 1
        successes += 1

    if successes == 0:
        return MarkovBoundaryReport(
            target=target,
            mb=[],
            method=f"stability+{method}",
            backend=used_backend,
            n=int(n),
            alpha=alpha,
            test=used_test,
            notes=["all bootstrap iterations failed"],
            stability_scores={},
            bootstrap_iterations=bootstrap_iterations,
        )

    scores = {c: counts.get(c, 0) / successes for c in counts}
    stable = sorted(
        [c for c, freq in scores.items() if freq >= stability_threshold],
        key=lambda c: -scores[c],
    )
    if max_size is not None:
        stable = stable[:max_size]

    return MarkovBoundaryReport(
        target=target,
        mb=stable,
        method=f"stability+{method}",
        backend=used_backend,
        n=int(n),
        alpha=alpha,
        test=used_test,
        notes=[
            f"B={successes}/{bootstrap_iterations} bootstraps succeeded, "
            f"subsample_frac={subsample_fraction}, "
            f"stability_threshold={stability_threshold}"
        ],
        stability_scores=scores,
        bootstrap_iterations=successes,
    )


__all__ = [
    "MarkovBoundaryReport",
    "discover_markov_boundary",
    "discover_multiple_mbs",
    "discover_stable_mb",
]
