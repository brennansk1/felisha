"""Uplift modelling, Qini / AUUC, and policy targeting (Sprint 5.5).

Given a fitted CATE-capable estimator (or just an array of per-row CATE
predictions), this module answers "who should we treat?" by producing:

1. A recommended policy :math:`\\pi(X) = 1\\{\\widehat{\\tau}(X) > c\\}` for
   some user-supplied threshold ``c`` (default 0 — treat anyone with a
   positive expected effect).
2. A **Qini curve** (Radcliffe 2007) — the cumulative uplift achieved as
   we treat the top-fraction of units ranked by CATE, compared against
   the random-targeting baseline.
3. **AUUC** (Area Under the Uplift Curve) — a single-number summary of
   policy quality; integrates the cumulative uplift over the targeting
   fraction in ``[0, 1]``.
4. **Expected policy value (EPV)** via ERUPT
   (:func:`causalrag.estimators.causaltune_select.erupt`-style IPW).
5. A small CART-style **policy tree** when ``econml.policy.PolicyTree``
   or the R ``policytree`` package is installed. Returns ``None`` if
   neither is available — the rest of the report still functions.

Algorithm — cumulative uplift / Qini
------------------------------------
Sort all rows in descending order of predicted CATE. For each fraction
:math:`f \\in [0, 1]` (one breakpoint per row in the sorted index) we
compute the per-unit cumulative uplift::

    uplift(f) = mean(Y | T = 1, top-f) - mean(Y | T = 0, top-f)

multiplied by the number of units in the top-f bucket — i.e. the
Radcliffe "Qini" curve in *gain* units. The random-targeting baseline
is the line from ``(0, 0)`` to ``(1, n * ATT)`` (treating everyone gives
the full ATT). The **Qini coefficient** is the area between the actual
uplift curve and the random baseline, normalized by the area under the
random baseline so a perfect-targeting policy approaches 1 and random
targeting yields 0. **AUUC** is the unnormalized area under the actual
uplift curve.

We use the standard trapezoidal rule along the (fraction, lift) curve
to integrate.

Notes
-----
* Empty top-arm buckets (no treated or no control in the top-f set) are
  handled by carrying forward the most recent valid mean — this matches
  the convention used by `causalml` / `scikit-uplift`. Without it the
  curve has visible jumps on small datasets.
* When all rows have the same predicted CATE (e.g. a constant
  estimator), the Qini coefficient is ~0 because sorting is arbitrary
  — that's the right answer: a constant CATE provides no targeting
  signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class UpliftCurve:
    """Per-fraction cumulative uplift + the random-targeting baseline."""

    fraction_treated: np.ndarray  # length n+1, increasing from 0 to 1
    lift: np.ndarray  # cumulative uplift at each fraction (gain units)
    random_baseline: np.ndarray  # what random targeting would achieve
    qini_coefficient: float
    auuc: float  # area under the (fraction, lift) curve


@dataclass
class PolicyTreeResult:
    """Wrapper around a fitted PolicyTree (econml or rpy2-policytree)."""

    backend: str  # "econml" or "policytree-r"
    max_depth: int
    n_leaves: int
    feature_names: list[str]
    tree: Any  # the underlying fitted object (kept opaque)
    predict_fn: Any = None  # callable(X) -> np.ndarray of recommended actions
    notes: list[str] = field(default_factory=list)


@dataclass
class TargetingReport:
    n_total: int
    n_recommended_treat: int
    fraction_recommended_treat: float
    expected_policy_value: float
    expected_random_value: float
    policy_lift_over_random: float
    threshold_used: float
    qini: UpliftCurve
    quantile_atts: dict[float, float]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_1d(arr: Any, name: str) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).reshape(-1)
    if out.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    return out


def _check_lengths(*arrs: tuple[str, np.ndarray]) -> int:
    lengths = {name: arr.shape[0] for name, arr in arrs}
    sizes = set(lengths.values())
    if len(sizes) != 1:
        raise ValueError(f"Length mismatch: {lengths}")
    return sizes.pop()


def _cumulative_uplift_curve(
    cate: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sort rows by descending CATE, walk down the list, return the gain
    curve (fraction, cumulative_uplift_gain) and the total ATT.

    Cumulative uplift gain at the top-k subset is::

        gain_k = k * ( mean(Y_top-k | T=1) - mean(Y_top-k | T=0) )

    so by construction gain_0 = 0 and gain_n = n * total_ATT.
    """
    order = np.argsort(-cate, kind="stable")
    y_s = y[order]
    t_s = t[order]
    n = y_s.shape[0]

    # Running sums of treated and control outcomes (and counts).
    is_treated = (t_s > 0.5).astype(np.float64)
    is_control = 1.0 - is_treated
    cum_y_t = np.cumsum(y_s * is_treated)
    cum_y_c = np.cumsum(y_s * is_control)
    cum_n_t = np.cumsum(is_treated)
    cum_n_c = np.cumsum(is_control)

    # Carry-forward means when a bucket lacks one arm.
    mean_t = np.where(cum_n_t > 0, cum_y_t / np.maximum(cum_n_t, 1.0), 0.0)
    mean_c = np.where(cum_n_c > 0, cum_y_c / np.maximum(cum_n_c, 1.0), 0.0)

    # Gain in *count of units* * mean-uplift = the Radcliffe Qini "lift"
    # axis. Scale by k = (i + 1).
    k = np.arange(1, n + 1, dtype=np.float64)
    lift_inner = k * (mean_t - mean_c)

    # Prepend the origin (0, 0) so the integral is well-defined.
    lift = np.concatenate([[0.0], lift_inner])
    fraction = np.concatenate([[0.0], k / n])

    # Total ATT: the final point on the curve divided by n. (More precisely,
    # the unconditional treated-vs-control mean difference on the full
    # sample, which equals lift[-1] / n by construction.)
    total_att = float(lift[-1] / n) if n > 0 else 0.0
    return fraction, lift, total_att


def _trapz_area(x: np.ndarray, y: np.ndarray) -> float:
    """Plain trapezoidal-rule area under ``(x, y)``."""
    if x.shape[0] < 2:
        return 0.0
    return float(np.trapezoid(y, x))


def _erupt_ipw(
    y: np.ndarray,
    t: np.ndarray,
    pi: np.ndarray,
    propensity: np.ndarray,
) -> float:
    """IPW ERUPT — see :func:`causalrag.estimators.causaltune_select.erupt`.

    Computes ``mean_i [ Y_i * 1{T_i == pi_i} / P(T_i = T_i | X_i) ]``.
    ``propensity`` is :math:`P(T_i = 1 | X_i)`; we flip it for control units.
    """
    propensity = np.clip(propensity, 0.05, 0.95)
    e_obs = np.where(t > 0.5, propensity, 1.0 - propensity)
    match = (t == pi).astype(np.float64)
    contributions = y * match / e_obs
    return float(np.mean(contributions))


def _quantile_atts(
    cate: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_deciles: int,
) -> dict[float, float]:
    """ATT within each CATE quantile bucket.

    Splits rows into ``n_deciles`` equally-sized buckets by descending
    CATE, returns ``{midpoint: ATT}`` where ``midpoint`` is the bucket's
    fractional midpoint in ``[0, 1]`` so callers can sort the dict by
    targeting order (top decile first).
    """
    n = cate.shape[0]
    if n == 0 or n_deciles < 1:
        return {}
    order = np.argsort(-cate, kind="stable")
    y_s = y[order]
    t_s = t[order]
    out: dict[float, float] = {}
    edges = np.linspace(0, n, n_deciles + 1, dtype=int)
    for i in range(n_deciles):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        sub_y = y_s[lo:hi]
        sub_t = t_s[lo:hi]
        mask_t = sub_t > 0.5
        mask_c = ~mask_t
        if mask_t.sum() == 0 or mask_c.sum() == 0:
            att = float("nan")
        else:
            att = float(sub_y[mask_t].mean() - sub_y[mask_c].mean())
        # Midpoint of the targeting fraction, e.g. decile 1 -> 0.05.
        midpoint = (lo + hi) / 2.0 / n
        out[round(midpoint, 6)] = att
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_targeting_report(
    *,
    cate_predictions: np.ndarray,
    observed_outcomes: np.ndarray,
    observed_treatments: np.ndarray,
    propensity: np.ndarray | None = None,
    threshold: float = 0.0,
    n_deciles: int = 10,
) -> TargetingReport:
    """Build the full uplift / Qini / policy-value report.

    Parameters
    ----------
    cate_predictions
        Per-row predicted CATE :math:`\\widehat{\\tau}(X_i)`.
    observed_outcomes
        Observed :math:`Y_i`.
    observed_treatments
        Observed :math:`T_i \\in \\{0, 1\\}`.
    propensity
        Optional per-row :math:`P(T_i = 1 | X_i)`. Defaults to the
        marginal :math:`\\bar T` (a uniform propensity is fine when the
        data come from a randomized experiment).
    threshold
        Policy threshold ``c`` in :math:`\\pi(X) = 1\\{\\tau(X) > c\\}`.
    n_deciles
        Number of CATE-rank quantile buckets to report ATT within.

    Returns
    -------
    :class:`TargetingReport`
    """
    cate = _coerce_1d(cate_predictions, "cate_predictions")
    y = _coerce_1d(observed_outcomes, "observed_outcomes")
    t = _coerce_1d(observed_treatments, "observed_treatments")
    n = _check_lengths(
        ("cate_predictions", cate),
        ("observed_outcomes", y),
        ("observed_treatments", t),
    )

    treat_set = set(np.unique(t).tolist())
    if not treat_set.issubset({0.0, 1.0}):
        raise ValueError(
            f"observed_treatments must be binary 0/1; got {treat_set}"
        )

    if propensity is None:
        p_bar = float(t.mean()) if n > 0 else 0.5
        prop = np.full(n, p_bar, dtype=np.float64)
    else:
        prop = _coerce_1d(propensity, "propensity")
        if prop.shape[0] != n:
            raise ValueError(
                f"propensity length {prop.shape[0]} != cate length {n}"
            )

    # ------------------------------------------------------------------
    # Uplift curve, AUUC, Qini coefficient
    # ------------------------------------------------------------------
    fraction, lift, total_att = _cumulative_uplift_curve(cate, y, t)
    # Random-targeting baseline goes linearly from 0 to n * total_att.
    random_baseline = fraction * n * total_att
    auuc = _trapz_area(fraction, lift)
    auuc_random = _trapz_area(fraction, random_baseline)
    # Qini coefficient = (area between curve & random) / |area under random|.
    # We use abs to handle negative-ATT regimes sensibly.
    if abs(auuc_random) > 0:
        qini = (auuc - auuc_random) / abs(auuc_random)
    else:
        # No average effect to normalize against — fall back to the raw
        # area difference so we can still distinguish zero vs nonzero.
        qini = auuc - auuc_random

    curve = UpliftCurve(
        fraction_treated=fraction,
        lift=lift,
        random_baseline=random_baseline,
        qini_coefficient=float(qini),
        auuc=float(auuc),
    )

    # ------------------------------------------------------------------
    # Policy + EPV via ERUPT
    # ------------------------------------------------------------------
    pi = (cate > threshold).astype(np.float64)
    n_treat = int(pi.sum())
    frac_treat = float(n_treat / n) if n else 0.0

    epv = _erupt_ipw(y, t, pi, prop)
    # Random policy: treat-all + treat-none averaged, evaluated under IPW.
    pi_all = np.ones(n, dtype=np.float64)
    pi_none = np.zeros(n, dtype=np.float64)
    epv_random = 0.5 * (
        _erupt_ipw(y, t, pi_all, prop) + _erupt_ipw(y, t, pi_none, prop)
    )
    if abs(epv_random) > 0:
        policy_lift = (epv - epv_random) / abs(epv_random)
    else:
        policy_lift = epv - epv_random

    # ------------------------------------------------------------------
    # Quantile ATTs
    # ------------------------------------------------------------------
    qatts = _quantile_atts(cate, y, t, n_deciles=n_deciles)

    notes: list[str] = []
    if propensity is None:
        notes.append(
            "propensity not provided; assumed uniform P(T=1) = marginal mean"
        )
    if abs(total_att) < 1e-12:
        notes.append(
            "total ATT ~ 0; Qini coefficient is degenerate (no signal to "
            "normalize against)"
        )
    if n_treat == 0:
        notes.append(f"policy recommends treating nobody (threshold={threshold})")
    elif n_treat == n:
        notes.append(f"policy recommends treating everyone (threshold={threshold})")

    return TargetingReport(
        n_total=int(n),
        n_recommended_treat=n_treat,
        fraction_recommended_treat=frac_treat,
        expected_policy_value=float(epv),
        expected_random_value=float(epv_random),
        policy_lift_over_random=float(policy_lift),
        threshold_used=float(threshold),
        qini=curve,
        quantile_atts=qatts,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Optional policy tree
# ---------------------------------------------------------------------------


def policy_tree(
    *,
    X: pd.DataFrame,
    cate_predictions: np.ndarray,
    max_depth: int = 3,
) -> PolicyTreeResult | None:
    """Fit a small CART-style policy tree on the predicted CATE.

    Tries ``econml.policy.PolicyTree`` first (pure Python). If that's not
    importable, falls back to the R ``policytree`` package via ``rpy2``.
    Returns ``None`` when neither is available — callers should treat
    that as a soft failure (the rest of the targeting report is still
    valid).

    Parameters
    ----------
    X
        Feature matrix used to split on. Must be a pandas DataFrame so
        we can preserve feature names for the tree summary.
    cate_predictions
        Per-row predicted CATE, length must match ``len(X)``.
    max_depth
        Maximum tree depth.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    cate = np.asarray(cate_predictions, dtype=np.float64).reshape(-1)
    if cate.shape[0] != len(X):
        raise ValueError(
            f"cate_predictions length {cate.shape[0]} != len(X) {len(X)}"
        )

    feature_names = [str(c) for c in X.columns]
    X_arr = X.to_numpy(dtype=np.float64)

    # Reward matrix for a binary policy tree: column 0 = reward under
    # action 0 (= 0 — the baseline never gains anything), column 1 =
    # reward under action 1 (= CATE).
    reward = np.column_stack([np.zeros_like(cate), cate])

    # ---- econml backend ------------------------------------------------
    try:
        from econml.policy import PolicyTree as _EconPolicyTree
    except Exception:
        _EconPolicyTree = None  # type: ignore[assignment]

    if _EconPolicyTree is not None:
        try:
            tree = _EconPolicyTree(max_depth=max_depth, min_samples_leaf=5)
            tree.fit(X_arr, reward)

            def _predict(X_new: np.ndarray | pd.DataFrame) -> np.ndarray:
                arr = (
                    X_new[list(X.columns)].to_numpy(dtype=np.float64)
                    if isinstance(X_new, pd.DataFrame)
                    else np.asarray(X_new, dtype=np.float64)
                )
                return np.asarray(tree.predict(arr), dtype=np.int64).reshape(-1)

            try:
                n_leaves = int(tree.get_n_leaves())
            except Exception:
                n_leaves = -1
            return PolicyTreeResult(
                backend="econml",
                max_depth=int(max_depth),
                n_leaves=n_leaves,
                feature_names=feature_names,
                tree=tree,
                predict_fn=_predict,
            )
        except Exception as exc:  # pragma: no cover - econml runtime failure
            return PolicyTreeResult(
                backend="econml",
                max_depth=int(max_depth),
                n_leaves=0,
                feature_names=feature_names,
                tree=None,
                predict_fn=None,
                notes=[f"econml PolicyTree fit failed: {exc!r}"],
            )

    # ---- R policytree fallback ----------------------------------------
    try:  # pragma: no cover - exercised only when rpy2+policytree installed
        import rpy2.robjects as ro  # type: ignore[import-not-found]
        from rpy2.robjects import numpy2ri, pandas2ri  # type: ignore[import-not-found]
        from rpy2.robjects.packages import importr  # type: ignore[import-not-found]

        numpy2ri.activate()
        pandas2ri.activate()
        policytree_r = importr("policytree")
        r_tree = policytree_r.policy_tree(X_arr, reward, depth=int(max_depth))
        return PolicyTreeResult(
            backend="policytree-r",
            max_depth=int(max_depth),
            n_leaves=-1,
            feature_names=feature_names,
            tree=r_tree,
            predict_fn=None,
            notes=["R policytree fitted; predict_fn requires rpy2 call-site"],
        )
    except Exception:
        return None


__all__ = [
    "UpliftCurve",
    "PolicyTreeResult",
    "TargetingReport",
    "build_targeting_report",
    "policy_tree",
]
