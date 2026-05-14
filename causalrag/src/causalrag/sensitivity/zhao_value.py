"""Zhao 2019 (JASA) sensitivity value Γ — matched-pair causal inference.

Zhao, Small & Bhattacharya (2019, *JASA*) define the **sensitivity value**
as the smallest Γ ≥ 1 at which the matched-pair test of the sharp null no
longer rejects at level α under the worst-case Rosenbaum-style bias model.

Interpretation
--------------
Γ is the *bias factor*: a hidden confounder U would need to multiply the
odds of treatment between two matched units by Γ before our rejection of
the null can be explained away. So:

  - Γ = 1.0  → even a vanishingly small bias overturns the finding.
  - Γ = 2.0  → a confounder doubling the odds of treatment is required.
  - Γ = 5.0  → smoking-and-lung-cancer territory.

This is **only meaningful when the estimator path was matching**
(``rbridge.matchit``). For any other estimator we return
``verdict='unknown'`` with a rationale and do not compute.

Asymptotic normality (Zhao 2019, Thm 1) gives a CI for Γ itself, which is
the headline contribution of the paper relative to plain Rosenbaum bounds.
We implement the delta-method/normal-approximation CI by linearizing the
bisection target around the solved Γ.

Math
----
For matched pairs i = 1..N with treated-minus-control outcome differences
d_i, the Wilcoxon signed-rank statistic is

    W = Σ_i q_i · 1{d_i > 0}     where q_i = rank(|d_i|).

Under the Rosenbaum sensitivity model with parameter Γ, the worst-case
null distribution of W is approximately normal with

    E_max(Γ)   = (Γ / (1+Γ)) · Σ q_i
    Var_max(Γ) = (Γ / (1+Γ)^2) · Σ q_i^2

(The familiar Γ=1 case recovers E = ½ Σ q_i and Var = ¼ Σ q_i^2.)

The sensitivity value Γ* solves

    (W - E_max(Γ)) / sqrt(Var_max(Γ)) = z_{1-α}.

For Γ that small that the LHS at Γ=1 is already < z_{1-α} (i.e. we already
fail to reject without any bias), Γ* = 1 by convention and the verdict is
``red``.

Pure Python — no R bridge required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ZhaoVerdict = Literal["green", "yellow", "red", "unknown"]
ZhaoTestStat = Literal["t", "wilcoxon", "sign"]
ZhaoMethod = Literal["zhao_normal", "grid_search"]

_GAMMA_MIN = 1.0
_GAMMA_MAX = 10.0
_BISECTION_TOL = 1e-5
_BISECTION_MAX_ITER = 200


@dataclass
class ZhaoSensitivityValue:
    """Zhao 2019 sensitivity value Γ — the smallest Γ ≥ 1 at which the
    matched-pair test no longer rejects the null at level α.

    Larger Γ = more robust to hidden bias. Γ=1 means even one unit of
    hidden bias overturns the finding; Γ=2 means a confounder that
    doubles the odds of treatment between matched units would be needed;
    Γ ≥ 5 is "smoking causes lung cancer" territory.
    """

    gamma: float
    gamma_se: float | None
    gamma_ci_low: float | None
    gamma_ci_high: float | None
    n_matched_pairs: int
    alpha: float
    verdict: ZhaoVerdict
    rationale: str
    backend: str  # "python.zhao" or "rbridge.crossscreening"
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------


def _verdict_for_gamma(gamma: float) -> ZhaoVerdict:
    """Map Γ to a traffic-light verdict per the sprint spec.

    Γ = 1.0 exactly is "red" — no bias at all is required to overturn.
    Γ between 1.0 and 1.5 is also "red". 1.5–2.0 yellow. ≥2.0 green.
    """
    if gamma <= 1.0:
        return "red"
    if gamma < 1.5:
        return "red"
    if gamma < 2.0:
        return "yellow"
    return "green"


def _rationale_for_gamma(gamma: float, alpha: float) -> str:
    if gamma <= 1.0:
        return (
            f"Γ = 1.00: the matched-pair test at α={alpha} already fails (or only "
            "barely passes) under zero hidden bias; even a vanishingly small "
            "unmeasured confounder reverses the conclusion."
        )
    return (
        f"Γ = {gamma:.2f}: a hidden confounder would need to multiply the odds "
        f"of treatment between matched units by ≈{gamma:.2f} before the "
        f"matched-pair test at α={alpha} would fail to reject the null."
    )


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank machinery for the Γ-bias upper-bound
# ---------------------------------------------------------------------------


def _signed_rank_components(d: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (W, ranks_of_|d|, positivity_mask) for d_i ≠ 0.

    Zero differences are dropped (standard Wilcoxon convention). Ties in
    |d| get average ranks — same as ``scipy.stats.rankdata(..., 'average')``
    so we mirror that behavior without importing scipy.
    """
    d = np.asarray(d, dtype=float)
    nz = d[d != 0.0]
    if nz.size == 0:
        return 0.0, np.array([]), np.array([])
    abs_d = np.abs(nz)
    # average-rank for ties — match scipy.stats.rankdata default.
    order = np.argsort(abs_d, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_abs = abs_d[order]
    i = 0
    n = sorted_abs.size
    while i < n:
        j = i
        while j + 1 < n and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        avg = (i + j + 2) / 2.0  # ranks are 1-indexed
        ranks[order[i : j + 1]] = avg
        i = j + 1
    pos = (nz > 0).astype(float)
    w = float(np.sum(ranks * pos))
    return w, ranks, pos


def _bias_deviate(gamma: float, w: float, ranks: np.ndarray, *, upper: bool) -> float:
    """Standardized deviate of W under the Γ-worst-case null.

    With ``upper=True`` we use E_max = (Γ/(1+Γ)) · Σq_i (upper-tail tested)
    and the result is positive when W exceeds the bias-shifted expectation.
    With ``upper=False`` we mirror for a lower-tail test.
    """
    sum_q = float(np.sum(ranks))
    sum_q2 = float(np.sum(ranks**2))
    if sum_q2 <= 0.0:
        return 0.0
    p = gamma / (1.0 + gamma)
    if upper:
        mean = p * sum_q
    else:
        mean = (1.0 / (1.0 + gamma)) * sum_q
    var = (gamma / (1.0 + gamma) ** 2) * sum_q2
    if var <= 0.0:
        return 0.0
    return (w - mean) / math.sqrt(var)


def _normal_inv_cdf(p: float) -> float:
    """Inverse standard-normal CDF — Beasley-Springer-Moro approximation.

    Hand-rolled so we don't need scipy at runtime.
    """
    # Constants from Peter Acklam's algorithm — accurate to ~1e-9.
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    q = math.sqrt(-2 * math.log(1 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def _bisect_gamma(
    w: float,
    ranks: np.ndarray,
    z_alpha: float,
    *,
    upper: bool,
) -> tuple[float, list[str]]:
    """Find Γ* such that the Γ-bias deviate equals z_alpha.

    Returns (gamma_star, notes). If the deviate is already below z_alpha
    at Γ=1, we return Γ*=1 — the test does not reject even under no bias,
    so the sensitivity value is degenerate.
    """
    notes: list[str] = []
    f1 = _bias_deviate(_GAMMA_MIN, w, ranks, upper=upper) - z_alpha
    if f1 <= 0.0:
        notes.append(
            "Test does not reject at α even with Γ=1 (no hidden bias); "
            "returning Γ*=1 by convention."
        )
        return _GAMMA_MIN, notes

    f_hi = _bias_deviate(_GAMMA_MAX, w, ranks, upper=upper) - z_alpha
    if f_hi > 0.0:
        notes.append(
            f"Γ bisection bracket [1, {_GAMMA_MAX}] does not contain the root; "
            f"the matched-pair test is robust beyond Γ={_GAMMA_MAX}. Reporting "
            "the upper bracket as a lower bound on Γ*."
        )
        return _GAMMA_MAX, notes

    lo, hi = _GAMMA_MIN, _GAMMA_MAX
    for _ in range(_BISECTION_MAX_ITER):
        mid = 0.5 * (lo + hi)
        f_mid = _bias_deviate(mid, w, ranks, upper=upper) - z_alpha
        if abs(f_mid) < _BISECTION_TOL or (hi - lo) < _BISECTION_TOL:
            return mid, notes
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi), notes


def _gamma_se_delta_method(
    gamma_star: float,
    w: float,
    ranks: np.ndarray,
    *,
    upper: bool,
) -> float | None:
    """Asymptotic SE of Γ* via the delta method on the bisection target.

    Let g(Γ) = (W - E_max(Γ)) / sqrt(Var_max(Γ)). At the solution g(Γ*) = z.
    Locally W is approximately normal with mean E_max(Γ_true) and variance
    Var_max(Γ_true) under the worst-case null, so

        Var(g(Γ*)) ≈ 1   (W is on the standardized scale),

    and by the delta method

        SE(Γ*) ≈ 1 / |g'(Γ*)|.

    We compute g'(Γ*) by finite difference. Returns None if the derivative
    is too small to invert numerically.
    """
    if gamma_star <= _GAMMA_MIN or gamma_star >= _GAMMA_MAX:
        return None
    eps = max(1e-4, 1e-3 * gamma_star)
    g_hi = _bias_deviate(gamma_star + eps, w, ranks, upper=upper)
    g_lo = _bias_deviate(gamma_star - eps, w, ranks, upper=upper)
    deriv = (g_hi - g_lo) / (2 * eps)
    if not math.isfinite(deriv) or abs(deriv) < 1e-6:
        return None
    return 1.0 / abs(deriv)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def zhao_sensitivity_value(
    *,
    treated_outcomes: np.ndarray,
    matched_control_outcomes: np.ndarray,
    alpha: float = 0.05,
    test_statistic: ZhaoTestStat = "wilcoxon",
    method: ZhaoMethod = "zhao_normal",
) -> ZhaoSensitivityValue:
    """Compute the Zhao 2019 sensitivity value Γ for matched-pair data.

    Parameters
    ----------
    treated_outcomes, matched_control_outcomes:
        Same-length 1-D arrays of outcomes for treated units and their
        matched control units (pair i ↔ index i).
    alpha:
        Significance level for the matched-pair test. Default 0.05.
    test_statistic:
        ``"wilcoxon"`` (default) — Wilcoxon signed-rank, Zhao 2019's
        primary case. ``"t"`` and ``"sign"`` are accepted but routed
        through the same upper-bound machinery using either signed
        differences or signs of differences as the "rank" weights; this
        preserves the asymptotic-normal CI structure.
    method:
        ``"zhao_normal"`` (default) — bisection plus Zhao 2019 Thm 1
        normal-approximation CI. ``"grid_search"`` falls back to a fine
        grid over Γ ∈ [1, 10] (no CI).

    Returns
    -------
    ZhaoSensitivityValue with the bisected Γ*, its asymptotic SE/CI, and
    a verdict.

    Raises
    ------
    ValueError if the input arrays have mismatched lengths or are not 1-D.
    """
    t = np.asarray(treated_outcomes, dtype=float)
    c = np.asarray(matched_control_outcomes, dtype=float)
    if t.ndim != 1 or c.ndim != 1:
        raise ValueError(
            f"treated_outcomes and matched_control_outcomes must be 1-D arrays; "
            f"got shapes {t.shape} and {c.shape}."
        )
    if t.shape[0] != c.shape[0]:
        raise ValueError(
            f"treated_outcomes and matched_control_outcomes must have the same "
            f"length (matched-pair index); got {t.shape[0]} vs {c.shape[0]}."
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")

    d = t - c
    notes: list[str] = []

    # Decide which "rank" weights and statistic to use.
    if test_statistic == "wilcoxon":
        w, ranks, _pos = _signed_rank_components(d)
    elif test_statistic == "sign":
        nz = d[d != 0.0]
        ranks = np.ones_like(nz)
        w = float(np.sum(ranks * (nz > 0)))
        notes.append("Using sign test: ranks fixed at 1, statistic = #(d_i > 0).")
    elif test_statistic == "t":
        nz = d[d != 0.0]
        # Use |d_i| as continuous "rank" weights — preserves Zhao's
        # upper-bound structure while leaning on the magnitude of d.
        ranks = np.abs(nz)
        w = float(np.sum(ranks * (nz > 0)))
        notes.append(
            "Using |d_i|-weighted variant of the signed-rank upper bound."
        )
    else:
        raise ValueError(
            f"test_statistic must be 'wilcoxon', 'sign', or 't'; got {test_statistic!r}."
        )

    n_pairs = int(ranks.size)
    if n_pairs == 0:
        return ZhaoSensitivityValue(
            gamma=1.0,
            gamma_se=None,
            gamma_ci_low=None,
            gamma_ci_high=None,
            n_matched_pairs=0,
            alpha=alpha,
            verdict="red",
            rationale=(
                "All matched-pair differences are zero (or all pairs dropped); "
                "no information for a Zhao sensitivity value."
            ),
            backend="python.zhao",
            notes=["empty non-zero differences"],
        )

    # Determine direction: if the median difference is negative, mirror
    # to an upper-tail test so the bias-deviate is positive.
    upper = bool(np.sum(d > 0) >= np.sum(d < 0))
    if not upper:
        # Flip and treat the negative side as upper-tail by symmetry.
        w_flip, ranks_flip, _ = _signed_rank_components(-d)
        w = w_flip
        ranks = ranks_flip
        upper = True
        notes.append("Effect points downward; mirrored to upper-tail bias bound.")

    z_alpha = _normal_inv_cdf(1.0 - alpha)

    if method == "grid_search":
        grid = np.linspace(_GAMMA_MIN, _GAMMA_MAX, 2001)
        deviates = np.array(
            [_bias_deviate(g, w, ranks, upper=upper) for g in grid]
        )
        below = np.where(deviates <= z_alpha)[0]
        if below.size == 0:
            gamma_star = _GAMMA_MAX
            notes.append(
                f"Grid search did not find Γ ≤ {_GAMMA_MAX}; reporting upper bracket."
            )
        else:
            gamma_star = float(grid[below[0]])
        gamma_se = None
        gamma_ci_low = None
        gamma_ci_high = None
    else:
        gamma_star, bnotes = _bisect_gamma(w, ranks, z_alpha, upper=upper)
        notes.extend(bnotes)
        gamma_se = _gamma_se_delta_method(gamma_star, w, ranks, upper=upper)
        if gamma_se is not None and gamma_star > _GAMMA_MIN and gamma_star < _GAMMA_MAX:
            z_ci = _normal_inv_cdf(1.0 - alpha / 2.0)
            gamma_ci_low = max(_GAMMA_MIN, gamma_star - z_ci * gamma_se)
            gamma_ci_high = gamma_star + z_ci * gamma_se
        else:
            gamma_ci_low = None
            gamma_ci_high = None

    verdict = _verdict_for_gamma(gamma_star)
    rationale = _rationale_for_gamma(gamma_star, alpha)

    return ZhaoSensitivityValue(
        gamma=float(gamma_star),
        gamma_se=gamma_se,
        gamma_ci_low=gamma_ci_low,
        gamma_ci_high=gamma_ci_high,
        n_matched_pairs=n_pairs,
        alpha=alpha,
        verdict=verdict,
        rationale=rationale,
        backend="python.zhao",
        notes=notes,
    )


def zhao_sensitivity_value_unknown(
    *,
    estimator_id: str,
    alpha: float = 0.05,
) -> ZhaoSensitivityValue:
    """Build an ``unknown``-verdict result for non-matching estimator paths.

    Use this from the dashboard whenever the chosen estimator id is not
    ``rbridge.matchit``. The Zhao sensitivity value is only defined for the
    matched-pair design; reporting it for a DML/forest/etc. run would be
    methodologically wrong.
    """
    return ZhaoSensitivityValue(
        gamma=float("nan"),
        gamma_se=None,
        gamma_ci_low=None,
        gamma_ci_high=None,
        n_matched_pairs=0,
        alpha=alpha,
        verdict="unknown",
        rationale=(
            f"Zhao 2019 sensitivity value is only defined for matched-pair "
            f"designs; estimator path was {estimator_id!r}, not "
            "'rbridge.matchit'. Skipping computation."
        ),
        backend="python.zhao",
        notes=["non-matching estimator path"],
    )


__all__ = [
    "ZhaoSensitivityValue",
    "ZhaoVerdict",
    "ZhaoTestStat",
    "ZhaoMethod",
    "zhao_sensitivity_value",
    "zhao_sensitivity_value_unknown",
]
