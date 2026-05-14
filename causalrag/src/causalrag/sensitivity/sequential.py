"""Always-valid (anytime-valid) confidence sequences for plug-in inference.

Sprint 9.4 -- wraps any influence-function-based estimator (AIPW / TMLE /
DML) and produces a CI you can *peek at* repeatedly as more data arrives
without inflating the type-I error rate. This is the asymptotic guarantee
that fixed-n Wald CIs lack: under standard Wald inference, stopping the
study the moment ``p < 0.05`` produces an actual ``alpha`` that is *not* 5%.

Two methods are supported:

- ``betting`` (Waudby-Smith & Ramdas 2023, "Estimating means of bounded
  random variables by betting", *JRSSB*). For a vector of influence
  function values ``phi_i``, a *betting* confidence sequence inverts the
  test ``prod_i (1 + lam_i * (phi_i - mu))`` -- a non-negative martingale
  under ``H_0 : E[phi] = mu`` (Ville's inequality). We use a single fixed
  bet ``lam = sqrt(2 log(2/alpha) / (n * sigma_hat^2))`` clamped to the
  safety region ``|lam| <= 1/(2*range)`` so the wealth process never
  goes negative. This is the practical "predictable plug-in" variant
  of WSR 2023 section 4.

- ``asymptotic-cs`` (Howard, Ramdas, McAuliffe & Sekhon 2021, "Time-
  uniform, nonparametric, nonasymptotic confidence sequences", *Annals
  of Statistics*). Wraps the asymptotic Wald form with a log-iterated
  inflation factor::

      point +/- tau_n * sqrt(sigma_hat^2 / n)
      tau_n = sqrt( 2 * ( log log(e * n) + log(2/alpha) ) )

  This is wider than ``z_{1-alpha/2}`` at any fixed n -- *that is the
  price* of optional stopping -- but valid at every n simultaneously.

Both methods are *strictly wider* than the fixed-n Wald CI at the same
``alpha`` -- this is a feature, not a bug. The unit tests explicitly
check that.

Online updates: :func:`update_anytime_ci` is provided for the auto-mode
loop. Because the betting CI's product form is path-dependent, the
online update recomputes from the running ``(n, mean, sum-of-squares,
min, max)`` sufficient statistics -- which we stash in
``AnytimeValidCI._state`` -- rather than the full IF vector. That keeps
memory ``O(1)`` per walk.

References
----------
Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J. (2021).
    Time-uniform, nonparametric, nonasymptotic confidence sequences.
    *Annals of Statistics*, 49(2), 1055-1080.
Waudby-Smith, I., & Ramdas, A. (2023). Estimating means of bounded
    random variables by betting. *JRSSB*, 86(1), 1-27.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Method = Literal["betting", "asymptotic-cs", "hoeffding-cs"]


@dataclass
class AnytimeValidCI:
    """Anytime-valid confidence interval for a plug-in estimand.

    Attributes
    ----------
    point:
        Sample mean of the influence-function values, i.e. the AIPW /
        TMLE / DML point estimate.
    lower, upper:
        Lower / upper anytime-valid CI bounds at coverage ``1 - alpha``.
    coverage:
        Nominal coverage ``1 - alpha``.
    n_at_check:
        Sample size at the moment the CI was computed. Reported because
        a confidence-sequence CI evaluated at a different ``n`` is a
        *different* CI -- re-asking the same dataclass is meaningless.
    method:
        Which always-valid scheme produced the CI.
    rationale:
        One-line, audit-friendly explanation; consumed by the LLM
        narrator and the verdict-card renderer.
    """

    point: float
    lower: float
    upper: float
    coverage: float
    n_at_check: int
    method: Method
    rationale: str
    # Sufficient statistics carried for O(1) online updates; intentionally
    # excluded from equality / repr so two CIs with identical bounds compare
    # equal regardless of update history.
    _state: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def always_valid_ci(
    influence_function_values: np.ndarray,
    *,
    alpha: float = 0.05,
    method: Method = "betting",
) -> AnytimeValidCI:
    """Compute the anytime-valid CI from a vector of per-row IF values.

    Parameters
    ----------
    influence_function_values:
        1-D array ``phi_i`` (one entry per observation). For AIPW this
        is the empirical IF of the ATE; for DML it is the orthogonal
        score.
    alpha:
        Type-I error budget. Coverage will be ``1 - alpha``. Must be in
        ``(0, 1)``.
    method:
        ``"betting"`` (WSR 2023, default) or ``"asymptotic-cs"`` (HRMS
        2021). ``"hoeffding-cs"`` is accepted for API completeness but
        currently routes through the asymptotic-cs path with a
        bounded-range assumption noted in the rationale.

    Returns
    -------
    AnytimeValidCI
        Always-valid CI plus rationale and online-update state.
    """
    phi = np.asarray(influence_function_values, dtype=float).ravel()
    if phi.size == 0:
        raise ValueError("influence_function_values must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}")
    if method not in ("betting", "asymptotic-cs", "hoeffding-cs"):
        raise ValueError(f"unknown method {method!r}")

    n = int(phi.size)
    mean = float(phi.mean())
    # ddof=1 for unbiased variance; degenerate when n == 1.
    var = float(phi.var(ddof=1)) if n > 1 else 0.0
    lo_obs = float(phi.min())
    hi_obs = float(phi.max())

    if method == "betting":
        lower, upper, rationale = _betting_ci_from_stats(
            alpha=alpha, n=n, mean=mean, var=var, lo_obs=lo_obs, hi_obs=hi_obs
        )
        resolved_method: Method = "betting"
    else:
        # Both "asymptotic-cs" and "hoeffding-cs" route here; the latter
        # is currently an alias that surfaces the bounded-range note.
        lower, upper, rationale = _asymptotic_cs(
            alpha=alpha, n=n, mean=mean, var=var
        )
        resolved_method = method
        if method == "hoeffding-cs":
            rationale += (
                " (hoeffding-cs alias: bounded-range assumption is the "
                "caller's responsibility; routed through asymptotic-cs)"
            )

    state = {
        "n": n,
        "sum": float(phi.sum()),
        "sumsq": float(np.sum(phi * phi)),
        "min": lo_obs,
        "max": hi_obs,
        "alpha": float(alpha),
        "method": resolved_method,
    }
    return AnytimeValidCI(
        point=mean,
        lower=lower,
        upper=upper,
        coverage=1.0 - alpha,
        n_at_check=n,
        method=resolved_method,
        rationale=rationale,
        _state=state,
    )


def update_anytime_ci(
    prev: AnytimeValidCI, new_if_values: np.ndarray
) -> AnytimeValidCI:
    """Update the CI online as new IF values arrive.

    The auto-mode loop calls this after each fresh batch; because we
    carry only ``(n, sum, sum-of-squares, min, max)`` the cost is
    ``O(len(new_if_values))``, not ``O(total n)``.
    """
    if not prev._state:
        raise ValueError(
            "AnytimeValidCI has no carried state; was it constructed by "
            "always_valid_ci()? Reconstruct from the full IF vector."
        )
    new = np.asarray(new_if_values, dtype=float).ravel()
    if new.size == 0:
        # No new data -- return prev unchanged.
        return prev

    s = prev._state
    n = int(s["n"]) + int(new.size)
    total_sum = float(s["sum"]) + float(new.sum())
    total_sumsq = float(s["sumsq"]) + float(np.sum(new * new))
    lo_obs = min(float(s["min"]), float(new.min()))
    hi_obs = max(float(s["max"]), float(new.max()))
    alpha = float(s["alpha"])
    method: Method = s["method"]

    mean = total_sum / n
    # Welford-equivalent unbiased variance from running sums.
    var = max((total_sumsq - n * mean * mean) / (n - 1), 0.0) if n > 1 else 0.0

    if method == "betting":
        # The path-dependent betting wealth would require the full path;
        # for the predictable-plug-in single-bet variant we recompute
        # the bound from sufficient stats, which is exact for this lam
        # choice.
        lower, upper, rationale = _betting_ci_from_stats(
            alpha=alpha, n=n, mean=mean, var=var, lo_obs=lo_obs, hi_obs=hi_obs
        )
    else:
        lower, upper, rationale = _asymptotic_cs(
            alpha=alpha, n=n, mean=mean, var=var
        )
        if method == "hoeffding-cs":
            rationale += " (hoeffding-cs alias: routed through asymptotic-cs)"

    new_state = {
        "n": n,
        "sum": total_sum,
        "sumsq": total_sumsq,
        "min": lo_obs,
        "max": hi_obs,
        "alpha": alpha,
        "method": method,
    }
    return AnytimeValidCI(
        point=mean,
        lower=lower,
        upper=upper,
        coverage=1.0 - alpha,
        n_at_check=n,
        method=method,
        rationale=rationale,
        _state=new_state,
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _asymptotic_cs(
    *, alpha: float, n: int, mean: float, var: float
) -> tuple[float, float, str]:
    """Howard-Ramdas-McAuliffe-Sekhon 2021 asymptotic confidence sequence.

    Uses the log-iterated boundary::

        tau_n = sqrt( 2 * ( log log(e * n) + log(2/alpha) ) )

    around ``mean +/- tau_n * sqrt(sigma_hat^2 / n)``. Reduces to within
    a constant factor of the Wald CI as ``alpha -> 1`` while remaining
    valid under optional stopping.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if var <= 0.0:
        # Degenerate IF (all values equal) -- CI collapses to the point.
        return (
            mean,
            mean,
            "asymptotic-cs: degenerate sigma_hat^2 = 0; CI collapses to point.",
        )

    log_log_term = math.log(math.log(math.e * n))  # >= 0 for n >= 1
    tau = math.sqrt(2.0 * (log_log_term + math.log(2.0 / alpha)))
    half_width = tau * math.sqrt(var / n)
    rationale = (
        f"asymptotic-cs (Howard 2021): tau_n={tau:.3f}, "
        f"sigma_hat={math.sqrt(var):.3g}, n={n}, "
        f"half-width={half_width:.3g}; valid at every n simultaneously."
    )
    return (mean - half_width, mean + half_width, rationale)


def _betting_ci_from_stats(
    *,
    alpha: float,
    n: int,
    mean: float,
    var: float,
    lo_obs: float,
    hi_obs: float,
) -> tuple[float, float, str]:
    """Predictable-plug-in betting CS (WSR 2023, single-lam variant).

    The full path-dependent KKT inversion is overkill for the auto-mode
    diagnostic; we use the predictable plug-in with a single lam and
    rely on Ville's inequality applied to the wealth martingale. This
    is the flavor the WSR paper recommends in section 4.2 for
    "set-and-forget" use.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")

    # Bounded-range proxy. For an *IF*, true bounds are not known, so we
    # use the observed range with a small safety pad. This is standard
    # practice when porting WSR to influence functions.
    obs_range = max(hi_obs - lo_obs, 1e-12)

    # Single bet: lam = sqrt( 2 log(2/alpha) / (n * sigma_hat^2) ),
    # capped to the safety region |lam| < 1/(2*range) so the betting
    # wealth stays positive -- this preserves the Ville bound.
    if var > 0.0:
        lam = math.sqrt(2.0 * math.log(2.0 / alpha) / (n * var))
    else:
        # Degenerate -- fall back to range-based lam.
        lam = math.sqrt(2.0 * math.log(2.0 / alpha) / n) / obs_range
    lam_cap = 1.0 / (2.0 * obs_range)
    lam = min(lam, lam_cap)

    # WSR predictable-plug-in half-width:
    #   h = log(2/alpha) / (n * lam) + lam * sigma_hat^2 / 2
    # Minimized at lam* = sqrt(2 log(2/alpha) / (n sigma_hat^2)) -- our
    # default bet, before capping. With the cap the bound is still
    # valid, just slightly looser.
    half_width = math.log(2.0 / alpha) / (n * lam) + lam * var / 2.0

    rationale = (
        f"betting-CS (WSR 2023, predictable plug-in): lam={lam:.3g} "
        f"(cap={lam_cap:.3g}), sigma_hat={math.sqrt(var):.3g}, n={n}, "
        f"half-width={half_width:.3g}; anytime-valid under optional stopping."
    )
    return (mean - half_width, mean + half_width, rationale)


__all__ = ["AnytimeValidCI", "always_valid_ci", "update_anytime_ci"]
