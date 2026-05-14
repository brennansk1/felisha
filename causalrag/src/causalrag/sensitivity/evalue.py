"""E-value — VanderWeele & Ding 2017 sensitivity to unmeasured confounding.

The E-value answers a single question: *how strong would an unmeasured
confounder have to be — on both the treatment and outcome — to fully
explain away the observed effect?*

It has a closed form for several effect-size scales. Because no single scale
is universally agreed-best, we expose the user's choice via ``scale=`` with
sensible auto-defaults:

- ``risk_ratio`` (binary outcome): canonical VanderWeele-Ding form. The
  caller must pass a risk ratio (RR), not a risk difference.
- ``odds_ratio`` (binary outcome): VanderWeele-Ding approximation
  ``e = max(OR^0.5, 1)`` valid for rare outcomes (~<15%). The caller must
  pass an odds ratio, not a log-odds and not a risk difference.
- ``hazard_ratio`` (survival outcome): same closed form as risk_ratio for
  hazard ratios > 1; mirrored for hazard ratios < 1. The caller must pass
  a hazard ratio (HR), not a log-HR.
- ``risk_difference`` (binary outcome): caller passes a risk difference
  (e.g. from LinearDML on a 0/1 outcome). The function converts internally
  to an RR using the baseline outcome risk ``p0``: ``RR = (p0 + rd)/p0``
  (Ding & VanderWeele 2017). ``baseline_risk`` is required; without it we
  cannot recover a scale and return ``verdict='unknown'``.
- ``standardized`` (continuous outcome): VanderWeele-Ding 2019 standardized-
  effect form ``RR_approx ≈ exp(0.91 * d)`` where ``d`` is Cohen's d —
  i.e. the input is a *standardized mean difference*, NOT a raw mean
  difference. If the caller hands us an implausible ``|d| > 5`` we refuse
  to compute and return ``verdict='unknown'`` rather than silently clamp.

Backwards compatibility: ``evalue(point_estimate, scale=...)`` keeps its
v0 signature; only the internal handling of ``standardized`` changed (no
more silent clamp) and new scales/kwargs are additive.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.result import EstimationResult

EvalueScale = Literal[
    "risk_ratio",
    "odds_ratio",
    "hazard_ratio",
    "standardized",
    "risk_difference",
]

# Magnitude beyond which a "standardized mean difference" is implausible and
# almost certainly indicates the caller handed us a value on the wrong scale
# (e.g. a raw mean difference or log-odds). Returning a wildly large e-value
# here would mislead the verdict aggregator into a falsely-robust call.
_STANDARDIZED_MAX_ABS = 5.0


class EValueResult(BaseModel):
    """Single E-value computation result."""

    model_config = ConfigDict(extra="forbid")

    scale: EvalueScale
    point_estimate: float
    ci_low: float | None = None
    ci_high: float | None = None
    e_value: float
    e_value_ci: float | None = Field(
        default=None,
        description="E-value applied to the CI bound closer to the null — the "
        "more conservative interpretation that some methodologists prefer.",
    )
    verdict: str
    reason: str | None = Field(
        default=None,
        description="If the verdict is 'unknown' (we refused to compute), the "
        "diagnostic explaining why — bad scale, missing baseline, implausible "
        "magnitude, etc.",
    )


def _evalue_rr(rr: float) -> float:
    rr = max(rr, 1 / rr)  # mirror around 1
    return float(rr + math.sqrt(rr * (rr - 1)))


def _unknown(
    *,
    scale: EvalueScale,
    point_estimate: float,
    ci_low: float | None,
    ci_high: float | None,
    reason: str,
) -> EValueResult:
    """Build a result that says 'we cannot compute an E-value for this input'.

    By convention we set ``e_value=1.0`` (the null) so any downstream colorer
    that does not know about the ``reason`` field will err on the pessimistic
    side rather than report a falsely-confident "robust" verdict.
    """
    return EValueResult(
        scale=scale,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        e_value=1.0,
        e_value_ci=None,
        verdict=f"Unknown — {reason}",
        reason=reason,
    )


def evalue(
    point_estimate: float,
    *,
    scale: EvalueScale = "risk_ratio",
    ci_low: float | None = None,
    ci_high: float | None = None,
    outcome_prevalence: float | None = None,
    baseline_risk: float | None = None,
) -> EValueResult:
    """Compute the E-value for an observed effect.

    Parameters
    ----------
    point_estimate:
        The effect on the scale named by ``scale``. **The unit must match**
        — e.g. if ``scale='odds_ratio'`` pass an OR, not a log-OR or a risk
        difference. Mismatched scales are the single most common silent
        failure mode for this function.
    scale:
        How to interpret ``point_estimate``. See module docstring.
    ci_low, ci_high:
        Lower / upper CI bounds on the same scale as ``point_estimate``.
    outcome_prevalence:
        Optional, only used for ``scale='odds_ratio'`` when the outcome is
        non-rare (>15%); enables the VanderWeele-Vansteelandt OR→RR
        conversion.
    baseline_risk:
        Required for ``scale='risk_difference'``. The baseline outcome
        probability under control; together with the risk difference this
        recovers the risk ratio ``RR = (p0 + rd)/p0``.

    Returns
    -------
    EValueResult — including a ``verdict='Unknown — ...'`` and ``reason``
    when we refused to compute (e.g. implausible standardized magnitude or
    missing ``baseline_risk`` for a risk-difference input). Callers can
    detect refusal via ``result.reason is not None``.
    """
    if scale == "standardized":
        # VanderWeele & Ding 2019: RR_approx ≈ exp(0.91 * d), where d is a
        # standardized mean difference (Cohen's d). Anything beyond ~|d|=5
        # is implausible for a well-specified effect and almost always means
        # the caller passed a raw (un-standardized) mean difference. Refuse
        # rather than silently clamp — a clamp would produce e^(0.91 * 10) ≈
        # 9000 and a falsely "robust" verdict.
        if abs(point_estimate) > _STANDARDIZED_MAX_ABS:
            return _unknown(
                scale=scale,
                point_estimate=point_estimate,
                ci_low=ci_low,
                ci_high=ci_high,
                reason=(
                    f"|standardized mean difference| = {abs(point_estimate):.2f} "
                    f"exceeds plausible bound {_STANDARDIZED_MAX_ABS}; the input "
                    "is probably on the wrong scale (raw mean diff, log-odds, "
                    "etc.) rather than Cohen's d. Pre-standardize before "
                    "calling, or pick a different scale."
                ),
            )

        rr = math.exp(0.91 * point_estimate)
        rr_low = math.exp(0.91 * ci_low) if ci_low is not None else None
        rr_high = math.exp(0.91 * ci_high) if ci_high is not None else None
    elif scale == "risk_difference":
        if baseline_risk is None or not (0.0 < baseline_risk < 1.0):
            return _unknown(
                scale=scale,
                point_estimate=point_estimate,
                ci_low=ci_low,
                ci_high=ci_high,
                reason=(
                    "risk_difference scale requires baseline_risk in (0, 1) to "
                    "convert RD → RR via (p0 + rd)/p0 (Ding & VanderWeele "
                    "2017). None supplied — refusing to fabricate one."
                ),
            )
        p0 = baseline_risk

        def _rd_to_rr(rd: float) -> float | None:
            p1 = p0 + rd
            if p1 <= 0.0 or p1 >= 1.0 or p0 <= 0.0:
                return None
            return p1 / p0

        rr_maybe = _rd_to_rr(point_estimate)
        if rr_maybe is None:
            return _unknown(
                scale=scale,
                point_estimate=point_estimate,
                ci_low=ci_low,
                ci_high=ci_high,
                reason=(
                    f"risk_difference {point_estimate:+.3f} combined with "
                    f"baseline_risk={p0} produces an implied p1 outside (0,1); "
                    "either the RD or the baseline is inconsistent."
                ),
            )
        rr = rr_maybe
        rr_low = _rd_to_rr(ci_low) if ci_low is not None else None
        rr_high = _rd_to_rr(ci_high) if ci_high is not None else None
    elif scale == "odds_ratio":
        if outcome_prevalence is not None and outcome_prevalence > 0.15:
            # VanderWeele-Vansteelandt approximation
            denom = 1 - outcome_prevalence + outcome_prevalence * point_estimate
            rr = point_estimate / denom
            rr_low = ci_low / (1 - outcome_prevalence + outcome_prevalence * ci_low) if ci_low else None
            rr_high = ci_high / (1 - outcome_prevalence + outcome_prevalence * ci_high) if ci_high else None
        else:
            rr = math.sqrt(point_estimate)
            rr_low = math.sqrt(ci_low) if ci_low is not None and ci_low > 0 else None
            rr_high = math.sqrt(ci_high) if ci_high is not None and ci_high > 0 else None
    else:  # risk_ratio, hazard_ratio
        rr, rr_low, rr_high = point_estimate, ci_low, ci_high

    e_main = _evalue_rr(rr)
    # Apply the E-value to the CI bound that is closer to the null (1) — the
    # conservative "confounding to nullify even the optimistic boundary" view.
    bounds = [b for b in (rr_low, rr_high) if b is not None]
    e_ci: float | None
    if bounds:
        closer = min(bounds, key=lambda b: abs(math.log(max(b, 1e-9))))
        if (rr > 1 and closer <= 1) or (rr < 1 and closer >= 1):
            # CI crosses the null — no meaningful E-value for the CI side.
            e_ci = 1.0
        else:
            e_ci = _evalue_rr(closer)
    else:
        e_ci = None

    verdict = _verdict(e_main, e_ci)
    return EValueResult(
        scale=scale,
        point_estimate=point_estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        e_value=e_main,
        e_value_ci=e_ci,
        verdict=verdict,
    )


def _verdict(e_main: float, e_ci: float | None) -> str:
    if e_ci is not None and e_ci <= 1.25:
        return "Effect is fragile — a weak unmeasured confounder could nullify it."
    if e_main >= 4.0:
        return "Robust — only a very strong unmeasured confounder could overturn the finding."
    if e_main >= 2.0:
        return "Moderately robust — a moderately strong unmeasured confounder would be required."
    return "Sensitive — a modest unmeasured confounder could explain the result."


# ---------------------------------------------------------------------------
# Estimator-aware dispatcher
# ---------------------------------------------------------------------------

# Estimator id prefixes that return effects already on a *log* scale (e.g.
# log-hazard or log-RMST contrast) — these are typically handled by the
# survival branch and should be exponentiated before being passed to the
# hazard_ratio E-value branch.
_SURVIVAL_PREFIXES = (
    "rbridge.grf.causal_survival_forest",
    "rbridge.survrm2",
)

# Estimator ids whose ``point_estimate`` is a *risk difference* on a binary
# outcome. These come from linear / partialing-out estimators applied to a
# 0/1 outcome (LinearDML, plain OLS, simple meta-learners, etc.) where the
# regression coefficient is interpretable as a difference in probabilities.
_RISK_DIFFERENCE_ESTIMATORS = (
    "python.dml.linear",
    "python.ols",
    "python.metalearner.s",
    "python.metalearner.t",
    "python.metalearner.x",
    "rbridge.grf.causal_forest",  # CATE on probability scale for binary Y
)

# Estimator ids that emit log-odds on a binary outcome.
_LOG_ODDS_ESTIMATORS = (
    "python.bart",
    "rbridge.bart",
)


def evalue_for_estimator(
    result: EstimationResult,
    *,
    outcome_dtype: str,
    baseline_risk: float | None = None,
) -> EValueResult:
    """Pick the right E-value scale from estimator metadata, then compute.

    The point of this helper is to keep the scale-selection burden out of
    ``master_loop.py``. The decision is driven by three things:

    1. ``result.estimator_id`` — tells us *what kind of number*
       ``point_estimate`` is on. A coefficient from LinearDML on a 0/1
       outcome is a risk difference; a coefficient from a Cox-style survival
       model is a log-hazard; a BART posterior mean on a binary outcome is
       on the log-odds scale.
    2. ``outcome_dtype`` — caller-provided. Allowed values:
       ``"binary"``, ``"continuous"``, ``"survival"``.
    3. ``baseline_risk`` — required when we end up needing the
       risk-difference branch.

    Returns
    -------
    EValueResult. If no sensible scale can be chosen, returns an unknown
    verdict (``result.reason is not None``) rather than guessing.
    """
    eid = result.estimator_id
    point = result.point_estimate
    ci_low = result.ci_low
    ci_high = result.ci_high

    # 1. Survival estimators — convert log-hazard / log-RMST contrast to a
    #    hazard ratio (or RMST ratio) and route to the hazard_ratio branch.
    if any(eid.startswith(p) for p in _SURVIVAL_PREFIXES):
        try:
            hr = math.exp(point)
            hr_low = math.exp(ci_low) if ci_low is not None else None
            hr_high = math.exp(ci_high) if ci_high is not None else None
        except (OverflowError, ValueError):
            return _unknown(
                scale="hazard_ratio",
                point_estimate=point,
                ci_low=ci_low,
                ci_high=ci_high,
                reason=(
                    f"survival estimator {eid} produced log-hazard {point} "
                    "that overflows when exponentiated."
                ),
            )
        return evalue(hr, scale="hazard_ratio", ci_low=hr_low, ci_high=hr_high)

    # 2. Binary outcome routing
    if outcome_dtype == "binary":
        if any(eid.startswith(p) for p in _LOG_ODDS_ESTIMATORS):
            try:
                or_ = math.exp(point)
                or_low = math.exp(ci_low) if ci_low is not None else None
                or_high = math.exp(ci_high) if ci_high is not None else None
            except (OverflowError, ValueError):
                return _unknown(
                    scale="odds_ratio",
                    point_estimate=point,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    reason=(
                        f"log-odds estimator {eid} produced {point} that "
                        "overflows when exponentiated."
                    ),
                )
            return evalue(or_, scale="odds_ratio", ci_low=or_low, ci_high=or_high)

        if any(eid.startswith(p) for p in _RISK_DIFFERENCE_ESTIMATORS):
            return evalue(
                point,
                scale="risk_difference",
                ci_low=ci_low,
                ci_high=ci_high,
                baseline_risk=baseline_risk,
            )

        # Binary outcome but estimator id unknown to us — refuse rather than
        # guess. The wrong guess silently mis-scales the E-value.
        return _unknown(
            scale="risk_difference",
            point_estimate=point,
            ci_low=ci_low,
            ci_high=ci_high,
            reason=(
                f"estimator id {eid!r} is not registered as risk-difference, "
                "log-odds, or survival; cannot infer the scale of its "
                "point estimate for a binary outcome."
            ),
        )

    # 3. Survival outcome but the estimator wasn't matched above — refuse.
    if outcome_dtype == "survival":
        return _unknown(
            scale="hazard_ratio",
            point_estimate=point,
            ci_low=ci_low,
            ci_high=ci_high,
            reason=(
                f"estimator id {eid!r} is not a registered survival "
                "estimator; refusing to assume hazard-ratio scale."
            ),
        )

    # 4. Continuous outcome — the caller is responsible for pre-standardizing
    #    (we cannot, since we don't see the data here). If the magnitude is
    #    implausible the `evalue()` standardized branch will return unknown.
    if outcome_dtype == "continuous":
        return evalue(point, scale="standardized", ci_low=ci_low, ci_high=ci_high)

    return _unknown(
        scale="standardized",
        point_estimate=point,
        ci_low=ci_low,
        ci_high=ci_high,
        reason=f"unrecognized outcome_dtype={outcome_dtype!r}.",
    )


__all__ = ["EvalueScale", "EValueResult", "evalue", "evalue_for_estimator"]
