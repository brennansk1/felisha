"""Expected Information Gain (EIG) chain-continuation scoring.

Implements Sprints 3.1 and 3.6 of PDD §33:

* :func:`expected_information_gain` — Lindley-style EIG (1956 /
  Chaloner-Verdinelli 1995) under a Gaussian-approximate posterior
  on the treatment effect τ.
* :func:`saturation_probability` — Monte-Carlo posterior probability
  that the next chain step would shrink the credible interval by less
  than a relative threshold ε.
* :func:`should_continue_chain_eig` — combined stopping rule that
  retires a chain when *both* signals say further runs are unlikely
  to be informative.

All functions are pure: no LLM, no R bridge, no I/O. They take the
chain state plus anticipated next-step parameters and return numbers
plus a continuation decision. They replace the legacy
``info_gain_streak_below_eps`` heuristic in
:class:`causalrag.master_loop.ChainState` — but never edit it.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

__all__ = [
    "ChainStateLike",
    "expected_information_gain",
    "saturation_probability",
    "should_continue_chain_eig",
]


class ChainStateLike(Protocol):
    """Minimal structural protocol matching :class:`master_loop.ChainState`.

    We only need the most recent point estimate and SE to score the
    chain — anything else is the master loop's business.
    """

    last_point: float | None
    last_se: float | None


# ─────────── Expected Information Gain ───────────────────────────────────


def expected_information_gain(
    *,
    current_point: float,
    current_se: float,
    anticipated_se: float,
) -> float:
    """Lindley-style EIG (Chaloner & Verdinelli 1995) for a Gaussian
    posterior on τ.

    Under a Gaussian-approximate posterior with current variance
    ``σ² = current_se²`` and an anticipated next-run likelihood with
    variance ``s² = anticipated_se²``, the posterior after a
    conjugate Gaussian update has variance ``σ² · s² / (σ² + s²)``.
    The expected reduction in differential entropy is therefore

    .. math::

        \\mathrm{EIG} = \\tfrac{1}{2}\\log\\frac{\\sigma^2}
                                                {\\sigma^2 s^2/(\\sigma^2+s^2)}
                     = \\tfrac{1}{2}\\log\\!\\left(1 + \\sigma^2/s^2\\right).

    Parameters
    ----------
    current_point:
        Current point estimate (unused in the variance-only EIG, but
        accepted to keep the call signature stable for future
        extensions that condition on effect size).
    current_se:
        Current posterior standard error σ.
    anticipated_se:
        Anticipated next-run standard error s. Callers typically pass
        the parent's SE as a conservative anchor unless they have a
        learned shrinkage estimate.

    Returns
    -------
    float
        Expected nats of entropy reduction. Always non-negative.
        Returns ``0.0`` if either SE is non-finite, non-positive, or
        the anticipated step is degenerate.
    """
    del current_point  # currently unused; documented above

    if (
        not math.isfinite(current_se)
        or not math.isfinite(anticipated_se)
        or current_se <= 0.0
    ):
        return 0.0
    # anticipated_se → ∞ ⇒ no information ⇒ EIG → 0.
    if anticipated_se <= 0.0:
        # A perfectly-informative next step (s=0) gives infinite EIG;
        # clamp to a large finite value so downstream comparisons are
        # well-defined. In practice s=0 never occurs.
        return float("inf")

    ratio = (current_se / anticipated_se) ** 2
    return 0.5 * math.log1p(ratio)


# ─────────── Bayesian saturation probability ─────────────────────────────


def saturation_probability(
    *,
    current_point: float,
    current_se: float,
    epsilon_ci_width: float,
    n_simulations: int = 1000,
    seed: int = 42,
) -> float:
    """Posterior probability the next chain step shrinks the CI by < ε.

    We simulate the next step's SE under a conservative half-normal
    prior centered on ``current_se`` (scale = ``current_se``); the
    posited next CI width is ``2 · 1.96 · s_next`` and the current CI
    width is ``2 · 1.96 · current_se``. The *relative* shrinkage is

    .. math::

        \\Delta\\mathrm{CI} = 1 - s_{\\text{next}} / s_{\\text{current}}.

    Parameters
    ----------
    current_point:
        Current point estimate (accepted for API stability; not used
        in the relative-shrinkage calculation).
    current_se:
        Current posterior standard error σ.
    epsilon_ci_width:
        Minimum *relative* CI shrinkage the user considers
        informative; e.g. ``0.10`` means "at least a 10 % narrower
        CI."
    n_simulations:
        Monte-Carlo draws of the next-step SE.
    seed:
        Deterministic seed for reproducibility.

    Returns
    -------
    float
        ``P(ΔCI < epsilon_ci_width)`` in [0, 1]. High values mean the
        chain has saturated and additional runs are unlikely to add
        precision.
    """
    del current_point  # currently unused

    if not math.isfinite(current_se) or current_se <= 0.0:
        # No posterior to shrink; the chain is trivially saturated.
        return 1.0
    if n_simulations <= 0:
        return 1.0
    if not math.isfinite(epsilon_ci_width):
        return 1.0

    rng = np.random.default_rng(seed)
    # Half-normal with scale = current_se: |N(0, current_se²)|.
    # Mean ≈ current_se · sqrt(2/π) ≈ 0.798 · current_se, encoding
    # "the next step rarely shrinks the SE by more than half."
    s_next = np.abs(rng.normal(loc=0.0, scale=current_se, size=n_simulations))
    relative_shrinkage = 1.0 - (s_next / current_se)
    # Negative shrinkages (the next step widens the CI) also count as
    # "below ε" — they're a fortiori uninformative.
    return float(np.mean(relative_shrinkage < epsilon_ci_width))


# ─────────── Combined continuation rule ──────────────────────────────────


def should_continue_chain_eig(
    *,
    chain_state: ChainStateLike,
    epsilon_eig: float = 0.05,
    epsilon_ci: float = 0.10,
    saturation_threshold: float = 0.9,
    anticipated_se: float | None = None,
) -> tuple[bool, str]:
    """Combined Sprint 3.1 + 3.6 continuation rule.

    Stops the chain when EIG falls below ``epsilon_eig`` **and** the
    saturation probability exceeds ``saturation_threshold``. Both
    signals must agree, so a chain is never retired on a single noisy
    measurement.

    Parameters
    ----------
    chain_state:
        Anything with ``last_point`` and ``last_se`` attributes —
        nominally :class:`causalrag.master_loop.ChainState`.
    epsilon_eig:
        Minimum EIG in nats to justify another step (Sprint 3.1).
    epsilon_ci:
        Minimum relative CI shrinkage that counts as informative
        (Sprint 3.6).
    saturation_threshold:
        Stop if ``P(ΔCI < epsilon_ci) > saturation_threshold``.
    anticipated_se:
        Anticipated next-run SE. Defaults to ``chain_state.last_se``
        (the Chaloner-Verdinelli conservative anchor: assume the next
        step is no better than the last).

    Returns
    -------
    (bool, str)
        ``(should_continue, reason)`` — the second element is a
        short human-readable explanation suitable for the
        observability stream.
    """
    last_point = chain_state.last_point
    last_se = chain_state.last_se

    if last_point is None or last_se is None:
        # No prior estimate — let the loop take its first swing.
        return True, "chain has no prior estimate; allow first step"

    if not math.isfinite(last_se) or last_se <= 0.0:
        # Degenerate SE: we can't score, so we conservatively allow
        # one more step rather than silently truncating the chain.
        return True, f"last_se={last_se!r} is non-positive; allow step"

    s_next = anticipated_se if anticipated_se is not None else last_se
    eig = expected_information_gain(
        current_point=last_point,
        current_se=last_se,
        anticipated_se=s_next,
    )
    sat_p = saturation_probability(
        current_point=last_point,
        current_se=last_se,
        epsilon_ci_width=epsilon_ci,
    )

    eig_below = eig < epsilon_eig
    saturated = sat_p > saturation_threshold

    if eig_below and saturated:
        return False, (
            f"chain saturated: EIG={eig:.4f} nats < {epsilon_eig} "
            f"and P(ΔCI<{epsilon_ci})={sat_p:.2f} > {saturation_threshold}"
        )

    if eig_below:
        return True, (
            f"EIG={eig:.4f} nats below {epsilon_eig} but saturation "
            f"P={sat_p:.2f} ≤ {saturation_threshold} — continue"
        )
    if saturated:
        return True, (
            f"saturation P={sat_p:.2f} above {saturation_threshold} "
            f"but EIG={eig:.4f} nats ≥ {epsilon_eig} — continue"
        )
    return True, (
        f"EIG={eig:.4f} nats ≥ {epsilon_eig}, "
        f"saturation P={sat_p:.2f} ≤ {saturation_threshold}"
    )
