"""Thompson sampling + UCB1 bandit allocation across chain roots.

Implements Sprint 3.2 of PDD §33: when the foundation-recursion loop
has multiple live chains, allocate the next experimental "pull" to the
chain whose expected payoff is most uncertain *and* most promising.

Payoff for a chain is defined as :math:`|\\mathrm{point}/\\mathrm{SE}|`
of that chain's most recent walk — a standardised effect size proxy
for "evidence quality." A chain that has produced a strong signal
(``|point/SE| ≫ 1``) will get pulled more often; a chain whose root
keeps coming back null will be pulled less but never zero.

The module is intentionally side-effect-free — it never mutates
:class:`causalrag.master_loop.ChainState`. It only reads two
attributes (``last_point``, ``last_se``) plus the chain id and a
"depth-like" pull count, treating its input as a structural
:class:`ChainStateLike`.

Two strategies are exposed:

* :func:`thompson_sample_chain` — Bayesian; draws once from each
  chain's posterior on ``|point/SE|`` and returns the argmax.
* :func:`ucb1_chain_choice` — deterministic Auer et al. (2002) UCB1
  for reproducible runs (e.g. CI).

Both return ``(chain_id, [BanditArm, ...])`` so the TUI / decision
ledger can show *why* the loop picked the chain it did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

__all__ = [
    "BanditArm",
    "ChainStateLike",
    "thompson_sample_chain",
    "ucb1_chain_choice",
]


class ChainStateLike(Protocol):
    """Minimal structural protocol matching :class:`master_loop.ChainState`.

    We need the chain identifier, the most recent point estimate and
    SE, and a "pull count" — depth is a good proxy because each
    completed walk increments it.
    """

    chain_id: str
    last_point: float | None
    last_se: float | None
    depth: int


# ─────────── Arm view (TUI / ledger surface) ─────────────────────────────


@dataclass
class BanditArm:
    """One chain's bandit posterior + summary, for the TUI / ledger."""

    chain_id: str
    posterior_mean: float
    posterior_variance: float
    n_pulls: int  # number of completed walks in this chain
    prior_used: str  # "uniform" / "user_specified" / "ucb_seeded"


# ─────────── Helpers ─────────────────────────────────────────────────────


def _safe_payoff(chain: ChainStateLike) -> float | None:
    """Return ``|point/SE|`` for the chain or ``None`` if undefined.

    Treats non-finite or non-positive SE as "no usable observation."
    A null-result chain (``point ≈ 0``) returns a small but finite
    value, which is exactly what we want — its posterior shrinks
    toward zero.
    """
    point = getattr(chain, "last_point", None)
    se = getattr(chain, "last_se", None)
    if point is None or se is None:
        return None
    if not (math.isfinite(point) and math.isfinite(se)):
        return None
    if se <= 0.0:
        return None
    return abs(point / se)


def _n_pulls(chain: ChainStateLike) -> int:
    """How many completed walks the chain has — ``depth`` is the
    canonical counter on :class:`master_loop.ChainState`. Defaults to
    0 for freshly-rooted chains."""
    depth = getattr(chain, "depth", 0) or 0
    return max(int(depth), 0)


def _posterior(
    payoff: float | None,
    n_pulls: int,
    prior_mean: float,
    prior_variance: float,
    obs_variance: float = 1.0,
) -> tuple[float, float, str]:
    """Conjugate Normal-Normal update with a *known* observation
    variance ``obs_variance``.

    Treating each completed walk as one noisy observation of the
    chain's true ``|point/SE|``, the posterior after ``n_pulls``
    observations of (approximate) sample mean ``payoff`` is

    .. math::

        \\mu_{\\text{post}} = \\frac{\\mu_0/\\tau_0^2 + n\\,\\bar y/\\sigma^2}
                                   {1/\\tau_0^2 + n/\\sigma^2}, \\quad
        \\tau_{\\text{post}}^2 = (1/\\tau_0^2 + n/\\sigma^2)^{-1}.

    We only know the *latest* payoff, not the running mean — so for
    ``n ≥ 1`` we treat ``payoff`` as the sample mean of the
    observations to date. This is the standard "summary-statistic"
    approximation used in bandit posteriors when only the most
    recent estimate is retained.
    """
    if payoff is None or n_pulls <= 0:
        return prior_mean, prior_variance, "uniform"

    prior_precision = 1.0 / prior_variance
    likelihood_precision = n_pulls / obs_variance
    post_precision = prior_precision + likelihood_precision
    post_variance = 1.0 / post_precision
    post_mean = (
        prior_mean * prior_precision + payoff * likelihood_precision
    ) / post_precision
    return post_mean, post_variance, "user_specified"


# ─────────── Thompson sampling ───────────────────────────────────────────


def thompson_sample_chain(
    *,
    chains: Iterable[ChainStateLike],
    rng: np.random.Generator | None = None,
    prior_mean: float = 1.0,
    prior_variance: float = 4.0,
) -> tuple[str, list[BanditArm]]:
    """Thompson-sample which chain to drill into next.

    Algorithm
    ---------
    1. For each chain, compute a Normal posterior on
       :math:`|\\mathrm{point}/\\mathrm{SE}|` from its last walk and
       its current pull count (``depth``).
    2. Draw one sample from each chain's posterior.
    3. Return the ``chain_id`` whose draw is largest, plus a per-arm
       view (for the decision-ledger / TUI surface).

    Edge cases
    ----------
    * Empty ``chains`` → :class:`ValueError`.
    * A just-rooted chain (no ``last_point``) keeps the prior — so
      brand-new chains are *exploration-favoured* by virtue of having
      the widest posterior.
    * A chain whose last walk was null (``|point/SE|`` small) sees its
      posterior pulled toward 0, dampening future pulls but never
      eliminating them.
    * NaN / non-finite point or SE → treated as "no observation."

    Parameters
    ----------
    chains:
        Iterable of objects matching :class:`ChainStateLike`. A
        ``list[ChainState]`` from :mod:`master_loop` works directly.
    rng:
        Optional :class:`numpy.random.Generator`. The master loop is
        expected to seed this from its own RNG for reproducibility.
        Defaults to :func:`numpy.random.default_rng()` (non-seeded).
    prior_mean:
        Starting belief about ``|point/SE|``. ``1.0`` ≈ borderline
        significant, which encodes a healthy skepticism without
        choking off exploration.
    prior_variance:
        Prior variance. Wider → more exploration on fresh chains.

    Returns
    -------
    (str, list[BanditArm])
        Chosen chain id and the per-arm posterior view (in the same
        order as ``chains``).
    """
    chain_list = list(chains)
    if not chain_list:
        raise ValueError("thompson_sample_chain: chains is empty")
    if prior_variance <= 0.0:
        raise ValueError(
            f"thompson_sample_chain: prior_variance must be > 0; got {prior_variance!r}"
        )

    rng = rng if rng is not None else np.random.default_rng()

    arms: list[BanditArm] = []
    samples: list[float] = []
    for chain in chain_list:
        payoff = _safe_payoff(chain)
        n_pulls = _n_pulls(chain)
        post_mean, post_var, prior_tag = _posterior(
            payoff=payoff,
            n_pulls=n_pulls,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
        )
        # Sample from the posterior on the payoff measure. We sample
        # on the real line and use the raw draw for ranking — the
        # ordering is what matters for argmax, and clipping at zero
        # would bias ties toward the first arm.
        draw = float(rng.normal(loc=post_mean, scale=math.sqrt(post_var)))
        samples.append(draw)
        arms.append(
            BanditArm(
                chain_id=getattr(chain, "chain_id", f"<unknown:{id(chain)}>"),
                posterior_mean=post_mean,
                posterior_variance=post_var,
                n_pulls=n_pulls,
                prior_used=prior_tag,
            )
        )

    best_idx = int(np.argmax(samples))
    return arms[best_idx].chain_id, arms


# ─────────── UCB1 (deterministic) ────────────────────────────────────────


def ucb1_chain_choice(
    chains: Iterable[ChainStateLike],
    *,
    c: float = 1.41,
) -> tuple[str, list[BanditArm]]:
    """Auer et al. (2002) UCB1 chain choice — deterministic.

    The upper-confidence-bound score for chain ``c`` is

    .. math::

        \\mathrm{UCB}_c = \\bar y_c + c \\, \\sqrt{\\frac{\\ln N}{n_c}},

    where :math:`\\bar y_c` is the chain's last ``|point/SE|``,
    :math:`n_c` is its pull count, and :math:`N = \\sum_c n_c`.

    Unobserved chains (``n_c == 0``) are pulled first, preserving the
    UCB1 invariant that every arm must be played at least once.
    Ties are broken by the order chains appear in the input — this
    keeps the function bit-exact reproducible.

    The arms returned here are filled in with ``posterior_mean`` =
    UCB score and ``posterior_variance`` = bonus squared, so the
    same :class:`BanditArm` surface can be reused by the TUI.

    Parameters
    ----------
    chains:
        Iterable of objects matching :class:`ChainStateLike`.
    c:
        Exploration constant. ``sqrt(2) ≈ 1.41`` is the textbook
        choice for UCB1.

    Returns
    -------
    (str, list[BanditArm])
        Chosen chain id and the per-arm UCB view.
    """
    chain_list = list(chains)
    if not chain_list:
        raise ValueError("ucb1_chain_choice: chains is empty")

    n_total = sum(_n_pulls(ch) for ch in chain_list)
    # ln(N) is undefined at N=0; UCB1 conventionally treats this as
    # "everyone explores" so we set the log term to 0 and let the
    # n_pulls==0 fast-path below pick the first chain.
    log_n = math.log(n_total) if n_total > 0 else 0.0

    arms: list[BanditArm] = []
    scores: list[float] = []
    first_unpulled: int | None = None
    for idx, chain in enumerate(chain_list):
        payoff = _safe_payoff(chain)
        n_pulls = _n_pulls(chain)
        if n_pulls <= 0 or payoff is None:
            # Unobserved arm: infinite score. We record finite numbers
            # in the arm view to keep the surface JSON-serialisable,
            # but mark this index as the deterministic winner.
            if first_unpulled is None:
                first_unpulled = idx
            scores.append(float("inf"))
            arms.append(
                BanditArm(
                    chain_id=getattr(chain, "chain_id", f"<unknown:{id(chain)}>"),
                    posterior_mean=payoff if payoff is not None else 0.0,
                    posterior_variance=float("inf"),
                    n_pulls=n_pulls,
                    prior_used="ucb_seeded",
                )
            )
            continue
        bonus = c * math.sqrt(log_n / n_pulls) if n_pulls > 0 else float("inf")
        ucb = payoff + bonus
        scores.append(ucb)
        arms.append(
            BanditArm(
                chain_id=getattr(chain, "chain_id", f"<unknown:{id(chain)}>"),
                posterior_mean=ucb,
                posterior_variance=bonus * bonus,
                n_pulls=n_pulls,
                prior_used="ucb_seeded",
            )
        )

    if first_unpulled is not None:
        return arms[first_unpulled].chain_id, arms

    # All chains have ≥ 1 pull — pick deterministically by argmax,
    # breaking ties by input order (np.argmax already returns the
    # first max).
    best_idx = int(np.argmax(scores))
    return arms[best_idx].chain_id, arms
