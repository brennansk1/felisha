"""E-value closed testing for sequential FWER control.

The master loop runs K hypothesis tests (one per roadmap walk, per
sensitivity slice, etc.) and we want family-wise error-rate (FWER)
control while allowing the loop to *peek* at every intermediate result
without inflating α. Classical Bonferroni does this with p-values but
loses a lot of power; the e-value variant of closed testing — Vovk &
Wang 2024 (`arXiv 2501.09015`) — keeps FWER control while being valid
under optional stopping and is uniformly at least as powerful as the
e-Bonferroni baseline.

Concept in one paragraph
------------------------
An e-value ``e`` for a hypothesis ``H`` is a non-negative random
variable with ``E[e | H true] ≤ 1``. By Markov's inequality, rejecting
``H`` when ``e ≥ 1/α`` controls the type-I error at level α — even
under optional stopping, which is exactly what the master loop does
when it peeks. The *closure principle* extends this to a family of K
hypotheses: for each subset ``S ⊆ {1,…,K}`` form a *combined* e-value
``e_S`` (we use the arithmetic mean — the GRO-optimal merging function
under no further dependence assumptions), and reject the individual
``H_i`` iff every subset ``S ∋ i`` has ``e_S ≥ 1/α``. The resulting
procedure controls FWER strongly in the family.

Implementation notes
--------------------
- For ``K ≤ 16`` we enumerate all ``2^K`` subsets exhaustively. That is
  65 536 subsets at the boundary — comfortably fast in pure Python.
- For ``K > 16`` we use the **e-Bonferroni-Holm shortcut**: sort
  e-values in descending order and reject the i-th largest
  ``e_(i)`` iff ``e_(i) ≥ (K - i + 1) / α``. This is a known
  valid (if slightly conservative) shortcut for the closure; see Wang
  & Ramdas 2022 §5 and Vovk-Wang 2024 §3.
- ``compute_evalue_from_pvalue`` implements the Vovk-Wang 2024
  conservative calibrator ``e = (1 - 1/p) / log(p)`` for ``p < 1/e``
  and ``e = 0`` otherwise. The calibrator is monotone decreasing in
  ``p`` on its support and yields an e-value such that controlling on
  e at level ``1/α`` implies the corresponding p-value test rejects at
  level ``α``. We only need it when a downstream test reports a
  p-value rather than a native e-value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations


# Hard cap for exhaustive closure. 2**16 = 65 536 subsets — still
# sub-second in pure Python. Above this we fall back to the
# e-Bonferroni-Holm shortcut.
_EXHAUSTIVE_MAX_K = 16


@dataclass
class EValueClosedTest:
    """Result of running e-value closed testing on a hypothesis family.

    Attributes
    ----------
    family_size:
        K — number of hypotheses tested jointly.
    alpha:
        Family-wise type-I error level the rejection set controls.
    e_values:
        The input ``{hypothesis_id: e_value}`` mapping, copied so the
        caller can inspect what was tested.
    rejected_at_alpha:
        Hypothesis ids rejected by the closure at the given ``alpha``.
        FWER over this set is ≤ ``alpha``.
    adjusted_alphas:
        Per-hypothesis adjusted thresholds — the smallest α at which
        each hypothesis would be rejected. ``rejected_at_alpha`` is
        exactly the set of ids whose adjusted α is ≤ ``alpha``. Useful
        for sorting / reporting even when no rejection is made at the
        nominal α.
    method:
        ``"exhaustive_closure"`` or ``"e_bonferroni_holm_shortcut"``,
        depending on family size.
    """

    family_size: int
    alpha: float
    e_values: dict[str, float]
    rejected_at_alpha: list[str]
    adjusted_alphas: dict[str, float]
    method: str = field(default="exhaustive_closure")


def compute_evalue_from_pvalue(p: float) -> float:
    """Conservative p → e-value calibration (Vovk-Wang 2024).

    Implements ``e = (1 - 1/p) / log(p)`` for ``p < 1/e``, and ``e = 0``
    otherwise. The calibrator is monotone *decreasing* in p on ``(0,
    1/e)``: smaller p → larger e. Rejecting the corresponding e-value
    test at level ``1/α`` implies a p-value test at level α would also
    reject, so feeding calibrated e-values into :func:`closed_testing`
    is at least as conservative as a p-value-based FWER procedure.

    Parameters
    ----------
    p:
        A p-value in ``[0, 1]``.

    Returns
    -------
    The calibrated e-value, ``≥ 0``. Returns ``+inf`` for ``p == 0``
    (a degenerate but well-defined limit).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p-value must lie in [0, 1]; got {p!r}")
    if p == 0.0:
        return math.inf
    threshold = 1.0 / math.e
    if p >= threshold:
        return 0.0
    # log(p) is negative on (0, 1); (1 - 1/p) is also negative — their
    # ratio is positive.
    return (1.0 - 1.0 / p) / math.log(p)


def _mean(values: list[float]) -> float:
    """Arithmetic mean — the GRO-optimal e-value merging function under
    no further dependence assumptions (Vovk-Wang 2024 Prop. 3.1)."""
    return sum(values) / len(values)


def _exhaustive_closure(
    e_values: dict[str, float],
    *,
    alpha: float,
) -> tuple[list[str], dict[str, float]]:
    """Exact closure: reject H_i iff every subset S ∋ i has e_S ≥ 1/α.

    Also computes per-hypothesis adjusted α — the worst (smallest)
    ``mean(e on S) for S ∋ i`` inverted into an α threshold via
    ``α_i = 1 / min_S mean(e on S)``.
    """
    ids = list(e_values.keys())
    K = len(ids)
    # For each hypothesis, track the minimum mean-e across all subsets
    # containing it. That minimum determines the adjusted α.
    min_mean_e: dict[str, float] = {h: math.inf for h in ids}

    for size in range(1, K + 1):
        for subset in combinations(range(K), size):
            mean_e = _mean([e_values[ids[j]] for j in subset])
            for j in subset:
                if mean_e < min_mean_e[ids[j]]:
                    min_mean_e[ids[j]] = mean_e

    adjusted_alphas: dict[str, float] = {}
    rejected: list[str] = []
    for h in ids:
        min_e = min_mean_e[h]
        # α_adj = 1 / min_e; if min_e == 0 the hypothesis is never
        # rejected (α_adj = ∞). If min_e == ∞ (shouldn't happen
        # because singleton {h} is always included) treat as 0.
        if min_e <= 0.0:
            adj = math.inf
        elif math.isinf(min_e):
            adj = 0.0
        else:
            adj = 1.0 / min_e
        adjusted_alphas[h] = adj
        if adj <= alpha:
            rejected.append(h)
    return rejected, adjusted_alphas


def _e_bonferroni_holm(
    e_values: dict[str, float],
    *,
    alpha: float,
) -> tuple[list[str], dict[str, float]]:
    """Shortcut for K > 16: sort e-values descending, reject the i-th
    largest if ``e_(i) ≥ (K - i + 1) / α``.

    Per-hypothesis adjusted α is the smallest α for which that
    hypothesis would still pass the e-Holm step it sits at:
    ``α_i = (K - rank_i + 1) / e_i``, with rank counted in the
    *descending* sort (largest e gets rank 1).
    """
    K = len(e_values)
    # Sort hypotheses by e-value, descending. Stable on ties so the
    # ranking is deterministic given the input ordering.
    ordered = sorted(e_values.items(), key=lambda kv: kv[1], reverse=True)

    adjusted_alphas: dict[str, float] = {}
    rejected: list[str] = []
    # e-Holm: walk top-down; once a hypothesis fails the threshold,
    # neither it nor any smaller-e hypothesis is rejected.
    stopped = False
    for i, (h, e) in enumerate(ordered, start=1):
        threshold_recip = (K - i + 1)  # the (K-i+1) in (K-i+1)/α
        # α at which this rank-i hypothesis would just barely reject:
        # e ≥ (K-i+1)/α ⇔ α ≥ (K-i+1)/e.
        if e <= 0.0:
            adj = math.inf
        else:
            adj = threshold_recip / e
        adjusted_alphas[h] = adj
        if not stopped and e >= threshold_recip / alpha:
            rejected.append(h)
        else:
            stopped = True
    return rejected, adjusted_alphas


def closed_testing(
    e_values: dict[str, float],
    *,
    alpha: float = 0.05,
) -> EValueClosedTest:
    """Run e-value closed testing across the K hypotheses.

    For ``K ≤ 16`` we enumerate every non-empty subset and build the
    full closure (arithmetic-mean merging of e-values). For ``K > 16``
    we fall back to the e-Bonferroni-Holm shortcut — slightly
    conservative but linear in K log K rather than exponential.

    Parameters
    ----------
    e_values:
        ``{hypothesis_id: e_value}``. All e-values must be non-negative.
    alpha:
        Family-wise type-I error level. Must lie in (0, 1).

    Returns
    -------
    :class:`EValueClosedTest` with ``rejected_at_alpha`` listing the
    hypotheses the closure rejects at the given α.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}")
    if not e_values:
        return EValueClosedTest(
            family_size=0,
            alpha=alpha,
            e_values={},
            rejected_at_alpha=[],
            adjusted_alphas={},
            method="exhaustive_closure",
        )
    for h, e in e_values.items():
        if e < 0.0 or math.isnan(e):
            raise ValueError(
                f"e-value for hypothesis {h!r} must be non-negative; got {e!r}"
            )

    K = len(e_values)
    if K <= _EXHAUSTIVE_MAX_K:
        rejected, adjusted = _exhaustive_closure(e_values, alpha=alpha)
        method = "exhaustive_closure"
    else:
        rejected, adjusted = _e_bonferroni_holm(e_values, alpha=alpha)
        method = "e_bonferroni_holm_shortcut"

    return EValueClosedTest(
        family_size=K,
        alpha=alpha,
        e_values=dict(e_values),
        rejected_at_alpha=rejected,
        adjusted_alphas=adjusted,
        method=method,
    )


__all__ = [
    "EValueClosedTest",
    "closed_testing",
    "compute_evalue_from_pvalue",
]
