"""Threshold-setting modes for the feasibility filter (PDD §8.3).

Three modes:

- ``default`` — statistician-set per-family thresholds (alpha=0.05,
  target_power=0.80, plausible-band by outcome dtype).
- ``manual`` — user overrides via the StudyProtocol or CLI flags.
- ``llm_calibrated`` — LLM proposes a plausible band based on domain
  context; statistically validated before being applied.

Returns a :class:`Thresholds` object consumed by ``feasibility.report``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from causalrag.core.flags import DataFlag

Mode = Literal["default", "manual", "llm_calibrated"]


@dataclass
class Thresholds:
    mode: Mode
    alpha: float = 0.05
    target_power: float = 0.80
    n_floor: int = 200
    plausible_band: tuple[float, float] | None = None
    rationale: str | None = None


def default_thresholds(flags: frozenset[DataFlag] | set[DataFlag]) -> Thresholds:
    """Statistician-defensible defaults given the situation flags."""
    band: tuple[float, float] | None = None
    rationale = "PDD §8.3 statistician defaults: α=0.05, target_power=0.80."
    if DataFlag.BINARY_OUTCOME in flags:
        # An absolute-risk difference of 0.05 is a defensible default for
        # binary outcomes in clinical effectiveness work.
        band = (0.02, 0.10)
        rationale += " Plausible band 2-10 pp risk difference (binary outcome)."
    elif DataFlag.CONTINUOUS_OUTCOME in flags:
        # Cohen's d ≈ 0.2-0.5 is the standard "small-to-medium" range; we
        # express in outcome units only after seeing the data SD.
        band = None
        rationale += " Plausible band defaults to Cohen's d ∈ [0.2, 0.5] (resolved at run time)."
    return Thresholds(
        mode="default",
        alpha=0.05,
        target_power=0.80,
        plausible_band=band,
        rationale=rationale,
    )


def manual_thresholds(
    alpha: float = 0.05,
    target_power: float = 0.80,
    plausible_band: tuple[float, float] | None = None,
    n_floor: int = 200,
) -> Thresholds:
    return Thresholds(
        mode="manual",
        alpha=alpha,
        target_power=target_power,
        plausible_band=plausible_band,
        n_floor=n_floor,
        rationale=f"Analyst-set thresholds (α={alpha}, power={target_power}).",
    )


__all__ = ["Mode", "Thresholds", "default_thresholds", "manual_thresholds"]
