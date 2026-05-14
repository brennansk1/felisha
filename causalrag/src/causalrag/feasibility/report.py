"""Compose the feasibility report from per-(treatment, outcome) power calcs.

Iterates the candidate (treatment, outcome) pairs from the StudyProtocol's
discovery report (or supplied explicitly), runs the appropriate power
calculator for each, and returns a :class:`FeasibilityReportFull` — a richer
object than the protocol's lean :class:`causalrag.core.protocol.FeasibilityReport`.

The richer object is what the CLI / TUI / report renderer want; the lean
projection is what is persisted to ``study.causalrag.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import FeasibilityReport, StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.feasibility.power import (
    PowerResult,
    power_binary_ate,
    power_continuous_ate,
)
from causalrag.feasibility.thresholds import Thresholds, default_thresholds


@dataclass
class FeasibilityReportFull:
    thresholds: Thresholds
    results: list[PowerResult] = field(default_factory=list)

    @property
    def admissible(self) -> list[PowerResult]:
        return [r for r in self.results if r.verdict == "admissible"]

    @property
    def borderline(self) -> list[PowerResult]:
        return [r for r in self.results if r.verdict == "borderline"]

    @property
    def underpowered(self) -> list[PowerResult]:
        return [r for r in self.results if r.verdict == "underpowered"]

    def to_protocol(self) -> FeasibilityReport:
        return FeasibilityReport(
            admissible_pairs=tuple(
                (r.treatment, r.outcome) for r in self.admissible
            ),
            n_floor=self.thresholds.n_floor,
            power_target=self.thresholds.target_power,
            alpha=self.thresholds.alpha,
            notes=self.thresholds.rationale,
        )


def candidate_pairs(protocol: StudyProtocol) -> list[tuple[str, str]]:
    """Read (treatment, outcome) candidates from the discovery report or the
    protocol's expert brief. When neither is available, return an empty list
    and let the caller error out with an actionable message.
    """
    if protocol.discovery is None:
        return []
    treatments = [
        v.name for v in protocol.discovery.columns if v.role is VariableRole.TREATMENT
    ]
    outcomes = [
        v.name for v in protocol.discovery.columns if v.role is VariableRole.OUTCOME
    ]
    if not treatments or not outcomes:
        return []
    return [(t, y) for t in treatments for y in outcomes]


def run_feasibility(
    df: pd.DataFrame,
    protocol: StudyProtocol,
    *,
    pairs: list[tuple[str, str]] | None = None,
    thresholds: Thresholds | None = None,
) -> FeasibilityReportFull:
    pairs = pairs if pairs is not None else candidate_pairs(protocol)
    flags = frozenset(protocol.flags)
    thresholds = thresholds or default_thresholds(flags)
    out: list[PowerResult] = []
    for treatment, outcome in pairs:
        if treatment not in df.columns or outcome not in df.columns:
            continue
        t_unique = df[treatment].dropna().nunique()
        # Default plausible band — for continuous outcome we resolve a small
        # Cohen-d-equivalent band post-hoc using outcome SD.
        band = thresholds.plausible_band
        is_binary_treatment = (
            DataFlag.BINARY_TREATMENT in flags or t_unique <= 2
        )
        if not is_binary_treatment and t_unique > 2:
            # Continuous treatment path
            if band is None:
                sd_y = float(df[outcome].std(ddof=1) or 1.0)
                band = (0.2 * sd_y, 0.5 * sd_y)
            res = power_continuous_ate(
                df,
                treatment,
                outcome,
                alpha=thresholds.alpha,
                target_power=thresholds.target_power,
                plausible_band=band,
            )
        else:
            if band is None and DataFlag.CONTINUOUS_OUTCOME in flags:
                sd_y = float(df[outcome].std(ddof=1) or 1.0)
                band = (0.2 * sd_y, 0.5 * sd_y)
            res = power_binary_ate(
                df,
                treatment,
                outcome,
                alpha=thresholds.alpha,
                target_power=thresholds.target_power,
                plausible_band=band,
            )
        out.append(res)
    return FeasibilityReportFull(thresholds=thresholds, results=out)


__all__ = ["FeasibilityReportFull", "candidate_pairs", "run_feasibility"]
