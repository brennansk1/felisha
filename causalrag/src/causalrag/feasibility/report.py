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
    # Memoize per-column scans (nunique / SD) so datasets with many
    # treatment × outcome pairs do not repeat full-column passes per pair.
    nunique_cache: dict[str, int] = {}
    sd_cache: dict[str, float] = {}

    def _t_unique(col: str) -> int:
        if col not in nunique_cache:
            nunique_cache[col] = int(df[col].dropna().nunique())
        return nunique_cache[col]

    def _sd_y(col: str) -> float:
        if col not in sd_cache:
            sd_cache[col] = float(df[col].std(ddof=1) or 1.0)
        return sd_cache[col]

    import pandas.api.types as _pdt

    for treatment, outcome in pairs:
        if treatment not in df.columns or outcome not in df.columns:
            continue
        # A non-numeric (string/categorical) treatment cannot be fed to the
        # numeric power calculations — the continuous path would cast the labels
        # to float and raise ``ValueError: could not convert string to float``
        # (G7). Binary string treatments should be 0/1-encoded upstream; a
        # >2-level categorical treatment needs an explicit contrast (which arm
        # vs which) that power analysis cannot infer. Skip rather than crash.
        if not _pdt.is_numeric_dtype(df[treatment]):
            continue
        t_unique = _t_unique(treatment)
        # Default plausible band — for continuous outcome we resolve a small
        # Cohen-d-equivalent band post-hoc using outcome SD.
        band = thresholds.plausible_band
        is_binary_treatment = (
            DataFlag.BINARY_TREATMENT in flags or t_unique <= 2
        )
        if not is_binary_treatment and t_unique > 2:
            # Continuous treatment path
            if band is None:
                sd_y = _sd_y(outcome)
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
                sd_y = _sd_y(outcome)
                band = (0.2 * sd_y, 0.5 * sd_y)
            res = power_binary_ate(
                df,
                treatment,
                outcome,
                alpha=thresholds.alpha,
                target_power=thresholds.target_power,
                plausible_band=band,
            )
        # Enforce the configured sample-size floor: a pair that does not meet
        # n_floor cannot be admissible/borderline regardless of MDE.
        if (
            res.verdict in ("admissible", "borderline")
            and res.n_used < thresholds.n_floor
        ):
            res.verdict = "underpowered"
            floor_note = f"n_used={res.n_used} < n_floor={thresholds.n_floor}"
            res.notes = f"{res.notes}; {floor_note}" if res.notes else floor_note
        out.append(res)
    return FeasibilityReportFull(thresholds=thresholds, results=out)


__all__ = ["FeasibilityReportFull", "candidate_pairs", "run_feasibility"]
