"""Phase 2 — feasibility filter (PDD §8).

The filter scores each candidate (treatment, outcome) pair against a power
target and emits admissible / borderline / underpowered verdicts. Only
admissible pairs flow into Phase 3 hypothesis generation by default.
"""

from causalrag.feasibility.power import (
    PowerResult,
    power_binary_ate,
    power_continuous_ate,
    power_subgroup_cate,
)
from causalrag.feasibility.report import (
    FeasibilityReportFull,
    candidate_pairs,
    run_feasibility,
)
from causalrag.feasibility.thresholds import (
    Thresholds,
    default_thresholds,
    manual_thresholds,
)

__all__ = [
    "FeasibilityReportFull",
    "PowerResult",
    "Thresholds",
    "candidate_pairs",
    "default_thresholds",
    "manual_thresholds",
    "power_binary_ate",
    "power_continuous_ate",
    "power_subgroup_cate",
    "run_feasibility",
]
