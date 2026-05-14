"""Survival outcome routing — ``RIGHT_CENSORED_OUTCOME`` should block
non-survival estimators and surface a useful error.

There is no R-bridged survival estimator yet (v0.5), so the selector should
raise ``LookupError`` with a message that lists the missing R-side estimators.
This is the architecturally-correct failure mode per PDD §15.2.
"""

from __future__ import annotations

import pytest

from causalrag.core.flags import DataFlag
from causalrag.data.profiler import profile_dataframe
from causalrag.estimators.python.select import select_estimator

pytestmark = pytest.mark.integration


def test_censoring_pair_detected_in_profile(survival_synthetic) -> None:
    p = profile_dataframe(survival_synthetic)
    assert ("overall_time_days", "overall_event") in p.censoring_pairs


def test_survival_routing_excludes_linear_dml() -> None:
    """LinearDML is excluded by ``RIGHT_CENSORED_OUTCOME``; v0.1 has no
    survival estimator registered, so select_estimator must raise.
    """
    with pytest.raises(LookupError):
        select_estimator(
            estimand="RMST_CONTRAST",
            flags={DataFlag.BINARY_TREATMENT, DataFlag.RIGHT_CENSORED_OUTCOME},
            n=600,
        )


def test_continuous_outcome_does_not_inherit_censoring() -> None:
    """A continuous outcome on the same dataset should still route to
    LinearDML — the censoring flag only attaches to the time/event pair."""
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        n=600,
    )
    assert entry.id == "python.dml.linear"
