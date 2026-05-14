"""Routing tests for the new DataFlags wired in Week 3.

Each test exercises one branch of ``select._rule_cascade``.
"""

from __future__ import annotations

from causalrag.core.flags import DataFlag
from causalrag.estimators.python.select import select_estimator

# Importing estimators triggers registration of the full Python catalog.
import causalrag.estimators  # noqa: F401


def test_rare_outcome_routes_to_dr_learner() -> None:
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.BINARY_OUTCOME, DataFlag.RARE_OUTCOME},
        n=800,
    )
    assert entry.id == "python.dr.dr_learner"


def test_imbalanced_treatment_flag_routes_to_x_learner() -> None:
    """Without a numeric ``treatment_prevalence`` the legacy check can't
    fire; the IMBALANCED_TREATMENT flag must be sufficient on its own."""
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.IMBALANCED_TREATMENT},
        n=800,
    )
    assert entry.id == "python.meta.x_learner"


def test_imbalanced_treatment_fallback_via_prevalence_high() -> None:
    """Prevalence > 0.85 should also trip the legacy fallback."""
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT},
        n=800,
        treatment_prevalence=0.92,
    )
    assert entry.id == "python.meta.x_learner"


def test_bounded_outcome_excludes_ols() -> None:
    """OLS must not win on a [0, 1]-bounded outcome even when small-n would
    normally route there."""
    entry = select_estimator(
        estimand="ATE",
        flags={
            DataFlag.BINARY_TREATMENT,
            DataFlag.CONTINUOUS_OUTCOME,
            DataFlag.BOUNDED_OUTCOME,
            DataFlag.SMALL_SAMPLE,
        },
        n=150,
    )
    assert entry.id != "python.linear.ols"


def test_effect_modification_lowers_forest_threshold() -> None:
    """With the effect-modification-of-interest flag, n_modifiers=1 must be
    enough to route to a forest estimator."""
    entry = select_estimator(
        estimand="CATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.EFFECT_MODIFICATION_OF_INTEREST},
        n=800,
        n_modifiers=1,
    )
    assert entry.id in ("rbridge.grf.causal_forest", "python.dml.causal_forest")


def test_zero_inflated_outcome_falls_through_to_dml() -> None:
    """No dedicated estimator yet; the default ladder still serves DML."""
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.ZERO_INFLATED_OUTCOME},
        n=800,
    )
    assert entry.id == "python.dml.linear"


def test_did_candidate_falls_through_to_default_ladder() -> None:
    """DiD has no Python-side estimator; expected behavior is fall-through."""
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.DIFF_IN_DIFF_CANDIDATE},
        n=800,
    )
    assert entry.id == "python.dml.linear"
