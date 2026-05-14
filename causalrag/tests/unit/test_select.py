from __future__ import annotations

import pytest

from causalrag.core.flags import DataFlag
from causalrag.estimators.python.select import select_estimator

# Importing estimators triggers registration of the full Python catalog.
import causalrag.estimators  # noqa: F401


def test_high_dimensional_routes_to_sparse() -> None:
    entry = select_estimator(
        estimand="ATE", flags={DataFlag.BINARY_TREATMENT, DataFlag.HIGH_DIMENSIONAL}, n=1000
    )
    assert entry.id == "python.dml.sparse_linear"


def test_small_sample_routes_to_ols() -> None:
    """SMALL_SAMPLE routes to OLS — the only estimator with a viable
    min_sample_size for n<100 (the DML family needs ~100 for stable
    cross-fitting)."""
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.SMALL_SAMPLE},
        n=40,
    )
    assert entry.id == "python.linear.ols"


def test_moderate_n_with_small_sample_flag_still_picks_ols() -> None:
    """At n=120 with SMALL_SAMPLE flag explicitly set, OLS still wins for
    transparency and finite-sample robustness."""
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.SMALL_SAMPLE},
        n=120,
    )
    assert entry.id == "python.linear.ols"


def test_many_modifiers_route_to_causal_forest() -> None:
    """At n≥500 with ≥3 modifiers, prefer the R-bridged grf reference
    implementation; if R isn't available, falls back to EconML's
    CausalForestDML."""
    entry = select_estimator(
        estimand="CATE", flags={DataFlag.BINARY_TREATMENT}, n=600, n_modifiers=4
    )
    assert entry.id in ("rbridge.grf.causal_forest", "python.dml.causal_forest")


def test_rare_treatment_routes_to_x_learner() -> None:
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT},
        n=800,
        treatment_prevalence=0.08,
    )
    assert entry.id == "python.meta.x_learner"


def test_default_is_linear_dml() -> None:
    entry = select_estimator(estimand="ATE", flags={DataFlag.BINARY_TREATMENT}, n=400)
    assert entry.id == "python.dml.linear"


def test_user_preference_overrides_auto() -> None:
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT},
        n=600,
        prefer="python.meta.t_learner",
    )
    assert entry.id == "python.meta.t_learner"


def test_user_family_preference() -> None:
    entry = select_estimator(
        estimand="ATE", flags={DataFlag.BINARY_TREATMENT}, n=600, prefer="forest"
    )
    assert entry.id == "python.dml.causal_forest"


def test_no_supported_estimator_raises() -> None:
    with pytest.raises(LookupError):
        select_estimator(estimand="DOES_NOT_EXIST")
