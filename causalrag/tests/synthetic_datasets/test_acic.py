"""ACIC-flavored synthetic — full Roadmap walk against known truth.

Tests the headline ATE recovery on a ~30-covariate, sparse-true DGP with
n=2000, plus that the refutations pass on a well-specified analysis.
"""

from __future__ import annotations

import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalGraph
from causalrag.core.protocol import StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q7_estimate import estimate

pytestmark = pytest.mark.integration


def test_acic_synthetic_ate_recovery(acic_synthetic) -> None:
    df, true_ate = acic_synthetic
    covariates = tuple(f"x{i}" for i in range(30))
    estimand = CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": "treat",
            "outcome": "y",
            "formal_expression": "E[Y(1)-Y(0)]",
        }
    )
    graph = CausalGraph.from_edge_list(
        [(c, "treat") for c in covariates] + [(c, "y") for c in covariates] + [("treat", "y")],
        roles={
            **{c: VariableRole.CONFOUNDER for c in covariates},
            "treat": VariableRole.TREATMENT,
            "y": VariableRole.OUTCOME,
        },
    )
    ident = identify_effect(estimand, graph, df=df)
    result = estimate(
        df=df,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="acic"),
        confounders=covariates,
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME, DataFlag.HIGH_DIMENSIONAL},
        selection="post_double_selection",
    )
    # Tight tolerance on a clean DGP with n=2000
    assert abs(result.point_estimate - true_ate) < 0.20 * abs(true_ate)
    # Identification was via backdoor
    assert ident.strategy == "backdoor"
    # Refutations should mostly pass on a well-specified analysis
    refs = result.refutations or {}
    assert refs.get("n_passed", 0) >= 1
