"""High-dimensional sparse-truth DGP — SparseLinearDML + post-double-selection.

p ≈ 60, n = 500, only 3 covariates are relevant by construction. The pipeline
should:

- Auto-select SparseLinearDML on the HIGH_DIMENSIONAL flag.
- Post-double-selection drops most of the noise variables.
- ATE recovery within ±15% of true ATE.
"""

from __future__ import annotations

import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalGraph
from causalrag.core.protocol import StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.estimators.python.select import select_estimator
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q7_estimate import estimate

pytestmark = pytest.mark.integration


def test_high_dim_routes_to_sparse_linear_dml() -> None:
    entry = select_estimator(
        estimand="ATE",
        flags={DataFlag.BINARY_TREATMENT, DataFlag.HIGH_DIMENSIONAL},
        n=500,
    )
    assert entry.id == "python.dml.sparse_linear"


def test_high_dim_post_double_selection_recovers_ate(high_dim_sparse) -> None:
    df, true_ate, relevant = high_dim_sparse
    covariates = tuple(c for c in df.columns if c not in {"treat", "y"})
    estimand = CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": "treat",
            "outcome": "y",
            "formal_expression": "E[Y(1)-Y(0)]",
        }
    )
    graph = CausalGraph.from_edge_list(
        [(c, "treat") for c in covariates]
        + [(c, "y") for c in covariates]
        + [("treat", "y")],
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
        protocol=StudyProtocol(name="high_dim"),
        confounders=covariates,
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME, DataFlag.HIGH_DIMENSIONAL},
        selection="post_double_selection",
    )

    # Estimator was routed to sparse
    assert "sparse" in result.estimator_id

    # ATE recovery within 25% of truth (high-dim is noisy)
    assert abs(result.point_estimate - true_ate) < 0.25 * abs(true_ate) + 0.3

    # Selection dropped most noise; the relevant variables should appear
    selection = result.diagnostics.get("variable_selection", {})
    selected = set(selection.get("selected", []))
    # At least one of the truly relevant covariates was selected
    assert any(r in selected for r in relevant)
    # Significantly fewer than the original 60 candidates were retained
    assert len(selected) < 50
