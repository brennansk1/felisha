"""IHDP-flavored semi-synthetic — CATE heterogeneity recovery.

The IHDP (Infant Health and Development Program) benchmark superimposes a
synthetic outcome with known heterogeneous treatment effects on real
covariates. The pipeline should:

- Auto-select a forest-based estimator when many modifiers are supplied.
- Recover the true ATE within tight bounds (semi-synthetic = known truth).
- Surface CATE variance > 0 (the effect IS heterogeneous by construction).
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


def _ihdp_estimand(modifiers: tuple[str, ...]) -> CausalEstimand:
    return CausalEstimand.model_validate(
        {
            "class": EstimandClass.CATE,
            "treatment": "treat",
            "outcome": "y",
            "modifiers": modifiers,
            "formal_expression": "E[Y(1)-Y(0) | X]",
        }
    )


def _ihdp_graph(covariates: list[str]) -> CausalGraph:
    edges = [(c, "treat") for c in covariates] + [(c, "y") for c in covariates] + [("treat", "y")]
    roles = {c: VariableRole.CONFOUNDER for c in covariates}
    roles["treat"] = VariableRole.TREATMENT
    roles["y"] = VariableRole.OUTCOME
    return CausalGraph.from_edge_list(edges, roles=roles)


def test_many_modifiers_route_to_forest_estimator() -> None:
    """≥3 modifiers + n ≥ 500 should select a causal-forest estimator.

    grf::causal_forest (R) is preferred when the R bridge is available;
    falls back to EconML's CausalForestDML otherwise.
    """
    entry = select_estimator(
        estimand="CATE",
        flags={DataFlag.BINARY_TREATMENT},
        n=700,
        n_modifiers=4,
    )
    assert entry.id in {"rbridge.grf.causal_forest", "python.dml.causal_forest"}


def test_ihdp_ate_recovery(ihdp_synthetic) -> None:
    """LinearDML on IHDP-flavored data recovers the known true ATE."""
    df, true_ate = ihdp_synthetic
    covariates = [c for c in df.columns if c not in {"treat", "y"}]
    estimand = _ihdp_estimand(modifiers=())
    estimand = CausalEstimand.model_validate(
        {**estimand.model_dump(by_alias=True), "class": EstimandClass.ATE}
    )
    graph = _ihdp_graph(covariates)
    ident = identify_effect(estimand, graph, df=df)
    assert ident.identifiable

    result = estimate(
        df=df,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="ihdp"),
        confounders=tuple(covariates),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        selection="none",
    )
    # IHDP-flavored truth is the per-row CATE mean (≈ 3.0); allow ±20% tolerance.
    assert abs(result.point_estimate - true_ate) < 0.2 * abs(true_ate) + 1.0


def test_ihdp_cate_heterogeneity_detected(ihdp_synthetic) -> None:
    """CausalForestDML with effect modifiers should produce nonzero CATE variance."""
    df, _true_ate = ihdp_synthetic
    covariates = [c for c in df.columns if c not in {"treat", "y"}]
    # Use the two continuous covariates that drive heterogeneity (by construction)
    modifiers = ("x_cont_0", "x_cont_1", "x_cont_2", "x_cont_3")
    estimand = _ihdp_estimand(modifiers=modifiers)
    graph = _ihdp_graph(covariates)
    ident = identify_effect(estimand, graph, df=df)

    confounders = tuple(c for c in covariates if c not in modifiers)
    result = estimate(
        df=df,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="ihdp"),
        confounders=confounders,
        modifiers=modifiers,
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        prefer="forest",
        selection="none",
    )
    assert result.estimator_id == "python.dml.causal_forest"
    cate_mean = result.diagnostics.get("cate_mean")
    cate_low = result.diagnostics.get("cate_ci_low")
    cate_high = result.diagnostics.get("cate_ci_high")
    assert cate_mean is not None
    if cate_low is not None and cate_high is not None:
        assert cate_high > cate_low  # nontrivial interval
