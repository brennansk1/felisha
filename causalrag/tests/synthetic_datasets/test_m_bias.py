"""M-bias collider — pipeline must NOT silently adjust for a collider.

DGP: U1 → X ← U2, T → X, X → Y. ``x_collider`` is a post-treatment collider;
adjusting for it induces bias. Our Layer-3 temporal check + Layer-4 CI test
should catch the situation. At minimum the audit should mark T → X as a
*supported* edge (because T does cause X), which surfaces to the analyst.
"""

from __future__ import annotations

import pytest

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.llm.guards import audit_dag_edges

pytestmark = pytest.mark.integration


def test_collider_edge_is_supported_in_audit(m_bias_collider) -> None:
    """The audit must mark T → X_collider as supported — that edge IS real,
    and surfacing it forces the analyst to confront the collider question."""
    df, _ = m_bias_collider
    g = CausalGraph(
        nodes=("treat", "x_collider", "u1_observed", "y"),
        edges=(
            CausalEdge(source="treat", target="x_collider", llm_proposed=True),
            CausalEdge(source="treat", target="y", llm_proposed=True),
        ),
        roles={
            "treat": VariableRole.TREATMENT,
            "x_collider": VariableRole.COLLIDER,
            "y": VariableRole.OUTCOME,
            "u1_observed": VariableRole.CONFOUNDER,
        },
    )
    audits = audit_dag_edges(g, df)
    by_pair = {(a.source, a.target): a for a in audits}
    collider_audit = by_pair[("treat", "x_collider")]
    # T → X_collider is a real causal edge in the DGP — the audit must
    # detect it.
    assert collider_audit.verdict == "supported"


def test_adjusting_for_collider_changes_estimate_meaningfully(m_bias_collider) -> None:
    """Empirical demonstration of why collider control is harmful: the
    estimate with the collider in the adjustment set should differ from the
    estimate without it."""
    df, _true_ate = m_bias_collider
    from causalrag.core.estimand import CausalEstimand, EstimandClass
    from causalrag.core.flags import DataFlag
    from causalrag.core.protocol import StudyProtocol
    from causalrag.roadmap.q5_identify import IdentificationResult
    from causalrag.roadmap.q7_estimate import estimate

    estimand = CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": "treat",
            "outcome": "y",
            "formal_expression": "E[Y(1)-Y(0)]",
        }
    )
    ident = IdentificationResult(identifiable=True, strategy="backdoor")
    # No-collider run: simple T → Y comparison (Lalonde NSW is randomized in
    # spirit — empty adjustment is unbiased)
    no_collider = estimate(
        df=df,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="m_bias"),
        confounders=(),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        selection="none",
        preprocess=False,
    )
    # With-collider run: include x_collider in the adjustment set
    with_collider = estimate(
        df=df,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="m_bias"),
        confounders=("x_collider",),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        selection="none",
        preprocess=False,
    )
    # The two estimates should differ — that's the empirical signature of
    # collider bias. The point estimates need not differ massively; even a
    # 15% shift demonstrates the phenomenon.
    diff = abs(no_collider.point_estimate - with_collider.point_estimate)
    assert diff > 0.1, (
        f"Collider adjustment should perturb the estimate; got identical results "
        f"({no_collider.point_estimate:.3f} vs {with_collider.point_estimate:.3f})"
    )
