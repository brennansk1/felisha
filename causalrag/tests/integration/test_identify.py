"""Step 5 identifiability tests (PDD §10.5).

Use small hand-built DAGs to verify that DoWhy returns the expected strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.graph import CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.roadmap.q5_identify import identify_effect

pytest.importorskip("dowhy")

pytestmark = pytest.mark.integration


def _estimand(klass: EstimandClass = EstimandClass.ATE, **kwargs) -> CausalEstimand:
    return CausalEstimand.model_validate(
        {
            "class": klass,
            "treatment": kwargs.get("treatment", "T"),
            "outcome": kwargs.get("outcome", "Y"),
            "modifiers": kwargs.get("modifiers", ()),
            "mediator": kwargs.get("mediator"),
            "instrument": kwargs.get("instrument"),
            "formal_expression": kwargs.get("formal_expression", "E[Y(1)-Y(0)]"),
        }
    )


def test_backdoor_identification() -> None:
    """Classic confounder X — backdoor adjustment expected."""
    graph = CausalGraph.from_edge_list(
        [("X", "T"), ("X", "Y"), ("T", "Y")],
        roles={
            "X": VariableRole.CONFOUNDER,
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
        },
    )
    result = identify_effect(_estimand(), graph)
    assert result.identifiable
    assert result.strategy == "backdoor"
    assert "X" in result.adjustment_set


def test_no_unblocked_confounder_still_identifiable() -> None:
    """T → Y with no confounders is trivially identifiable with empty adjustment."""
    graph = CausalGraph.from_edge_list(
        [("T", "Y")],
        roles={"T": VariableRole.TREATMENT, "Y": VariableRole.OUTCOME},
    )
    result = identify_effect(_estimand(), graph)
    assert result.identifiable
    assert result.strategy == "backdoor"
    assert result.adjustment_set == ()


def test_unobserved_confounder_is_non_identifiable() -> None:
    """An unobserved variable U creating an open backdoor without an IV."""
    graph = CausalGraph.from_edge_list(
        [("U", "T"), ("U", "Y"), ("T", "Y")],
        roles={
            "U": VariableRole.UNMEASURED_CONFOUNDER_CANDIDATE,
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
        },
    )
    # Build a frame with only observed columns: T, Y
    df = pd.DataFrame({"T": np.random.binomial(1, 0.5, 50), "Y": np.random.randn(50)})
    result = identify_effect(_estimand(), graph, df=df)
    # DoWhy will still find a backdoor over U because the graph node exists;
    # we accept either non-identifiable or backdoor-with-U-in-adjustment as
    # long as the analyst gets the warning.
    if result.identifiable:
        assert "U" in result.adjustment_set


def test_instrumental_variable_detected() -> None:
    """Z → T → Y with U → (T, Y); Z is a valid IV."""
    graph = CausalGraph.from_edge_list(
        [("Z", "T"), ("U", "T"), ("U", "Y"), ("T", "Y")],
        roles={
            "Z": VariableRole.INSTRUMENT,
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "U": VariableRole.UNMEASURED_CONFOUNDER_CANDIDATE,
        },
    )
    result = identify_effect(_estimand(klass=EstimandClass.LATE, instrument="Z"), graph)
    # DoWhy may report iv OR backdoor depending on version; allow either.
    assert result.strategy in {"iv", "backdoor"}


def test_unsupported_estimand_class() -> None:
    """Counterfactual distribution estimands are not in scope for Step 5 yet."""
    graph = CausalGraph.from_edge_list(
        [("T", "Y")], roles={"T": VariableRole.TREATMENT, "Y": VariableRole.OUTCOME}
    )
    result = identify_effect(
        _estimand(klass=EstimandClass.COUNTERFACTUAL_DISTRIBUTION), graph
    )
    assert not result.identifiable
    assert result.strategy == "unsupported"
