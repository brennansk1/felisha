"""Unit tests for the collider / descendant / mediator safety filter applied
to DoWhy's backdoor adjustment set in :mod:`causalrag.roadmap.q5_identify`.

These tests stub DoWhy out so they can run without the optional dependency
installed; only the post-hoc filter and candidate-graph selection logic are
exercised here.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.graph import CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.roadmap import q5_identify
from causalrag.roadmap.q5_identify import (
    IdentificationResult,
    _filter_adjustment_set,
    identify_effect,
)


def _estimand(
    treatment: str = "T", outcome: str = "Y", klass: EstimandClass = EstimandClass.ATE
) -> CausalEstimand:
    return CausalEstimand.model_validate(
        {
            "class": klass,
            "treatment": treatment,
            "outcome": outcome,
            "modifiers": (),
            "mediator": None,
            "instrument": None,
            "formal_expression": "E[Y(1)-Y(0)]",
        }
    )


@dataclass
class _FakeIdentified:
    """Mimics the small surface of DoWhy's IdentifiedEstimand we read."""

    backdoor_variables: dict[str, list[str]]
    estimand_expression: str = "E[Y|do(T)]"
    estimand_type: str = "nonparametric-ate"

    @property
    def estimands(self) -> dict[str, Any]:
        return {"backdoor": {"some": "thing"}}

    @property
    def instrumental_variables(self) -> list[str]:
        return []

    @property
    def frontdoor_variables(self) -> list[str]:
        return []


class _FakeCausalModel:
    """Returns a pre-canned IdentifiedEstimand controlled by ``_FAKE_ADJ``."""

    _FAKE_ADJ: list[str] = []

    def __init__(self, **_: Any) -> None:
        pass

    def identify_effect(self, proceed_when_unidentifiable: bool = False) -> _FakeIdentified:
        return _FakeIdentified(backdoor_variables={"backdoor_set1": list(self._FAKE_ADJ)})


@pytest.fixture
def fake_dowhy(monkeypatch):
    """Install a fake ``dowhy`` module exposing ``CausalModel``."""
    module = types.ModuleType("dowhy")
    module.CausalModel = _FakeCausalModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dowhy", module)

    def set_adjustment(values: list[str]) -> None:
        _FakeCausalModel._FAKE_ADJ = list(values)

    return set_adjustment


# --- Pure-function filter ----------------------------------------------------


def test_filter_drops_descendant_of_treatment() -> None:
    graph = CausalGraph.from_edge_list(
        [("T", "D"), ("T", "Y"), ("Z", "T"), ("Z", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "D": VariableRole.AUXILIARY,
            "Z": VariableRole.CONFOUNDER,
        },
    )
    filtered, drops, weak, warns = _filter_adjustment_set(("D", "Z"), graph, _estimand())
    assert "D" in drops["descendants"]
    assert filtered == ("Z",)
    assert not weak
    assert any("descendants" in w for w in warns)


def test_filter_drops_mediator() -> None:
    graph = CausalGraph.from_edge_list(
        [("T", "M"), ("M", "Y"), ("Z", "T"), ("Z", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "M": VariableRole.MEDIATOR,
            "Z": VariableRole.CONFOUNDER,
        },
    )
    filtered, drops, weak, warns = _filter_adjustment_set(("M", "Z"), graph, _estimand())
    assert "M" in drops["mediators"]
    assert filtered == ("Z",)
    assert any("mediators" in w for w in warns)


def test_filter_drops_collider() -> None:
    # T -> C <- Y :: C is a collider on the T-Y path
    graph = CausalGraph.from_edge_list(
        [("T", "Y"), ("T", "C"), ("Y", "C"), ("Z", "T"), ("Z", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "C": VariableRole.COLLIDER,
            "Z": VariableRole.CONFOUNDER,
        },
    )
    filtered, drops, weak, warns = _filter_adjustment_set(("C", "Z"), graph, _estimand())
    assert "C" in drops["colliders"]
    assert filtered == ("Z",)
    assert any("collider" in w.lower() for w in warns)


def test_filter_empty_after_drop_marks_weak() -> None:
    graph = CausalGraph.from_edge_list(
        [("T", "Y"), ("T", "C"), ("Y", "C")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "C": VariableRole.COLLIDER,
        },
    )
    filtered, _drops, weak, _warns = _filter_adjustment_set(("C",), graph, _estimand())
    assert filtered == ()
    assert weak is True


# --- identify_effect end-to-end (with fake DoWhy) ----------------------------


def test_identify_effect_drops_collider(fake_dowhy) -> None:
    fake_dowhy(["C", "Z"])
    graph = CausalGraph.from_edge_list(
        [("T", "Y"), ("T", "C"), ("Y", "C"), ("Z", "T"), ("Z", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "C": VariableRole.COLLIDER,
            "Z": VariableRole.CONFOUNDER,
        },
    )
    result = identify_effect(_estimand(), graph, df=pd.DataFrame({c: [0.0] for c in graph.nodes}))
    assert result.identifiable
    assert result.strategy == "backdoor"
    assert "C" not in result.adjustment_set
    assert "Z" in result.adjustment_set
    assert "C" in result.diagnostics["dropped_colliders"]
    assert any("collider" in w.lower() for w in result.warnings)


def test_identify_effect_drops_descendant(fake_dowhy) -> None:
    fake_dowhy(["D", "Z"])
    graph = CausalGraph.from_edge_list(
        [("T", "D"), ("T", "Y"), ("Z", "T"), ("Z", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "D": VariableRole.AUXILIARY,
            "Z": VariableRole.CONFOUNDER,
        },
    )
    result = identify_effect(_estimand(), graph, df=pd.DataFrame({c: [0.0] for c in graph.nodes}))
    assert "D" not in result.adjustment_set
    assert "D" in result.diagnostics["dropped_descendants"]


def test_identify_effect_drops_mediator(fake_dowhy) -> None:
    fake_dowhy(["M", "Z"])
    graph = CausalGraph.from_edge_list(
        [("T", "M"), ("M", "Y"), ("Z", "T"), ("Z", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "M": VariableRole.MEDIATOR,
            "Z": VariableRole.CONFOUNDER,
        },
    )
    result = identify_effect(_estimand(), graph, df=pd.DataFrame({c: [0.0] for c in graph.nodes}))
    assert "M" not in result.adjustment_set
    assert "M" in result.diagnostics["dropped_mediators"]


def test_identify_effect_picks_second_candidate(monkeypatch, fake_dowhy) -> None:
    """First candidate non-identifiable, second identifiable -> chose_dag == 1."""
    fake_dowhy(["Z"])  # whatever DoWhy returns; doesn't matter for the first call

    bad = CausalGraph.from_edge_list(
        [("T", "Y")],
        roles={"T": VariableRole.TREATMENT, "Y": VariableRole.OUTCOME},
    )
    good = CausalGraph.from_edge_list(
        [("Z", "T"), ("Z", "Y"), ("T", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "Z": VariableRole.CONFOUNDER,
        },
    )

    # Make identify_effect return non-identifiable for ``bad`` and identifiable for ``good``
    real_identify = q5_identify.identify_effect

    def fake_single(estimand, graph, df=None, *, candidate_graphs=None):
        if candidate_graphs is not None:
            return real_identify(
                estimand, graph, df=df, candidate_graphs=candidate_graphs
            )
        if graph is bad:
            return IdentificationResult(identifiable=False, strategy="non-identifiable")
        return IdentificationResult(
            identifiable=True, strategy="backdoor", adjustment_set=("Z",)
        )

    monkeypatch.setattr(q5_identify, "identify_effect", fake_single)

    result = q5_identify.identify_effect(
        _estimand(), bad, candidate_graphs=(bad, good)
    )
    assert result.identifiable
    assert result.diagnostics["chose_dag"] == 1
    assert result.adjustment_set == ("Z",)
