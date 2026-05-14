from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.protocol import DatasetSpec, Hypothesis, LLMConfig, StudyProtocol
from causalrag.core.roles import VariableRole, VariableSpec

# ----- Strategies ---------------------------------------------------------------

identifier = st.text(
    alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    min_size=1,
    max_size=8,
).filter(lambda s: s.isidentifier())

tier = st.sampled_from(["data-scientist", "academic", "domain-expert"])


@st.composite
def variable_specs(draw: st.DrawFn) -> VariableSpec:
    return VariableSpec(
        name=draw(identifier),
        role=draw(st.sampled_from(list(VariableRole))),
        dtype=draw(st.sampled_from(["int64", "float64", "bool", "str", "datetime64"])),
        nullable=draw(st.booleans()),
        semantic_description=draw(st.one_of(st.none(), st.text(max_size=40))),
        unit=draw(st.one_of(st.none(), st.text(max_size=12))),
        llm_confidence=draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))),
        analyst_override=draw(st.booleans()),
    )


treatment_only_flags = st.sets(
    st.sampled_from(
        [
            DataFlag.PANEL_STRUCTURE,
            DataFlag.LONGITUDINAL,
            DataFlag.CLUSTERED,
            DataFlag.SMALL_SAMPLE,
            DataFlag.HIGH_DIMENSIONAL,
            DataFlag.HEAVY_MISSINGNESS,
            DataFlag.MEDIATOR_PROPOSED,
            DataFlag.NEGATIVE_CONTROL_AVAILABLE,
        ]
    ),
    max_size=4,
)


@st.composite
def study_protocols(draw: st.DrawFn) -> StudyProtocol:
    cols = tuple(draw(st.lists(variable_specs(), max_size=3, unique_by=lambda v: v.name)))
    return StudyProtocol(
        name=draw(identifier),
        tier=draw(tier),
        research_question=draw(st.one_of(st.none(), st.text(max_size=80))),
        dataset=DatasetSpec(source="csv://example.csv", n_rows=draw(st.integers(0, 10_000)), columns=cols),
        flags=draw(treatment_only_flags),
        llm=LLMConfig(seed=draw(st.integers(0, 2**31 - 1))),
        counterfactual_ratio=draw(st.floats(min_value=0.0, max_value=1.0)),
    )


# ----- Tests --------------------------------------------------------------------


def test_minimal_protocol_yaml_roundtrip() -> None:
    p = StudyProtocol(name="demo")
    text = p.to_yaml()
    p2 = StudyProtocol.from_yaml(text)
    assert p2.name == "demo"
    assert p2.tier == "academic"
    assert p2.flags == set()
    assert p2.llm.backend == "ollama"


def test_protocol_writes_and_reads_file(tmp_path: Path) -> None:
    p = StudyProtocol(
        name="demo",
        research_question="Does coffee cause focus?",
        dataset=DatasetSpec(source="csv://focus.csv"),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
    )
    path = tmp_path / "study.causalrag.yaml"
    p.write_yaml(path)
    p2 = StudyProtocol.read_yaml(path)
    assert p2.research_question == "Does coffee cause focus?"
    assert p2.flags == {DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME}


def test_hypothesis_with_estimand_roundtrip() -> None:
    est = CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": "T",
            "outcome": "Y",
            "formal_expression": "E[Y(1) - Y(0)]",
        }
    )
    h = Hypothesis(id="h1", treatment="T", outcome="Y", estimand=est, flags={DataFlag.BINARY_TREATMENT})
    p = StudyProtocol(name="demo", hypothesis_queue=(h,))
    p2 = StudyProtocol.from_yaml(p.to_yaml())
    assert p2.hypothesis_queue[0].estimand is not None
    assert p2.hypothesis_queue[0].estimand.klass is EstimandClass.ATE
    assert p2.hypothesis_queue[0].flags == {DataFlag.BINARY_TREATMENT}


def test_rejects_conflicting_treatment_flags() -> None:
    import pytest

    with pytest.raises(ValueError):
        StudyProtocol(
            name="demo",
            flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_TREATMENT},
        )


@given(study_protocols())
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_roundtrip(p: StudyProtocol) -> None:
    text = p.to_yaml()
    p2 = StudyProtocol.from_yaml(text)
    assert p2.name == p.name
    assert p2.tier == p.tier
    assert p2.flags == p.flags
    assert p2.counterfactual_ratio == p.counterfactual_ratio
    assert p2.llm.seed == p.llm.seed
    if p.dataset and p.dataset.columns:
        assert len(p2.dataset.columns) == len(p.dataset.columns)
        assert {c.name for c in p2.dataset.columns} == {c.name for c in p.dataset.columns}


def test_timestamps_serialize_isoformat() -> None:
    now = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc).replace(tzinfo=None)
    p = StudyProtocol(name="demo", created=now, updated=now)
    text = p.to_yaml()
    assert "2026-05-13" in text
