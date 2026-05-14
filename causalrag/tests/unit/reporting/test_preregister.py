"""Tests for the OSF / AsPredicted / target-trial preregistration exporters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.protocol import (
    DatasetSpec,
    DiscoveryReport,
    Hypothesis,
    StudyProtocol,
)
from causalrag.core.roles import VariableRole, VariableSpec
from causalrag.reporting.preregister import (
    TTE_ELEMENT_HEADINGS,
    export_aspredicted_markdown,
    export_osf_preregistration,
    export_target_trial_protocol,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_protocol() -> StudyProtocol:
    return StudyProtocol(name="EmptyStudy")


@pytest.fixture
def populated_protocol() -> StudyProtocol:
    columns = (
        VariableSpec(name="age", role=VariableRole.CONFOUNDER, dtype="int64"),
        VariableSpec(name="sex", role=VariableRole.CONFOUNDER, dtype="str"),
        VariableSpec(name="drug", role=VariableRole.TREATMENT, dtype="bool"),
        VariableSpec(name="dose", role=VariableRole.TREATMENT, dtype="float64"),
        VariableSpec(name="recovered", role=VariableRole.OUTCOME, dtype="bool"),
        VariableSpec(name="ttr", role=VariableRole.OUTCOME, dtype="float64"),
    )
    dataset = DatasetSpec(
        source="csv://study.csv", n_rows=2048, n_cols=6, columns=columns
    )
    discovery = DiscoveryReport(columns=columns)
    estimand1 = CausalEstimand(
        **{"class": EstimandClass.ATE},
        treatment="drug",
        outcome="recovered",
        formal_expression="E[Y(1) - Y(0)]",
    )
    estimand2 = CausalEstimand(
        **{"class": EstimandClass.ATT},
        treatment="dose",
        outcome="ttr",
        formal_expression="E[Y(1) - Y(0) | T=1]",
    )
    hypotheses = (
        Hypothesis(
            id="H1",
            treatment="drug",
            outcome="recovered",
            estimand=estimand1,
            rationale="Primary efficacy",
        ),
        Hypothesis(
            id="H2",
            treatment="dose",
            outcome="ttr",
            estimand=estimand2,
            rationale="Dose-response",
        ),
    )
    return StudyProtocol(
        name="DrugStudy",
        dataset=dataset,
        discovery=discovery,
        hypothesis_queue=hypotheses,
    )


# ---------------------------------------------------------------------------
# OSF preregistration
# ---------------------------------------------------------------------------


def test_osf_empty_protocol_does_not_crash(
    empty_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = export_osf_preregistration(empty_protocol, tmp_path / "osf.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["registration_schema"] == "openEnded-2.0"
    assert data["title"] == "EmptyStudy"
    assert data["study_information"]["hypotheses"] == []


def test_osf_populated_protocol_has_two_hypotheses(
    populated_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = export_osf_preregistration(populated_protocol, tmp_path / "osf.json")
    data = json.loads(out.read_text())
    hyps = data["study_information"]["hypotheses"]
    assert len(hyps) == 2
    ids = {h["id"] for h in hyps}
    assert ids == {"H1", "H2"}


def test_osf_round_trip_json_shape(
    populated_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = export_osf_preregistration(populated_protocol, tmp_path / "osf.json")
    raw = out.read_text()
    # Round-trip through json.loads.
    data = json.loads(raw)
    for key in (
        "registration_schema",
        "title",
        "study_information",
        "design_plan",
        "sampling_plan",
        "variables",
        "analysis_plan",
    ):
        assert key in data
    assert "drug" in data["variables"]["manipulated_variables"]
    assert "dose" in data["variables"]["manipulated_variables"]
    assert "recovered" in data["variables"]["measured_variables"]
    assert "ATE" in data["analysis_plan"]["statistical_models"]
    assert "bh" in data["analysis_plan"]["inference_criteria"]
    assert data["sampling_plan"]["sample_size"] == 2048


# ---------------------------------------------------------------------------
# AsPredicted markdown
# ---------------------------------------------------------------------------


def test_aspredicted_empty_protocol_does_not_crash(
    empty_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = export_aspredicted_markdown(empty_protocol, tmp_path / "ap.md")
    assert out.exists()
    text = out.read_text()
    assert "AsPredicted Preregistration" in text
    # All 9 numbered questions present.
    for i in range(1, 10):
        assert f"## {i}." in text


def test_aspredicted_populated_protocol(
    populated_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = export_aspredicted_markdown(populated_protocol, tmp_path / "ap.md")
    text = out.read_text()
    assert "H1" in text
    assert "H2" in text
    assert "drug" in text
    assert "recovered" in text
    for i in range(1, 10):
        assert f"## {i}." in text


# ---------------------------------------------------------------------------
# Target-trial emulation
# ---------------------------------------------------------------------------


def test_tte_empty_protocol_does_not_crash(
    empty_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = export_target_trial_protocol(empty_protocol, tmp_path / "tte.md")
    assert out.exists()
    text = out.read_text()
    for heading in TTE_ELEMENT_HEADINGS:
        assert f"## {heading}" in text
    # Inferred fields must be flagged.
    assert "[INFERRED]" in text


def test_tte_populated_has_seven_headings_and_specifics(
    populated_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = export_target_trial_protocol(populated_protocol, tmp_path / "tte.md")
    text = out.read_text()
    assert len(TTE_ELEMENT_HEADINGS) == 7
    for heading in TTE_ELEMENT_HEADINGS:
        assert f"## {heading}" in text
    assert "[SPECIFIED]" in text
    assert "drug" in text
    assert "recovered" in text
    assert "ATE" in text


def test_returned_path_matches_output_path(
    empty_protocol: StudyProtocol, tmp_path: Path
) -> None:
    target_osf = tmp_path / "deep" / "nested" / "osf.json"
    target_ap = tmp_path / "deep" / "nested" / "ap.md"
    target_tte = tmp_path / "deep" / "nested" / "tte.md"
    assert export_osf_preregistration(empty_protocol, target_osf) == target_osf
    assert export_aspredicted_markdown(empty_protocol, target_ap) == target_ap
    assert export_target_trial_protocol(empty_protocol, target_tte) == target_tte
    # The exporters should create missing parent dirs.
    assert target_osf.exists()
    assert target_ap.exists()
    assert target_tte.exists()
