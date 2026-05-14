"""Tests for the DAG-mismatch alert layer (Sprint 6.5.10)."""

from __future__ import annotations

import pytest

from causalrag.core.roles import VariableRole, VariableSpec
from causalrag.discovery.dag_conflicts import (
    DAGConflict,
    DAGMismatchReport,
    detect_conflicts,
)


def _spec(name: str, role: VariableRole = VariableRole.AUXILIARY) -> VariableSpec:
    return VariableSpec(name=name, dtype="float64", role=role)


def test_all_agree_no_conflicts() -> None:
    columns = [
        _spec("age", VariableRole.CONFOUNDER),
        _spec("educ", VariableRole.CONFOUNDER),
        _spec("treat", VariableRole.TREATMENT),
        _spec("y", VariableRole.OUTCOME),
    ]
    brief = [{"name": "age"}, {"name": "educ"}]
    mb = ["age", "educ"]
    report = detect_conflicts(
        target="y",
        columns=columns,
        brief_confounders=brief,
        markov_boundary=mb,
    )
    assert report.conflicts == []
    assert report.high_severity_count == 0
    assert report.low_severity_count == 2


def test_high_severity_when_only_stats_flag_it() -> None:
    columns = [_spec("age"), _spec("y", VariableRole.OUTCOME)]
    brief: list = []
    mb = ["age"]
    report = detect_conflicts(
        target="y",
        columns=columns,
        brief_confounders=brief,
        markov_boundary=mb,
    )
    assert len(report.conflicts) == 1
    c = report.conflicts[0]
    assert c.column == "age"
    assert c.markov_says is True
    assert c.investigator_says is False
    assert c.brief_says is False
    assert c.severity == "high"
    assert any("Statistical MB flagged" in n for n in c.notes)


def test_high_severity_when_only_investigator_says() -> None:
    columns = [_spec("age", VariableRole.CONFOUNDER), _spec("y", VariableRole.OUTCOME)]
    report = detect_conflicts(
        target="y",
        columns=columns,
        brief_confounders=[],
        markov_boundary=[],
    )
    assert len(report.conflicts) == 1
    c = report.conflicts[0]
    assert c.investigator_says is True
    assert c.severity == "high"


def test_medium_severity_two_in_one_out() -> None:
    columns = [_spec("age", VariableRole.CONFOUNDER), _spec("y", VariableRole.OUTCOME)]
    brief = [{"name": "age"}]
    mb: list[str] = []
    report = detect_conflicts(
        target="y",
        columns=columns,
        brief_confounders=brief,
        markov_boundary=mb,
    )
    assert len(report.conflicts) == 1
    c = report.conflicts[0]
    assert c.severity == "medium"


def test_brief_extraction_handles_strings_dicts_and_objects() -> None:
    class _BriefConfounder:
        def __init__(self, name: str) -> None:
            self.name = name

    columns = [
        _spec("age"), _spec("educ"), _spec("married"), _spec("y", VariableRole.OUTCOME)
    ]
    brief = ["age", {"name": "educ"}, _BriefConfounder("married")]
    mb: list[str] = []
    report = detect_conflicts(
        target="y",
        columns=columns,
        brief_confounders=brief,
        markov_boundary=mb,
    )
    # Three columns the brief flagged but neither investigator nor stats did
    flagged = {c.column for c in report.conflicts}
    assert flagged == {"age", "educ", "married"}


def test_target_self_excluded() -> None:
    columns = [_spec("y", VariableRole.CONFOUNDER), _spec("treat", VariableRole.TREATMENT)]
    report = detect_conflicts(
        target="y",
        columns=columns,
        brief_confounders=[],
        markov_boundary=["y"],  # nonsense MB; we should ignore the target
    )
    # The target shouldn't appear in conflicts about itself
    columns_in_report = {c.column for c in report.conflicts}
    assert "y" not in columns_in_report


def test_summary_includes_counts() -> None:
    columns = [
        _spec("age"), _spec("educ", VariableRole.CONFOUNDER),
        _spec("treat", VariableRole.TREATMENT), _spec("y", VariableRole.OUTCOME),
    ]
    report = detect_conflicts(
        target="y",
        columns=columns,
        brief_confounders=[],
        markov_boundary=["age", "educ"],
    )
    assert "high-severity" in report.summary
    assert "medium-severity" in report.summary
    assert "y" in report.summary
