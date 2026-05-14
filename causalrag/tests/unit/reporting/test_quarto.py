"""Tests for ``causalrag.reporting.quarto`` (Sprint 1.4)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.protocol import (
    DatasetSpec,
    DiscoveryReport,
    Hypothesis,
    RoadmapWalk,
    StudyProtocol,
)
from causalrag.core.result import EstimationResult
from causalrag.core.roles import VariableRole, VariableSpec
from causalrag.reporting.quarto import (
    SENSITIVITY_CALLOUT,
    render_quarto,
)
from causalrag.reporting.synthesis import ExecutiveSynthesis, Insight


# ─────────────────────────── fixtures ──────────────────────────────────


@pytest.fixture
def empty_protocol() -> StudyProtocol:
    return StudyProtocol(name="EmptyStudy")


@pytest.fixture
def two_hypothesis_protocol() -> StudyProtocol:
    columns = (
        VariableSpec(name="age", role=VariableRole.CONFOUNDER, dtype="int64"),
        VariableSpec(name="drug", role=VariableRole.TREATMENT, dtype="bool"),
        VariableSpec(name="recovered", role=VariableRole.OUTCOME, dtype="bool"),
    )
    dataset = DatasetSpec(
        source="csv://study.csv", n_rows=1024, n_cols=3, columns=columns
    )
    discovery = DiscoveryReport(columns=columns, domain_brief="Drug efficacy study.")
    e1 = CausalEstimand(
        **{"class": EstimandClass.ATE},
        treatment="drug",
        outcome="recovered",
        formal_expression="E[Y(1) - Y(0)]",
    )
    e2 = CausalEstimand(
        **{"class": EstimandClass.ATE},
        treatment="age",
        outcome="recovered",
        formal_expression="E[Y(1) - Y(0)]",
    )
    hypotheses = (
        Hypothesis(id="H1", treatment="drug", outcome="recovered", estimand=e1),
        Hypothesis(id="H2", treatment="age", outcome="recovered", estimand=e2),
    )
    walks = {
        "H1": RoadmapWalk(
            hypothesis_id="H1",
            q3_estimand=e1,
            q5_identification={
                "strategy": "back-door adjustment",
                "identifiable": True,
                "adjustment_set": ["age"],
            },
            q7_estimates=(
                EstimationResult(
                    estimator_id="python.tmle3.ate",
                    estimand_class="ATE",
                    point_estimate=0.12,
                    ci_low=0.04,
                    ci_high=0.20,
                    p_value=0.003,
                    n_used=1024,
                ),
            ),
            q8_interpretation="Effect is stable under plausible confounding.",
            sensitivity_verdict="green",
        ),
        "H2": RoadmapWalk(
            hypothesis_id="H2",
            q3_estimand=e2,
            q7_estimates=(
                EstimationResult(
                    estimator_id="python.dml.linear",
                    estimand_class="ATE",
                    point_estimate=0.01,
                    ci_low=-0.02,
                    ci_high=0.04,
                    n_used=1024,
                ),
            ),
            sensitivity_verdict="red",
        ),
    }
    return StudyProtocol(
        name="DrugStudy",
        dataset=dataset,
        discovery=discovery,
        hypothesis_queue=hypotheses,
        roadmap_walks=walks,
    )


@pytest.fixture
def example_synthesis() -> ExecutiveSynthesis:
    return ExecutiveSynthesis(
        inferred_domain="clinical",
        tldr="Drug X meaningfully improves recovery in adults.",
        findings=[
            Insight(
                rank=1,
                hypothesis_id="H1",
                headline="Drug X raises recovery by 12 percentage points.",
                quantified_effect="+12 pp recovery rate",
                domain_implication="Patients on drug X may benefit clinically.",
                suggested_next_step="Confirm in an RCT.",
                confidence="high",
                estimator_used="python.tmle3.ate",
            ),
        ],
        overall_caveats=["Observational data only."],
    )


# ─────────────────────────── tests ────────────────────────────────────


def _extract_yaml_header(text: str) -> dict:
    assert text.startswith("---"), "qmd must open with a YAML header"
    end = text.find("\n---", 3)
    assert end != -1, "qmd must close the YAML header"
    header = text[3:end].strip()
    return yaml.safe_load(header)


def test_empty_protocol_produces_valid_qmd(
    empty_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = render_quarto(empty_protocol, output_dir=tmp_path)
    assert out.exists()
    assert out.suffix == ".qmd"
    text = out.read_text(encoding="utf-8")
    assert "EmptyStudy" in text
    assert "no experiments run yet" in text.lower()


def test_yaml_header_is_parseable(
    empty_protocol: StudyProtocol, tmp_path: Path
) -> None:
    out = render_quarto(empty_protocol, output_dir=tmp_path)
    text = out.read_text(encoding="utf-8")
    header = _extract_yaml_header(text)
    assert header["title"] == "EmptyStudy"
    assert "date" in header
    fmt = header["format"]
    # html / pdf / docx all configured.
    assert "html" in fmt
    assert "pdf" in fmt
    assert "docx" in fmt
    assert fmt["html"]["embed-resources"] is True


def test_two_hypotheses_plus_synthesis(
    two_hypothesis_protocol: StudyProtocol,
    example_synthesis: ExecutiveSynthesis,
    tmp_path: Path,
) -> None:
    out = render_quarto(
        two_hypothesis_protocol,
        executive_synthesis=example_synthesis,
        output_dir=tmp_path,
    )
    text = out.read_text(encoding="utf-8")
    # TL;DR present.
    assert "Drug X meaningfully improves recovery" in text
    # Both hypothesis sections present.
    assert "### H1" in text
    assert "### H2" in text
    # Estimator ids surfaced in their sections.
    assert "python.tmle3.ate" in text
    assert "python.dml.linear" in text
    # Domain brief rendered.
    assert "Drug efficacy study." in text


def test_run_quarto_no_op_without_cli(
    empty_protocol: StudyProtocol,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Force ``which('quarto')`` to return None.
    monkeypatch.setattr(shutil, "which", lambda name: None)
    out = render_quarto(empty_protocol, output_dir=tmp_path, run_quarto=True)
    # Returns the .qmd path, no crash.
    assert out.suffix == ".qmd"
    assert out.exists()


def test_sensitivity_verdict_colour_maps_to_callout(
    two_hypothesis_protocol: StudyProtocol, tmp_path: Path
) -> None:
    # Sanity-check the mapping constants.
    assert SENSITIVITY_CALLOUT["green"] == "note"
    assert SENSITIVITY_CALLOUT["yellow"] == "warning"
    assert SENSITIVITY_CALLOUT["red"] == "important"

    out = render_quarto(two_hypothesis_protocol, output_dir=tmp_path)
    text = out.read_text(encoding="utf-8")
    # H1 verdict = green → callout-note. H2 verdict = red → callout-important.
    assert "{.callout-note}" in text
    assert "{.callout-important}" in text


def test_yellow_verdict_maps_to_warning(tmp_path: Path) -> None:
    walks = {
        "H1": RoadmapWalk(
            hypothesis_id="H1",
            q7_estimates=(
                EstimationResult(
                    estimator_id="python.tmle3.ate",
                    estimand_class="ATE",
                    point_estimate=0.05,
                    n_used=300,
                ),
            ),
            q8_interpretation="Mixed signal under stress tests.",
            sensitivity_verdict="yellow",
        ),
    }
    protocol = StudyProtocol(name="YellowStudy", roadmap_walks=walks)
    out = render_quarto(protocol, output_dir=tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "{.callout-warning}" in text
    assert "Mixed signal" in text
