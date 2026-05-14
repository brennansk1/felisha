"""End-to-end discovery test (PDD §33.105).

Uses a synthetic Lalonde-like dataset and a FakeOllamaTransport scripted with
a valid InvestigatorReport. Demonstrates Stages 1a–1d end-to-end without
hitting Ollama.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalrag.core.flags import DataFlag
from causalrag.discovery import run_discovery
from causalrag.llm.ollama_client import FakeOllamaTransport, OllamaClient

pytestmark = pytest.mark.integration


def _synthetic_lalonde(n: int = 600, seed: int = 1) -> pd.DataFrame:
    """Lalonde-style synthetic: pre-treatment income, employment, age, race,
    a binary training program indicator, post-treatment income outcome."""
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 55, size=n)
    educ = rng.integers(7, 16, size=n)
    black = rng.integers(0, 2, size=n)
    hispanic = rng.integers(0, 2, size=n)
    married = rng.integers(0, 2, size=n)
    re74 = rng.gamma(2.0, 5000, size=n)  # pre-treatment income 1974
    re75 = re74 * rng.uniform(0.6, 1.4, size=n)
    nodegree = (educ < 12).astype(int)
    treat = (rng.uniform(size=n) < 0.4).astype(int)
    re78 = re75 + 2000 * treat + rng.normal(0, 500, size=n)
    return pd.DataFrame(
        {
            "age": age,
            "educ": educ,
            "black": black,
            "hispanic": hispanic,
            "married": married,
            "nodegree": nodegree,
            "re74": re74,
            "re75": re75,
            "treat": treat,
            "re78": re78,
        }
    )


def _investigator_response(columns: list[str]) -> dict:
    """A minimal, schema-valid InvestigatorReport for the synthetic dataset."""
    meaning = {
        "age": ("subject age in years at baseline", "pre_treatment"),
        "educ": ("years of education at baseline", "pre_treatment"),
        "black": ("indicator: subject is Black", "baseline"),
        "hispanic": ("indicator: subject is Hispanic", "baseline"),
        "married": ("indicator: subject is married at baseline", "baseline"),
        "nodegree": ("indicator: no high-school degree", "baseline"),
        "re74": ("real earnings in 1974 (pre-treatment)", "pre_treatment"),
        "re75": ("real earnings in 1975 (pre-treatment)", "pre_treatment"),
        "treat": ("binary training program enrollment indicator", "treatment_era"),
        "re78": ("real earnings in 1978 (post-treatment outcome)", "outcome"),
    }
    return {
        "domain_tag": "social_science",
        "columns": [
            {
                "column": c,
                "domain_meaning": meaning.get(c, ("unknown", "unknown"))[0],
                "domain_tag": "social_science",
                "temporal_position": meaning.get(c, ("unknown", "unknown"))[1],
                "watch_for": [],
                "proposed_role": _proposed_role(c),
            }
            for c in columns
        ],
    }


def _expert_response() -> dict:
    """A schema-valid DomainExpertBrief for the synthetic Lalonde data."""
    confounders = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]
    return {
        "domain_summary": "Job-training evaluation dataset (Lalonde NSW family). "
        "Binary treatment is enrollment in a training program; "
        "outcome is 1978 earnings.",
        "treatments": [
            {
                "column": "treat",
                "rationale": "explicit binary training-program indicator",
                "suitability": 0.95,
                "typical_questions": ["Does enrollment increase post-program earnings?"],
            }
        ],
        "outcomes": [
            {
                "column": "re78",
                "rationale": "Post-treatment earnings, continuous, well-measured",
                "measurement_notes": "Real 1978 dollars",
                "censoring_notes": None,
            }
        ],
        "confounders": [
            {
                "treatment": "treat",
                "outcome": "re78",
                "confounders": confounders,
                "rationale": "Standard Lalonde adjustment set",
            }
        ],
        "mediators": [],
        "effect_modifiers": ["age", "educ"],
        "unmeasured_confounders": [
            {
                "name": "motivation_to_train",
                "reason": "Self-selection into the program is driven by latent motivation",
                "observed_proxies": ["re74", "re75"],
            }
        ],
        "candidate_dags": [
            {
                "rank": 1,
                "rationale": "Baseline covariates → treat → re78, with re74/re75 also feeding re78",
                "edges": [
                    ["age", "treat"],
                    ["educ", "treat"],
                    ["re74", "treat"],
                    ["re75", "treat"],
                    ["treat", "re78"],
                    ["age", "re78"],
                    ["educ", "re78"],
                    ["re74", "re78"],
                    ["re75", "re78"],
                ],
                "distinguishing_edges": [],
            }
        ],
        "identification_warnings": ["Self-selection threatens unconfoundedness"],
    }


def _scripted_transport() -> "FakeOllamaTransport":
    """A transport that dispatches based on the role keyword the prompt contains."""
    return FakeOllamaTransport(
        {
            "Per-column investigator output": json.dumps(_expert_response()),
            "Statistical profile": json.dumps(_investigator_response([
                "age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75", "treat", "re78"
            ])),
        }
    )


def _proposed_role(name: str) -> str | None:
    if name == "treat":
        return "treatment"
    if name == "re78":
        return "outcome"
    if name in {"age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"}:
        return "confounder"
    return None


def test_discovery_with_fake_llm(tmp_path: Path) -> None:
    df = _synthetic_lalonde()
    client = OllamaClient(
        model="qwen3:14b-q4_K_M",
        seed=42,
        cassette_dir=tmp_path / "cassettes",
        transport=_scripted_transport(),
        allow_live=True,
    )

    result = run_discovery(
        source=df, client=client, treatment="treat", outcome="re78"
    )

    # Flags from the deterministic emitter
    assert DataFlag.BINARY_TREATMENT in result.flags
    assert DataFlag.CONTINUOUS_OUTCOME in result.flags
    assert DataFlag.SMALL_SAMPLE not in result.flags  # n=600

    # Investigator columns match the profile
    assert result.investigator is not None
    assert len(result.investigator.columns) == df.shape[1]
    assert result.investigator.domain_tag == "social_science"

    # Expert brief landed and produced a candidate DAG
    assert result.expert is not None
    assert result.expert.treatments[0].column == "treat"
    assert len(result.candidate_graphs) == 1
    dag = result.candidate_graphs[0]
    assert dag.rank == 1
    assert ("treat", "re78") in {(e.source, e.target) for e in dag.edges}

    # Variable specs carry roles + semantic descriptions
    by_name = {v.name: v for v in result.columns}
    assert by_name["treat"].role.value == "treatment"
    assert by_name["re78"].role.value == "outcome"
    assert by_name["age"].semantic_description == "subject age in years at baseline"

    # Confounder audit ran and produced rows
    assert any(c.confounder == "age" for c in result.confounder_audit)

    # Second run replays from cassettes (no live)
    client2 = OllamaClient(
        model="qwen3:14b-q4_K_M",
        seed=42,
        cassette_dir=tmp_path / "cassettes",
        transport=FakeOllamaTransport({"x": '{"impossible": true}'}),
        allow_live=False,
    )
    result2 = run_discovery(source=df, client=client2, treatment="treat", outcome="re78")
    assert result2.investigator is not None
    assert result2.expert is not None
    assert {v.name for v in result2.columns} == {v.name for v in result.columns}


def test_discovery_without_llm() -> None:
    df = _synthetic_lalonde(n=180)  # n < 200 triggers SMALL_SAMPLE
    result = run_discovery(source=df, treatment="treat", outcome="re78")
    assert result.investigator is None
    assert DataFlag.SMALL_SAMPLE in result.flags
    assert DataFlag.BINARY_TREATMENT in result.flags


def test_discovery_from_csv(tmp_path: Path) -> None:
    df = _synthetic_lalonde()
    csv_path = tmp_path / "lalonde.csv"
    df.to_csv(csv_path, index=False)
    result = run_discovery(source=csv_path, treatment="treat", outcome="re78")
    assert result.dataframe.shape == df.shape
    assert "csv" in result.source_describe["source"]


def test_discovery_report_round_trips_through_studyprotocol(tmp_path: Path) -> None:
    """The discovery report must survive YAML round-trip via the StudyProtocol."""
    from causalrag.core.protocol import StudyProtocol

    df = _synthetic_lalonde(n=300)
    result = run_discovery(source=df, treatment="treat", outcome="re78")
    p = StudyProtocol(
        name="lalonde_demo", discovery=result.to_report(), flags=set(result.flags)
    )
    yaml_text = p.to_yaml()
    p2 = StudyProtocol.from_yaml(yaml_text)
    assert p2.discovery is not None
    assert len(p2.discovery.columns) == df.shape[1]
    assert p2.flags == result.flags
