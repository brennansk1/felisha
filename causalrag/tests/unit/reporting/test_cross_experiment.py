"""Tests for the cross-experiment synthesis pre-pass."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.protocol import DatasetSpec, RoadmapWalk, StudyProtocol
from causalrag.core.result import EstimationResult
from causalrag.reporting.cross_experiment import (
    ChainNarrative,
    Contradiction,
    CrossExperimentAnalysis,
    Reinforcement,
    analyze_cross_experiment,
    cross_experiment_block_for_prompt,
)


# ─────────── helpers ─────────────────────────────────────────────────────


def _make_walk(
    *,
    hid: str,
    treatment: str = "T",
    outcome: str = "Y",
    klass: EstimandClass = EstimandClass.ATE,
    point: float = 0.10,
    ci_low: float | None = 0.05,
    ci_high: float | None = 0.15,
    n_used: int = 1000,
    estimator_id: str = "python.dml.linear",
    chain_id: str | None = None,
    parent_id: str | None = None,
    modifiers: tuple[str, ...] = (),
    sensitivity: str | None = "green — robust",
    diagnostics: dict[str, Any] | None = None,
) -> RoadmapWalk:
    return RoadmapWalk(
        hypothesis_id=hid,
        q3_estimand=CausalEstimand.model_validate(
            {
                "class": klass.value,
                "treatment": treatment,
                "outcome": outcome,
                "modifiers": list(modifiers),
                "formal_expression": "E[Y(1)-Y(0)]",
            }
        ),
        q7_estimates=(
            EstimationResult(
                estimator_id=estimator_id,
                estimand_class=klass.value,
                point_estimate=point,
                ci_low=ci_low,
                ci_high=ci_high,
                p_value=0.01,
                n_used=n_used,
                diagnostics=diagnostics or {},
            ),
        ),
        q8_interpretation=sensitivity,
        chain_id=chain_id,
        parent_id=parent_id,
    )


def _protocol_with(walks: list[RoadmapWalk]) -> StudyProtocol:
    return StudyProtocol(
        name="t",
        research_question="does T cause Y?",
        dataset=DatasetSpec(source="csv://x.csv", n_rows=1000, n_cols=3),
        roadmap_walks={w.hypothesis_id: w for w in walks},
    )


def _empty_analysis() -> CrossExperimentAnalysis:
    return CrossExperimentAnalysis(
        contradictions=[],
        reinforcements=[],
        chain_narratives=[],
        overall_theme="",
    )


# ─────────── deterministic pre-pass: contradictions ──────────────────────


def test_opposite_sign_pair_surfaces_candidate_contradiction() -> None:
    """Two walks with opposite-sign point estimates on the same (T, Y)."""
    walks = [
        _make_walk(hid="H1", point=0.10),
        _make_walk(hid="H2", point=-0.08),
    ]
    proto = _protocol_with(walks)

    # Make the LLM step trivially echo an empty analysis so we are testing
    # ONLY the deterministic candidates surfacing into the prompt.
    response = MagicMock()
    response.parsed = _empty_analysis()
    client = MagicMock()
    client.parse.return_value = response

    analysis = analyze_cross_experiment(protocol=proto, client=client)
    # The LLM step replaced everything with empty, so we instead inspect
    # the deterministic-only pathway by short-circuiting via a raise.
    assert isinstance(analysis, CrossExperimentAnalysis)

    # Now exercise the deterministic-only path by forcing an LLM error.
    client2 = MagicMock()
    client2.parse.side_effect = RuntimeError("boom")
    analysis2 = analyze_cross_experiment(protocol=proto, client=client2)

    assert len(analysis2.contradictions) == 1
    c = analysis2.contradictions[0]
    assert {c.exp_a_id, c.exp_b_id} == {"H1", "H2"}
    # Same (T, Y) and same estimand class with no modifiers → structural.
    assert c.severity == "structural"


def test_opposite_sign_different_estimand_class_is_surface() -> None:
    walks = [
        _make_walk(hid="H1", point=0.10, klass=EstimandClass.ATE),
        _make_walk(
            hid="H2",
            point=-0.08,
            klass=EstimandClass.ATT,
        ),
    ]
    proto = _protocol_with(walks)
    client = MagicMock()
    client.parse.side_effect = RuntimeError("boom")
    analysis = analyze_cross_experiment(protocol=proto, client=client)
    assert len(analysis.contradictions) == 1
    assert analysis.contradictions[0].severity == "surface"


def test_same_sign_pair_does_not_surface_contradiction() -> None:
    walks = [
        _make_walk(hid="H1", point=0.10),
        _make_walk(hid="H2", point=0.08),
    ]
    proto = _protocol_with(walks)
    client = MagicMock()
    client.parse.side_effect = RuntimeError("boom")
    analysis = analyze_cross_experiment(protocol=proto, client=client)
    assert analysis.contradictions == []


# ─────────── deterministic pre-pass: chain narratives ────────────────────


def test_three_walks_in_chain_surface_ordered_narrative() -> None:
    walks = [
        _make_walk(hid="H1", chain_id="C1"),  # root
        _make_walk(hid="H2", chain_id="C1", parent_id="H1"),
        _make_walk(hid="H3", chain_id="C1", parent_id="H2"),
    ]
    proto = _protocol_with(walks)
    client = MagicMock()
    client.parse.side_effect = RuntimeError("boom")
    analysis = analyze_cross_experiment(protocol=proto, client=client)
    assert len(analysis.chain_narratives) == 1
    cn = analysis.chain_narratives[0]
    assert cn.chain_id == "C1"
    assert cn.root_hypothesis_id == "H1"
    assert cn.walk_ids_in_order == ["H1", "H2", "H3"]


def test_chain_with_branching_includes_all_walks_root_first() -> None:
    walks = [
        _make_walk(hid="H1", chain_id="C1"),
        _make_walk(hid="H2", chain_id="C1", parent_id="H1"),
        _make_walk(hid="H3", chain_id="C1", parent_id="H1"),
    ]
    proto = _protocol_with(walks)
    client = MagicMock()
    client.parse.side_effect = RuntimeError("boom")
    analysis = analyze_cross_experiment(protocol=proto, client=client)
    assert len(analysis.chain_narratives) == 1
    cn = analysis.chain_narratives[0]
    assert cn.root_hypothesis_id == "H1"
    assert cn.walk_ids_in_order[0] == "H1"
    assert set(cn.walk_ids_in_order) == {"H1", "H2", "H3"}


# ─────────── LLM fabricated-id filtering ─────────────────────────────────


def test_fabricated_hypothesis_ids_are_filtered_out_with_warning(caplog) -> None:
    walks = [
        _make_walk(hid="H1", point=0.10),
        _make_walk(hid="H2", point=-0.08),
    ]
    proto = _protocol_with(walks)

    # LLM invents a non-existent id "H_FAKE" in a contradiction and a
    # reinforcement, plus claims a fake chain root.
    bad_analysis = CrossExperimentAnalysis(
        contradictions=[
            Contradiction(
                exp_a_id="H1",
                exp_b_id="H_FAKE",
                description="fabricated reference",
                severity="surface",
            ),
            Contradiction(
                exp_a_id="H1",
                exp_b_id="H2",
                description="real",
                severity="structural",
            ),
        ],
        reinforcements=[
            Reinforcement(
                exp_ids=["H1", "H_FAKE"],
                description="fabricated",
                strength="weak",
            ),
        ],
        chain_narratives=[
            ChainNarrative(
                chain_id="C1",
                root_hypothesis_id="H_FAKE",
                walk_ids_in_order=["H_FAKE", "H1"],
                story="fabricated chain",
            ),
        ],
        overall_theme="theme",
    )
    response = MagicMock()
    response.parsed = bad_analysis
    client = MagicMock()
    client.parse.return_value = response

    with caplog.at_level(logging.WARNING, logger="causalrag.reporting.cross_experiment"):
        analysis = analyze_cross_experiment(protocol=proto, client=client)

    # Only the real contradiction survived.
    assert len(analysis.contradictions) == 1
    assert analysis.contradictions[0].exp_b_id == "H2"

    # The fabricated reinforcement was dropped.
    assert analysis.reinforcements == []

    # The fabricated chain was dropped.
    assert analysis.chain_narratives == []

    # At least one warning was logged.
    assert any(
        "fabricated" in record.getMessage().lower()
        for record in caplog.records
    )


# ─────────── LLM failure → deterministic-only ────────────────────────────


def test_llm_raise_returns_deterministic_only_analysis() -> None:
    walks = [
        _make_walk(hid="H1", chain_id="C1", point=0.10),
        _make_walk(hid="H2", chain_id="C1", parent_id="H1", point=-0.08),
    ]
    proto = _protocol_with(walks)
    client = MagicMock()
    client.parse.side_effect = RuntimeError("ollama dead")

    analysis = analyze_cross_experiment(protocol=proto, client=client)

    # Must not raise; must contain the deterministic pre-pass output.
    assert isinstance(analysis, CrossExperimentAnalysis)
    assert len(analysis.contradictions) == 1
    assert len(analysis.chain_narratives) == 1
    # overall_theme stays empty because no LLM ran.
    assert analysis.overall_theme == ""


def test_no_completed_walks_returns_empty_analysis_without_calling_llm() -> None:
    proto = _protocol_with(walks=[])
    client = MagicMock()
    analysis = analyze_cross_experiment(protocol=proto, client=client)
    assert isinstance(analysis, CrossExperimentAnalysis)
    assert analysis.contradictions == []
    assert analysis.chain_narratives == []
    client.parse.assert_not_called()


# ─────────── prompt-block formatter ──────────────────────────────────────


def test_cross_experiment_block_for_prompt_is_non_empty_markdown() -> None:
    analysis = CrossExperimentAnalysis(
        contradictions=[
            Contradiction(
                exp_a_id="H1",
                exp_b_id="H2",
                description="opposite signs on same target",
                severity="structural",
            ),
        ],
        reinforcements=[
            Reinforcement(
                exp_ids=["H3", "H4"],
                description="agree on positive effect",
                strength="strong",
            ),
        ],
        chain_narratives=[
            ChainNarrative(
                chain_id="C1",
                root_hypothesis_id="H1",
                walk_ids_in_order=["H1", "H2"],
                story="The chain pivoted from ATE to CATE.",
            ),
        ],
        overall_theme="Mixed evidence with one structural conflict.",
    )
    block = cross_experiment_block_for_prompt(analysis)
    assert block.strip() != ""
    assert "## Cross-experiment context" in block
    assert "Contradictions" in block
    assert "Reinforcements" in block
    assert "Foundation-chain narratives" in block
    assert "H1" in block and "H2" in block and "H3" in block and "H4" in block
    assert "Mixed evidence" in block


def test_cross_experiment_block_empty_analysis_still_produces_markdown() -> None:
    block = cross_experiment_block_for_prompt(_empty_analysis())
    assert "## Cross-experiment context" in block
    # Each section reports "(none)" / "(no chains)" rather than disappearing.
    assert "(none)" in block
