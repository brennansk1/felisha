"""Robustness tests for the executive-synthesis layer.

Covers classifier dtype/flag awareness, estimand-aware magnitude
scaling (ATE vs ATT vs ATC), LLM-failure stub generation, and
deterministic validation of fabricated fields and confidence flags.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.protocol import DatasetSpec, RoadmapWalk, StudyProtocol
from causalrag.core.result import EstimationResult
from causalrag.core.roles import VariableSpec
from causalrag.reporting.synthesis import (
    ExecutiveSynthesis,
    Insight,
    _classify_outcome_units,
    _magnitude,
    synthesize_insights,
)


# ─────────── helpers ─────────────────────────────────────────────────────


def _make_walk(
    *,
    hid: str = "H1",
    treatment: str = "T",
    outcome: str = "Y",
    klass: EstimandClass = EstimandClass.ATE,
    point: float = 0.10,
    ci_low: float | None = 0.05,
    ci_high: float | None = 0.15,
    n_used: int = 1000,
    estimator_id: str = "python.dml.linear",
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
    )


def _protocol_with(walks: list[RoadmapWalk], *, flags: set[DataFlag] | None = None,
                   columns: tuple[VariableSpec, ...] = ()) -> StudyProtocol:
    return StudyProtocol(
        name="t",
        research_question="does T cause Y?",
        dataset=DatasetSpec(source="csv://x.csv", n_rows=1000, n_cols=3, columns=columns),
        flags=flags or set(),
        roadmap_walks={w.hypothesis_id: w for w in walks},
    )


def _good_synthesis(walk_id: str = "H1", estimator: str = "python.dml.linear",
                    confidence: str = "high") -> ExecutiveSynthesis:
    return ExecutiveSynthesis(
        inferred_domain="business",
        tldr="A finding.",
        findings=[
            Insight(
                rank=1,
                hypothesis_id=walk_id,
                headline="Treatment T raises outcome Y.",
                quantified_effect="+10 percentage points",
                domain_implication="Operators should consider pilots.",
                suggested_next_step="Run a holdout test.",
                confidence=confidence,  # type: ignore[arg-type]
                caveats=[],
                estimator_used=estimator,
            )
        ],
    )


# ─────────── classifier ──────────────────────────────────────────────────


def test_classify_flag_overrides_regex_units_sold_is_time_when_censored() -> None:
    # 'units_sold' would normally hit the count regex; the censoring flag
    # must win and route to 'time'.
    kind = _classify_outcome_units(
        "units_sold", frozenset({DataFlag.RIGHT_CENSORED_OUTCOME})
    )
    assert kind == "time"


def test_classify_binary_flag_overrides_monetary_regex() -> None:
    kind = _classify_outcome_units(
        "cost_to_complete", frozenset({DataFlag.BINARY_OUTCOME})
    )
    assert kind == "rate"


def test_classify_dtype_bool_routes_to_rate_regardless_of_name() -> None:
    proto = _protocol_with(
        walks=[],
        columns=(
            VariableSpec(name="cost_to_complete", dtype="bool"),
        ),
    )
    kind = _classify_outcome_units(
        "cost_to_complete", frozenset(), protocol=proto
    )
    assert kind == "rate"


def test_classify_profile_indicator_routes_to_rate() -> None:
    # Integer column whose profile says values are in [0, 1] → rate.
    spec = VariableSpec(name="cost_to_complete", dtype="int64")
    # The codebase's VariableSpec doesn't yet carry a `profile` attr;
    # we stash one via setattr so the classifier can read it.
    object.__setattr__(spec, "profile", {"min": 0, "max": 1, "pct_zeros": 0.5})
    proto = _protocol_with(walks=[], columns=(spec,))
    kind = _classify_outcome_units(
        "cost_to_complete", frozenset(), protocol=proto
    )
    assert kind == "rate"


def test_classify_falls_through_to_monetary_regex_when_no_profile() -> None:
    # No flags, no dtype info → fall through to name regex.
    kind = _classify_outcome_units("cost_total", frozenset())
    assert kind == "monetary"


def test_classify_t2e_with_censoring_flag_is_time() -> None:
    # The pathological case the audit called out: a survival outcome
    # named 't2e' matches no regex.
    kind = _classify_outcome_units(
        "t2e", frozenset({DataFlag.RIGHT_CENSORED_OUTCOME})
    )
    assert kind == "time"


# ─────────── magnitude scaling ───────────────────────────────────────────


def test_magnitude_att_uses_n_treated_not_n_used() -> None:
    info = _magnitude(
        "converted",
        point=0.10,
        n_used=1000,
        flags=frozenset({DataFlag.BINARY_OUTCOME}),
        estimand_klass=EstimandClass.ATT.value,
        n_treated=200,
        n_control=800,
    )
    # ATT must multiply by n_treated (200), not n_used (1000).
    assert "expected_count_att_n_treated" in info
    assert info["expected_count_att_n_treated"] == pytest.approx(20.0, rel=1e-6)
    assert "expected_count_ate_n_used" not in info


def test_magnitude_atc_uses_n_control() -> None:
    info = _magnitude(
        "converted",
        point=0.10,
        n_used=1000,
        flags=frozenset({DataFlag.BINARY_OUTCOME}),
        estimand_klass=EstimandClass.ATC.value,
        n_treated=200,
        n_control=800,
    )
    assert info["expected_count_atc_n_control"] == pytest.approx(80.0, rel=1e-6)


def test_magnitude_ate_uses_n_used() -> None:
    info = _magnitude(
        "converted",
        point=0.10,
        n_used=1000,
        flags=frozenset({DataFlag.BINARY_OUTCOME}),
        estimand_klass=EstimandClass.ATE.value,
    )
    assert info["expected_count_ate_n_used"] == pytest.approx(100.0, rel=1e-6)


def test_magnitude_unknown_population_labels_uncertain_with_caveat() -> None:
    info = _magnitude(
        "converted",
        point=0.10,
        n_used=1000,
        flags=frozenset({DataFlag.BINARY_OUTCOME}),
        estimand_klass=EstimandClass.LATE.value,
    )
    assert "expected_count_uncertain_population" in info
    assert "population_scale_caveat" in info


def test_magnitude_currency_uses_renamed_key_with_caveat() -> None:
    info = _magnitude(
        "revenue",
        point=2.5,
        n_used=500,
        flags=frozenset(),
    )
    assert "effect_at_analysis_sample_currency" in info
    # The old, misleading key must be gone.
    assert "aggregate_currency_change_at_analysis_n" not in info
    assert "effect_at_analysis_sample_currency_caveat" in info


# ─────────── synthesize_insights failure path ────────────────────────────


def test_synthesize_returns_stub_when_client_raises(tmp_path) -> None:
    walk = _make_walk()
    proto = _protocol_with(walks=[walk])
    client = MagicMock()
    client.parse.side_effect = RuntimeError("boom")

    err_log = tmp_path / "executive_synthesis_error.txt"
    synth = synthesize_insights(
        protocol=proto,
        df=pd.DataFrame({"T": [0, 1], "Y": [0, 1]}),
        client=client,
        error_log_path=err_log,
    )
    assert synth.inferred_domain == "other"
    assert "synthesis failed" in synth.tldr
    assert "RuntimeError" in synth.tldr
    assert len(synth.findings) == 1
    f = synth.findings[0]
    assert f.hypothesis_id == "<system>"
    assert f.confidence == "low"
    # Error file was written.
    assert err_log.exists()
    assert "RuntimeError" in err_log.read_text()


def test_synthesize_failure_without_error_log_path_does_not_raise() -> None:
    walk = _make_walk()
    proto = _protocol_with(walks=[walk])
    client = MagicMock()
    client.parse.side_effect = ValueError("nope")
    synth = synthesize_insights(
        protocol=proto,
        df=pd.DataFrame({"T": [0, 1], "Y": [0, 1]}),
        client=client,
    )
    assert "synthesis failed" in synth.tldr


# ─────────── validation: fabricated fields ───────────────────────────────


def test_synthesize_drops_fabricated_hypothesis_id() -> None:
    walk = _make_walk(hid="H1")
    proto = _protocol_with(walks=[walk])
    bad = ExecutiveSynthesis(
        inferred_domain="business",
        tldr="…",
        findings=[
            Insight(
                rank=1,
                hypothesis_id="H_FAKE",
                headline="x",
                quantified_effect="y",
                domain_implication="z",
                suggested_next_step="w",
                confidence="high",
                estimator_used="python.dml.linear",
            ),
            Insight(
                rank=2,
                hypothesis_id="H1",
                headline="x",
                quantified_effect="y",
                domain_implication="z",
                suggested_next_step="w",
                confidence="high",
                estimator_used="python.dml.linear",
            ),
        ],
    )
    response = MagicMock()
    response.parsed = bad
    client = MagicMock()
    client.parse.return_value = response

    synth = synthesize_insights(
        protocol=proto,
        df=pd.DataFrame({"T": [0, 1], "Y": [0, 1]}),
        client=client,
    )
    ids = [f.hypothesis_id for f in synth.findings]
    assert ids == ["H1"]
    assert any("H_FAKE" in w for w in synth.validation_warnings)


def test_synthesize_corrects_fabricated_estimator_id() -> None:
    walk = _make_walk(hid="H1", estimator_id="python.dml.linear")
    proto = _protocol_with(walks=[walk])
    bad = _good_synthesis(walk_id="H1", estimator="r.bart.fake")
    response = MagicMock()
    response.parsed = bad
    client = MagicMock()
    client.parse.return_value = response

    synth = synthesize_insights(
        protocol=proto,
        df=pd.DataFrame({"T": [0, 1], "Y": [0, 1]}),
        client=client,
    )
    assert synth.findings[0].estimator_used == "python.dml.linear"
    assert any("r.bart.fake" in w for w in synth.validation_warnings)


# ─────────── deterministic confidence enforcement ────────────────────────


def test_confidence_forced_low_when_ci_crosses_zero() -> None:
    walk = _make_walk(hid="H1", ci_low=-0.05, ci_high=0.15)
    proto = _protocol_with(walks=[walk])
    bad = _good_synthesis(walk_id="H1", confidence="high")
    response = MagicMock()
    response.parsed = bad
    client = MagicMock()
    client.parse.return_value = response

    synth = synthesize_insights(
        protocol=proto,
        df=pd.DataFrame({"T": [0, 1], "Y": [0, 1]}),
        client=client,
    )
    assert synth.findings[0].confidence == "low"
    assert any("CI crosses zero" in w for w in synth.validation_warnings)


def test_confidence_forced_low_when_sensitivity_red() -> None:
    walk = _make_walk(hid="H1", sensitivity="red — unmeasured confounding")
    proto = _protocol_with(walks=[walk])
    bad = _good_synthesis(walk_id="H1", confidence="high")
    response = MagicMock()
    response.parsed = bad
    client = MagicMock()
    client.parse.return_value = response

    synth = synthesize_insights(
        protocol=proto,
        df=pd.DataFrame({"T": [0, 1], "Y": [0, 1]}),
        client=client,
    )
    assert synth.findings[0].confidence == "low"


def test_confidence_capped_medium_when_n_used_small() -> None:
    walk = _make_walk(hid="H1", n_used=50)
    proto = _protocol_with(walks=[walk])
    bad = _good_synthesis(walk_id="H1", confidence="high")
    response = MagicMock()
    response.parsed = bad
    client = MagicMock()
    client.parse.return_value = response

    synth = synthesize_insights(
        protocol=proto,
        df=pd.DataFrame({"T": [0, 1], "Y": [0, 1]}),
        client=client,
    )
    assert synth.findings[0].confidence == "medium"
    assert any("n_used" in w for w in synth.validation_warnings)
