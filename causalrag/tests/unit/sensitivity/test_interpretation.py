"""Tests for the LLM-driven, domain-aware sensitivity interpreter."""

from __future__ import annotations

import warnings
from typing import Any

import pytest

from causalrag.llm.ollama_client import FakeOllamaTransport, OllamaClient
from causalrag.sensitivity.evalue import EValueResult
from causalrag.sensitivity.interpretation import (
    SensitivityInterpretation,
    interpret_sensitivity,
)
from causalrag.sensitivity.sensemakr_py import SensemakrResult


# ─────────── Fixtures ────────────────────────────────────────────────────


def _make_client(responses: dict[str, str | dict | list]) -> OllamaClient:
    """Build an OllamaClient backed by a scripted FakeOllamaTransport.

    No cassettes — direct call path so the test fully controls what the
    LLM 'returns'.
    """
    return OllamaClient(
        model="test:fake",
        cassette_dir=None,
        transport=FakeOllamaTransport(responses=responses),
        allow_live=True,
    )


def _exploding_client() -> OllamaClient:
    """Client whose transport raises on every call — exercises the
    failure-safe fallback path in ``interpret_sensitivity``."""

    class _BoomTransport:
        def generate(self, **_kw: Any) -> str:  # noqa: D401 — test stub
            raise RuntimeError("simulated LLM outage")

        def show(self, model: str) -> dict[str, Any]:
            return {"digest": "sha256:boom"}

    return OllamaClient(
        model="test:fake",
        cassette_dir=None,
        transport=_BoomTransport(),
        allow_live=True,
    )


def _good_evalue() -> EValueResult:
    return EValueResult(
        scale="risk_ratio",
        point_estimate=1.4,
        ci_low=1.1,
        ci_high=1.8,
        e_value=2.4,
        e_value_ci=1.45,
        verdict="Moderately robust",
    )


def _unknown_evalue() -> EValueResult:
    return EValueResult(
        scale="standardized",
        point_estimate=50.0,
        ci_low=None,
        ci_high=None,
        e_value=1.0,
        e_value_ci=None,
        verdict="Unknown — implausible magnitude",
        reason="|standardized| too large; probably wrong scale.",
    )


def _good_sensemakr() -> SensemakrResult:
    return SensemakrResult(
        treatment="dose",
        outcome="recovery",
        estimate=0.4,
        se=0.1,
        t_value=4.0,
        robustness_value=0.18,
        robustness_value_q=0.12,
    )


def _valid_payload(color: str = "yellow") -> dict[str, str]:
    return {
        "verdict_color": color,
        "plain_language": (
            "An unmeasured confounder would need substantial strength "
            "to explain this away."
        ),
        "what_it_rules_out": "Weak confounders comparable to age alone.",
        "what_it_does_not_rule_out": (
            "Strong unmeasured selection on disease severity."
        ),
        "plausibility_of_threshold_confounder": (
            "Plausible in clinical work but uncommon."
        ),
        "rationale": "Yellow verdict: moderately robust.",
    }


# ─────────── 1. Happy path: stubbed JSON round-trips ─────────────────────


def test_stubbed_client_returns_valid_interpretation() -> None:
    client = _make_client({"": _valid_payload("yellow")})
    result = interpret_sensitivity(
        evalue_result=_good_evalue(),
        sensemakr_result=_good_sensemakr(),
        deterministic_verdict="yellow",
        point_estimate=0.4,
        ci_low=0.2,
        ci_high=0.6,
        treatment="dose",
        outcome="recovery",
        domain_brief="A randomized trial of patients with hypertension.",
        outcome_dtype="binary",
        client=client,
    )
    assert isinstance(result, SensitivityInterpretation)
    assert result.verdict_color == "yellow"
    assert "confounder" in result.plain_language.lower()


# ─────────── 2. Color mismatch → deterministic override ──────────────────


def test_color_mismatch_is_overridden_to_deterministic() -> None:
    # LLM emits 'green' but the deterministic verdict is 'red'. The
    # output must be forced back to 'red' and a RuntimeWarning must fire.
    client = _make_client({"": _valid_payload("green")})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = interpret_sensitivity(
            evalue_result=_good_evalue(),
            sensemakr_result=_good_sensemakr(),
            deterministic_verdict="red",
            point_estimate=0.4,
            ci_low=0.2,
            ci_high=0.6,
            treatment="dose",
            outcome="recovery",
            domain_brief="Clinical trial of patients with diabetes.",
            outcome_dtype="binary",
            client=client,
        )
    assert result.verdict_color == "red"
    assert any(
        "overriding" in str(w.message).lower() for w in caught
    ), f"expected override warning, got {[str(w.message) for w in caught]}"


# ─────────── 3. LLM error → safe fallback ────────────────────────────────


def test_llm_error_returns_default_interpretation() -> None:
    client = _exploding_client()
    deterministic_rationale = (
        "Sensitivity yellow. Moderately robust. E-value=2.40 (risk_ratio)."
    )
    result = interpret_sensitivity(
        evalue_result=_good_evalue(),
        sensemakr_result=_good_sensemakr(),
        deterministic_verdict="yellow",
        point_estimate=0.4,
        ci_low=0.2,
        ci_high=0.6,
        treatment="dose",
        outcome="recovery",
        domain_brief="Patients enrolled in a clinical study.",
        outcome_dtype="binary",
        client=client,
        deterministic_rationale=deterministic_rationale,
    )
    assert isinstance(result, SensitivityInterpretation)
    assert result.verdict_color == "yellow"
    # The fallback must surface the deterministic rationale somewhere
    # so the synthesis layer still has quotable text.
    assert "Moderately robust" in result.rationale
    assert "simulated LLM outage" in result.rationale.lower() or \
        "runtimeerror" in result.rationale.lower()


# ─────────── 4. E-value 'unknown' path short-circuits ────────────────────


def test_evalue_unknown_returns_unknown_interpretation() -> None:
    # Build a client whose response would BE valid if it were called —
    # this lets us prove the function never invoked the LLM by asserting
    # the result is the 'unknown' template, not the scripted payload.
    transport = FakeOllamaTransport(responses={"": _valid_payload("green")})
    client = OllamaClient(
        model="test:fake",
        cassette_dir=None,
        transport=transport,
        allow_live=True,
    )
    result = interpret_sensitivity(
        evalue_result=_unknown_evalue(),
        sensemakr_result=None,
        deterministic_verdict="yellow",  # caller's color — ignored
        point_estimate=50.0,
        ci_low=None,
        ci_high=None,
        treatment="dose",
        outcome="recovery",
        domain_brief="Clinical trial in oncology patients.",
        outcome_dtype="continuous",
        client=client,
    )
    assert result.verdict_color == "unknown"
    assert "could not be" in result.plain_language.lower() or \
        "cannot" in result.plain_language.lower()
    # Critical: the LLM must not have been called.
    assert transport.calls == [], (
        "Unknown E-value path must short-circuit BEFORE calling the LLM "
        f"— but transport recorded {len(transport.calls)} call(s)."
    )


def test_deterministic_unknown_color_also_short_circuits() -> None:
    transport = FakeOllamaTransport(responses={"": _valid_payload("green")})
    client = OllamaClient(
        model="test:fake",
        cassette_dir=None,
        transport=transport,
        allow_live=True,
    )
    # E-value is "fine" but the aggregator gave 'unknown' anyway (e.g.
    # everything errored). We should still refuse to call the LLM.
    result = interpret_sensitivity(
        evalue_result=_good_evalue(),
        sensemakr_result=None,
        deterministic_verdict="unknown",
        point_estimate=0.4,
        ci_low=0.2,
        ci_high=0.6,
        treatment="dose",
        outcome="recovery",
        domain_brief="A general dataset.",
        outcome_dtype="binary",
        client=client,
    )
    assert result.verdict_color == "unknown"
    assert transport.calls == []


# ─────────── 5. Domain inference appears in the prompt ───────────────────


def test_clinical_brief_produces_clinical_prompt() -> None:
    transport = FakeOllamaTransport(responses={"": _valid_payload("yellow")})
    client = OllamaClient(
        model="test:fake",
        cassette_dir=None,
        transport=transport,
        allow_live=True,
    )
    interpret_sensitivity(
        evalue_result=_good_evalue(),
        sensemakr_result=_good_sensemakr(),
        deterministic_verdict="yellow",
        point_estimate=0.4,
        ci_low=0.2,
        ci_high=0.6,
        treatment="dose",
        outcome="recovery",
        domain_brief=(
            "A multicenter clinical trial of patients with diabetes "
            "evaluated a new drug. Hospital records were used to track "
            "mortality and comorbidities."
        ),
        outcome_dtype="binary",
        client=client,
    )
    assert len(transport.calls) == 1
    call = transport.calls[0]
    blob = (call["system"] + "\n" + call["prompt"]).lower()
    assert (
        "clinical" in blob or "patients" in blob
    ), f"expected clinical-flavored language in prompt; got: {blob[:400]}"


def test_marketing_brief_produces_marketing_prompt() -> None:
    transport = FakeOllamaTransport(responses={"": _valid_payload("yellow")})
    client = OllamaClient(
        model="test:fake",
        cassette_dir=None,
        transport=transport,
        allow_live=True,
    )
    interpret_sensitivity(
        evalue_result=_good_evalue(),
        sensemakr_result=_good_sensemakr(),
        deterministic_verdict="yellow",
        point_estimate=0.4,
        ci_low=0.2,
        ci_high=0.6,
        treatment="campaign_exposure",
        outcome="conversion",
        domain_brief=(
            "A marketing study of customers exposed to a new ad campaign "
            "across multiple channels. Conversion rate and click-through "
            "are the primary outcomes."
        ),
        outcome_dtype="binary",
        client=client,
    )
    assert len(transport.calls) == 1
    call = transport.calls[0]
    blob = (call["system"] + "\n" + call["prompt"]).lower()
    assert (
        "marketing" in blob or "customers" in blob
    ), f"expected marketing-flavored language in prompt; got: {blob[:400]}"


# ─────────── 6. Prompt restates the FIXED color ─────────────────────────


@pytest.mark.parametrize("color", ["green", "yellow", "red"])
def test_prompt_pins_deterministic_verdict_color(color: str) -> None:
    transport = FakeOllamaTransport(responses={"": _valid_payload(color)})
    client = OllamaClient(
        model="test:fake",
        cassette_dir=None,
        transport=transport,
        allow_live=True,
    )
    interpret_sensitivity(
        evalue_result=_good_evalue(),
        sensemakr_result=_good_sensemakr(),
        deterministic_verdict=color,  # type: ignore[arg-type]
        point_estimate=0.4,
        ci_low=0.2,
        ci_high=0.6,
        treatment="t",
        outcome="y",
        domain_brief="patients in a clinical trial",
        outcome_dtype="binary",
        client=client,
    )
    call = transport.calls[0]
    # The color must appear in both system and user prompts (defense in
    # depth — see _build_prompt + _SYSTEM_PROMPT_TEMPLATE).
    assert color in call["system"]
    assert color in call["prompt"]
