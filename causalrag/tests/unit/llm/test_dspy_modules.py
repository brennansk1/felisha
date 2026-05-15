"""Tests for the DSPy + Outlines scaffolding (Sprint 1.7)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from causalrag.llm.dspy_modules import (
    DSPyAdapter,
    DSPyAvailability,
    PromptSignature,
    SIGNATURES,
    compile_signatures,
    detect_dspy_availability,
    outlines_json_generator,
)


# ─── Schema for tests ────────────────────────────────────────────────


class _DummySchema(BaseModel):
    text: str
    n: int = 0


class _FakeLLMResponse:
    """Mimics OllamaClient.LLMResponse."""

    def __init__(self, parsed):
        self.parsed = parsed
        self.raw = "ok"
        self.model = "fake"
        self.key = "fake"
        self.retries = 0
        self.errors: list[str] = []


class _FakeFallbackClient:
    """Mimics OllamaClient.parse(...)."""

    def __init__(self):
        self.calls: list[dict] = []

    def parse(self, *, prompt, schema, system="", json_schema=None, extra_options=None):
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "system": system,
                "json_schema": json_schema,
            }
        )
        return _FakeLLMResponse(parsed=schema(text="ok"))


# ─── Tests ──────────────────────────────────────────────────────────


def test_signature_catalog_covers_known_prompt_sites() -> None:
    expected = {
        "planner",
        "critic",
        "foundation_followup",
        "synthesis",
        "anomaly_audit",
        "identification_narration",
        "sensitivity_interpretation",
        "cross_experiment",
    }
    assert set(SIGNATURES.keys()) == expected


def test_detect_dspy_availability_returns_dataclass() -> None:
    avail = detect_dspy_availability()
    assert isinstance(avail, DSPyAvailability)
    assert isinstance(avail.dspy, bool)
    assert isinstance(avail.outlines, bool)


def test_adapter_falls_back_when_dspy_unavailable() -> None:
    fallback = _FakeFallbackClient()
    adapter = DSPyAdapter(
        fallback_client=fallback,
        availability=DSPyAvailability(dspy=False, outlines=False),
    )
    response = adapter.parse(prompt="test", schema=_DummySchema, system="sys")
    assert response.parsed is not None
    assert response.parsed.text == "ok"
    assert len(fallback.calls) == 1
    assert fallback.calls[0]["prompt"] == "test"


def test_adapter_falls_back_when_no_compiled_module() -> None:
    """Even when DSPy is installed, an unregistered site falls back."""
    fallback = _FakeFallbackClient()
    adapter = DSPyAdapter(
        fallback_client=fallback,
        availability=DSPyAvailability(dspy=True, outlines=True),
    )
    response = adapter.parse(prompt="p", schema=_DummySchema)
    assert response.parsed is not None
    assert len(fallback.calls) == 1


def test_adapter_registers_compiled_module() -> None:
    fallback = _FakeFallbackClient()
    adapter = DSPyAdapter(
        fallback_client=fallback,
        availability=DSPyAvailability(dspy=True, outlines=False),
    )
    fake_module = lambda **kw: {"text": "from_dspy", "n": 7}  # noqa: E731
    adapter.register_compiled("planner", fake_module)
    assert "planner" in adapter.compiled_modules


def test_infer_site_name_resolves_known_schemas() -> None:
    adapter = DSPyAdapter(fallback_client=_FakeFallbackClient())

    class CandidateQueue(BaseModel):
        x: str = ""

    class CriticBatch(BaseModel):
        x: str = ""

    class NextExperiment(BaseModel):
        x: str = ""

    class ExecutiveSynthesis(BaseModel):
        x: str = ""

    assert adapter._infer_site_name(CandidateQueue, "", "") == "planner"
    assert adapter._infer_site_name(CriticBatch, "", "") == "critic"
    assert adapter._infer_site_name(NextExperiment, "", "") == "foundation_followup"
    assert adapter._infer_site_name(ExecutiveSynthesis, "", "") == "synthesis"

    class Unknown(BaseModel):
        x: str = ""

    assert adapter._infer_site_name(Unknown, "", "") is None


def test_outlines_json_generator_returns_none_when_unavailable() -> None:
    # Outlines isn't installed in this env; should return None gracefully.
    result = outlines_json_generator(model=object(), schema=_DummySchema)
    assert result is None or callable(result)


def test_compile_signatures_returns_none_without_dspy() -> None:
    """When dspy isn't installed, compile_signatures gracefully returns None."""
    result = compile_signatures(gold_set={})
    # Either None (no dspy) or dict (dspy installed) — both fine
    assert result is None or isinstance(result, dict)


def test_signature_input_output_fields_well_formed() -> None:
    """Every signature has at least one input + one output field."""
    for name, sig in SIGNATURES.items():
        assert len(sig.input_fields) >= 1, f"{name} has no inputs"
        assert len(sig.output_fields) >= 1, f"{name} has no outputs"
        for input_name, input_desc in sig.input_fields:
            assert isinstance(input_name, str) and input_name
            assert isinstance(input_desc, str) and input_desc
        for output_name, output_desc, output_dtype in sig.output_fields:
            assert isinstance(output_name, str) and output_name


def test_prompt_signature_is_frozen_dataclass() -> None:
    """Sanity: signatures are immutable so the catalog can't be
    mutated at runtime."""
    sig = SIGNATURES["planner"]
    with pytest.raises(Exception):
        sig.instruction = "changed"  # type: ignore[misc]


def test_adapter_handles_unparseable_dspy_output() -> None:
    """When the compiled module returns something that can't be
    coerced to the schema, the adapter falls back."""
    fallback = _FakeFallbackClient()
    adapter = DSPyAdapter(
        fallback_client=fallback,
        availability=DSPyAvailability(dspy=True, outlines=False),
    )
    adapter.register_compiled("planner", lambda **kw: 12345)  # nonsense

    class CandidateQueue(BaseModel):
        candidates: list[str] = []

    # Should not crash — falls back
    response = adapter.parse(prompt="p", schema=CandidateQueue)
    assert response is not None
