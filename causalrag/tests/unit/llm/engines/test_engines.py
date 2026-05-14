"""Unit tests for the engine-adapter layer (Sprint 8.1).

We don't run any real model — every test stubs the network at the
adapter's transport seam. The goal is to prove:

  * ``select_engine`` dispatches to the right adapter class by short name;
  * each adapter constructs cleanly with a fake model identifier;
  * ``parse(...)`` returns an ``LLMResponse`` with the same shape the
    existing ``OllamaClient`` produces (so call-sites are interchangeable);
  * ``healthcheck()`` is callable and stubs out the network;
  * missing-backend conditions raise ``EngineNotAvailable`` rather than
    silently degrading.
"""

from __future__ import annotations

import json
import platform
from typing import Any

import pytest
from pydantic import BaseModel

from causalrag.llm.engines import (
    EngineNotAvailable,
    LlamaCppServerAdapter,
    MlxLmAdapter,
    OllamaAdapter,
    VllmAdapter,
    available_engines,
    select_engine,
)
from causalrag.llm.ollama_client import FakeOllamaTransport, LLMResponse


class Toy(BaseModel):
    answer: str
    score: int


# ─────────── select_engine() dispatch ─────────────────────────────────────


def test_available_engines_lists_all_four() -> None:
    names = set(available_engines())
    assert {"ollama", "llamacpp", "vllm", "mlx"}.issubset(names)


def test_select_engine_unknown_name_raises() -> None:
    with pytest.raises(EngineNotAvailable) as exc:
        select_engine("not-a-real-engine", model="toy")
    assert "unknown engine" in exc.value.reason


def test_select_engine_ollama_returns_adapter() -> None:
    transport = FakeOllamaTransport({"": {"answer": "ok", "score": 1}})
    engine = select_engine("ollama", model="toy", transport=transport, allow_live=True)
    assert isinstance(engine, OllamaAdapter)
    assert engine.model == "toy"


def test_select_engine_llamacpp_returns_adapter() -> None:
    engine = select_engine("llamacpp", model="toy", base_url="http://x:8080")
    assert isinstance(engine, LlamaCppServerAdapter)
    assert engine.base_url == "http://x:8080"


def test_select_engine_vllm_returns_adapter() -> None:
    engine = select_engine("vllm", model="toy")
    assert isinstance(engine, VllmAdapter)


def test_select_engine_mlx_requires_apple_silicon() -> None:
    if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
        # On Apple Silicon we can't construct without mlx-lm; verify it
        # raises EngineNotAvailable cleanly when the package is missing.
        pytest.importorskip("mlx_lm")
        engine = select_engine("mlx", model="toy")
        assert isinstance(engine, MlxLmAdapter)
    else:
        with pytest.raises(EngineNotAvailable):
            select_engine("mlx", model="toy")


# ─────────── OllamaAdapter — parse() shape ────────────────────────────────


def test_ollama_adapter_parse_returns_llm_response() -> None:
    transport = FakeOllamaTransport({"": {"answer": "hello", "score": 42}})
    engine = OllamaAdapter(model="toy", transport=transport, allow_live=True)
    resp = engine.parse(prompt="hi", schema=Toy)
    assert isinstance(resp, LLMResponse)
    assert isinstance(resp.parsed, Toy)
    assert resp.parsed.answer == "hello"
    assert resp.parsed.score == 42
    assert resp.model == "toy"
    assert resp.source == "live"
    assert resp.retries == 0


def test_ollama_adapter_healthcheck_handles_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    # No server running on the random port -> healthcheck returns False, no raise.
    engine = OllamaAdapter(
        model="toy",
        base_url="http://127.0.0.1:1",  # unbound
        transport=FakeOllamaTransport({"": {"answer": "x", "score": 0}}),
        allow_live=True,
    )
    assert engine.healthcheck() is False


# ─────────── llama.cpp adapter ────────────────────────────────────────────


class _FakeChatTransport:
    """Mimics what the adapters use: a ``.chat(body) -> raw_str`` callable."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def chat(self, body: dict[str, Any]) -> str:
        self.calls.append(body)
        return json.dumps(self.payload)


def test_llamacpp_adapter_parse_shape() -> None:
    transport = _FakeChatTransport({"answer": "cpp", "score": 7})
    engine = LlamaCppServerAdapter(model="toy", transport=transport)
    resp = engine.parse(prompt="hi", schema=Toy, system="be helpful")
    assert isinstance(resp, LLMResponse)
    assert resp.parsed.answer == "cpp"
    assert resp.parsed.score == 7
    assert resp.model == "toy"
    assert resp.retries == 0
    # Verify the system prompt + response_format made it onto the wire.
    body = transport.calls[0]
    assert body["messages"][0] == {"role": "system", "content": "be helpful"}
    assert body["messages"][-1]["role"] == "user"
    assert body["response_format"]["type"] == "json_object"


def test_llamacpp_adapter_json_schema_sets_response_format() -> None:
    transport = _FakeChatTransport({"answer": "ok", "score": 1})
    engine = LlamaCppServerAdapter(model="toy", transport=transport)
    engine.parse(
        prompt="hi",
        schema=Toy,
        json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    rf = transport.calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["schema"]["type"] == "object"


def test_llamacpp_adapter_healthcheck_offline() -> None:
    engine = LlamaCppServerAdapter(model="toy", base_url="http://127.0.0.1:1")
    assert engine.healthcheck() is False


# ─────────── vLLM adapter ─────────────────────────────────────────────────


def test_vllm_adapter_parse_shape() -> None:
    transport = _FakeChatTransport({"answer": "vllm", "score": 9})
    engine = VllmAdapter(model="toy", transport=transport)
    resp = engine.parse(prompt="hi", schema=Toy)
    assert isinstance(resp, LLMResponse)
    assert resp.parsed.answer == "vllm"
    assert resp.parsed.score == 9
    # vLLM uses response_format when no schema is given.
    assert transport.calls[0]["response_format"]["type"] == "json_object"


def test_vllm_adapter_json_schema_uses_guided_json() -> None:
    transport = _FakeChatTransport({"answer": "ok", "score": 1})
    engine = VllmAdapter(model="toy", transport=transport)
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    engine.parse(prompt="hi", schema=Toy, json_schema=schema)
    body = transport.calls[0]
    assert body["guided_json"] == schema
    # When guided_json is set, response_format must NOT be sent (vLLM treats
    # them as mutually exclusive).
    assert "response_format" not in body


def test_vllm_adapter_healthcheck_offline() -> None:
    engine = VllmAdapter(model="toy", base_url="http://127.0.0.1:1")
    assert engine.healthcheck() is False


# ─────────── MLX adapter ──────────────────────────────────────────────────


class _FakeMlxRunner:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, prompt: str, **kw: Any) -> str:
        self.calls.append({"prompt": prompt, **kw})
        return json.dumps(self.payload)


def test_mlx_adapter_parse_shape_with_fake_runner() -> None:
    runner = _FakeMlxRunner({"answer": "mlx", "score": 3})
    engine = MlxLmAdapter(model="toy", runner=runner, skip_hardware_check=True)
    resp = engine.parse(prompt="hi", schema=Toy, system="sys")
    assert isinstance(resp, LLMResponse)
    assert resp.parsed.answer == "mlx"
    assert resp.parsed.score == 3
    assert engine.healthcheck() is True
    # System block must precede the user prompt in the composed string.
    sent = runner.calls[0]["prompt"]
    assert "sys" in sent
    assert "hi" in sent
    assert sent.index("sys") < sent.index("hi")


def test_mlx_adapter_non_apple_raises() -> None:
    if platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}:
        pytest.skip("This host IS Apple Silicon — hardware check would pass.")
    with pytest.raises(EngineNotAvailable) as exc:
        MlxLmAdapter(model="toy", runner=_FakeMlxRunner({}))
    assert exc.value.engine == "mlx"


def test_mlx_adapter_missing_package_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirm the loader raises ``EngineNotAvailable`` when ``mlx_lm`` is absent."""
    import builtins

    real_import = builtins.__import__

    def _fail(name: str, *a: Any, **kw: Any) -> Any:
        if name == "mlx_lm" or name.startswith("mlx_lm."):
            raise ImportError("simulated missing mlx-lm")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fail)
    with pytest.raises(EngineNotAvailable) as exc:
        # skip_hardware_check so we exercise the import path on any OS.
        MlxLmAdapter(model="toy", skip_hardware_check=True)
    assert "mlx-lm not installed" in exc.value.reason


# ─────────── Cross-adapter LLMResponse parity ────────────────────────────


@pytest.mark.parametrize(
    "make_engine",
    [
        lambda: OllamaAdapter(
            model="toy",
            transport=FakeOllamaTransport({"": {"answer": "a", "score": 1}}),
            allow_live=True,
        ),
        lambda: LlamaCppServerAdapter(
            model="toy",
            transport=_FakeChatTransport({"answer": "a", "score": 1}),
        ),
        lambda: VllmAdapter(
            model="toy",
            transport=_FakeChatTransport({"answer": "a", "score": 1}),
        ),
        lambda: MlxLmAdapter(
            model="toy",
            runner=_FakeMlxRunner({"answer": "a", "score": 1}),
            skip_hardware_check=True,
        ),
    ],
    ids=["ollama", "llamacpp", "vllm", "mlx"],
)
def test_all_adapters_return_identical_response_shape(make_engine: Any) -> None:
    engine = make_engine()
    resp = engine.parse(prompt="hi", schema=Toy)
    # Every adapter MUST populate the same set of fields the master loop
    # reads from ``LLMResponse`` today (see ollama_client.LLMResponse).
    expected = {
        "parsed", "raw", "model", "model_digest", "seed",
        "cassette_key", "source", "retries", "options",
    }
    assert expected.issubset(resp.__dict__.keys())
    assert isinstance(resp.parsed, Toy)
    assert resp.model == "toy"
    assert isinstance(resp.raw, str)
    assert isinstance(resp.retries, int)
