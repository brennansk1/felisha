"""Unit tests for the EAGLE-2/3 speculative-decoding adapter (Sprint 8.2).

Coverage:
  * Registry lookup picks the curated draft model for known targets.
  * Unknown targets return ``None`` (graceful fall-through, not an error).
  * Engines without spec-decoding support (Ollama, MLX) return ``None``.
  * Engines with support (vLLM, llama.cpp) return a config.
  * ``estimate_speedup`` is monotone-increasing in ``spec_n`` up to the
    plateau, and decays in ``batch_size``.
  * Dataclass validation rejects malformed inputs.
  * ``to_extra_options`` emits the OpenAI-compatible ``speculation``
    shape that both vLLM and llama.cpp honor.
"""

from __future__ import annotations

import pytest

from causalrag.llm.spec_decoding import (
    SPEC_DECODING_ENGINES,
    SpecDecodingConfig,
    estimate_speedup,
    recommend_spec_config,
    to_extra_options,
)


# --- Registry lookup ----------------------------------------------------------


def test_registry_lookup_qwen3_14b_picks_1_7b_draft() -> None:
    cfg = recommend_spec_config("qwen3:14b-q4_K_M", engine="vllm")
    assert cfg is not None
    assert cfg.draft_model == "qwen3:1.7b-instruct"
    assert cfg.spec_n == 4
    assert 0.0 <= cfg.accept_threshold <= 1.0


def test_registry_lookup_llama_70b_picks_3b_draft() -> None:
    cfg = recommend_spec_config(
        "llama3.3:70b-instruct-q4_K_M", engine="llamacpp"
    )
    assert cfg is not None
    assert cfg.draft_model == "llama3.2:3b-instruct"


def test_registry_lookup_deepseek_distill_uses_qwen_draft() -> None:
    # DeepSeek-R1 distills inherit the Qwen tokenizer, so a Qwen draft works.
    cfg = recommend_spec_config("deepseek-r1-distill-qwen-32b", engine="vllm")
    assert cfg is not None
    assert cfg.draft_model.startswith("qwen3:")


def test_unknown_target_returns_none() -> None:
    assert recommend_spec_config("totally-fictional-model:1b", engine="vllm") is None


# --- Engine gating ------------------------------------------------------------


def test_engine_ollama_returns_none() -> None:
    # Ollama (as of 0.6.x / 2026-05) does not expose a draft-model slot.
    assert recommend_spec_config("qwen3:14b-q4_K_M", engine="ollama") is None


def test_engine_mlx_returns_none() -> None:
    assert recommend_spec_config("qwen3:14b-q4_K_M", engine="mlx") is None


def test_engine_llamacpp_returns_config() -> None:
    cfg = recommend_spec_config("qwen3:14b-q4_K_M", engine="llamacpp")
    assert cfg is not None
    assert cfg.draft_model == "qwen3:1.7b-instruct"


def test_engine_vllm_returns_config() -> None:
    cfg = recommend_spec_config("qwen3:14b-q4_K_M", engine="vllm")
    assert cfg is not None


def test_spec_decoding_engines_set_is_minimal() -> None:
    # Documents the supported set so a future maintainer who adds a new
    # engine knows where to wire it in.
    assert SPEC_DECODING_ENGINES == frozenset({"vllm", "llamacpp"})


def test_unknown_engine_returns_none() -> None:
    assert recommend_spec_config("qwen3:14b-q4_K_M", engine="not-a-real-engine") is None


# --- Method override ----------------------------------------------------------


def test_method_override_is_propagated() -> None:
    cfg = recommend_spec_config("qwen3:14b-q4_K_M", engine="vllm", method="eagle3")
    assert cfg is not None
    assert cfg.method == "eagle3"


def test_method_n_gram_uses_lower_spec_n() -> None:
    # n-gram speculation drafts shorter sequences (the source-text
    # n-grams) — verify the defaults table is consulted.
    cfg = recommend_spec_config("qwen3:14b-q4_K_M", engine="vllm", method="n_gram")
    assert cfg is not None
    assert cfg.spec_n == 3
    assert cfg.method == "n_gram"


# --- Dataclass validation -----------------------------------------------------


def test_spec_decoding_config_rejects_spec_n_zero() -> None:
    with pytest.raises(ValueError, match="spec_n"):
        SpecDecodingConfig(draft_model="x", spec_n=0)


def test_spec_decoding_config_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ValueError, match="accept_threshold"):
        SpecDecodingConfig(draft_model="x", accept_threshold=1.5)
    with pytest.raises(ValueError, match="accept_threshold"):
        SpecDecodingConfig(draft_model="x", accept_threshold=-0.1)


def test_spec_decoding_config_rejects_empty_draft_model() -> None:
    with pytest.raises(ValueError, match="draft_model"):
        SpecDecodingConfig(draft_model="")


def test_spec_decoding_config_is_frozen() -> None:
    cfg = SpecDecodingConfig(draft_model="x")
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        cfg.spec_n = 8  # type: ignore[misc]


# --- estimate_speedup ---------------------------------------------------------


def test_estimate_speedup_returns_speedup_at_least_one() -> None:
    out = estimate_speedup(
        target_model="qwen3:14b-q4_K_M",
        draft_model="qwen3:1.7b-instruct",
        batch_size=1,
        avg_tokens_out=256,
    )
    assert out["speedup"] >= 1.0


def test_estimate_speedup_monotone_in_spec_n() -> None:
    base_kw = dict(
        target_model="qwen3:14b-q4_K_M",
        draft_model="qwen3:1.7b-instruct",
        batch_size=1,
        avg_tokens_out=256,
    )
    speedups = [
        estimate_speedup(**base_kw, spec_n=n)["speedup"]  # type: ignore[arg-type]
        for n in (1, 2, 4, 6, 8)
    ]
    # Non-decreasing (the model plateaus at spec_n=8 but never regresses
    # within the supported range).
    for a, b in zip(speedups, speedups[1:]):
        assert b >= a, f"speedup not monotone in spec_n: {speedups}"
    # And strictly increasing between the extremes — otherwise the test
    # would tolerate a degenerate constant function.
    assert speedups[-1] > speedups[0]


def test_estimate_speedup_decays_with_batch_size() -> None:
    # arXiv:2508.08192 fig 6: spec-decoding wins shrink as batch grows.
    base_kw = dict(
        target_model="qwen3:14b-q4_K_M",
        draft_model="qwen3:1.7b-instruct",
        avg_tokens_out=256,
        spec_n=4,
    )
    s1 = estimate_speedup(**base_kw, batch_size=1)["speedup"]
    s32 = estimate_speedup(**base_kw, batch_size=32)["speedup"]
    assert s1 > s32
    # Floor: even at batch=32 we should not predict a slowdown.
    assert s32 >= 1.0


def test_estimate_speedup_regime_label() -> None:
    out_lat = estimate_speedup(
        target_model="x", draft_model="y", batch_size=1, avg_tokens_out=128
    )
    out_thru = estimate_speedup(
        target_model="x", draft_model="y", batch_size=32, avg_tokens_out=128
    )
    assert out_lat["regime"] == "latency"
    assert out_thru["regime"] == "throughput"


def test_estimate_speedup_rejects_zero_batch() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        estimate_speedup(
            target_model="x", draft_model="y", batch_size=0, avg_tokens_out=128
        )


def test_estimate_speedup_rejects_zero_tokens() -> None:
    with pytest.raises(ValueError, match="avg_tokens_out"):
        estimate_speedup(
            target_model="x", draft_model="y", batch_size=1, avg_tokens_out=0
        )


def test_estimate_speedup_echoes_inputs() -> None:
    out = estimate_speedup(
        target_model="qwen3:14b-q4_K_M",
        draft_model="qwen3:1.7b-instruct",
        batch_size=4,
        avg_tokens_out=512,
        spec_n=4,
        accept_threshold=0.5,
        method="eagle3",
    )
    assert out["target_model"] == "qwen3:14b-q4_K_M"
    assert out["draft_model"] == "qwen3:1.7b-instruct"
    assert out["batch_size"] == 4
    assert out["avg_tokens_out"] == 512
    assert out["spec_n"] == 4
    assert out["method"] == "eagle3"


# --- to_extra_options payload shape ------------------------------------------


def test_to_extra_options_emits_speculation_block() -> None:
    cfg = SpecDecodingConfig(
        draft_model="qwen3:1.7b-instruct",
        spec_n=4,
        accept_threshold=0.5,
        method="eagle3",
    )
    payload = to_extra_options(cfg)
    # The OpenAI-compatible ``speculation`` block is what vLLM and
    # llama.cpp actually consume.
    assert "speculation" in payload
    spec = payload["speculation"]
    assert spec["model"] == "qwen3:1.7b-instruct"
    assert spec["num_speculative_tokens"] == 4
    assert spec["acceptance_threshold"] == 0.5
    assert spec["method"] == "eagle3"
    # And we round-trip the dataclass for observability.
    assert payload["spec_decoding"]["draft_model"] == "qwen3:1.7b-instruct"


def test_to_extra_options_is_adapter_ready() -> None:
    # The adapter merges ``extra_options`` into the request body with
    # ``body.update(extra_options)``. Verify the keys we emit don't
    # collide with the adapter's reserved fields.
    cfg = recommend_spec_config("qwen3:14b-q4_K_M", engine="vllm")
    assert cfg is not None
    payload = to_extra_options(cfg)
    reserved = {
        "model",
        "messages",
        "temperature",
        "seed",
        "max_tokens",
        "stream",
        "response_format",
        "guided_json",
    }
    assert reserved.isdisjoint(payload.keys())
