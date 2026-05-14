"""EAGLE-2/3 speculative-decoding adapter layer (Sprint 8.2).

Speculative decoding accelerates autoregressive LLM inference by using a
small *draft* model to propose ``spec_n`` tokens per step which the larger
*target* model verifies in a single forward pass. The technique is
loss-less (the verified output matches the target's distribution) and
yields 1.4-2x throughput at batch size 1 on EAGLE-3 published benchmarks
(see Llama-at-scale, arXiv:2508.08192). Speedup degrades as batch size
grows because the target model is already compute-bound at batch>16.

This module is a *thin* helper: it picks a sensible draft model for a
given target, returns a :class:`SpecDecodingConfig` dataclass, and emits
the OpenAI-compatible ``speculation`` payload shape that both vLLM and
llama.cpp's server honor. The engine adapters
(:mod:`causalrag.llm.engines.llamacpp_adapter`,
:mod:`causalrag.llm.engines.vllm_adapter`) simply forward whatever the
caller sets in ``extra_options["spec_decoding"]``; no adapter changes
are required to consume the helper.

Usage::

    from causalrag.llm.spec_decoding import (
        recommend_spec_config,
        to_extra_options,
    )

    cfg = recommend_spec_config("qwen3:14b-q4_K_M", engine="vllm")
    if cfg is not None:
        adapter.parse(
            prompt=...,
            schema=...,
            extra_options=to_extra_options(cfg),
        )

Engines without spec-decoding support (Ollama, MLX-LM as of 2026-05)
return ``None`` from :func:`recommend_spec_config` — callers should
treat that as "skip the feature, fall back to vanilla decoding".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

__all__ = [
    "SpecDecodingConfig",
    "recommend_spec_config",
    "estimate_speedup",
    "to_extra_options",
    "SPEC_DECODING_ENGINES",
]


# Engines that honor the OpenAI-compatible ``speculation`` payload.
# Ollama (as of 0.6.x / 2026-05) and MLX-LM do not expose a draft-model
# slot, so we refuse to emit a config for them.
SPEC_DECODING_ENGINES: frozenset[str] = frozenset({"vllm", "llamacpp"})


SpecMethod = Literal["eagle2", "eagle3", "n_gram", "auto"]


@dataclass(frozen=True)
class SpecDecodingConfig:
    """Speculative-decoding configuration for a single ``parse(...)`` call.

    Attributes:
        draft_model: Short name of the small draft model. Must be loadable
            by the same engine that serves the target model.
        spec_n: Tokens the draft proposes per verification step. 4 is the
            EAGLE-3 sweet spot; values >8 rarely help and can hurt because
            rejection probability compounds.
        accept_threshold: Probability floor below which a proposed token
            is rejected even if it appears in the target's top-k. 0.5 is
            a balanced default.
        method: Speculation algorithm. ``"auto"`` lets the server choose
            (typically EAGLE-3 for transformer pairs, n-gram for code).
    """

    draft_model: str
    spec_n: int = 4
    accept_threshold: float = 0.5
    method: SpecMethod = "auto"

    def __post_init__(self) -> None:
        if self.spec_n < 1:
            raise ValueError(f"spec_n must be >= 1, got {self.spec_n}")
        if not 0.0 <= self.accept_threshold <= 1.0:
            raise ValueError(
                f"accept_threshold must be in [0,1], got {self.accept_threshold}"
            )
        if not self.draft_model:
            raise ValueError("draft_model must be a non-empty string")


# Curated draft-model registry: target model → recommended draft.
#
# The picks follow two rules: (a) same tokenizer family (Qwen<->Qwen,
# Llama<->Llama, DeepSeek<->Qwen since DeepSeek-R1 distills use Qwen
# tokenizers), and (b) ~10-20x parameter ratio (sweet spot for EAGLE-3
# acceptance rate per the published benchmarks).
_DRAFT_MODEL_REGISTRY: dict[str, str] = {
    # Qwen3 family — 1.7B drafts the 14B and 32B variants well.
    "qwen3:14b-q4_K_M": "qwen3:1.7b-instruct",
    "qwen3:14b-instruct": "qwen3:1.7b-instruct",
    "qwen3:32b-q4_K_M": "qwen3:1.7b-instruct",
    "qwen3:32b-instruct": "qwen3:1.7b-instruct",
    # Llama 3.x family — 3B drafts 70B and 8B targets.
    "llama3.3:70b-instruct-q4_K_M": "llama3.2:3b-instruct",
    "llama3.3:70b-instruct": "llama3.2:3b-instruct",
    "llama3.1:70b-instruct-q4_K_M": "llama3.2:3b-instruct",
    "llama3.1:8b-instruct": "llama3.2:1b-instruct",
    # DeepSeek-R1 distills are Qwen-tokenized, so Qwen3 drafts work.
    "deepseek-r1-distill-qwen-32b": "qwen3:1.7b-instruct",
    "deepseek-r1-distill-qwen-14b": "qwen3:1.7b-instruct",
    "deepseek-r1-distill-llama-70b": "llama3.2:3b-instruct",
}


# Per-method tuning. EAGLE-3 is the default since it dominates EAGLE-2 in
# published benchmarks; n-gram is reserved for code-heavy workloads where
# the target's own context is the best draft.
_METHOD_DEFAULTS: dict[SpecMethod, dict[str, Any]] = {
    "eagle2": {"spec_n": 4, "accept_threshold": 0.5},
    "eagle3": {"spec_n": 4, "accept_threshold": 0.5},
    "n_gram": {"spec_n": 3, "accept_threshold": 0.6},
    "auto": {"spec_n": 4, "accept_threshold": 0.5},
}


def recommend_spec_config(
    target_model: str,
    engine: str,
    *,
    method: SpecMethod = "auto",
) -> SpecDecodingConfig | None:
    """Return a sensible spec-decoding config for ``(target_model, engine)``.

    Args:
        target_model: Short name of the *target* (large) model that will
            actually serve the request.
        engine: Engine short name (``"vllm"``, ``"llamacpp"``,
            ``"ollama"``, ``"mlx"``).
        method: Override the speculation method. Defaults to ``"auto"``
            (EAGLE-3 in practice).

    Returns:
        A :class:`SpecDecodingConfig` when both the engine supports
        speculation and we have a registered draft model for the target,
        or ``None`` otherwise. ``None`` is a soft signal to fall back to
        vanilla decoding — it is *not* an error.
    """
    if engine not in SPEC_DECODING_ENGINES:
        return None
    draft = _DRAFT_MODEL_REGISTRY.get(target_model)
    if draft is None:
        return None
    defaults = _METHOD_DEFAULTS[method]
    return SpecDecodingConfig(
        draft_model=draft,
        spec_n=int(defaults["spec_n"]),
        accept_threshold=float(defaults["accept_threshold"]),
        method=method,
    )


def estimate_speedup(
    *,
    target_model: str,
    draft_model: str,
    batch_size: int,
    avg_tokens_out: int,
    spec_n: int = 4,
    accept_threshold: float = 0.5,
    method: SpecMethod = "auto",
) -> dict[str, Any]:
    """Rough speedup estimate based on EAGLE-3 published benchmarks.

    The model: peak speedup ``S_peak`` is 1.4-2.0x at batch=1 and decays
    geometrically with batch size (the target becomes compute-bound).
    Speedup is approximately monotone-increasing in ``spec_n`` up to
    ``spec_n=8`` (where rejection probability flattens the curve), and
    monotone-increasing in ``avg_tokens_out`` (amortizes the draft
    warmup). We do *not* claim calibration — call this a planning hint,
    not a benchmark.

    Reference: Llama-at-scale, arXiv:2508.08192 (EAGLE-3 evaluation).

    Args:
        target_model: Target model short name (used only for metadata).
        draft_model: Draft model short name (used only for metadata).
        batch_size: Concurrent decode requests on the server.
        avg_tokens_out: Average completion length.
        spec_n: Tokens drafted per step.
        accept_threshold: Acceptance probability floor.
        method: Speculation method.

    Returns:
        Dict with ``speedup`` (float, ``>= 1.0``), ``regime``
        (``"latency"`` / ``"throughput"``), and the inputs echoed for
        log-friendliness.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if avg_tokens_out < 1:
        raise ValueError(f"avg_tokens_out must be >= 1, got {avg_tokens_out}")

    # Peak speedup at batch=1. EAGLE-3 reports 1.4-2.0x depending on
    # workload; we anchor at 1.8x and let ``spec_n`` and acceptance
    # threshold slide it within [1.4, 2.0].
    base_peak = 1.8
    # Monotone-increasing in spec_n: each extra speculative token adds
    # ~0.08x up to spec_n=8 (then plateaus).
    spec_n_bonus = 0.08 * min(spec_n, 8)
    # Higher acceptance thresholds reject more drafts → smaller speedup.
    threshold_penalty = 0.4 * accept_threshold
    # Method nudge: EAGLE-3 > EAGLE-2 > n-gram for transformer targets.
    method_bonus = {"eagle3": 0.1, "auto": 0.1, "eagle2": 0.0, "n_gram": -0.1}[method]

    peak = base_peak + spec_n_bonus - threshold_penalty + method_bonus
    peak = max(1.0, min(peak, 2.5))

    # Batch decay: ~half-life at batch=16 per the arXiv:2508.08192 fig 6.
    # speedup(batch) = 1 + (peak - 1) * 2^(-(batch-1)/16)
    decay = 0.5 ** ((batch_size - 1) / 16.0)
    speedup = 1.0 + (peak - 1.0) * decay

    # Amortization: very short completions (<32 tokens) lose some gain
    # to draft-model warmup. Scale linearly from 0.7x at 1 token to 1.0x
    # at 32 tokens.
    if avg_tokens_out < 32:
        amortize = 0.7 + 0.3 * (avg_tokens_out / 32.0)
        speedup = 1.0 + (speedup - 1.0) * amortize

    regime = "latency" if batch_size <= 4 else "throughput"

    return {
        "speedup": round(speedup, 3),
        "regime": regime,
        "target_model": target_model,
        "draft_model": draft_model,
        "batch_size": batch_size,
        "avg_tokens_out": avg_tokens_out,
        "spec_n": spec_n,
        "accept_threshold": accept_threshold,
        "method": method,
    }


def to_extra_options(cfg: SpecDecodingConfig) -> dict[str, Any]:
    """Emit the OpenAI-compatible ``speculation`` payload.

    Both vLLM (``--speculative_model``) and llama.cpp's server
    (``--model-draft``) accept this shape inside the chat-completions
    body. Callers pass the result through the adapter's
    ``extra_options=`` kwarg; the adapter merges it into the request
    body verbatim.
    """
    return {
        "spec_decoding": asdict(cfg),
        "speculation": {
            "model": cfg.draft_model,
            "num_speculative_tokens": cfg.spec_n,
            "acceptance_threshold": cfg.accept_threshold,
            "method": cfg.method,
        },
    }


# Registry version stamp — bump when entries are added or retired so log
# consumers can correlate observed speedups with the recommendation table.
_REGISTRY_VERSION: str = "2026-05"
