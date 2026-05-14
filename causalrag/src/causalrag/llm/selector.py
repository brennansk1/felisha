"""Model selection — map a HardwareProfile to default model slots.

PDD §16.2 (tier defaults) and §16.5 (three-slot split: discovery /
hypothesize / utility). The selector returns string identifiers in the
Ollama-compatible ``family:size-quant`` form; resolving them against the
local Ollama install (and prompting the user to ``ollama pull`` missing
models) is the caller's job.

Slot routing in this codebase:

- ``discovery`` (general, instruction-tuned, fast JSON): Stage 1c investigator
  (column-by-column, narrow extraction task — speed and structured-output
  reliability matter more than depth).
- ``hypothesize`` (reasoning / "thinking" model): Stage 1e domain expert
  (cross-column synthesis, DAG proposals, identification warnings) AND
  Phase 3 hypothesis generation. Both are reasoning-heavy tasks that benefit
  from deliberate step-by-step inference.
- ``utility`` (smallest available): JSON repair, schema fix, brief
  summarization, retry-with-feedback corrections.

The hardware tier table below assigns slots so that the FLOOR tier (T2,
12-16 GB VRAM or 32 GB unified) still gets a genuine reasoning model in the
hypothesize slot — the package's design principle is that hypothesis quality
must not degrade silently on low-end hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

from causalrag.llm.hardware import HardwareProfile


@dataclass(frozen=True)
class ModelSlots:
    """Three-slot model assignment (PDD §16.5).

    - ``discovery``: general-purpose instruction-tuned, fast structured-JSON.
    - ``hypothesize``: reasoning model — slower, deeper proposals.
    - ``utility``: smallest available — JSON repair, schema fix, summaries.
    """

    discovery: str
    hypothesize: str
    utility: str
    tier: int
    quantization: str = "Q4_K_M"


# Per §16.2; quantization defaults follow §16.4 (Q4_K_M floor; Q5/Q8 optional at T4+).
# Below T2, no genuine reasoning model fits in VRAM. We still split slots so
# downstream code reads correct slot names; we issue a warning at run_doctor
# time so the user knows hypothesis quality will be degraded.
_TIER_TABLE: dict[int, ModelSlots] = {
    0: ModelSlots(
        discovery="qwen3:4b-q4_K_M",          # general, instruction-tuned, fast
        hypothesize="qwen3:8b-q4_K_M",        # best available "deep" model at this tier
        utility="qwen3:1.7b-q4_K_M",          # tiny — JSON repair only
        tier=0,
    ),
    1: ModelSlots(
        discovery="qwen3:8b-q4_K_M",          # general
        hypothesize="qwen3:14b-q4_K_M",       # quantized 14B as cheapest reasoning-grade fit
        utility="qwen3:4b-q4_K_M",
        tier=1,
    ),
    2: ModelSlots(
        discovery="qwen3:14b-q4_K_M",         # FLOOR: general 14B
        hypothesize="deepseek-r1:14b-q4_K_M", # FLOOR: reasoning model
        utility="qwen3:4b-q4_K_M",
        tier=2,
    ),
    3: ModelSlots(
        discovery="qwen3:32b-q4_K_M",
        hypothesize="deepseek-r1:32b-q5_K_M",
        utility="qwen3:8b-q4_K_M",
        tier=3,
        quantization="Q5_K_M",
    ),
    4: ModelSlots(
        discovery="llama3.3:70b-q4_K_M",
        hypothesize="deepseek-r1:70b-distill-q4_K_M",
        utility="qwen3:14b-q4_K_M",
        tier=4,
        quantization="Q5_K_M",
    ),
    5: ModelSlots(
        discovery="llama3.3:70b-q4_K_M",
        hypothesize="deepseek-r1:70b-distill-q8_0",
        utility="qwen3:32b-q4_K_M",
        tier=5,
        quantization="Q8_0",
    ),
}


def select_slots(profile: HardwareProfile) -> ModelSlots:
    """Return the recommended ModelSlots for the given HardwareProfile.

    Falls back to an installed model when the tier-default isn't present
    locally. The fallback ranks installed models by family preference
    (reasoning > general 14B+ > general 8B) so a reasonable substitute is
    chosen rather than silently failing later at the first /api/generate
    call.
    """
    recommended = _TIER_TABLE.get(profile.tier, _TIER_TABLE[0])
    installed = list(profile.ollama.models)
    if not installed:
        return recommended

    discovery = _resolve_with_fallback(recommended.discovery, installed, _DISCOVERY_FALLBACKS)
    hypothesize = _resolve_with_fallback(recommended.hypothesize, installed, _HYPOTHESIZE_FALLBACKS)
    utility = _resolve_with_fallback(recommended.utility, installed, _UTILITY_FALLBACKS)
    return ModelSlots(
        discovery=discovery,
        hypothesize=hypothesize,
        utility=utility,
        tier=recommended.tier,
        quantization=recommended.quantization,
    )


# Ordered fallback families per slot. Each entry is a substring matched against
# the installed model name. We prefer reasoning-grade models (deepseek-r1, qwq)
# for the hypothesize slot and general instruction-tuned models for discovery.
_DISCOVERY_FALLBACKS: tuple[str, ...] = (
    "qwen3:14b",
    "qwen3:8b",
    "qwen2.5:14b",
    "llama3.3",
    "llama3.1:8b",
    "gemma2:27b",
    "mistral-small:24b",
)
_HYPOTHESIZE_FALLBACKS: tuple[str, ...] = (
    "deepseek-r1",
    "qwq",
    "qwen3:14b",  # qwen3 has thinking mode
    "gemma2:27b",
    "mistral-small:24b",
    "qwen2.5:14b",
    "llama3.3",
    "llama3.1:8b",
)
_UTILITY_FALLBACKS: tuple[str, ...] = (
    "qwen3:4b",
    "qwen3:8b",
    "llama3.1:8b",
    "qwen2.5:14b",
)


def _resolve_with_fallback(target: str, installed: list[str], fallbacks: tuple[str, ...]) -> str:
    """Return ``target`` if installed, otherwise the first fallback that is.

    Matching is by substring against the installed model name (handles tag
    variants like ``qwen3:14b-q4_K_M`` vs ``qwen3:14b``).
    """
    if target in installed:
        return target
    target_family = target.split("-")[0]
    for name in installed:
        if name.startswith(target_family):
            return name
    for fb in fallbacks:
        for name in installed:
            if fb in name:
                return name
    # Last resort: return the first installed model so the call doesn't blow up.
    return installed[0] if installed else target


def missing_models(profile: HardwareProfile, slots: ModelSlots) -> tuple[str, ...]:
    """Return slot models that are not yet present in the local Ollama install.

    Comparison is by model family + size prefix (e.g., ``qwen3:14b``); the exact
    quantization tag is matched if Ollama reports it, otherwise just the family.
    """
    installed = {m.split(":")[0]: m for m in profile.ollama.models}
    installed_full = set(profile.ollama.models)
    out: list[str] = []
    for slot in (slots.discovery, slots.hypothesize, slots.utility):
        if slot in installed_full:
            continue
        family = slot.split(":")[0]
        if family in installed and installed[family].startswith(slot.split("-")[0]):
            continue
        out.append(slot)
    return tuple(out)
