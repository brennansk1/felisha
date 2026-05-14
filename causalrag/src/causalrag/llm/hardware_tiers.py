"""Refreshed T0-T5 hardware-tier map (Sprint 8.3).

A lightweight, self-contained companion to :mod:`causalrag.llm.hardware` /
:mod:`causalrag.llm.selector`. Where ``hardware.py`` returns a heavyweight
:class:`HardwareProfile` (Ollama probes, pynvml, R, disk free), this module
focuses on the *recommendation* axis: given a minimal hardware fingerprint,
which three-slot model bundle (discovery / hypothesize / synthesis) should
we suggest for the May 2026 open-weight landscape?

Tiers are curated to:

* T0 — laptop CPU only (8-16 GB RAM). Sub-floor for genuine reasoning;
  we route the user to small instruction-tuned models for guard-rail tasks
  only.
* T1 — laptop with iGPU or modest dGPU (8-12 GB VRAM, or Apple Silicon
  with 16-24 GB unified memory).
* T2 — desktop with a single 24 GB consumer GPU (RTX 3090 / 4090) or
  M-series Mac with 32 GB unified memory. This is the FLOOR for "real"
  reasoning models in the local stack.
* T3 — desktop with 48 GB VRAM (RTX 6000 Ada, dual 4090, or 64-96 GB
  Apple Silicon).
* T4 — workstation: 80 GB VRAM (single H100/A100/H200) or 128+ GB Apple
  Silicon (M3 Ultra). Enough headroom for a 70B model at Q5_K_M.
* T5 — datacenter: dual H100 / H200, enabling speculative-decoding-backed
  serving of the 70B with a small drafter for low latency.

The module exposes three pure functions — ``probe_hardware``,
``assign_tier``, and ``render_recommendation`` — so it is trivially
testable without touching the heavier probes in ``hardware.py``.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from typing import Literal

TierName = Literal["T0", "T1", "T2", "T3", "T4", "T5"]


# --- Data classes -----------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Minimal hardware fingerprint used for tier assignment.

    Distinct from :class:`causalrag.llm.hardware.HardwareProfile`, which is
    the canonical "doctor" artifact. This struct is intentionally lean so
    synthetic profiles can be constructed inline in tests.
    """

    cpu_logical_cores: int
    cpu_physical_cores: int
    ram_gb: float
    gpu_vram_gb: float | None  # None = no GPU detected
    apple_silicon: bool


@dataclass(frozen=True)
class ModelSlot:
    """One recommended model for a given slot (discovery / hypothesize / synthesis)."""

    name: str
    parameters_b: float
    quantization: str
    context: int
    notes: str


@dataclass(frozen=True)
class TierAssignment:
    """Output of :func:`assign_tier` — a tier + three model slots + caveats."""

    tier: TierName
    profile_summary: str
    discovery_model: ModelSlot
    hypothesize_model: ModelSlot
    synthesis_model: ModelSlot
    notes: list[str] = field(default_factory=list)


# --- Curated tier map (May 2026 open-weight landscape) ----------------------
#
# Slot semantics mirror :mod:`causalrag.llm.selector`:
#   discovery   — fast, instruction-tuned, reliable structured-JSON output
#   hypothesize — reasoning / "thinking" model for DAGs + identification
#   synthesis   — long-context summarization / final-report drafting
#
# Quantization defaults are Q4_K_M for the floor, stepping up to Q5_K_M /
# Q8_0 / FP16 as headroom allows. Context windows are conservative — the
# numbers below are what each model can be served at on the corresponding
# tier without OOM during 1-2 concurrent agent loops.


T0_LAPTOP_CPU = TierAssignment(
    tier="T0",
    profile_summary="Laptop / CPU-only, <16 GB RAM",
    discovery_model=ModelSlot(
        name="phi-4-mini-instruct",
        parameters_b=3.8,
        quantization="Q4_K_M",
        context=8192,
        notes="Microsoft Phi-4-Mini; strong JSON output for sub-4B class.",
    ),
    hypothesize_model=ModelSlot(
        name="qwen3-1.7b-thinking",
        parameters_b=1.7,
        quantization="Q4_K_M",
        context=8192,
        notes="Qwen3 1.7B with thinking trace; quality-degraded fallback only.",
    ),
    synthesis_model=ModelSlot(
        name="qwen3-1.7b-instruct",
        parameters_b=1.7,
        quantization="Q4_K_M",
        context=8192,
        notes="Same family as hypothesize slot; minimizes weight-swap thrash.",
    ),
    notes=[
        "Sub-floor: hypothesis generation will be slow and shallow.",
        "Recommend upgrading to >=16 GB VRAM (or 32 GB Apple unified) for real reasoning.",
    ],
)


T1_LAPTOP_GPU = TierAssignment(
    tier="T1",
    profile_summary="Laptop GPU or 16-24 GB Apple Silicon",
    discovery_model=ModelSlot(
        name="qwen3-8b-instruct",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context=16384,
        notes="Qwen3 8B Instruct; reliable structured JSON, fits in 8-12 GB VRAM.",
    ),
    hypothesize_model=ModelSlot(
        name="qwen3-8b-thinking",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context=16384,
        notes="Qwen3 8B with thinking traces; still sub-floor for full DAG synthesis.",
    ),
    synthesis_model=ModelSlot(
        name="qwen3-8b-instruct",
        parameters_b=8.0,
        quantization="Q4_K_M",
        context=16384,
        notes="Shared with discovery slot to avoid model reload on small VRAM.",
    ),
    notes=[
        "Below the 14B reasoning floor; expect weaker identification reasoning.",
    ],
)


T2_DESKTOP_24GB_VRAM = TierAssignment(
    tier="T2",
    profile_summary="Desktop with 24 GB VRAM or 32 GB Apple unified memory",
    discovery_model=ModelSlot(
        name="qwen3-14b-instruct",
        parameters_b=14.0,
        quantization="Q4_K_M",
        context=32768,
        notes="Qwen3 14B Instruct; primary Stage 1c investigator model.",
    ),
    hypothesize_model=ModelSlot(
        name="qwen3-14b-thinking",
        parameters_b=14.0,
        quantization="Q4_K_M",
        context=32768,
        notes="Qwen3 14B with thinking traces — the FLOOR for genuine reasoning.",
    ),
    synthesis_model=ModelSlot(
        name="phi-4-14b",
        parameters_b=14.0,
        quantization="Q4_K_M",
        context=16384,
        notes="Phi-4 14B; strong synthesis/summarization at this tier.",
    ),
    notes=[
        "T2 is the reasoning floor — DAG quality and identification warnings are dependable here.",
    ],
)


T3_DESKTOP_48GB = TierAssignment(
    tier="T3",
    profile_summary="Desktop / prosumer with 48 GB VRAM (or 64-96 GB Apple Silicon)",
    discovery_model=ModelSlot(
        name="mistral-small-3-24b-instruct",
        parameters_b=24.0,
        quantization="Q5_K_M",
        context=32768,
        notes="Mistral-Small-3 24B; faster structured JSON than Gemma at this size.",
    ),
    hypothesize_model=ModelSlot(
        name="gemma-3-27b-instruct",
        parameters_b=27.0,
        quantization="Q5_K_M",
        context=32768,
        notes="Gemma 3 27B; strong causal-language understanding for DAG proposals.",
    ),
    synthesis_model=ModelSlot(
        name="mistral-small-3-24b-instruct",
        parameters_b=24.0,
        quantization="Q5_K_M",
        context=32768,
        notes="Reused for synthesis to avoid double-residency of two 24B+ models.",
    ),
    notes=[
        "Comfortable headroom for 1-2 concurrent agent loops at Q5_K_M.",
    ],
)


T4_WORKSTATION = TierAssignment(
    tier="T4",
    profile_summary="Workstation with 80 GB VRAM (H100/A100) or 128+ GB Apple Silicon",
    discovery_model=ModelSlot(
        name="llama-3.3-70b-instruct",
        parameters_b=70.0,
        quantization="Q4_K_M",
        context=32768,
        notes="Llama 3.3 70B Instruct at Q4_K_M; ~40 GB resident.",
    ),
    hypothesize_model=ModelSlot(
        name="llama-3.3-70b-instruct",
        parameters_b=70.0,
        quantization="Q5_K_M",
        context=32768,
        notes="Same family at Q5_K_M for hypothesize slot; ~48 GB resident.",
    ),
    synthesis_model=ModelSlot(
        name="llama-3.3-70b-instruct",
        parameters_b=70.0,
        quantization="Q4_K_M",
        context=65536,
        notes="Long-context profile for final-report drafting.",
    ),
    notes=[
        "Single-GPU 70B serving; speculative decoding optional but not required.",
    ],
)


T5_DATACENTER = TierAssignment(
    tier="T5",
    profile_summary="Datacenter: dual H100/H200 (160+ GB VRAM)",
    discovery_model=ModelSlot(
        name="llama-3.3-70b-instruct",
        parameters_b=70.0,
        quantization="FP16",
        context=65536,
        notes="Full-precision 70B with low-latency batched serving.",
    ),
    hypothesize_model=ModelSlot(
        name="llama-3.3-70b-instruct+spec-decoding",
        parameters_b=70.0,
        quantization="FP16",
        context=65536,
        notes="70B target + Llama 3.2 3B drafter; ~2-3x throughput on dual H100.",
    ),
    synthesis_model=ModelSlot(
        name="llama-3.3-70b-instruct",
        parameters_b=70.0,
        quantization="FP16",
        context=131072,
        notes="Max-context serving for end-to-end roadmap synthesis.",
    ),
    notes=[
        "Speculative decoding via causalrag.llm.spec_decoding; expect 2-3x tokens/sec.",
    ],
)


_ALL_TIERS: dict[TierName, TierAssignment] = {
    "T0": T0_LAPTOP_CPU,
    "T1": T1_LAPTOP_GPU,
    "T2": T2_DESKTOP_24GB_VRAM,
    "T3": T3_DESKTOP_48GB,
    "T4": T4_WORKSTATION,
    "T5": T5_DATACENTER,
}


# --- Probing ----------------------------------------------------------------


def probe_hardware() -> HardwareProfile:
    """Probe the host machine and return a :class:`HardwareProfile`.

    Uses ``psutil`` (already in deps) for CPU + RAM, and
    ``platform.processor()`` / ``platform.machine()`` for Apple Silicon
    detection. GPU VRAM is best-effort: pynvml first (NVIDIA), then a
    unified-memory estimate for Apple Silicon (~80% of RAM). On systems
    where no GPU is detectable, ``gpu_vram_gb`` is ``None``.
    """
    import psutil  # local import — keeps the module importable in tests w/o psutil

    logical = psutil.cpu_count(logical=True) or 0
    physical = psutil.cpu_count(logical=False) or logical
    vm = psutil.virtual_memory()
    ram_gb = round(vm.total / (1024**3), 2)

    is_apple = (
        platform.system() == "Darwin"
        and (platform.machine() == "arm64" or "Apple" in (platform.processor() or ""))
    )

    gpu_vram_gb: float | None = None
    # Try NVIDIA first.
    try:  # pragma: no cover - depends on host
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            best = 0.0
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                best = max(best, mem.total / (1024**3))
            if count:
                gpu_vram_gb = round(best, 2)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # Apple Silicon: unified memory acts as VRAM (~80% practical budget).
    if gpu_vram_gb is None and is_apple:
        gpu_vram_gb = round(0.8 * ram_gb, 2)

    return HardwareProfile(
        cpu_logical_cores=logical,
        cpu_physical_cores=physical,
        ram_gb=ram_gb,
        gpu_vram_gb=gpu_vram_gb,
        apple_silicon=is_apple,
    )


# --- Assignment -------------------------------------------------------------


def assign_tier(profile: HardwareProfile) -> TierAssignment:
    """Map a :class:`HardwareProfile` to a :class:`TierAssignment`.

    Decision order (Apple Silicon uses RAM as effective VRAM):

    1. dual H100-class (>= 160 GB effective VRAM) ............... T5
    2. single H100-class / M3 Ultra (>= 80 GB) .................. T4
    3. 48 GB VRAM (RTX 6000 Ada, dual 4090) or 64-96 GB Apple ... T3
    4. 24 GB VRAM (RTX 3090/4090) or 32 GB Apple unified ........ T2
    5. small GPU / 16-24 GB Apple unified ....................... T1
    6. CPU-only / <16 GB RAM .................................... T0
    """
    # Apple Silicon: unified memory wins.
    if profile.apple_silicon:
        effective = profile.gpu_vram_gb if profile.gpu_vram_gb is not None else 0.8 * profile.ram_gb
    elif profile.gpu_vram_gb is not None:
        effective = profile.gpu_vram_gb
    else:
        effective = 0.0

    if effective >= 160:
        return T5_DATACENTER
    if effective >= 80:
        return T4_WORKSTATION
    if effective >= 48:
        return T3_DESKTOP_48GB
    if effective >= 24:
        return T2_DESKTOP_24GB_VRAM
    if effective >= 8:
        return T1_LAPTOP_GPU
    # No usable GPU / unified memory budget — fall back on RAM heuristic.
    if profile.ram_gb >= 32 and profile.gpu_vram_gb is None:
        # 32+ GB RAM CPU-only box: T1 quality is realistic if slow.
        return T1_LAPTOP_GPU
    return T0_LAPTOP_CPU


# --- Rendering --------------------------------------------------------------


def _fmt_slot(label: str, slot: ModelSlot) -> str:
    return (
        f"- **{label}**: `{slot.name}` "
        f"({slot.parameters_b:g}B, {slot.quantization}, ctx={slot.context}) — {slot.notes}"
    )


def render_recommendation(t: TierAssignment) -> str:
    """Return a markdown rendering of a :class:`TierAssignment`.

    Used by the ``doctor`` CLI subcommand and by status panes; tests just
    assert it is non-empty markdown.
    """
    lines: list[str] = [
        f"## Hardware tier: {t.tier} — {t.profile_summary}",
        "",
        "### Recommended models",
        _fmt_slot("discovery", t.discovery_model),
        _fmt_slot("hypothesize", t.hypothesize_model),
        _fmt_slot("synthesis", t.synthesis_model),
    ]
    if t.notes:
        lines.extend(["", "### Notes"])
        for note in t.notes:
            lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


__all__ = [
    "HardwareProfile",
    "ModelSlot",
    "TierAssignment",
    "T0_LAPTOP_CPU",
    "T1_LAPTOP_GPU",
    "T2_DESKTOP_24GB_VRAM",
    "T3_DESKTOP_48GB",
    "T4_WORKSTATION",
    "T5_DATACENTER",
    "probe_hardware",
    "assign_tier",
    "render_recommendation",
]
