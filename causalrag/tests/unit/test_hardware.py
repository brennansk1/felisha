from __future__ import annotations

from causalrag.llm.hardware import (
    GPUDevice,
    HardwareProfile,
    OllamaStatus,
    _effective_vram,
    _select_tier,
    probe,
)
from causalrag.llm.selector import missing_models, select_slots


def test_probe_returns_structured_profile() -> None:
    p = probe()
    assert isinstance(p, HardwareProfile)
    assert p.tier in {0, 1, 2, 3, 4, 5}
    assert p.tier_label.startswith(f"T{p.tier}")
    assert p.effective_vram_gb >= 0.0
    assert isinstance(p.warnings, list)


def test_apple_silicon_uses_unified_ram_as_effective_vram() -> None:
    gpus = [GPUDevice(name="m_pro", vram_total_gb=24.0, backend="apple_silicon")]
    assert _effective_vram(total_ram_gb=24.0, gpus=gpus) == round(0.8 * 24.0, 2)


def test_discrete_nvidia_uses_largest_device() -> None:
    gpus = [
        GPUDevice(name="3090", vram_total_gb=24.0, backend="nvidia"),
        GPUDevice(name="3060", vram_total_gb=12.0, backend="nvidia"),
    ]
    assert _effective_vram(total_ram_gb=64.0, gpus=gpus) == 24.0


def test_tier_floor_is_t2_at_12gb_vram() -> None:
    spec = _select_tier(effective_vram_gb=12.0, total_ram_gb=16.0)
    assert spec.tier == 2


def test_tier_t0_for_thin_laptop() -> None:
    spec = _select_tier(effective_vram_gb=0.0, total_ram_gb=8.0)
    assert spec.tier == 0


def test_tier_ram_fallback_lifts_cpu_box_to_t2() -> None:
    """A 32-GB CPU-only box still qualifies for T2 via the RAM fallback."""
    spec = _select_tier(effective_vram_gb=0.0, total_ram_gb=32.0)
    assert spec.tier == 2


def test_selector_returns_qwen3_14b_at_tier2() -> None:
    fake = _fake_profile(tier=2, effective_vram_gb=16.0)
    slots = select_slots(fake)
    assert "14b" in slots.discovery or "14B" in slots.discovery
    assert "deepseek-r1" in slots.hypothesize


def test_missing_models_omits_installed() -> None:
    """When the tier-default isn't installed, select_slots now substitutes an
    installed model from the same family. missing_models then operates on the
    substituted slots and reports only models that genuinely aren't present.
    """
    profile = _fake_profile(
        tier=2,
        effective_vram_gb=16.0,
        installed=("qwen3:14b-q4_K_M", "deepseek-r1:14b-q4_K_M"),
    )
    slots = select_slots(profile)
    # Discovery + hypothesize map to installed models
    assert slots.discovery in profile.ollama.models
    assert slots.hypothesize in profile.ollama.models
    # missing_models returns slot models that are not present locally
    missing = missing_models(profile, slots)
    for m in missing:
        assert m not in profile.ollama.models


def test_fallback_substitutes_when_tier_default_absent() -> None:
    """T2 default discovery is qwen3:14b-q4_K_M. With only llama3.1:8b
    installed, the selector falls back to it rather than returning a
    non-existent model."""
    profile = _fake_profile(
        tier=2,
        effective_vram_gb=16.0,
        installed=("llama3.1:8b",),
    )
    slots = select_slots(profile)
    assert slots.discovery == "llama3.1:8b"
    assert slots.hypothesize == "llama3.1:8b"


def _fake_profile(
    *,
    tier: int,
    effective_vram_gb: float,
    installed: tuple[str, ...] = (),
) -> HardwareProfile:
    return HardwareProfile(
        python_version="3.12.0",
        platform="test",
        cpu_logical=8,
        cpu_physical=8,
        total_ram_gb=32.0,
        available_ram_gb=24.0,
        disk_free_gb=500.0,
        gpus=[],
        ollama=OllamaStatus(
            reachable=True, base_url="http://127.0.0.1:11434", version="0.4", models=list(installed)
        ),
        rpy2_importable=False,
        r_binary=None,
        effective_vram_gb=effective_vram_gb,
        tier=tier,
        tier_label=f"T{tier} test",
        warnings=[],
    )
