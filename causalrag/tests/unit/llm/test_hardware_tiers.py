"""Unit tests for ``causalrag.llm.hardware_tiers`` (Sprint 8.3).

We don't exercise ``probe_hardware`` against the real host (that would
make the test bed-dependent). Instead we:

* construct synthetic :class:`HardwareProfile` values for each T0-T5
  scenario and assert :func:`assign_tier` returns the expected tier;
* assert :func:`render_recommendation` returns a non-empty markdown
  string containing the tier label and all three slot names;
* sanity-check that :func:`probe_hardware` returns a structurally valid
  profile on whatever host the tests happen to run on.
"""

from __future__ import annotations

import pytest

from causalrag.llm.hardware_tiers import (
    HardwareProfile,
    ModelSlot,
    T0_LAPTOP_CPU,
    T1_LAPTOP_GPU,
    T2_DESKTOP_24GB_VRAM,
    T3_DESKTOP_48GB,
    T4_WORKSTATION,
    T5_DATACENTER,
    TierAssignment,
    assign_tier,
    probe_hardware,
    render_recommendation,
)


# --- Synthetic profiles -----------------------------------------------------


def _profile(
    *,
    cores: int = 8,
    ram: float = 16.0,
    vram: float | None = None,
    apple: bool = False,
) -> HardwareProfile:
    return HardwareProfile(
        cpu_logical_cores=cores,
        cpu_physical_cores=max(1, cores // 2),
        ram_gb=ram,
        gpu_vram_gb=vram,
        apple_silicon=apple,
    )


# --- assign_tier ------------------------------------------------------------


@pytest.mark.parametrize(
    "profile, expected",
    [
        # T0: laptop CPU-only, 8 GB RAM, no GPU.
        (_profile(cores=4, ram=8.0, vram=None), T0_LAPTOP_CPU),
        # T0: even a 16 GB CPU-only box stays sub-floor (no usable VRAM budget).
        (_profile(cores=8, ram=16.0, vram=None), T0_LAPTOP_CPU),
        # T1: laptop GPU with 8-12 GB VRAM.
        (_profile(cores=8, ram=16.0, vram=8.0), T1_LAPTOP_GPU),
        (_profile(cores=8, ram=32.0, vram=12.0), T1_LAPTOP_GPU),
        # T1: Apple Silicon with 16 GB unified (~12.8 effective).
        (_profile(cores=8, ram=16.0, vram=12.8, apple=True), T1_LAPTOP_GPU),
        # T2: 24 GB consumer GPU.
        (_profile(cores=16, ram=64.0, vram=24.0), T2_DESKTOP_24GB_VRAM),
        # T2: Apple Silicon 32 GB unified (~25.6 effective).
        (_profile(cores=10, ram=32.0, vram=25.6, apple=True), T2_DESKTOP_24GB_VRAM),
        # T3: 48 GB VRAM workstation.
        (_profile(cores=24, ram=128.0, vram=48.0), T3_DESKTOP_48GB),
        # T3: M2 Max 64 GB unified (~51 effective).
        (_profile(cores=12, ram=64.0, vram=51.2, apple=True), T3_DESKTOP_48GB),
        # T4: single H100 80 GB.
        (_profile(cores=64, ram=256.0, vram=80.0), T4_WORKSTATION),
        # T4: M3 Ultra 128 GB unified (~102 effective).
        (_profile(cores=24, ram=128.0, vram=102.4, apple=True), T4_WORKSTATION),
        # T5: dual H100 / H200 (160+ GB).
        (_profile(cores=128, ram=512.0, vram=160.0), T5_DATACENTER),
        (_profile(cores=192, ram=1024.0, vram=320.0), T5_DATACENTER),
    ],
)
def test_assign_tier_picks_expected_assignment(
    profile: HardwareProfile, expected: TierAssignment
) -> None:
    got = assign_tier(profile)
    assert got is expected, f"expected {expected.tier} for {profile}, got {got.tier}"


def test_assign_tier_cpu_only_with_32gb_ram_falls_back_to_t1() -> None:
    # Special-case: CPU-only box with >= 32 GB RAM still gets T1 model
    # recommendations (the user can run 8B quantized on CPU, slowly).
    p = _profile(cores=16, ram=64.0, vram=None)
    assert assign_tier(p) is T1_LAPTOP_GPU


def test_assign_tier_apple_silicon_without_vram_uses_unified_memory() -> None:
    # If gpu_vram_gb is unset on an Apple Silicon profile, the assignment
    # falls back to ~80% of RAM as effective VRAM.
    p = HardwareProfile(
        cpu_logical_cores=10,
        cpu_physical_cores=10,
        ram_gb=32.0,
        gpu_vram_gb=None,
        apple_silicon=True,
    )
    # 0.8 * 32 = 25.6 -> T2 threshold (>= 24).
    assert assign_tier(p) is T2_DESKTOP_24GB_VRAM


# --- Tier map sanity checks -------------------------------------------------


@pytest.mark.parametrize(
    "tier",
    [T0_LAPTOP_CPU, T1_LAPTOP_GPU, T2_DESKTOP_24GB_VRAM, T3_DESKTOP_48GB, T4_WORKSTATION, T5_DATACENTER],
)
def test_each_tier_has_three_distinct_slots(tier: TierAssignment) -> None:
    for slot in (tier.discovery_model, tier.hypothesize_model, tier.synthesis_model):
        assert isinstance(slot, ModelSlot)
        assert slot.name
        assert slot.parameters_b > 0
        assert slot.quantization
        assert slot.context > 0
        assert slot.notes


def test_tier_labels_are_unique() -> None:
    labels = {
        t.tier
        for t in (
            T0_LAPTOP_CPU,
            T1_LAPTOP_GPU,
            T2_DESKTOP_24GB_VRAM,
            T3_DESKTOP_48GB,
            T4_WORKSTATION,
            T5_DATACENTER,
        )
    }
    assert labels == {"T0", "T1", "T2", "T3", "T4", "T5"}


# --- render_recommendation --------------------------------------------------


@pytest.mark.parametrize(
    "tier",
    [T0_LAPTOP_CPU, T1_LAPTOP_GPU, T2_DESKTOP_24GB_VRAM, T3_DESKTOP_48GB, T4_WORKSTATION, T5_DATACENTER],
)
def test_render_recommendation_non_empty_markdown(tier: TierAssignment) -> None:
    md = render_recommendation(tier)
    assert md
    assert isinstance(md, str)
    # Tier header is present.
    assert f"## Hardware tier: {tier.tier}" in md
    # All three slot model names appear in the rendering.
    assert tier.discovery_model.name in md
    assert tier.hypothesize_model.name in md
    assert tier.synthesis_model.name in md
    # Notes section is rendered when present.
    if tier.notes:
        assert "### Notes" in md


def test_render_recommendation_handles_no_notes() -> None:
    # Construct a tier with empty notes to exercise the no-notes branch.
    bare = TierAssignment(
        tier="T2",
        profile_summary="bare-bones",
        discovery_model=T2_DESKTOP_24GB_VRAM.discovery_model,
        hypothesize_model=T2_DESKTOP_24GB_VRAM.hypothesize_model,
        synthesis_model=T2_DESKTOP_24GB_VRAM.synthesis_model,
        notes=[],
    )
    md = render_recommendation(bare)
    assert "### Notes" not in md
    assert md.strip().endswith(bare.synthesis_model.notes)


# --- probe_hardware (host-agnostic structural check) ------------------------


def test_probe_hardware_returns_structurally_valid_profile() -> None:
    p = probe_hardware()
    assert isinstance(p, HardwareProfile)
    assert p.cpu_logical_cores >= 1
    assert p.cpu_physical_cores >= 1
    assert p.ram_gb > 0
    # gpu_vram_gb is either None or a positive float.
    assert p.gpu_vram_gb is None or p.gpu_vram_gb > 0
    # apple_silicon is a bool.
    assert isinstance(p.apple_silicon, bool)
    # assign_tier must accept whatever the probe returns.
    assigned = assign_tier(p)
    assert assigned.tier in {"T0", "T1", "T2", "T3", "T4", "T5"}
