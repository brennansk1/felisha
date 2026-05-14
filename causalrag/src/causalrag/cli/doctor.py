"""Environment audit — PDD §6.2.

Thin shim around :mod:`causalrag.llm.hardware`. The hardware probe lives in the
``llm`` layer because it is consumed by both the doctor command and the model
selector at every LLM-using command (PDD §16.1: "Detection runs as part of
``doctor`` and again at any LLM-using command").
"""

from __future__ import annotations

from typing import Any

from causalrag.llm.hardware import HardwareProfile, probe
from causalrag.llm.selector import ModelSlots, missing_models, select_slots


def run_doctor(base_url: str = "http://127.0.0.1:11434") -> HardwareProfile:
    """Probe the environment and return a structured profile."""
    return probe(base_url=base_url)


def recommend(profile: HardwareProfile) -> tuple[ModelSlots, tuple[str, ...]]:
    slots = select_slots(profile)
    return slots, missing_models(profile, slots)


def report_dict(profile: HardwareProfile) -> dict[str, Any]:
    slots = select_slots(profile)
    return {
        **profile.to_dict(),
        "recommended_models": {
            "discovery": slots.discovery,
            "hypothesize": slots.hypothesize,
            "utility": slots.utility,
            "quantization": slots.quantization,
        },
        "missing_models": list(missing_models(profile, slots)),
    }
