"""Hardware detection + tier mapping for local LLM selection.

PDD §16.1 (detection order) and §16.2 (tier table). The HardwareProfile is the
canonical structured artifact; the doctor command and the model selector both
read from it. Tiers map to default model slots in :mod:`causalrag.llm.selector`.

Detection probes are independent and best-effort: a missing optional dependency
(pynvml on a CPU box, rpy2 on a no-R machine) degrades gracefully rather than
raising.
"""

from __future__ import annotations

import json
import platform
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GPUDevice:
    name: str
    vram_total_gb: float
    backend: str  # "nvidia" | "apple_silicon" | "amd_rocm"
    index: int = 0


@dataclass
class OllamaStatus:
    reachable: bool
    base_url: str
    version: str | None = None
    models: list[str] = field(default_factory=list)


@dataclass
class HardwareProfile:
    """Cross-platform machine fingerprint used by the model selector.

    ``effective_vram_gb`` is the unified-memory equivalent of GPU VRAM: on Apple
    Silicon it is taken as ~80% of total RAM (the practical Ollama budget); on
    discrete GPUs it is the largest single device's VRAM.
    """

    python_version: str
    platform: str
    cpu_logical: int
    cpu_physical: int | None
    total_ram_gb: float
    available_ram_gb: float
    disk_free_gb: float | None
    gpus: list[GPUDevice]
    ollama: OllamaStatus
    rpy2_importable: bool
    r_binary: str | None
    effective_vram_gb: float
    tier: int
    tier_label: str
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        return out


# --- Probes ------------------------------------------------------------------


def _probe_python() -> tuple[str, bool]:
    import sys

    version = ".".join(str(v) for v in sys.version_info[:3])
    return version, sys.version_info >= (3, 11)


def _probe_cpu_ram() -> tuple[int, int | None, float, float]:
    try:
        import psutil

        logical = psutil.cpu_count(logical=True) or 0
        physical = psutil.cpu_count(logical=False)
        vm = psutil.virtual_memory()
        return logical, physical, round(vm.total / (1024**3), 2), round(vm.available / (1024**3), 2)
    except Exception:
        return 0, None, 0.0, 0.0


def _probe_nvidia() -> list[GPUDevice]:
    try:
        import pynvml

        pynvml.nvmlInit()
        out: list[GPUDevice] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            out.append(
                GPUDevice(
                    name=name,
                    vram_total_gb=round(mem.total / (1024**3), 2),
                    backend="nvidia",
                    index=i,
                )
            )
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return []


def _probe_apple_silicon(total_ram_gb: float) -> list[GPUDevice]:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    chip = "Apple Silicon GPU"
    try:
        out = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if out.returncode == 0 and out.stdout.strip():
            chip = out.stdout.strip()
    except Exception:
        pass
    return [
        GPUDevice(
            name=f"{chip} (unified memory)",
            vram_total_gb=total_ram_gb,
            backend="apple_silicon",
        )
    ]


def _probe_amd() -> list[GPUDevice]:
    rocm = shutil.which("rocm-smi")
    if not rocm:
        return []
    try:
        out = subprocess.run(
            [rocm, "--showmeminfo", "vram", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if out.returncode != 0 or not out.stdout:
            return []
        data = json.loads(out.stdout)
    except Exception:
        return []
    devices: list[GPUDevice] = []
    for key, info in data.items():
        if not key.startswith("card"):
            continue
        total = info.get("VRAM Total Memory (B)") or info.get("vram_total")
        if total is None:
            continue
        try:
            vram = round(float(total) / (1024**3), 2)
        except (TypeError, ValueError):
            continue
        devices.append(
            GPUDevice(
                name=info.get("Card series", key),
                vram_total_gb=vram,
                backend="amd_rocm",
                index=int(key.replace("card", "")) if key[4:].isdigit() else 0,
            )
        )
    return devices


def _probe_ollama(base_url: str = "http://127.0.0.1:11434") -> OllamaStatus:
    out = OllamaStatus(reachable=False, base_url=base_url)
    host, _, port_s = base_url.replace("http://", "").replace("https://", "").partition(":")
    try:
        port = int(port_s or "11434")
        with socket.create_connection((host, port), timeout=0.5):
            out.reachable = True
    except OSError:
        return out
    try:
        import httpx
    except ImportError:
        return out
    try:
        r = httpx.get(f"{base_url}/api/version", timeout=1.0)
        if r.status_code == 200:
            out.version = r.json().get("version")
        r = httpx.get(f"{base_url}/api/tags", timeout=1.5)
        if r.status_code == 200:
            out.models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
    except Exception:
        pass
    return out


def _probe_r() -> tuple[bool, str | None]:
    r_bin = shutil.which("R")
    try:
        import rpy2  # noqa: F401

        return True, r_bin
    except Exception:
        return False, r_bin


def _probe_disk() -> float | None:
    try:
        usage = shutil.disk_usage(".")
        return round(usage.free / (1024**3), 2)
    except Exception:
        return None


# --- Tier mapping ------------------------------------------------------------


@dataclass(frozen=True)
class TierSpec:
    tier: int
    label: str
    min_effective_vram_gb: float
    min_ram_gb: float


# PDD §16.2 — the "16B floor" is Tier 2. Tier 0/1 are sub-floor.
TIERS: tuple[TierSpec, ...] = (
    TierSpec(tier=0, label="T0 (sub-floor: CPU-only, <16 GB RAM)", min_effective_vram_gb=0.0, min_ram_gb=0.0),
    TierSpec(tier=1, label="T1 (sub-floor: 16–32 GB RAM, no GPU)", min_effective_vram_gb=0.0, min_ram_gb=16.0),
    TierSpec(tier=2, label="T2 FLOOR (12–16 GB VRAM or 32 GB unified)", min_effective_vram_gb=12.0, min_ram_gb=0.0),
    TierSpec(tier=3, label="T3 prosumer (24 GB VRAM)", min_effective_vram_gb=24.0, min_ram_gb=0.0),
    TierSpec(tier=4, label="T4 workstation (48 GB VRAM)", min_effective_vram_gb=48.0, min_ram_gb=0.0),
    TierSpec(tier=5, label="T5 server (80+ GB VRAM)", min_effective_vram_gb=80.0, min_ram_gb=0.0),
)


def _effective_vram(total_ram_gb: float, gpus: list[GPUDevice]) -> float:
    """For Apple Silicon, use ~80% of unified RAM as the Ollama budget. For
    discrete GPUs, use the largest single device's VRAM. For CPU-only systems,
    use a fraction of free RAM rounded down to a tier-relevant amount."""
    if any(g.backend == "apple_silicon" for g in gpus):
        return round(0.8 * total_ram_gb, 2)
    if gpus:
        return max(g.vram_total_gb for g in gpus)
    # Pure CPU path — return total RAM divided down so a 32 GB box still maps to T2
    # via the RAM-only fallback in TIERS.
    return 0.0


def _select_tier(effective_vram_gb: float, total_ram_gb: float) -> TierSpec:
    candidate = TIERS[0]
    for spec in TIERS:
        if effective_vram_gb >= spec.min_effective_vram_gb and total_ram_gb >= spec.min_ram_gb:
            candidate = spec
    # RAM-only fallback: a CPU box with >=32 GB RAM still qualifies for T2.
    if candidate.tier < 2 and total_ram_gb >= 32:
        return TIERS[2]
    return candidate


# --- Top-level entrypoint ----------------------------------------------------


def probe(base_url: str = "http://127.0.0.1:11434") -> HardwareProfile:
    py_version, py_ok = _probe_python()
    logical, physical, total_gb, avail_gb = _probe_cpu_ram()
    disk = _probe_disk()
    gpus = _probe_nvidia() or _probe_apple_silicon(total_gb) or _probe_amd()
    ollama = _probe_ollama(base_url=base_url)
    rpy2_ok, r_bin = _probe_r()
    eff_vram = _effective_vram(total_gb, gpus)
    tier_spec = _select_tier(eff_vram, total_gb)

    warnings: list[str] = []
    if not py_ok:
        warnings.append(f"Python {py_version} detected; CausalRoadmap requires >= 3.11.")
    if not ollama.reachable:
        warnings.append(
            f"Ollama not reachable at {base_url}. Install from https://ollama.com or set --base-url."
        )
    if tier_spec.tier < 2:
        warnings.append(
            f"Hardware {tier_spec.label} is below the 16B reasoning floor (T2). "
            "Local hypothesis generation will be slow or use a smaller fallback."
        )

    return HardwareProfile(
        python_version=py_version,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        cpu_logical=logical,
        cpu_physical=physical,
        total_ram_gb=total_gb,
        available_ram_gb=avail_gb,
        disk_free_gb=disk,
        gpus=list(gpus),
        ollama=ollama,
        rpy2_importable=rpy2_ok,
        r_binary=r_bin,
        effective_vram_gb=eff_vram,
        tier=tier_spec.tier,
        tier_label=tier_spec.label,
        warnings=warnings,
    )
