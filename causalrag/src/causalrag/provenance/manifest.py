"""RunManifest — bit-reproducible record of a single /auto or /run invocation.

Sprint 1.2 (PDD design principle 10: 'Honest provenance').

A :class:`RunManifest` hashes everything a reviewer would need to replay or
contest a finding: dataset (over canonicalised rows), DAG (sorted edge
list), estimand (per-hypothesis Pydantic JSON), RNG seeds, code SHA,
Python + R lockfiles, model digests, and prompt hashes.

The :class:`ManifestBuilder` accumulates these fields incrementally during
a master-loop run so the resulting manifest can be persisted alongside the
study YAML.
"""

from __future__ import annotations

import hashlib
import json
import platform as _platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict

from causalrag.core.estimand import CausalEstimand
from causalrag.core.graph import CausalGraph

__all__ = ["RunManifest", "ManifestBuilder"]


# ── Schema ────────────────────────────────────────────────────────────────────


class RunManifest(BaseModel):
    """Bit-reproducible record of a single /auto or /run invocation.

    Hashes everything a reviewer would need to replay or contest a
    finding: dataset (hash over canonicalised rows), DAG, estimand,
    RNG seeds, code SHA, Python + R lockfiles, model digests, prompt
    hashes.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    run_id: str  # uuid4 hex
    timestamp: datetime
    pipeline_version: str  # pkg version
    git_sha: str | None = None  # current HEAD if in repo

    # Data
    dataset_path: str
    dataset_sha256: str  # sha256 of canonicalised csv
    dataset_n_rows: int
    dataset_n_cols: int
    dataset_columns: list[str]

    # DAG
    dag_hash: str  # sha256 of normalised edge list
    n_dag_nodes: int
    n_dag_edges: int

    # Estimand (per hypothesis)
    estimand_hashes: dict[str, str]  # hypothesis_id -> sha256(estimand)

    # RNG
    seeds: dict[str, int]

    # Environment
    python_version: str
    platform: str
    package_versions: dict[str, str]
    r_packages: dict[str, str] | None = None

    # LLM
    llm_model: str | None = None
    llm_digest: str | None = None
    prompt_hashes: dict[str, str]

    # Replay-friendly
    cli_command: str | None = None


# ── Hashing helpers (module-level so they're independently testable) ─────────


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _canonical_csv(df: pd.DataFrame) -> str:
    """Canonicalise a DataFrame to a stable CSV string for hashing."""
    return df.reset_index(drop=True).to_csv(index=False, lineterminator="\n")


def _canonical_edges(graph: CausalGraph) -> str:
    """Return a canonical edge-list string for hashing."""
    edges = sorted((e.source, e.target) for e in graph.edges)
    return "\n".join(f"{s}\t{t}" for s, t in edges)


def _detect_git_sha() -> str | None:
    """Return the current git HEAD, or None if not in a repo or git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _pipeline_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("causalrag")
        except PackageNotFoundError:
            pass
    except Exception:  # noqa: BLE001 — defensive; metadata can fail in editable installs
        pass
    try:
        from causalrag import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "0.0.0+unknown"


# ── Builder ──────────────────────────────────────────────────────────────────


class ManifestBuilder:
    """Incrementally accumulate manifest fields during a run."""

    def __init__(self, run_id: str | None = None) -> None:
        self._fields: dict[str, Any] = {
            "schema_version": "1",
            "run_id": run_id or uuid4().hex,
            "timestamp": datetime.now(UTC),
            "pipeline_version": _pipeline_version(),
            "git_sha": _detect_git_sha(),
            "estimand_hashes": {},
            "seeds": {},
            "package_versions": {},
            "r_packages": None,
            "prompt_hashes": {},
            "python_version": sys.version.split()[0],
            "platform": _platform.platform(),
        }

    # ─── Dataset ─────────────────────────────────────────────────────────────
    def hash_dataset(
        self, df: pd.DataFrame, *, path: str | Path | None = None
    ) -> str:
        """Hash *df* via canonical CSV and record dataset metadata."""
        sha = _sha256_text(_canonical_csv(df))
        self._fields["dataset_sha256"] = sha
        self._fields["dataset_n_rows"] = int(df.shape[0])
        self._fields["dataset_n_cols"] = int(df.shape[1])
        self._fields["dataset_columns"] = [str(c) for c in df.columns]
        if path is not None:
            self._fields["dataset_path"] = str(path)
        elif "dataset_path" not in self._fields:
            self._fields["dataset_path"] = ""
        return sha

    def set_dataset_path(self, path: str | Path) -> None:
        self._fields["dataset_path"] = str(path)

    # ─── DAG ─────────────────────────────────────────────────────────────────
    def hash_dag(self, graph: CausalGraph) -> str:
        """Hash a :class:`CausalGraph` via its sorted edge list."""
        sha = _sha256_text(_canonical_edges(graph))
        self._fields["dag_hash"] = sha
        self._fields["n_dag_nodes"] = len(graph.nodes)
        self._fields["n_dag_edges"] = len(graph.edges)
        return sha

    # ─── Estimand ────────────────────────────────────────────────────────────
    def hash_estimand(
        self, estimand: CausalEstimand, *, hypothesis_id: str | None = None
    ) -> str:
        """Hash a CausalEstimand via its Pydantic JSON dump."""
        sha = _sha256_text(estimand.model_dump_json())
        if hypothesis_id is not None:
            self._fields["estimand_hashes"][hypothesis_id] = sha
        return sha

    # ─── RNG ─────────────────────────────────────────────────────────────────
    def record_seed(self, name: str, seed: int) -> None:
        self._fields["seeds"][name] = int(seed)

    # ─── Environment ─────────────────────────────────────────────────────────
    def record_package_versions(self) -> None:
        """Snapshot installed pkg versions via importlib.metadata."""
        from importlib.metadata import distributions

        versions: dict[str, str] = {}
        for dist in distributions():
            try:
                name = dist.metadata["Name"]
                ver = dist.version
            except Exception:  # noqa: BLE001
                continue
            if name:
                versions[str(name).lower()] = str(ver)
        self._fields["package_versions"] = versions

    def record_r_packages(self) -> None:
        """Snapshot R-bridge packages via rpy2 if available, else None."""
        try:
            from rpy2.robjects import r  # type: ignore[import-not-found]
        except Exception:  # noqa: BLE001
            self._fields["r_packages"] = None
            return
        try:
            inst = r("as.data.frame(installed.packages()[, c('Package','Version')])")
            names = list(inst.rx2("Package"))  # type: ignore[union-attr]
            vers = list(inst.rx2("Version"))  # type: ignore[union-attr]
            self._fields["r_packages"] = {str(n): str(v) for n, v in zip(names, vers, strict=False)}
        except Exception:  # noqa: BLE001
            self._fields["r_packages"] = None

    # ─── LLM ─────────────────────────────────────────────────────────────────
    def record_llm(self, *, model: str | None = None, digest: str | None = None) -> None:
        if model is not None:
            self._fields["llm_model"] = model
        if digest is not None:
            self._fields["llm_digest"] = digest

    def record_prompt(self, name: str, text: str) -> str:
        """Record sha256 of the full system+user prompt text for *name*."""
        sha = _sha256_text(text)
        self._fields["prompt_hashes"][name] = sha
        return sha

    # ─── CLI ─────────────────────────────────────────────────────────────────
    def record_cli_command(self, command: str) -> None:
        self._fields["cli_command"] = command

    # ─── Build / persist ─────────────────────────────────────────────────────
    def build(self) -> RunManifest:
        # Provide minimum defaults for required fields the caller may have
        # neglected to set — fail loud only when actually missing critical
        # hashes.
        required = (
            "dataset_path",
            "dataset_sha256",
            "dataset_n_rows",
            "dataset_n_cols",
            "dataset_columns",
            "dag_hash",
            "n_dag_nodes",
            "n_dag_edges",
        )
        missing = [k for k in required if k not in self._fields]
        if missing:
            raise ValueError(
                f"ManifestBuilder.build(): missing required field(s) {missing}; "
                "call hash_dataset() and hash_dag() first."
            )
        return RunManifest(**self._fields)

    def save(self, path: Path) -> None:
        """Write the manifest as canonical JSON (sorted keys, UTF-8)."""
        manifest = self.build()
        data = manifest.model_dump(mode="json")
        text = json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2)
        Path(path).write_text(text + "\n", encoding="utf-8")
