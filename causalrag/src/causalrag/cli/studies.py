"""Study save/load/branch — Sprint 4.3 (PDD §33, week 4).

A *study* is a named, versioned snapshot of a project directory (the
``study.causalrag.yaml`` protocol, cassettes, reports, and any per-chain
artifacts produced by ``/auto``). Studies live under a registry root::

    studies/
        <name>/
            .studies.json                 # registry metadata (branches, parents)
            main/
                study.causalrag.yaml
                executive_synthesis.json
                runs/<chain_id>/...
                manifest.lock             # provenance hash bundle
            <branch>/
                ...

The registry is intentionally filesystem-native (no DB) so analysts can
inspect, diff, and check it into git like any other research artifact.

This module exposes:

* :class:`StudyBranch` — a single branch record.
* :class:`StudyRegistry` — programmatic API used by the TUI and CLI.
* Three thin ``causalrag_study_*_cli`` callables that map argparse-style
  kwargs onto the registry methods. The Typer commands are wired up
  separately in ``cli/main.py`` (touched in its own sprint).

We deliberately keep the manifest a JSON sidecar (not a Pydantic ``RunManifest``
instance forced into existence) because branch checkpoints often happen
before estimation has run, when the dataset/DAG hashes don't yet exist.
When a real :class:`RunManifest` *is* present in the source dir we copy
it through verbatim so reproducibility is preserved.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from causalrag.provenance.manifest import RunManifest

__all__ = [
    "StudyBranch",
    "StudyRegistry",
    "StudyError",
    "causalrag_study_save_cli",
    "causalrag_study_load_cli",
    "causalrag_study_branch_cli",
    "causalrag_study_list_cli",
]


# ── Errors ───────────────────────────────────────────────────────────────────


class StudyError(RuntimeError):
    """Raised on study-registry violations (missing name/branch, collisions)."""


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class StudyBranch:
    """A single named branch within a study."""

    name: str
    parent: str | None
    created_at: datetime
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent": self.parent,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudyBranch:
        ts = data["created_at"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        return cls(
            name=str(data["name"]),
            parent=data.get("parent"),
            created_at=ts,
            description=str(data.get("description", "")),
        )


@dataclass
class _StudyMeta:
    """In-memory view of ``<study>/.studies.json``."""

    name: str
    created_at: datetime
    description: str = ""
    branches: dict[str, StudyBranch] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "name": self.name,
            "created_at": self.created_at.astimezone(UTC).isoformat(),
            "description": self.description,
            "branches": [b.to_dict() for b in self.branches.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _StudyMeta:
        ts = data["created_at"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        branches = {
            b["name"]: StudyBranch.from_dict(b) for b in data.get("branches", [])
        }
        return cls(
            name=str(data["name"]),
            created_at=ts,
            description=str(data.get("description", "")),
            branches=branches,
        )


# ── Registry ─────────────────────────────────────────────────────────────────


@dataclass
class StudyRegistry:
    """Filesystem-native registry of named studies and their branches."""

    studies_dir: Path

    # -- public API ------------------------------------------------------------

    def list_studies(self) -> list[str]:
        """Return the alphabetised list of study names in the registry."""
        root = Path(self.studies_dir)
        if not root.exists():
            return []
        out: list[str] = []
        for child in sorted(root.iterdir()):
            if child.is_dir() and (child / ".studies.json").exists():
                out.append(child.name)
        return out

    def list_branches(self, study: str) -> list[StudyBranch]:
        """Return branches for *study* (chronologically — main first)."""
        meta = self._read_meta(study)
        # main always first if present, then by created_at
        branches = list(meta.branches.values())
        branches.sort(key=lambda b: (0 if b.name == "main" else 1, b.created_at))
        return branches

    def save(
        self,
        source_dir: Path,
        name: str,
        *,
        description: str = "",
        branch: str = "main",
        overwrite: bool = False,
    ) -> Path:
        """Copy *source_dir* into ``studies/<name>/<branch>/`` and write metadata.

        Returns the path of the saved branch directory. Raises
        :class:`StudyError` on name collisions (unless ``overwrite=True``).
        """
        source = Path(source_dir)
        if not source.is_dir():
            raise StudyError(f"source_dir does not exist or is not a directory: {source}")

        root = Path(self.studies_dir)
        root.mkdir(parents=True, exist_ok=True)
        study_dir = root / name

        meta: _StudyMeta
        if study_dir.exists():
            if (study_dir / ".studies.json").exists():
                meta = self._read_meta(name)
                if branch in meta.branches and not overwrite:
                    raise StudyError(
                        f"study '{name}' already has branch '{branch}'. "
                        "Pass overwrite=True or use branch() to create a new one."
                    )
            else:
                raise StudyError(
                    f"path {study_dir} exists but is not a study (no .studies.json)."
                )
        else:
            study_dir.mkdir(parents=True)
            meta = _StudyMeta(
                name=name,
                created_at=_now(),
                description=description,
            )

        branch_dir = study_dir / branch
        if branch_dir.exists() and overwrite:
            shutil.rmtree(branch_dir)
        _copy_tree(source, branch_dir)

        # Manifest lock: prefer a pre-existing RunManifest if the source has one,
        # otherwise fall back to a lightweight study-level lock.
        _write_manifest_lock(source, branch_dir, study=name, branch=branch)

        # Register the branch
        if branch not in meta.branches:
            meta.branches[branch] = StudyBranch(
                name=branch,
                parent=None,
                created_at=_now(),
                description=description,
            )
        else:
            # overwrite: keep parent, refresh timestamp
            prev = meta.branches[branch]
            meta.branches[branch] = StudyBranch(
                name=branch,
                parent=prev.parent,
                created_at=_now(),
                description=description or prev.description,
            )
        self._write_meta(meta)
        return branch_dir

    def load(
        self,
        name: str,
        target_dir: Path,
        *,
        branch: str = "main",
        overwrite: bool = False,
    ) -> Path:
        """Copy a saved branch out to *target_dir*. Returns *target_dir*."""
        self._require_branch(name, branch)
        branch_dir = Path(self.studies_dir) / name / branch
        target = Path(target_dir)
        if target.exists() and any(target.iterdir()):
            if not overwrite:
                raise StudyError(
                    f"target {target} is not empty; pass overwrite=True to replace."
                )
            shutil.rmtree(target)
        _copy_tree(branch_dir, target)
        return target

    def branch(
        self,
        name: str,
        new_branch: str,
        *,
        from_branch: str = "main",
        description: str = "",
    ) -> Path:
        """Fork *from_branch* into *new_branch*. Returns the new branch dir."""
        self._require_branch(name, from_branch)
        meta = self._read_meta(name)
        if new_branch in meta.branches:
            raise StudyError(
                f"study '{name}' already has branch '{new_branch}'."
            )
        src = Path(self.studies_dir) / name / from_branch
        dst = Path(self.studies_dir) / name / new_branch
        _copy_tree(src, dst)
        # Refresh the manifest lock to record the fork relationship
        _write_manifest_lock(
            dst, dst, study=name, branch=new_branch, parent_branch=from_branch
        )
        meta.branches[new_branch] = StudyBranch(
            name=new_branch,
            parent=from_branch,
            created_at=_now(),
            description=description,
        )
        self._write_meta(meta)
        return dst

    # -- private helpers -------------------------------------------------------

    def _study_dir(self, name: str) -> Path:
        return Path(self.studies_dir) / name

    def _meta_path(self, name: str) -> Path:
        return self._study_dir(name) / ".studies.json"

    def _read_meta(self, name: str) -> _StudyMeta:
        path = self._meta_path(name)
        if not path.exists():
            raise StudyError(f"unknown study: {name!r}")
        return _StudyMeta.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_meta(self, meta: _StudyMeta) -> None:
        path = self._meta_path(meta.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(meta.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _require_branch(self, name: str, branch: str) -> None:
        meta = self._read_meta(name)
        if branch not in meta.branches:
            raise StudyError(
                f"study '{name}' has no branch '{branch}'. "
                f"Available: {sorted(meta.branches)}"
            )
        if not (self._study_dir(name) / branch).is_dir():
            raise StudyError(
                f"registry corruption: branch '{branch}' is listed but missing on disk."
            )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


# Directories we never copy into the study (caches, virtualenvs, scratch).
_IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    ".git",
    ".DS_Store",
}


def _copy_tree(src: Path, dst: Path) -> None:
    """Recursively copy *src* into *dst*, skipping caches and VCS dirs."""
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.name in _IGNORE_NAMES:
            continue
        target = dst / entry.name
        if entry.is_dir():
            _copy_tree(entry, target)
        else:
            shutil.copy2(entry, target)


def _write_manifest_lock(
    source: Path,
    branch_dir: Path,
    *,
    study: str,
    branch: str,
    parent_branch: str | None = None,
) -> Path:
    """Write ``manifest.lock`` describing the saved branch.

    If the *source* dir contains a ``manifest.lock`` produced by a real
    :class:`~causalrag.provenance.manifest.RunManifest` we round-trip it
    through Pydantic to verify the schema and re-emit canonical JSON.
    Otherwise we fall back to a lightweight study-level lock — enough for
    the registry to track what was saved, when, and from where, without
    forcing the analyst to have already run estimation.
    """
    branch_dir.mkdir(parents=True, exist_ok=True)
    out = branch_dir / "manifest.lock"

    # Try to inherit a real RunManifest if present at the source.
    candidate = Path(source) / "manifest.lock"
    if candidate.exists() and candidate.resolve() != out.resolve():
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            # If this looks like a RunManifest, validate via Pydantic.
            if "run_id" in raw and "dataset_sha256" in raw:
                validated = RunManifest.model_validate(raw)
                payload = validated.model_dump(mode="json")
                payload["__study__"] = {
                    "study": study,
                    "branch": branch,
                    "parent_branch": parent_branch,
                    "saved_at": _now().isoformat(),
                }
                out.write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return out
        except Exception:  # noqa: BLE001 — fall through to lightweight lock
            pass

    # Lightweight fallback — capture what we *do* know about the snapshot.
    payload = {
        "schema_version": "1",
        "kind": "study_branch_lock",
        "study": study,
        "branch": branch,
        "parent_branch": parent_branch,
        "saved_at": _now().isoformat(),
        "source": str(Path(source).resolve()),
        "files": sorted(_relative_files(branch_dir, out)),
    }
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def _relative_files(root: Path, exclude: Path) -> list[str]:
    """List files under *root* (excluding *exclude* itself), as POSIX paths."""
    root = Path(root)
    exclude_resolved = Path(exclude).resolve()
    out: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.resolve() == exclude_resolved:
            continue
        out.append(p.relative_to(root).as_posix())
    return out


# ── CLI shims ────────────────────────────────────────────────────────────────


def _registry(studies_dir: Path | None) -> StudyRegistry:
    base = Path(studies_dir) if studies_dir is not None else Path.cwd() / "studies"
    return StudyRegistry(studies_dir=base)


def causalrag_study_save_cli(
    source: Path,
    name: str,
    *,
    studies_dir: Path | None = None,
    description: str = "",
    branch: str = "main",
    overwrite: bool = False,
) -> Path:
    """Programmatic shim for ``causalrag study save``."""
    return _registry(studies_dir).save(
        source_dir=source,
        name=name,
        description=description,
        branch=branch,
        overwrite=overwrite,
    )


def causalrag_study_load_cli(
    name: str,
    target: Path,
    *,
    studies_dir: Path | None = None,
    branch: str = "main",
    overwrite: bool = False,
) -> Path:
    """Programmatic shim for ``causalrag study load``."""
    return _registry(studies_dir).load(
        name=name, target_dir=target, branch=branch, overwrite=overwrite
    )


def causalrag_study_branch_cli(
    name: str,
    new_branch: str,
    *,
    studies_dir: Path | None = None,
    from_branch: str = "main",
    description: str = "",
) -> Path:
    """Programmatic shim for ``causalrag study branch``."""
    return _registry(studies_dir).branch(
        name=name,
        new_branch=new_branch,
        from_branch=from_branch,
        description=description,
    )


def causalrag_study_list_cli(
    *,
    studies_dir: Path | None = None,
    study: str | None = None,
) -> list[str] | list[StudyBranch]:
    """List studies, or branches of one study when *study* is given."""
    reg = _registry(studies_dir)
    if study is None:
        return reg.list_studies()
    return reg.list_branches(study)
