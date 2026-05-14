"""Tests for ``causalrag.cli.studies`` — Sprint 4.3 study registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from causalrag.cli.studies import (
    StudyBranch,
    StudyError,
    StudyRegistry,
    causalrag_study_branch_cli,
    causalrag_study_list_cli,
    causalrag_study_load_cli,
    causalrag_study_save_cli,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _scaffold_project(root: Path, *, name: str = "demo") -> Path:
    """Build a minimal project dir resembling a freshly-run ``causalrag /auto``."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "study.causalrag.yaml").write_text(
        f"name: {name}\nversion: '0.1'\ntier: academic\n", encoding="utf-8"
    )
    (root / "executive_synthesis.json").write_text(
        json.dumps({"tldr": "a finding", "inferred_domain": "synthetic"}, indent=2),
        encoding="utf-8",
    )
    runs = root / "runs" / "chain_0"
    runs.mkdir(parents=True)
    (runs / "estimates.json").write_text(json.dumps({"point": 0.42}), encoding="utf-8")
    # A pycache that should *not* be copied through.
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "junk.pyc").write_bytes(b"\x00")
    return root


@pytest.fixture()
def source_project(tmp_path: Path) -> Path:
    return _scaffold_project(tmp_path / "src_project")


@pytest.fixture()
def registry(tmp_path: Path) -> StudyRegistry:
    return StudyRegistry(studies_dir=tmp_path / "studies")


# ── StudyBranch dataclass round-trip ─────────────────────────────────────────


def test_studybranch_roundtrips_via_json() -> None:
    b = StudyBranch(
        name="main",
        parent=None,
        created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
        description="initial",
    )
    blob = json.dumps(b.to_dict())
    revived = StudyBranch.from_dict(json.loads(blob))
    assert revived.name == "main"
    assert revived.parent is None
    assert revived.description == "initial"
    assert revived.created_at == b.created_at


# ── Save ──────────────────────────────────────────────────────────────────────


def test_save_copies_project_and_writes_manifest(
    registry: StudyRegistry, source_project: Path
) -> None:
    branch_dir = registry.save(source_project, "exp1", description="baseline")
    assert branch_dir == registry.studies_dir / "exp1" / "main"
    assert (branch_dir / "study.causalrag.yaml").exists()
    assert (branch_dir / "executive_synthesis.json").exists()
    assert (branch_dir / "runs" / "chain_0" / "estimates.json").exists()
    # Cache dirs are skipped.
    assert not (branch_dir / "__pycache__").exists()
    # Registry meta is written.
    meta_path = registry.studies_dir / "exp1" / ".studies.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["name"] == "exp1"
    assert [b["name"] for b in meta["branches"]] == ["main"]
    # Manifest lock present.
    lock_path = branch_dir / "manifest.lock"
    assert lock_path.exists()
    lock = json.loads(lock_path.read_text())
    assert lock["study"] == "exp1"
    assert lock["branch"] == "main"


def test_save_refuses_branch_collision(
    registry: StudyRegistry, source_project: Path
) -> None:
    registry.save(source_project, "exp1")
    with pytest.raises(StudyError, match="already has branch"):
        registry.save(source_project, "exp1")


def test_save_overwrite_replaces_branch(
    registry: StudyRegistry, source_project: Path, tmp_path: Path
) -> None:
    registry.save(source_project, "exp1")
    # Mutate the source, save again with overwrite=True.
    (source_project / "study.causalrag.yaml").write_text(
        "name: demo\nversion: '0.2'\ntier: academic\n", encoding="utf-8"
    )
    registry.save(source_project, "exp1", overwrite=True)
    content = (registry.studies_dir / "exp1" / "main" / "study.causalrag.yaml").read_text()
    assert "0.2" in content


def test_save_rejects_missing_source(registry: StudyRegistry, tmp_path: Path) -> None:
    with pytest.raises(StudyError, match="does not exist"):
        registry.save(tmp_path / "nope", "ghost")


# ── List ──────────────────────────────────────────────────────────────────────


def test_list_studies_empty_when_no_registry(tmp_path: Path) -> None:
    reg = StudyRegistry(studies_dir=tmp_path / "studies")
    assert reg.list_studies() == []


def test_list_studies_returns_saved(
    registry: StudyRegistry, source_project: Path
) -> None:
    registry.save(source_project, "alpha")
    registry.save(source_project, "beta")
    assert registry.list_studies() == ["alpha", "beta"]


def test_list_branches_main_first(
    registry: StudyRegistry, source_project: Path
) -> None:
    registry.save(source_project, "exp1")
    registry.branch("exp1", "robust", description="robustness fork")
    names = [b.name for b in registry.list_branches("exp1")]
    assert names[0] == "main"
    assert "robust" in names


def test_list_branches_missing_study_raises(registry: StudyRegistry) -> None:
    with pytest.raises(StudyError, match="unknown study"):
        registry.list_branches("nope")


# ── Branch ────────────────────────────────────────────────────────────────────


def test_branch_forks_main(registry: StudyRegistry, source_project: Path) -> None:
    registry.save(source_project, "exp1")
    new_dir = registry.branch("exp1", "robust", description="robustness")
    assert new_dir == registry.studies_dir / "exp1" / "robust"
    assert (new_dir / "study.causalrag.yaml").exists()
    branches = {b.name: b for b in registry.list_branches("exp1")}
    assert branches["robust"].parent == "main"
    assert branches["robust"].description == "robustness"
    # Manifest lock records the fork relationship.
    lock = json.loads((new_dir / "manifest.lock").read_text())
    assert lock["parent_branch"] == "main"


def test_branch_collision_raises(
    registry: StudyRegistry, source_project: Path
) -> None:
    registry.save(source_project, "exp1")
    registry.branch("exp1", "robust")
    with pytest.raises(StudyError, match="already has branch 'robust'"):
        registry.branch("exp1", "robust")


def test_branch_missing_source_branch_raises(
    registry: StudyRegistry, source_project: Path
) -> None:
    registry.save(source_project, "exp1")
    with pytest.raises(StudyError, match="no branch 'ghost'"):
        registry.branch("exp1", "fork", from_branch="ghost")


def test_branch_unknown_study_raises(registry: StudyRegistry) -> None:
    with pytest.raises(StudyError, match="unknown study"):
        registry.branch("nope", "new")


# ── Load ──────────────────────────────────────────────────────────────────────


def test_load_round_trips_project(
    registry: StudyRegistry, source_project: Path, tmp_path: Path
) -> None:
    registry.save(source_project, "exp1")
    target = tmp_path / "restored"
    out = registry.load("exp1", target)
    assert out == target
    assert (target / "study.causalrag.yaml").exists()
    assert (target / "runs" / "chain_0" / "estimates.json").exists()
    assert (target / "manifest.lock").exists()


def test_load_specific_branch(
    registry: StudyRegistry, source_project: Path, tmp_path: Path
) -> None:
    registry.save(source_project, "exp1")
    registry.branch("exp1", "robust")
    # Modify robust branch contents directly so we can tell them apart.
    (registry.studies_dir / "exp1" / "robust" / "marker.txt").write_text("robust")
    target = tmp_path / "restored"
    registry.load("exp1", target, branch="robust")
    assert (target / "marker.txt").read_text() == "robust"


def test_load_missing_branch_raises(
    registry: StudyRegistry, source_project: Path, tmp_path: Path
) -> None:
    registry.save(source_project, "exp1")
    with pytest.raises(StudyError, match="no branch 'ghost'"):
        registry.load("exp1", tmp_path / "out", branch="ghost")


def test_load_missing_study_raises(registry: StudyRegistry, tmp_path: Path) -> None:
    with pytest.raises(StudyError, match="unknown study"):
        registry.load("nope", tmp_path / "out")


def test_load_refuses_nonempty_target(
    registry: StudyRegistry, source_project: Path, tmp_path: Path
) -> None:
    registry.save(source_project, "exp1")
    target = tmp_path / "restored"
    target.mkdir()
    (target / "preexisting.txt").write_text("hi")
    with pytest.raises(StudyError, match="not empty"):
        registry.load("exp1", target)


def test_load_overwrite_replaces_target(
    registry: StudyRegistry, source_project: Path, tmp_path: Path
) -> None:
    registry.save(source_project, "exp1")
    target = tmp_path / "restored"
    target.mkdir()
    (target / "preexisting.txt").write_text("hi")
    registry.load("exp1", target, overwrite=True)
    assert not (target / "preexisting.txt").exists()
    assert (target / "study.causalrag.yaml").exists()


# ── End-to-end save → list → branch → load via CLI shims ──────────────────────


def test_cli_shims_full_lifecycle(tmp_path: Path) -> None:
    studies_dir = tmp_path / "studies"
    src = _scaffold_project(tmp_path / "proj")

    saved = causalrag_study_save_cli(
        source=src,
        name="trial",
        studies_dir=studies_dir,
        description="initial",
    )
    assert saved.exists()

    listing = causalrag_study_list_cli(studies_dir=studies_dir)
    assert listing == ["trial"]

    forked = causalrag_study_branch_cli(
        name="trial",
        new_branch="sensitivity",
        studies_dir=studies_dir,
        description="re-run with stricter rule",
    )
    assert forked.exists()

    branches = causalrag_study_list_cli(studies_dir=studies_dir, study="trial")
    assert {b.name for b in branches} == {"main", "sensitivity"}  # type: ignore[union-attr]

    out_dir = tmp_path / "restored"
    loaded = causalrag_study_load_cli(
        name="trial",
        target=out_dir,
        studies_dir=studies_dir,
        branch="sensitivity",
    )
    assert loaded == out_dir
    assert (loaded / "study.causalrag.yaml").exists()
    assert (loaded / "manifest.lock").exists()


# ── Manifest passthrough ─────────────────────────────────────────────────────


def test_save_inherits_lightweight_lock_when_no_runmanifest(
    registry: StudyRegistry, source_project: Path
) -> None:
    branch_dir = registry.save(source_project, "exp1")
    lock = json.loads((branch_dir / "manifest.lock").read_text())
    assert lock["kind"] == "study_branch_lock"
    # The lock enumerates copied files, excluding itself.
    assert "study.causalrag.yaml" in lock["files"]
    assert "manifest.lock" not in lock["files"]


def test_save_inherits_runmanifest_when_present(
    registry: StudyRegistry, source_project: Path
) -> None:
    # Drop a synthetic RunManifest into the source.
    fake_manifest = {
        "schema_version": "1",
        "run_id": "deadbeef",
        "timestamp": "2026-05-01T00:00:00+00:00",
        "pipeline_version": "0.0.0+test",
        "git_sha": None,
        "dataset_path": "data/foo.csv",
        "dataset_sha256": "a" * 64,
        "dataset_n_rows": 10,
        "dataset_n_cols": 3,
        "dataset_columns": ["t", "y", "x"],
        "dag_hash": "b" * 64,
        "n_dag_nodes": 3,
        "n_dag_edges": 2,
        "estimand_hashes": {},
        "seeds": {},
        "python_version": "3.12.0",
        "platform": "test",
        "package_versions": {},
        "r_packages": None,
        "llm_model": None,
        "llm_digest": None,
        "prompt_hashes": {},
        "cli_command": None,
    }
    (source_project / "manifest.lock").write_text(
        json.dumps(fake_manifest), encoding="utf-8"
    )
    branch_dir = registry.save(source_project, "exp1")
    lock = json.loads((branch_dir / "manifest.lock").read_text())
    # Inherited fields are present, and the study annotation was attached.
    assert lock["run_id"] == "deadbeef"
    assert lock["dataset_sha256"] == "a" * 64
    assert lock["__study__"]["study"] == "exp1"
    assert lock["__study__"]["branch"] == "main"
