"""Unit tests for the Sprint 1.2 reproducibility manifest."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.graph import CausalGraph
from causalrag.provenance.manifest import (
    ManifestBuilder,
    RunManifest,
    _canonical_csv,
    _canonical_edges,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "T": [0, 1, 0, 1, 1],
            "Y": [1.0, 2.5, 0.8, 3.1, 2.9],
            "X": [10, 12, 11, 13, 14],
        }
    )


def _dag() -> CausalGraph:
    return CausalGraph.from_edge_list([("X", "T"), ("X", "Y"), ("T", "Y")])


def _estimand() -> CausalEstimand:
    return CausalEstimand(
        **{"class": EstimandClass.ATE},
        treatment="T",
        outcome="Y",
        formal_expression="E[Y(1) - Y(0)]",
    )


def _populated_builder(tmp_path: Path) -> ManifestBuilder:
    b = ManifestBuilder()
    b.hash_dataset(_df(), path=tmp_path / "data.csv")
    b.hash_dag(_dag())
    b.hash_estimand(_estimand(), hypothesis_id="H1")
    b.record_seed("numpy", 0)
    b.record_seed("master_loop", 42)
    b.record_package_versions()
    b.record_prompt("planner", "system\nuser\n")
    b.record_llm(model="qwen3", digest="sha256:abc123")
    b.record_cli_command("auto run data.csv --experiments 5")
    return b


# ── Hashing idempotency ───────────────────────────────────────────────────────


def test_dataset_hash_idempotent(tmp_path: Path) -> None:
    b1 = ManifestBuilder()
    b2 = ManifestBuilder()
    assert b1.hash_dataset(_df(), path=tmp_path / "d.csv") == b2.hash_dataset(
        _df(), path=tmp_path / "d.csv"
    )


def test_dag_hash_idempotent() -> None:
    b1 = ManifestBuilder()
    b2 = ManifestBuilder()
    assert b1.hash_dag(_dag()) == b2.hash_dag(_dag())


def test_estimand_hash_idempotent() -> None:
    b1 = ManifestBuilder()
    b2 = ManifestBuilder()
    assert b1.hash_estimand(_estimand()) == b2.hash_estimand(_estimand())


# ── Hashing sensitivity ───────────────────────────────────────────────────────


def test_mutating_one_row_changes_dataset_sha(tmp_path: Path) -> None:
    df_a = _df()
    df_b = _df()
    df_b.loc[0, "Y"] = 99.9
    b = ManifestBuilder()
    assert b.hash_dataset(df_a, path=tmp_path / "d.csv") != b.hash_dataset(
        df_b, path=tmp_path / "d.csv"
    )


def test_adding_one_edge_changes_dag_hash() -> None:
    g1 = _dag()
    g2 = CausalGraph.from_edge_list(
        [("X", "T"), ("X", "Y"), ("T", "Y"), ("X", "Z")]
    )
    b = ManifestBuilder()
    assert b.hash_dag(g1) != b.hash_dag(g2)


def test_different_estimand_changes_hash() -> None:
    e1 = _estimand()
    e2 = CausalEstimand(
        **{"class": EstimandClass.ATT},
        treatment="T",
        outcome="Y",
        formal_expression="E[Y(1) - Y(0) | T=1]",
    )
    b = ManifestBuilder()
    assert b.hash_estimand(e1) != b.hash_estimand(e2)


# ── Canonicalisation surface ──────────────────────────────────────────────────


def test_canonical_csv_uses_lf_line_terminator() -> None:
    text = _canonical_csv(_df())
    assert "\r\n" not in text
    assert text.count("\n") == _df().shape[0] + 1  # header + rows


def test_canonical_edges_sorted() -> None:
    g1 = CausalGraph.from_edge_list([("X", "Y"), ("A", "B"), ("M", "N")])
    g2 = CausalGraph.from_edge_list([("M", "N"), ("A", "B"), ("X", "Y")])
    # Same edge set in different declaration orders → identical canonical form.
    assert _canonical_edges(g1) == _canonical_edges(g2)


# ── Builder fields ────────────────────────────────────────────────────────────


def test_record_package_versions_non_empty() -> None:
    b = ManifestBuilder()
    b.record_package_versions()
    versions = b._fields["package_versions"]
    assert isinstance(versions, dict)
    assert len(versions) > 0
    # Pydantic is a hard dep of the project — must show up.
    assert any(k.startswith("pydantic") for k in versions)


def test_record_prompt_returns_sha() -> None:
    b = ManifestBuilder()
    sha = b.record_prompt("critic", "hello world")
    assert len(sha) == 64
    assert b._fields["prompt_hashes"]["critic"] == sha


def test_record_seed_coerces_to_int() -> None:
    b = ManifestBuilder()
    b.record_seed("numpy", 7)
    assert b._fields["seeds"] == {"numpy": 7}


# ── Build & validation ────────────────────────────────────────────────────────


def test_build_requires_dataset_and_dag() -> None:
    b = ManifestBuilder()
    with pytest.raises(ValueError, match="missing required field"):
        b.build()


def test_build_returns_run_manifest(tmp_path: Path) -> None:
    manifest = _populated_builder(tmp_path).build()
    assert isinstance(manifest, RunManifest)
    assert manifest.schema_version == "1"
    assert len(manifest.run_id) == 32
    assert manifest.dataset_n_rows == 5
    assert manifest.n_dag_edges == 3
    assert "H1" in manifest.estimand_hashes
    assert manifest.seeds["master_loop"] == 42
    assert manifest.llm_model == "qwen3"
    assert manifest.prompt_hashes["planner"]


# ── JSON round-trip ───────────────────────────────────────────────────────────


def test_manifest_json_round_trip(tmp_path: Path) -> None:
    m1 = _populated_builder(tmp_path).build()
    text = m1.model_dump_json()
    m2 = RunManifest.model_validate_json(text)
    assert m1 == m2


def test_save_writes_canonical_sorted_json(tmp_path: Path) -> None:
    b = _populated_builder(tmp_path)
    out = tmp_path / "manifest.json"
    b.save(out)
    text = out.read_text(encoding="utf-8")
    data = json.loads(text)
    # Sorted keys → first key should come alphabetically first.
    keys = list(data.keys())
    assert keys == sorted(keys)
    # Round-trippable to a RunManifest.
    m = RunManifest.model_validate(data)
    assert m.schema_version == "1"


# ── R-package snapshot (rpy2 optional) ────────────────────────────────────────


def test_record_r_packages_handles_missing_rpy2(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name.startswith("rpy2"):
            raise ImportError("rpy2 not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    b = ManifestBuilder()
    b.record_r_packages()
    assert b._fields["r_packages"] is None
