"""Tests for the YAML FlagRegistry (Sprint 1.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causalrag.core.flag_registry import (
    FlagRegistry,
    FlagSpec,
    get_registry,
)
from causalrag.core.flags import DataFlag


def test_registry_bootstraps_from_descriptions() -> None:
    """When no YAML file is on disk, the registry should still build
    from the in-code `flag_descriptions` table so it's usable on
    first install."""
    r = FlagRegistry.load(path=None)
    # Every DataFlag in flag_descriptions should have a registry entry.
    enum_names = {f.value for f in DataFlag}
    registry_names = set(r.specs.keys())
    assert registry_names == enum_names, (
        f"missing: {enum_names - registry_names}; extra: {registry_names - enum_names}"
    )


def test_registry_consistency_check_clean() -> None:
    r = FlagRegistry.load(path=None)
    problems = r.check_consistency()
    assert problems == [], f"registry inconsistency: {problems}"


def test_registry_yaml_roundtrip(tmp_path: Path) -> None:
    r = FlagRegistry.load(path=None)
    yaml_path = tmp_path / "flags.yaml"
    r.save(yaml_path)
    assert yaml_path.exists()
    r2 = FlagRegistry.load(yaml_path)
    assert set(r2.specs.keys()) == set(r.specs.keys())
    for name in r.specs:
        a = r.specs[name]
        b = r2.specs[name]
        assert a.description == b.description
        assert a.implication == b.implication
        assert a.routes_to == b.routes_to


def test_registry_get_accepts_enum_or_string() -> None:
    r = FlagRegistry.load(path=None)
    by_enum = r.get(DataFlag.BINARY_TREATMENT)
    by_string = r.get("binary_treatment")
    assert by_enum is not None
    assert by_string is not None
    assert by_enum.name == by_string.name == "binary_treatment"


def test_registry_by_group() -> None:
    r = FlagRegistry.load(path=None)
    treatment_group = r.by_group("treatment")
    assert len(treatment_group) >= 4  # binary, categorical, continuous, mixture
    names = {s.name for s in treatment_group}
    assert "binary_treatment" in names
    assert "continuous_treatment" in names


def test_registry_closure_handles_implications() -> None:
    """If a flag declares 'implies', the closure should add it."""
    r = FlagRegistry(specs={
        "a": FlagSpec(name="a", group="treatment", description="d", implication="i", implies=["b"]),
        "b": FlagSpec(name="b", group="treatment", description="d", implication="i"),
        "c": FlagSpec(name="c", group="treatment", description="d", implication="i"),
    })
    closed = r.closure({"a"})
    assert closed == {"a", "b"}
    closed = r.closure({"c"})
    assert closed == {"c"}


def test_registry_closure_detects_cycle() -> None:
    r = FlagRegistry(specs={
        "a": FlagSpec(name="a", group="treatment", description="d", implication="i", implies=["b"]),
        "b": FlagSpec(name="b", group="treatment", description="d", implication="i", implies=["a"]),
    })
    # Closure on a cycle should terminate (set-union idempotent), not loop
    closed = r.closure({"a"})
    assert closed == {"a", "b"}


def test_registry_check_consistency_flags_dangling_implies() -> None:
    r = FlagRegistry(specs={
        "a": FlagSpec(name="a", group="treatment", description="d", implication="i", implies=["nonexistent"]),
    })
    problems = r.check_consistency()
    assert any("non-existent" in p.lower() or "no datafl" in p.lower() for p in problems)


def test_module_singleton_lazy() -> None:
    """get_registry() returns a cached instance on subsequent calls."""
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


def test_module_singleton_reload(tmp_path: Path) -> None:
    """reload=True forces a fresh registry construction."""
    r1 = get_registry()
    r2 = get_registry(reload=True)
    # New object, same content
    assert set(r1.specs.keys()) == set(r2.specs.keys())
