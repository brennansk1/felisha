"""Tests for the static island detector audit (Sprint 9.5.3)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import dedent

import pytest

from causalrag.audits.island_detector import (
    IslandReport,
    detect_islands,
)


# ─────────────────────────────────────────────────────────────────────
# Synthetic-package builder
# ─────────────────────────────────────────────────────────────────────


def _build_synthetic_package(root: Path) -> Path:
    """Lay out a tiny package + tests tree under ``root`` and return the
    package directory.

    The shape mirrors the real repo:

        <root>/
          src/
            mypkg/
              __init__.py
              wired.py        # imported + has a referenced function
              orphan_mod.py   # not imported anywhere
              mixed.py        # defines a referenced fn, a test-only
                              # fn, and a truly-orphaned fn
          tests/
            test_mixed.py
    """
    src = root / "src"
    pkg = src / "mypkg"
    pkg.mkdir(parents=True)
    tests = root / "tests"
    tests.mkdir()

    (pkg / "__init__.py").write_text(
        dedent(
            """
            from mypkg.wired import wired_fn
            from mypkg.mixed import referenced_fn
            __all__ = ["wired_fn", "referenced_fn"]
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (pkg / "wired.py").write_text(
        dedent(
            """
            def wired_fn():
                return referenced_fn() + 1


            from mypkg.mixed import referenced_fn
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    # orphan_mod.py defines a function that *nothing* imports — the
    # module itself is the orphan; its sole callable is also orphaned.
    (pkg / "orphan_mod.py").write_text(
        dedent(
            """
            def orphan_function():
                return 42
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (pkg / "mixed.py").write_text(
        dedent(
            """
            def referenced_fn():
                return 1


            def test_only_fn():
                return 2


            def truly_orphan_fn():
                return 3
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (tests / "__init__.py").write_text("", encoding="utf-8")
    (tests / "test_mixed.py").write_text(
        dedent(
            """
            from mypkg.mixed import test_only_fn

            def test_it():
                assert test_only_fn() == 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    return pkg


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def test_synthetic_layout_flags_one_orphan_one_test_only_one_orphan_module(tmp_path: Path):
    """Core acceptance test: the synthetic layout should produce exactly
    one truly orphaned callable (``truly_orphan_fn``), one test-only
    callable (``test_only_fn``), and at least one orphaned module
    (``mypkg.orphan_mod``)."""
    pkg = _build_synthetic_package(tmp_path)
    report = detect_islands(pkg)

    assert isinstance(report, IslandReport)
    assert isinstance(report.timestamp, datetime)

    truly = set(report.truly_orphaned)
    test_only = set(report.test_only)
    orphan_mods = set(report.orphaned_modules)

    assert "mypkg.mixed.truly_orphan_fn" in truly
    # orphan_function lives in orphan_mod, which is itself unreachable,
    # so it's also truly orphaned.
    assert "mypkg.orphan_mod.orphan_function" in truly

    assert "mypkg.mixed.test_only_fn" in test_only

    # referenced_fn is imported by both wired.py and __init__.py — must
    # not appear in any orphan list.
    assert "mypkg.mixed.referenced_fn" not in truly
    assert "mypkg.mixed.referenced_fn" not in test_only

    # wired_fn is re-exported through __init__.py only; that counts as
    # production wiring (the detector treats __init__ source as
    # production references even though __init__.py files are excluded
    # from the definition inventory).
    assert "mypkg.wired.wired_fn" not in truly

    assert "mypkg.orphan_mod" in orphan_mods
    # The wired modules must NOT be flagged as orphaned modules.
    assert "mypkg.wired" not in orphan_mods
    assert "mypkg.mixed" not in orphan_mods


def test_severity_red_when_many_orphans(tmp_path: Path):
    """5+ truly orphaned callables OR 3+ orphaned modules ⇒ red."""
    pkg = tmp_path / "src" / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    # 6 orphaned callables across separate modules.
    for i in range(6):
        (pkg / f"orph{i}.py").write_text(
            f"def orph_fn_{i}():\n    return {i}\n", encoding="utf-8"
        )

    report = detect_islands(pkg)
    assert report.severity == "red"
    assert len(report.truly_orphaned) >= 5


def test_severity_yellow_when_one_orphan(tmp_path: Path):
    """1-4 truly orphaned ⇒ yellow."""
    pkg = tmp_path / "src" / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from mypkg.wired import wired_fn\n", encoding="utf-8"
    )
    (pkg / "wired.py").write_text(
        "def wired_fn():\n    return 1\n", encoding="utf-8"
    )
    # Single orphan module + single orphan callable.
    (pkg / "orph.py").write_text(
        "def lonely():\n    return 1\n", encoding="utf-8"
    )

    report = detect_islands(pkg)
    assert report.severity == "yellow"
    assert "mypkg.orph.lonely" in report.truly_orphaned


def test_severity_green_when_nothing_orphaned(tmp_path: Path):
    """No truly-orphaned callables AND no orphaned modules ⇒ green."""
    pkg = tmp_path / "src" / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from mypkg.wired import wired_fn\n", encoding="utf-8"
    )
    (pkg / "wired.py").write_text(
        "def wired_fn():\n    return 1\n", encoding="utf-8"
    )

    report = detect_islands(pkg)
    assert report.severity == "green"
    assert report.truly_orphaned == []
    assert report.orphaned_modules == []


def test_framework_hooks_are_not_flagged(tmp_path: Path):
    """``__init__`` / ``main`` / ``_main`` are invoked implicitly and
    must not appear in the orphan list."""
    pkg = tmp_path / "src" / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "from mypkg.entry import main\n", encoding="utf-8"
    )
    (pkg / "entry.py").write_text(
        dedent(
            """
            class Thing:
                def __init__(self):
                    self.x = 1

            def main():
                return Thing()

            def _main():
                return 0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    report = detect_islands(pkg)
    qualnames = set(report.truly_orphaned) | set(report.test_only)
    # Hook names never appear as orphans.
    for hook in ("__init__", "main", "_main"):
        assert not any(q.endswith(f".{hook}") for q in qualnames), (
            f"{hook!r} should be exempt from orphan detection but appeared "
            f"in {qualnames}"
        )


def test_excluded_files_are_skipped(tmp_path: Path):
    """Files in ``excluded_files`` must not contribute definitions."""
    pkg = tmp_path / "src" / "mypkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "def should_be_ignored():\n    return 1\n", encoding="utf-8"
    )
    (pkg / "real.py").write_text(
        "def used():\n    return 1\n", encoding="utf-8"
    )
    # Real callable referenced from another file to keep it non-orphan.
    (pkg / "consumer.py").write_text(
        "from mypkg.real import used\n\n\ndef driver():\n    return used()\n",
        encoding="utf-8",
    )

    report = detect_islands(pkg)
    qualnames = set(report.truly_orphaned) | set(report.test_only)
    assert not any("should_be_ignored" in q for q in qualnames)


def test_report_counts_match_filesystem(tmp_path: Path):
    """``n_modules_scanned`` and ``n_callable_definitions`` reflect the
    actual scan."""
    pkg = _build_synthetic_package(tmp_path)
    report = detect_islands(pkg)

    # wired.py, orphan_mod.py, mixed.py — three modules (init excluded).
    assert report.n_modules_scanned == 3
    # wired_fn, orphan_function, referenced_fn, test_only_fn,
    # truly_orphan_fn → 5 callables.
    assert report.n_callable_definitions == 5
    assert "scanned 3 modules" in report.summary
    assert "5 definitions" in report.summary


def test_real_package_runs_without_crashing():
    """The detector must run end-to-end on the actual ``causalrag``
    package without raising. Severity is whatever the codebase happens
    to score — the test asserts only that we get a coherent report."""
    report = detect_islands()
    assert isinstance(report, IslandReport)
    assert report.severity in ("green", "yellow", "red")
    assert report.n_modules_scanned > 0
    assert report.n_callable_definitions > 0
    assert report.summary
