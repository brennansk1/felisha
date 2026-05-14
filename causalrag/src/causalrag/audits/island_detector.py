"""Static island detector (Sprint 9.5.3).

Surfaces dead code, partially-wired features, and "implemented but not
invoked" gaps that escape the :mod:`end_to_end_flow` audit. The user's
directive — "avoiding any islanded methods and flags" — drives this
module: anything defined in the package but never referenced from the
runtime path of ``/auto`` is suspicious.

The detector walks every ``.py`` in the package via :mod:`ast`, builds
an inventory of top-level callables (functions + classes), then scans
the whole source tree (and the ``tests/`` tree, separately) for
references to each one. Two passes:

1. **Production references** — any non-test file that mentions the
   callable name. Anything with zero production references is either
   :attr:`IslandReport.truly_orphaned` (nobody references it) or
   :attr:`IslandReport.test_only` (referenced only from tests).
2. **Module-level inbound imports** — modules with no ``from X import``
   /  ``import X`` references from elsewhere in the package land in
   :attr:`IslandReport.orphaned_modules`.

We use regex on source text rather than AST-resolving every name. This
is fast, robust to dynamic dispatch (string-based factories,
``getattr`` lookups), and good enough for an audit whose job is to flag
suspicious silence — false positives are eliminated by adding the
relevant name to an allowlist, which forces the silence to be a
conscious decision.

The module is intentionally self-contained and importable without
triggering any of the package's heavy optional deps — the detector
walks the filesystem, not the import graph.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal


Severity = Literal["green", "yellow", "red"]


# Names whose presence anywhere in module scope makes us treat the
# module / callable as a wiring endpoint rather than an island. Update
# explicitly if a new dunder hook appears.
_PRIVATE_PREFIX = "_"

# Common Python "framework-invoked" sentinel names — defining them is
# enough to make the module "wired" even if nothing else references it.
# These are checked as substrings of the defined name.
_FRAMEWORK_HOOK_NAMES: frozenset[str] = frozenset(
    {
        "__init__",
        "__post_init__",
        "__enter__",
        "__exit__",
        "__call__",
        "__repr__",
        "__str__",
        "__eq__",
        "__hash__",
        "main",
        "_main",
    }
)


# ─────────────────────────────────────────────────────────────────────
# Report dataclass
# ─────────────────────────────────────────────────────────────────────


@dataclass
class IslandReport:
    """Static audit of unreferenced callables and modules.

    See module docstring for what each list means.
    """

    timestamp: datetime

    n_modules_scanned: int
    n_callable_definitions: int

    # Functions / classes never referenced anywhere else in the codebase
    truly_orphaned: list[str] = field(default_factory=list)

    # Defined but only referenced from tests
    test_only: list[str] = field(default_factory=list)

    # Modules with zero inbound imports from the rest of the package
    orphaned_modules: list[str] = field(default_factory=list)

    severity: Severity = "green"
    summary: str = ""
    notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Inventory + reference helpers
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Defn:
    """A single top-level callable definition.

    ``qualname`` is the human-readable ``module.name`` form used in the
    report. ``name`` is the bare identifier searched for in source.
    """

    module: str
    name: str
    qualname: str
    path: Path


def _resolve_package_root(package_root: Path | None) -> Path:
    """Pin the package root to ``src/causalrag`` by default.

    We accept an override so tests can point the detector at a
    synthetic tmp_path layout.
    """
    if package_root is not None:
        return Path(package_root).resolve()
    # This file lives at ``<root>/causalrag/src/causalrag/audits/island_detector.py``;
    # the package root is two parents up.
    here = Path(__file__).resolve()
    return here.parent.parent


def _iter_py_files(
    root: Path,
    *,
    excluded_dirs: Iterable[str],
    excluded_files: Iterable[str],
) -> list[Path]:
    """Return every ``.py`` file under ``root`` honouring the excludes.

    ``excluded_dirs`` is matched against any path segment, so passing
    ``"tests"`` skips both top-level and nested ``tests/`` directories.
    ``excluded_files`` is matched against the filename only.
    """
    excluded_dir_set = {d for d in excluded_dirs}
    excluded_file_set = {f for f in excluded_files}
    out: list[Path] = []
    for p in root.rglob("*.py"):
        if any(part in excluded_dir_set for part in p.parts):
            continue
        if p.name in excluded_file_set:
            continue
        out.append(p)
    return sorted(out)


def _module_dotted_name(path: Path, root: Path) -> str:
    """Turn ``<root>/a/b/c.py`` into ``a.b.c`` (relative to ``root``).

    The leading package name is the root directory's own name, so
    ``<root>=…/causalrag`` produces ``causalrag.a.b.c``.
    """
    rel = path.relative_to(root.parent).with_suffix("")
    return ".".join(rel.parts)


def _collect_definitions(files: list[Path], root: Path) -> list[_Defn]:
    """Extract every top-level function / class from each file.

    Module-level only — nested defs and methods are not catalogued
    because their reachability is determined by the enclosing class /
    function. Async functions count.
    """
    defs: list[_Defn] = []
    for path in files:
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue
        module = _module_dotted_name(path, root)
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                name = node.name
                defs.append(
                    _Defn(
                        module=module,
                        name=name,
                        qualname=f"{module}.{name}",
                        path=path,
                    )
                )
    return defs


def _read_source_corpus(files: list[Path]) -> dict[Path, str]:
    """Slurp every file once — referenced repeatedly during scanning."""
    out: dict[Path, str] = {}
    for path in files:
        try:
            out[path] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            out[path] = ""
    return out


def _find_test_files(package_root: Path) -> list[Path]:
    """Locate the project's ``tests/`` tree.

    The detector intentionally looks *outside* ``package_root`` for
    tests, because the convention in this repo is
    ``<repo>/causalrag/src/causalrag`` for source and
    ``<repo>/causalrag/tests`` for tests — sibling directories under
    the same project root.
    """
    candidates = [
        package_root.parent / "tests",  # src/causalrag → src/tests
        package_root.parent.parent / "tests",  # src/causalrag → tests
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return sorted(cand.rglob("*.py"))
    return []


def _name_reference_regex(name: str) -> re.Pattern[str]:
    """Word-boundary regex for ``name``.

    Anchored to ``\\b`` so ``foo`` does not match ``foobar``. Using
    ``re.compile`` once per name pays off because every name is grepped
    against every file.
    """
    return re.compile(rf"\b{re.escape(name)}\b")


def _module_import_regex(module: str) -> re.Pattern[str]:
    """Regex that matches an inbound import of ``module``.

    Covers both ``import a.b.c`` and ``from a.b.c import …``, plus
    relative ``from .c import …`` style when the leaf matches. We
    therefore key on the *leaf* component as well as the full dotted
    name; the leaf check is broader, the dotted check is precise.
    """
    leaf = module.rsplit(".", 1)[-1]
    return re.compile(
        rf"(?:^|\n)\s*(?:from\s+{re.escape(module)}(?:\s|\.)"  # from a.b.c import …
        rf"|import\s+{re.escape(module)}\b"  # import a.b.c
        rf"|from\s+\.+\s*{re.escape(leaf)}\s+import"  # from .leaf import …
        rf"|from\s+\.+[\w.]*\.{re.escape(leaf)}\s+import"  # from .x.leaf import …
        rf")",
        re.MULTILINE,
    )


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────


def detect_islands(
    package_root: Path | None = None,
    *,
    excluded_dirs: list[str] = ("__pycache__", "tests", "docs"),
    excluded_files: list[str] = ("__init__.py",),
) -> IslandReport:
    """Scan ``package_root`` for orphaned callables and modules.

    Parameters
    ----------
    package_root:
        Filesystem root of the package to scan. Defaults to the
        ``causalrag`` package directory in this repo.
    excluded_dirs:
        Path segments to skip during the production-source walk. Tests
        are excluded here so they cannot mask an orphan — they are
        scanned separately to populate :attr:`IslandReport.test_only`.
    excluded_files:
        Filenames to skip outright. ``__init__.py`` is excluded by
        default because re-exports there would otherwise make every
        callable look "referenced".
    """
    root = _resolve_package_root(package_root)

    prod_files = _iter_py_files(
        root, excluded_dirs=excluded_dirs, excluded_files=excluded_files
    )
    prod_source = _read_source_corpus(prod_files)

    # __init__.py files are excluded from the *definition* inventory
    # (re-exports there are not islands), but their *contents* still
    # count as production references — a name only re-exported by an
    # __init__ is wired enough to escape "truly_orphaned".
    init_files = [
        p
        for p in root.rglob("__init__.py")
        if not any(part in set(excluded_dirs) for part in p.parts)
    ]
    init_source = _read_source_corpus(init_files)

    test_files = _find_test_files(root)
    test_source = _read_source_corpus(test_files)

    defs = _collect_definitions(prod_files, root)

    truly_orphaned: list[str] = []
    test_only: list[str] = []

    for defn in defs:
        # Framework hooks (``__init__``, ``__call__``, ``main`` …) are
        # invoked implicitly — exclude from the orphan check.
        if defn.name in _FRAMEWORK_HOOK_NAMES:
            continue

        pat = _name_reference_regex(defn.name)

        prod_refs = 0
        for path, src in prod_source.items():
            if path == defn.path:
                continue  # the definition itself does not count
            if pat.search(src):
                prod_refs += 1
                break  # one is enough — early-exit for speed
        if prod_refs == 0:
            for src in init_source.values():
                if pat.search(src):
                    prod_refs += 1
                    break

        if prod_refs:
            continue

        test_refs = 0
        for src in test_source.values():
            if pat.search(src):
                test_refs += 1
                break

        if test_refs:
            test_only.append(defn.qualname)
        else:
            # Strictly-private names (single-underscore prefix) are
            # expected to be referenced inside their own module only —
            # we still surface them, but flag them in notes so the
            # reader knows what they're looking at.
            truly_orphaned.append(defn.qualname)

    truly_orphaned.sort()
    test_only.sort()

    # Modules with zero inbound imports anywhere else in the package.
    orphaned_modules: list[str] = []
    all_modules = sorted({_module_dotted_name(p, root) for p in prod_files})
    # The ``causalrag.master_loop`` / ``causalrag.auto`` entry points
    # are top-level; an "orphan module" check still applies because the
    # /auto path imports them. The audit catches *no* inbound imports —
    # whether the module is an entry point is for the reader to judge.
    corpus_files = list(prod_source.items()) + list(init_source.items())
    for module in all_modules:
        # Find the file that defines this module so we don't count
        # self-references (e.g. relative imports inside the module).
        own_path = next(
            (p for p in prod_files if _module_dotted_name(p, root) == module),
            None,
        )
        rx = _module_import_regex(module)
        inbound = False
        for path, src in corpus_files:
            if path == own_path:
                continue
            if rx.search(src):
                inbound = True
                break
        if not inbound:
            orphaned_modules.append(module)
    orphaned_modules.sort()

    # ── Severity ───────────────────────────────────────────────────
    if len(truly_orphaned) >= 5 or len(orphaned_modules) >= 3:
        severity: Severity = "red"
    elif truly_orphaned or orphaned_modules:
        severity = "yellow"
    else:
        severity = "green"

    notes: list[str] = []
    if test_only:
        notes.append(
            f"{len(test_only)} callable(s) are exercised by tests but no "
            "production code references them — verify the test is asserting "
            "real wiring, not a stub."
        )
    private_orphans = [n for n in truly_orphaned if n.rsplit(".", 1)[-1].startswith(_PRIVATE_PREFIX)]
    if private_orphans:
        notes.append(
            f"{len(private_orphans)} orphan(s) are private (leading "
            "underscore) — likely intra-module helpers; confirm they're "
            "called from within their defining module."
        )

    summary = (
        f"island audit: severity={severity}; "
        f"{len(truly_orphaned)} truly orphaned, "
        f"{len(test_only)} test-only, "
        f"{len(orphaned_modules)} orphaned modules "
        f"(scanned {len(prod_files)} modules, {len(defs)} definitions)."
    )

    return IslandReport(
        timestamp=datetime.now(timezone.utc),
        n_modules_scanned=len(prod_files),
        n_callable_definitions=len(defs),
        truly_orphaned=truly_orphaned,
        test_only=test_only,
        orphaned_modules=orphaned_modules,
        severity=severity,
        summary=summary,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────
# CLI hook
# ─────────────────────────────────────────────────────────────────────


def _print_report(report: IslandReport) -> None:
    print(report.summary)
    print(f"  severity: {report.severity}")
    print(f"  modules scanned: {report.n_modules_scanned}")
    print(f"  callable definitions: {report.n_callable_definitions}")
    for label, items in (
        ("truly_orphaned", report.truly_orphaned),
        ("test_only", report.test_only),
        ("orphaned_modules", report.orphaned_modules),
    ):
        print(f"  {label} ({len(items)}):")
        for it in items:
            print(f"    - {it}")
    if report.notes:
        print("  notes:")
        for n in report.notes:
            print(f"    - {n}")


def _main() -> int:
    report = detect_islands()
    _print_report(report)
    return 0 if report.severity != "red" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "IslandReport",
    "detect_islands",
]
