"""Import-smoke tests for estimators/learned_router.py.

The Wave 7 agent for Sprint 9.2 was rate-limited before writing its
full test suite; this stub ensures the module is importable and the
public API exists.
"""

from __future__ import annotations


def test_module_imports() -> None:
    import causalrag.estimators.learned_router  # noqa: F401


def test_router_class_exists() -> None:
    from causalrag.estimators import learned_router as m
    # At least one of the documented entry points should exist
    has_class = any(
        hasattr(m, name)
        for name in ("LearnedDispatchRouter", "RouterPrediction", "explain")
    )
    assert has_class
