"""Import-smoke tests for estimators/rbridge/tmle3.py."""

from __future__ import annotations


def test_module_imports() -> None:
    import causalrag.estimators.rbridge.tmle3  # noqa: F401


def test_tmle3_classes_exist() -> None:
    from causalrag.estimators.rbridge import tmle3 as m
    assert any(
        hasattr(m, name)
        for name in ("TMLE3Estimator", "TMLE3MediationEstimator")
    )
