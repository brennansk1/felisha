"""Import-smoke tests for estimators/success_classifier.py."""

from __future__ import annotations


def test_module_imports() -> None:
    import causalrag.estimators.success_classifier  # noqa: F401


def test_classifier_class_exists() -> None:
    from causalrag.estimators import success_classifier as m
    assert any(
        hasattr(m, name)
        for name in (
            "EstimatorSuccessClassifier",
            "FleetSuccessClassifier",
            "SuccessPrediction",
        )
    )
