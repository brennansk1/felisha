from __future__ import annotations

import pytest

from causalrag.estimators.python.nuisance import (
    nuisance_models,
    resolve_library,
)


def test_resolve_library_falls_back_to_single_gbm_for_small_n() -> None:
    assert resolve_library("auto", n=100) == "single-gbm"
    assert resolve_library("auto", n=300) == "single-gbm"


def test_resolve_library_uses_hist_gbm_for_heavy_missing_small_n() -> None:
    assert resolve_library("auto", n=300, heavy_missing=True) == "hist-gbm"


def test_resolve_library_passes_through_explicit_choice() -> None:
    assert resolve_library("stacked-fast", n=100) == "stacked-fast"
    assert resolve_library("bart", n=10_000) == "bart"


def test_resolve_library_picks_stacked_at_large_n() -> None:
    out = resolve_library("auto", n=1000)
    assert out in {"stacked-default", "stacked-rich"}


def test_nuisance_models_returns_regressor_and_classifier() -> None:
    reg, clf = nuisance_models(random_state=0, library="single-gbm")
    assert hasattr(reg, "fit") and hasattr(reg, "predict")
    assert hasattr(clf, "fit") and hasattr(clf, "predict_proba")


@pytest.mark.parametrize("library", ["single-gbm", "hist-gbm", "stacked-default", "stacked-fast"])
def test_each_library_constructs(library: str) -> None:
    reg, clf = nuisance_models(random_state=0, library=library)  # type: ignore[arg-type]
    assert reg is not None and clf is not None
