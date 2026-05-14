"""Unit tests for the sign-anticipation analyzer in ``scripts/run_parrot_test.py``.

These don't need Ollama and run by default so the parrot heuristic
itself stays regression-tested even when the slow end-to-end tests are
skipped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_driver():
    import sys

    here = Path(__file__).resolve()
    scripts_dir = here.parents[2] / "scripts"
    name = "run_parrot_test_mod"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, scripts_dir / "run_parrot_test.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so dataclasses can resolve forward references.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def driver():
    return _load_driver()


def test_neutral_rationale_does_not_match(driver) -> None:
    total, hits, _ = driver.count_sign_anticipating(
        [
            "The data will reveal whether the treatment matters.",
            "Identification is supported by the observed confounders.",
            "Sample size is sufficient for a meaningful test.",
        ]
    )
    assert total == 3
    assert hits == 0


def test_directional_rationale_matches(driver) -> None:
    total, hits, matched = driver.count_sign_anticipating(
        [
            "Training increases earnings substantially.",
            "We expect a positive ATE here.",
            "Statins should lower cardiac risk by 5pp.",
        ]
    )
    assert total == 3
    assert hits == 3
    assert len(matched) == 3


def test_mixed_rationales(driver) -> None:
    total, hits, _ = driver.count_sign_anticipating(
        [
            "The intervention raises the outcome.",  # match
            "Estimand is well-identified by the backdoor set.",  # no match
            "Outcome is likely to decrease under treatment.",  # match
            "",  # skipped (empty)
            "Confounders are adequately observed.",  # no match
        ]
    )
    assert total == 4  # empty string skipped
    assert hits == 2


def test_ratio_zero_on_empty(driver) -> None:
    total, hits, _ = driver.count_sign_anticipating([])
    assert total == 0
    assert hits == 0


def test_parrot_result_summary_handles_none(driver) -> None:
    """The ``ParrotResult.summary()`` formatter must not blow up when the
    pipeline failed to produce an estimate (e.g. unidentifiable)."""
    r = driver.ParrotResult(
        dataset_label="degenerate",
        n_rationales=5,
        n_sign_anticipating=2,
    )
    s = r.summary()
    assert "degenerate" in s
    assert "40.00%" in s or "40%" in s
