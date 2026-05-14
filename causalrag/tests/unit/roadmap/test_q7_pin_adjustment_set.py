"""Pinned adjustment set — variable selection must not clobber a DAG-derived set.

Audit fix: when Step 5 identifies a backdoor-admissible adjustment set, Step 7
used to unconditionally run :func:`select_variables` over that set, which can
silently drop columns the DAG required for identification. The fix introduces a
``pin_adjustment_set`` kwarg (default ``True``) on :func:`estimate` that skips
data-driven selection on confirmatory backdoor adjustment sets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.roadmap.q5_identify import IdentificationResult
from causalrag.roadmap.q7_estimate import estimate


def _synthetic_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Confounders {C1, C2, C3} drive both T and Y, plus 5 irrelevant noise cols.

    Noise columns N1..N5 are mutually correlated so that
    ``correlation_pruning`` (the resolved selection method for 5 <= |W| <= 20)
    would drop several of them — exactly the silent-clobber path the audit
    flagged.
    """
    rng = np.random.default_rng(seed)
    C1 = rng.normal(size=n)
    C2 = rng.normal(size=n)
    C3 = rng.normal(size=n)
    # T and Y both depend on the true confounders.
    T = (0.4 * C1 + 0.3 * C2 - 0.2 * C3 + rng.normal(scale=0.5, size=n) > 0).astype(int)
    Y = 1.5 * T + 0.8 * C1 - 0.5 * C2 + 0.3 * C3 + rng.normal(scale=0.5, size=n)
    # Noise columns: highly intercorrelated so correlation_pruning would prune.
    base = rng.normal(size=n)
    N1 = base + 0.05 * rng.normal(size=n)
    N2 = base + 0.05 * rng.normal(size=n)
    N3 = base + 0.05 * rng.normal(size=n)
    N4 = rng.normal(size=n)
    N5 = rng.normal(size=n)
    return pd.DataFrame(
        {
            "T": T.astype(float),
            "Y": Y,
            "C1": C1,
            "C2": C2,
            "C3": C3,
            "N1": N1,
            "N2": N2,
            "N3": N3,
            "N4": N4,
            "N5": N5,
        }
    )


def _estimand() -> CausalEstimand:
    return CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": "T",
            "outcome": "Y",
            "formal_expression": "E[Y(1)-Y(0)]",
        }
    )


def _protocol() -> StudyProtocol:
    return StudyProtocol(name="pin_test")


def test_pin_adjustment_set_true_preserves_backdoor_set_verbatim() -> None:
    """With pin=True and a backdoor ID result, the estimator receives the
    DAG-derived adjustment set verbatim — no variable_selection step runs."""
    df = _synthetic_frame()
    ident = IdentificationResult(
        identifiable=True,
        strategy="backdoor",
        adjustment_set=("C1", "C2", "C3"),
    )
    # NOTE: deliberately do NOT pass confounders=; we want the function to
    # read identification.adjustment_set and respect the pin.
    result = estimate(
        df=df,
        estimand=_estimand(),
        identification=ident,
        protocol=_protocol(),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        preprocess=False,
        pin_adjustment_set=True,
    )
    assert tuple(result.diagnostics["adjustment_set_used"]) == ("C1", "C2", "C3")
    assert result.diagnostics.get("variable_selection_skipped") is True
    assert "variable_selection_skipped_reason" in result.diagnostics
    # The selection step itself should NOT have run.
    assert "variable_selection" not in result.diagnostics


def test_pin_adjustment_set_false_runs_variable_selection() -> None:
    """With pin=False, the legacy behavior is preserved: variable selection
    runs (and may modify the adjustment set)."""
    df = _synthetic_frame()
    # Adjustment set deliberately contains the 3 true confounders + 5 noise
    # columns so correlation_pruning has something to drop.
    full_set = ("C1", "C2", "C3", "N1", "N2", "N3", "N4", "N5")
    ident = IdentificationResult(
        identifiable=True,
        strategy="backdoor",
        adjustment_set=full_set,
    )
    result = estimate(
        df=df,
        estimand=_estimand(),
        identification=ident,
        protocol=_protocol(),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        preprocess=False,
        selection="correlation_pruning",
        pin_adjustment_set=False,
    )
    # Variable selection actually ran.
    assert "variable_selection" in result.diagnostics
    assert result.diagnostics.get("variable_selection_skipped") is not True
    sel = result.diagnostics["variable_selection"]
    # correlation_pruning on N1/N2/N3 (r ≈ 1) should drop at least one.
    assert len(sel["dropped"]) >= 1


def test_pin_has_no_effect_on_iv_strategy() -> None:
    """IV strategy does not use a backdoor adjustment set; the pin flag must
    be a no-op (variable selection still runs over whatever confounders were
    passed)."""
    df = _synthetic_frame()
    # An IV result carries an instrument, not a backdoor adjustment set.
    ident = IdentificationResult(
        identifiable=True,
        strategy="iv",
        adjustment_set=(),
        instrument="N4",
    )
    # Even with pin=True, since strategy != "backdoor" the kwarg must not
    # short-circuit selection. Provide a non-trivial confounders set so we
    # can observe selection running.
    result = estimate(
        df=df,
        estimand=_estimand(),
        identification=ident,
        protocol=_protocol(),
        confounders=("C1", "C2", "C3", "N1", "N2", "N3", "N4", "N5"),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        preprocess=False,
        selection="correlation_pruning",
        allow_nonidentifiable=False,
        pin_adjustment_set=True,
    )
    # With non-backdoor strategy, the pin must NOT skip selection.
    assert result.diagnostics.get("variable_selection_skipped") is not True
    assert "variable_selection" in result.diagnostics


def test_pin_with_empty_adjustment_set_does_not_crash() -> None:
    """When the backdoor adjustment set is empty, the pin must not short-
    circuit (there is nothing to pin). Variable selection / fallback must
    run without crashing."""
    df = _synthetic_frame()
    ident = IdentificationResult(
        identifiable=True,
        strategy="backdoor",
        adjustment_set=(),
    )
    # No confounders → adj_used is empty → variable selection is skipped
    # by the standard ``if adj_used:`` gate, not by the pin. The call must
    # complete without raising.
    result = estimate(
        df=df,
        estimand=_estimand(),
        identification=ident,
        protocol=_protocol(),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        preprocess=False,
        pin_adjustment_set=True,
    )
    # Pin diagnostic must NOT be set — the pin only engages when there is
    # an adjustment set worth pinning.
    assert result.diagnostics.get("variable_selection_skipped") is not True
    assert list(result.diagnostics["adjustment_set_used"]) == []
