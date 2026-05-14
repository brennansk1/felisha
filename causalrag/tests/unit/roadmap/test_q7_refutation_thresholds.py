"""SE-anchored refutation thresholds in q7 (audit fix).

The pre-fix code used absolute thresholds (e.g. ``|placebo| < 0.25*|orig| + 0.05``)
that mis-called e.g. ``placebo=0.04`` vs ``original=0.05`` as "passed" — same
order of magnitude, clearly residual confounding. The audit replaced these
with thresholds anchored to the original estimator's standard error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.protocol import StudyProtocol
from causalrag.core.result import EstimationResult
from causalrag.roadmap.q7_estimate import _run_refutations


# ---------------------------------------------------------------------------
# Scripted stub estimator: returns a different point estimate depending on
# what kind of refit it sees (placebo / RCC / subset), driven by a class-level
# script the test sets per-case.
# ---------------------------------------------------------------------------


class _StubEstimator:
    """Fake estimator wired to script per-refit estimates from test state."""

    SCRIPT: dict[str, float] = {}

    def __init__(self, treatment: str, outcome: str, confounders=(), modifiers=()) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = tuple(confounders)
        self.modifiers = tuple(modifiers)
        self._mode = "original"

    def fit(self, df: pd.DataFrame, protocol: StudyProtocol) -> "_StubEstimator":
        # Detect refit type from inputs: RCC adds a "_random_common_cause"
        # confounder; subset bootstrap shrinks the row count; placebo permutes
        # the treatment column but the test passes a frame whose treatment
        # column will be shuffled — we mark the fit_df shape as a proxy.
        if "_random_common_cause" in self.confounders:
            self._mode = "rcc"
        elif len(df) < self.SCRIPT.get("_n", len(df)):
            self._mode = "subset"
        else:
            # Could be original or placebo. We track call ordering: the
            # first fit() on a frame of full size with the original
            # treatment values is the "original"; any later same-sized fit
            # is the placebo.
            mode = "placebo" if self.SCRIPT.get("_original_consumed") else "original"
            if mode == "original":
                self.SCRIPT["_original_consumed"] = True
            self._mode = mode
        # The protocol flags must be preserved across refits (audit fix).
        self.SCRIPT.setdefault("_protocols_seen", []).append(protocol.name)
        return self

    def estimate(self) -> EstimationResult:
        value = self.SCRIPT[self._mode]
        return EstimationResult(
            estimator_id="stub",
            estimand_class="ATE",
            point_estimate=value,
            se=self.SCRIPT.get("se"),
            n_used=10,
        )


def _make_inputs(*, n: int = 50):
    df = pd.DataFrame(
        {
            "T": np.zeros(n, dtype=float),
            "Y": np.zeros(n, dtype=float),
            "X1": np.zeros(n, dtype=float),
        }
    )
    estimand = CausalEstimand(
        **{"class": EstimandClass.ATE},
        treatment="T",
        outcome="Y",
        formal_expression="E[Y(1) - Y(0)]",
    )
    protocol = StudyProtocol(name="study")
    return df, estimand, protocol


def _setup_script(*, original: float, se: float | None, placebo: float, rcc: float, subset: float, n: int = 50):
    _StubEstimator.SCRIPT.clear()
    _StubEstimator.SCRIPT.update(
        {
            "original": original,
            "placebo": placebo,
            "rcc": rcc,
            "subset": subset,
            "se": se,
            "_n": n,
            "_original_consumed": True,  # we manually fit "original" outside
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_placebo_fails_when_close_to_original_in_se_units():
    """Audit case: original=0.05, SE=0.01, placebo=0.04.

    Pre-fix: ``|0.04| < |0.05|*0.25 + 0.05 = 0.0625`` → "passed" (WRONG).
    Post-fix: ``|0.04| / 0.01 = 4.0 SE`` → fails.
    """
    df, estimand, protocol = _make_inputs(n=50)
    _setup_script(original=0.05, se=0.01, placebo=0.04, rcc=0.05, subset=0.05)
    original_result = EstimationResult(
        estimator_id="stub", estimand_class="ATE", point_estimate=0.05, se=0.01, n_used=50
    )
    stub = _StubEstimator(treatment="T", outcome="Y", confounders=("X1",))
    out = _run_refutations(stub, df, estimand, protocol, original_result)

    placebo = out["placebo_treatment"]
    assert placebo["passed"] is False
    assert placebo["original_se"] == 0.01
    assert placebo["refuted_estimate"] == 0.04
    # |0.04| / 0.01 = 4.0
    assert placebo["delta_in_se_units"] == 4.0


def test_placebo_passes_when_within_2_se_of_zero():
    """Case: original=0.05, SE=0.02, placebo=0.001. |0.001|/0.02 = 0.05 → passes."""
    df, estimand, protocol = _make_inputs(n=50)
    _setup_script(original=0.05, se=0.02, placebo=0.001, rcc=0.05, subset=0.05)
    original_result = EstimationResult(
        estimator_id="stub", estimand_class="ATE", point_estimate=0.05, se=0.02, n_used=50
    )
    stub = _StubEstimator(treatment="T", outcome="Y", confounders=("X1",))
    out = _run_refutations(stub, df, estimand, protocol, original_result)

    placebo = out["placebo_treatment"]
    assert placebo["passed"] is True
    assert placebo["delta_in_se_units"] < 2.0


def test_random_common_cause_passes_at_1_5_se_shift():
    """RCC shift of 1.5 SE should pass (within 2 SE band)."""
    df, estimand, protocol = _make_inputs(n=50)
    # original=0.10, SE=0.02 → 1.5 SE shift = 0.03 → rcc=0.13
    _setup_script(original=0.10, se=0.02, placebo=0.0, rcc=0.13, subset=0.10)
    original_result = EstimationResult(
        estimator_id="stub", estimand_class="ATE", point_estimate=0.10, se=0.02, n_used=50
    )
    stub = _StubEstimator(treatment="T", outcome="Y", confounders=("X1",))
    out = _run_refutations(stub, df, estimand, protocol, original_result)

    rcc = out["random_common_cause"]
    assert rcc["passed"] is True
    assert abs(rcc["delta_in_se_units"] - 1.5) < 1e-9


def test_random_common_cause_fails_at_2_5_se_shift():
    """RCC shift of 2.5 SE should fail (outside 2 SE band)."""
    df, estimand, protocol = _make_inputs(n=50)
    # original=0.10, SE=0.02 → 2.5 SE shift = 0.05 → rcc=0.15
    _setup_script(original=0.10, se=0.02, placebo=0.0, rcc=0.15, subset=0.10)
    original_result = EstimationResult(
        estimator_id="stub", estimand_class="ATE", point_estimate=0.10, se=0.02, n_used=50
    )
    stub = _StubEstimator(treatment="T", outcome="Y", confounders=("X1",))
    out = _run_refutations(stub, df, estimand, protocol, original_result)

    rcc = out["random_common_cause"]
    assert rcc["passed"] is False
    assert abs(rcc["delta_in_se_units"] - 2.5) < 1e-9


def test_no_se_yields_passed_none_with_reason():
    df, estimand, protocol = _make_inputs(n=50)
    _setup_script(original=0.05, se=None, placebo=0.04, rcc=0.05, subset=0.05)
    original_result = EstimationResult(
        estimator_id="stub", estimand_class="ATE", point_estimate=0.05, se=None, n_used=50
    )
    stub = _StubEstimator(treatment="T", outcome="Y", confounders=("X1",))
    out = _run_refutations(stub, df, estimand, protocol, original_result)

    for key in ("placebo_treatment", "random_common_cause", "subset_bootstrap"):
        assert out[key]["passed"] is None
        assert out[key]["reason"] == "no SE available"
    # n_passed counts only True; None should not count.
    assert out["n_passed"] == 0


def test_protocol_flags_preserved_in_refit():
    """Audit fix: refits must use the original protocol (with flags),
    not a stripped ``StudyProtocol(name="_refute")``."""
    from causalrag.core.flags import DataFlag

    df, estimand, _ = _make_inputs(n=50)
    protocol = StudyProtocol(name="my_study", flags={DataFlag.BINARY_TREATMENT})
    _setup_script(original=0.05, se=0.02, placebo=0.001, rcc=0.05, subset=0.05)
    original_result = EstimationResult(
        estimator_id="stub", estimand_class="ATE", point_estimate=0.05, se=0.02, n_used=50
    )
    stub = _StubEstimator(treatment="T", outcome="Y", confounders=("X1",))
    _run_refutations(stub, df, estimand, protocol, original_result)

    # The refit protocols should derive from the caller's protocol, never
    # the old stripped name.
    seen = _StubEstimator.SCRIPT["_protocols_seen"]
    assert seen, "expected at least one refit"
    for name in seen:
        assert name != "_refute", "refit must not use stripped placeholder protocol"
        assert "my_study" in name
