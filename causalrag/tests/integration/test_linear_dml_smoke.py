"""End-to-end smoke test for LinearDMLEstimator (PDD §33.99).

Synthesizes a small dataset with a known ATE, fits the registered LinearDML
wrapper, and verifies that the recovered point estimate is in the right ball
park and that the EstimationResult round-trips through Pydantic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.core.protocol import StudyProtocol

pytest.importorskip("econml")
pytest.importorskip("sklearn")


pytestmark = pytest.mark.integration


def _synthesize(n: int, true_ate: float = 2.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.5 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(float)
    noise = rng.normal(scale=1.0, size=n)
    y = true_ate * t + 1.5 * x1 + 0.5 * x2 + noise
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def test_linear_dml_recovers_synthetic_ate() -> None:
    from causalrag.estimators.python.dml import LinearDMLEstimator

    df = _synthesize(n=1500, true_ate=2.0)
    est = LinearDMLEstimator(
        treatment="t",
        outcome="y",
        confounders=("x1", "x2"),
        modifiers=(),
        random_state=42,
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()

    assert result.estimator_id == "python.dml.linear"
    assert result.estimand_class == "ATE"
    assert result.n_used == len(df)
    assert result.point_estimate == pytest.approx(2.0, abs=0.25)
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low < 2.0 < result.ci_high
    assert result.backend_version is not None and result.backend_version.startswith("econml")
    assert result.fit_seconds is not None and result.fit_seconds > 0


def test_linear_dml_with_modifiers_produces_cate() -> None:
    from causalrag.estimators.python.dml import LinearDMLEstimator

    df = _synthesize(n=1500, true_ate=1.5)
    est = LinearDMLEstimator(
        treatment="t",
        outcome="y",
        confounders=("x1",),
        modifiers=("x2",),
        random_state=42,
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()

    assert result.estimand_class == "CATE"
    assert "cate_mean" in result.diagnostics
    assert result.diagnostics["has_modifiers"] is True


def test_linear_dml_below_min_sample_size_raises() -> None:
    from causalrag.estimators.python.dml import LinearDMLEstimator

    df = _synthesize(n=50)
    est = LinearDMLEstimator(treatment="t", outcome="y", confounders=("x1",))
    with pytest.raises(ValueError, match="at least 100 rows"):
        est.fit(df, StudyProtocol(name="smoke"))


def test_estimation_result_yaml_safe() -> None:
    """The EstimationResult must serialize cleanly so it can land inside a
    RoadmapWalk inside a StudyProtocol."""
    from causalrag.estimators.python.dml import LinearDMLEstimator

    df = _synthesize(n=500, true_ate=1.0)
    est = LinearDMLEstimator(treatment="t", outcome="y", confounders=("x1", "x2"))
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()

    dumped = result.model_dump(mode="json")
    assert dumped["estimator_id"] == "python.dml.linear"
    assert isinstance(dumped["point_estimate"], float)
