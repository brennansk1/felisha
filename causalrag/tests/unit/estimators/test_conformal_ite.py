"""Tests for the weighted-conformal ITE estimator (Sprint 2.5).

Validates that the Lei-Candès construction:
1. Recovers the population ATE on a known DGP within statistical tolerance.
2. Achieves the targeted (1 - alpha) coverage on the calibration fold (with
   a small slack to absorb finite-sample variability).
3. Returns well-shaped per-row intervals on a query grid.
4. Enforces the documented ``min_sample_size`` guard.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from causalrag.core.protocol import StudyProtocol  # noqa: E402
from causalrag.estimators.python.conformal_ite import (  # noqa: E402
    ConformalITEEstimator,
)


def _synthesize(n: int = 800, seed: int = 11) -> pd.DataFrame:
    """Known CATE = 2 + 0.5 * x1 with confounding via the propensity."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.4 * x1 - 0.2 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(float)
    cate = 2.0 + 0.5 * x1
    noise = rng.normal(scale=1.0, size=n)
    y = cate * t + 0.8 * x1 + 0.3 * x2 + noise
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def test_conformal_ite_recovers_ate_within_2_se() -> None:
    df = _synthesize(n=800)
    est = ConformalITEEstimator(
        treatment="t",
        outcome="y",
        confounders=("x2",),
        modifiers=("x1",),
        alpha=0.10,
        calibration_split=0.3,
        base_learner="dr",
        seed=12345,
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()

    # True population ATE is E[2 + 0.5 * x1] = 2.0 (x1 ~ N(0, 1)).
    assert result.estimator_id == "python.conformal.ite"
    assert result.estimand_class == "CATE"  # modifiers present
    assert result.point_estimate == pytest.approx(2.0, abs=0.5)
    # 2-SE band recovery: point estimate within ~ 2 * se of truth.
    assert result.se is not None and result.se > 0
    assert abs(result.point_estimate - 2.0) <= 2 * max(result.se, 0.25)


def test_empirical_coverage_meets_target() -> None:
    df = _synthesize(n=800)
    alpha = 0.10
    est = ConformalITEEstimator(
        treatment="t",
        outcome="y",
        confounders=("x2",),
        modifiers=("x1",),
        alpha=alpha,
        seed=12345,
    )
    est.fit(df, StudyProtocol(name="smoke"))
    diag = est.diagnose()

    assert diag["fitted"] is True
    assert diag["base_learner"] == "dr"
    cov = diag["empirical_coverage_on_calibration"]
    # Weighted-conformal guarantee: coverage ≥ 1 - alpha with finite-sample
    # slack. We allow a small tolerance for the heavy IPW tails.
    assert cov >= (1.0 - alpha) - 0.05
    assert diag["calibration_n"] > 0
    assert diag["q_alpha"] > 0
    assert diag["interval_width_mean"] > 0


def test_per_row_intervals_shape_and_ordering() -> None:
    df = _synthesize(n=600)
    est = ConformalITEEstimator(
        treatment="t",
        outcome="y",
        confounders=("x2",),
        modifiers=("x1",),
        alpha=0.10,
        seed=12345,
    )
    est.fit(df, StudyProtocol(name="smoke"))

    query = pd.DataFrame(
        {"x1": np.linspace(-2, 2, 25), "x2": np.zeros(25)}
    )
    out = est.per_row_intervals(query)

    assert list(out.columns) == ["point", "lower", "upper"]
    assert out.shape == (25, 3)
    # Strict ordering for every row.
    assert bool((out["lower"] < out["point"]).all())
    assert bool((out["point"] < out["upper"]).all())
    # Interval width is constant (symmetric conformal interval).
    widths = (out["upper"] - out["lower"]).to_numpy()
    assert np.allclose(widths, widths[0], atol=1e-9)


def test_min_sample_size_guard_raises() -> None:
    df = _synthesize(n=150)  # below the 200 floor
    est = ConformalITEEstimator(
        treatment="t",
        outcome="y",
        confounders=("x2",),
        modifiers=("x1",),
        seed=12345,
    )
    with pytest.raises(ValueError, match="at least 200 rows"):
        est.fit(df, StudyProtocol(name="smoke"))


def test_estimator_registered_in_catalog() -> None:
    from causalrag.core.registry import get_registry

    reg = get_registry()
    entry = reg.get("python.conformal.ite")
    assert entry.backend == "python"
    assert "ATE" in entry.supported_estimands
    assert "CATE" in entry.supported_estimands
    assert entry.min_sample_size == 200
    assert entry.produces_cate is True
