"""Tests for meta-learner CI/SE fallback and confounder/modifier split.

These tests exercise the bootstrap fallback path that
:meth:`_MetaBase.estimate` and :meth:`_MetaBase.cate_predictions` use when
EconML's analytic ``effect_interval`` is unavailable, plus the W-vs-X split
that the audit flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("econml")
pytest.importorskip("sklearn")

from causalrag.core.protocol import StudyProtocol  # noqa: E402


def _synthesize(n: int, true_ate: float = 2.0, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    logits = 0.5 * x1 - 0.3 * x2
    p = 1.0 / (1.0 + np.exp(-logits))
    t = (rng.uniform(size=n) < p).astype(float)
    noise = rng.normal(scale=1.0, size=n)
    # Heterogeneous: effect varies with x2 (modifier).
    y = true_ate * t + 0.4 * t * x2 + 1.5 * x1 + 0.5 * x2 + noise
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


def test_xlearner_meta_returns_ate_se_and_ci_via_bootstrap() -> None:
    from causalrag.estimators.python.meta import XLearnerEstimator

    df = _synthesize(n=400, true_ate=2.0)
    est = XLearnerEstimator(
        treatment="t",
        outcome="y",
        confounders=("x1", "x2"),
        modifiers=(),
        random_state=42,
        bootstrap_iterations=25,  # keep test fast
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()

    assert result.estimator_id == "python.meta.x_learner"
    assert result.estimand_class == "ATE"
    assert result.point_estimate == pytest.approx(2.0, abs=0.6)
    # X-learner has no analytic interval, so we must have fallen back to bootstrap.
    assert result.diagnostics["bootstrap_used"] is True
    assert result.ci_low is not None and result.ci_high is not None
    assert result.ci_low < result.ci_high
    assert result.se is not None and result.se > 0
    # diagnostics surface the W/X split
    assert result.diagnostics["cate_available"] is False
    assert result.diagnostics["n_confounders"] == 2
    assert result.diagnostics["n_modifiers"] == 0


def test_xlearner_with_confounders_and_modifiers_provides_ate_and_cate_ci() -> None:
    from causalrag.estimators.python.meta import XLearnerEstimator

    df = _synthesize(n=400, true_ate=1.5)
    est = XLearnerEstimator(
        treatment="t",
        outcome="y",
        confounders=("x1",),
        modifiers=("x2",),
        random_state=42,
        bootstrap_iterations=25,
    )
    est.fit(df, StudyProtocol(name="smoke"))
    result = est.estimate()

    assert result.estimand_class == "CATE"
    assert result.diagnostics["cate_available"] is True
    assert result.diagnostics["n_confounders"] == 1
    assert result.diagnostics["n_modifiers"] == 1
    # ATE-level CI present from bootstrap fallback.
    assert result.ci_low is not None and result.ci_high is not None
    assert result.se is not None and result.se > 0

    # And a per-row CATE at a sample modifier value comes back with a CI.
    X_grid = pd.DataFrame({"x2": [-1.0, 0.0, 1.0]})
    preds = est.cate_predictions(X_grid)
    assert preds["point"].shape == (3,)
    assert preds["ci_low"].shape == (3,)
    assert preds["ci_high"].shape == (3,)
    assert np.all(preds["ci_low"] <= preds["ci_high"])
    assert not np.any(np.isnan(preds["ci_low"]))
    assert not np.any(np.isnan(preds["ci_high"]))


def test_drlearner_uses_cv_at_least_five() -> None:
    """Audit guard: DR-learner must use K>=5 cross-fits per DML literature."""
    from causalrag.estimators.python.meta import DRLearnerEstimator

    est = DRLearnerEstimator(
        treatment="t",
        outcome="y",
        confounders=("x1",),
        modifiers=(),
        random_state=0,
    )
    learner = est._build_learner()
    assert getattr(learner, "cv", None) == 5
