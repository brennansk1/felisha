"""Tests for the BART convergence diagnostics surfaced from
``BARTEstimator`` (PDD §13 / methodology audit).

PyMC sampling is slow, so the dataset and chain config are kept minimal — we
want to verify that R-hat / ESS / divergent-transition counts are wired into
``EstimationResult.diagnostics["bart"]``, not to benchmark the sampler.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Skip the whole module if the BART extras aren't available locally — this
# matches the same import guards used inside bart.py.
pytest.importorskip("pymc")
pytest.importorskip("pymc_bart")
pytest.importorskip("arviz")

from causalrag.core.protocol import StudyProtocol
from causalrag.estimators.python.bart import (
    BARTEstimator,
    _bart_convergence_diagnostics,
)


def _toy_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Small synthetic confounded-treatment dataset, large enough to clear
    ``BARTEstimator.min_sample_size`` (100)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Treatment depends on x1 -> confounding.
    p_t = 1.0 / (1.0 + np.exp(-0.6 * x1))
    t = rng.binomial(1, p_t).astype(float)
    y = 1.0 + 2.0 * t + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})


@pytest.fixture
def protocol() -> StudyProtocol:
    return StudyProtocol(name="bart-convergence-test")


def test_bart_estimator_emits_convergence_diagnostics(protocol: StudyProtocol) -> None:
    """Happy-path: a normal short BART run populates r_hat_max / ess_min /
    n_divergent in the EstimationResult diagnostics dict."""
    df = _toy_df(n=120, seed=0)
    est = BARTEstimator(
        treatment="t",
        outcome="y",
        confounders=("x1", "x2"),
        m=10,
        draws=50,
        tune=50,
        chains=2,
        random_state=7,
    )
    result = est.fit(df, protocol).estimate()

    assert "bart" in result.diagnostics, "diagnostics['bart'] block missing"
    bart_diag = result.diagnostics["bart"]
    for key in ("r_hat_max", "ess_min", "n_divergent"):
        assert key in bart_diag, f"diagnostics['bart'] missing {key!r}"

    # r_hat_max and ess_min should be finite floats on a real run; n_divergent
    # should be a non-negative int. We don't assert convergence quality — the
    # short draws config is intentionally weak.
    assert bart_diag["r_hat_max"] is None or np.isfinite(bart_diag["r_hat_max"])
    assert bart_diag["ess_min"] is None or np.isfinite(bart_diag["ess_min"])
    assert bart_diag["n_divergent"] is None or bart_diag["n_divergent"] >= 0


def test_pathological_bart_run_emits_warning(protocol: StudyProtocol) -> None:
    """A deliberately under-sampled BART (1 chain, 10 draws) should trip the
    ess_min < 100 threshold and produce a ``warning`` field."""
    df = _toy_df(n=120, seed=1)
    est = BARTEstimator(
        treatment="t",
        outcome="y",
        confounders=("x1", "x2"),
        m=10,
        draws=10,
        tune=20,
        chains=1,
        random_state=11,
    )
    result = est.fit(df, protocol).estimate()
    bart_diag = result.diagnostics["bart"]
    assert "warning" in bart_diag, (
        f"expected convergence warning for 1-chain/10-draw run; got {bart_diag!r}"
    )
    assert isinstance(bart_diag["warning"], str) and bart_diag["warning"]


def test_diagnostics_helper_handles_garbage_input() -> None:
    """``_bart_convergence_diagnostics`` must never raise — non-trace inputs
    just fall through to the all-None sentinel dict."""
    out = _bart_convergence_diagnostics(object())
    assert out["r_hat_max"] is None
    assert out["ess_min"] is None
    assert out["n_divergent"] is None
    # No warning should be emitted when every metric is None.
    assert "warning" not in out
