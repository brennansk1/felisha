"""Tests for the IV first-stage F diagnostic on GRFInstrumentalForest.

These exercise the pure-Python ``_partial_first_stage_F`` helper directly so
they run without R / rpy2 / grf installed. The full ``fit()`` path is left to
integration tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrag.estimators.rbridge.grf import (
    _IV_WEAK_WARNING,
    _iv_relevance_verdict,
    _partial_first_stage_F,
)


def _make_iv_data(
    n: int,
    instrument_strength: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic IV: T = pi * Z + gamma * X1 + U; X = (X1, X2)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n)
    X = rng.standard_normal((n, 2))
    U = rng.standard_normal(n)
    T = instrument_strength * Z + 0.5 * X[:, 0] + U
    return T, Z, X


def test_strong_instrument_yields_strong_verdict() -> None:
    T, Z, X = _make_iv_data(n=2000, instrument_strength=0.8, seed=1)
    F, q, n_used = _partial_first_stage_F(T, Z, X)
    assert q == 1
    assert n_used == 2000
    assert F > 23.1, f"expected F > 23.1, got {F}"
    assert _iv_relevance_verdict(F) == "strong"


def test_weak_instrument_yields_weak_verdict_and_warning_text() -> None:
    # tiny coefficient on Z relative to noise → F < 10
    T, Z, X = _make_iv_data(n=400, instrument_strength=0.05, seed=2)
    F, _q, _n = _partial_first_stage_F(T, Z, X)
    assert F < 10.0, f"expected F < 10, got {F}"
    assert _iv_relevance_verdict(F) == "weak"
    # The warning constant must mention the F<10 / Staiger-Stock threshold so
    # downstream surfaces can show it verbatim.
    assert "10" in _IV_WEAK_WARNING
    assert "weak" in _IV_WEAK_WARNING.lower()


def test_moderate_instrument_yields_adequate_verdict() -> None:
    # Hunt for a strength that lands F in [10, 23.1). Walk a small grid; this
    # is deterministic per (seed, n, strength).
    for strength in np.linspace(0.10, 0.30, 21):
        T, Z, X = _make_iv_data(n=600, instrument_strength=float(strength), seed=3)
        F, _q, _n = _partial_first_stage_F(T, Z, X)
        if 10.0 <= F < 23.1:
            assert _iv_relevance_verdict(F) == "adequate"
            return
    pytest.fail("could not synthesise an adequate-strength instrument in the grid")


def test_partial_F_matches_squared_t_for_single_instrument() -> None:
    """For one instrument, partial F == t_Z^2 in the unrestricted OLS."""
    T, Z, X = _make_iv_data(n=500, instrument_strength=0.4, seed=4)
    F, q, n = _partial_first_stage_F(T, Z, X)
    assert q == 1
    # Recompute OLS by hand and verify F == t^2 of the Z coefficient.
    R_u = np.hstack([Z.reshape(-1, 1), X, np.ones((n, 1))])
    beta, *_ = np.linalg.lstsq(R_u, T, rcond=None)
    resid = T - R_u @ beta
    rss = float(resid @ resid)
    k = R_u.shape[1]
    sigma2 = rss / (n - k)
    cov = sigma2 * np.linalg.inv(R_u.T @ R_u)
    se_z = float(np.sqrt(cov[0, 0]))
    t_z = beta[0] / se_z
    assert F == pytest.approx(t_z**2, rel=1e-6)


def test_no_controls_path_runs() -> None:
    T, Z, _X = _make_iv_data(n=300, instrument_strength=0.6, seed=5)
    F, q, n = _partial_first_stage_F(T, Z, None)
    assert q == 1 and n == 300
    assert F > 0


def test_multiple_instruments_q_reported() -> None:
    rng = np.random.default_rng(6)
    n = 800
    Z = rng.standard_normal((n, 3))
    X = rng.standard_normal((n, 2))
    U = rng.standard_normal(n)
    T = 0.5 * Z[:, 0] + 0.3 * Z[:, 1] + 0.1 * X[:, 0] + U
    F, q, _ = _partial_first_stage_F(T, Z, X)
    assert q == 3
    assert F > 10.0


# ---------------------------------------------------------------------------
# Optional: end-to-end fit through the R bridge. Skipped unless rpy2+grf live.
# ---------------------------------------------------------------------------


def test_fit_records_first_stage_F_when_grf_available() -> None:
    pytest.importorskip("rpy2")
    try:
        from causalrag.estimators.rbridge._r import require

        require("grf")
    except Exception:
        pytest.skip("R + grf package not available")

    import pandas as pd

    from causalrag.estimators.rbridge.grf import GRFInstrumentalForest

    T, Z, X = _make_iv_data(n=800, instrument_strength=0.8, seed=7)
    rng = np.random.default_rng(7)
    Y = 1.5 * T + 0.3 * X[:, 0] + rng.standard_normal(len(T))
    df = pd.DataFrame(
        {
            "y": Y,
            "t": T,
            "z": Z,
            "x1": X[:, 0],
            "x2": X[:, 1],
        }
    )
    est = GRFInstrumentalForest(
        treatment="t",
        outcome="y",
        instrument="z",
        confounders=("x1", "x2"),
        num_trees=200,
    )
    # min_sample_size default is 500; bump n if needed.
    est.min_sample_size = 100  # type: ignore[misc]
    est.fit(df, protocol=None)  # type: ignore[arg-type]
    diag = est.diagnose()
    assert diag["iv_first_stage_F"] is not None
    assert diag["iv_relevance_verdict"] in {"strong", "adequate", "weak"}
    assert diag["iv_instruments"] == ["z"]
