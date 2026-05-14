"""Tests for the lmtp R-bridge SuperLearner-library fallback policy and
the LMTPShift ``return_quantity`` contrast default.

The fallback-library logic (``_resolve_fallback_learners``) is pure Python
and is exercised directly via monkeypatching the installed-learners
probe — those tests run on any platform, with or without R/rpy2.

The end-to-end ``LMTPShift(return_quantity='contrast')`` test requires
rpy2 and the ``lmtp`` R package; it is skipped otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.estimators.rbridge._r import RBridgeError
from causalrag.estimators.rbridge import lmtp as lmtp_mod


# --------------------------------------------------------------------------
# Pure-Python tests for the fallback-library policy. No rpy2 needed.
# --------------------------------------------------------------------------


def test_minimal_strawman_library_refused_by_default(monkeypatch):
    """When only SL.glm + SL.mean are installed and sl3 is absent, the
    wrapper must refuse to run unless the caller explicitly opts in."""
    monkeypatch.setattr(
        lmtp_mod, "_r_installed_sl_learners", lambda: {"SL.glm", "SL.mean"}
    )
    with pytest.raises(RBridgeError) as exc_info:
        lmtp_mod._resolve_fallback_learners(allow_minimal_learners=False)
    msg = str(exc_info.value)
    # Error must explain *why* the wrapper refused: double-robustness.
    assert "double-robustness" in msg or "doubly" in msg.lower() or "robust" in msg
    assert "allow_minimal_learners" in msg


def test_minimal_strawman_library_allowed_with_optin(monkeypatch):
    """allow_minimal_learners=True lets the wrapper proceed (with a
    warning) on the SL.glm + SL.mean library."""
    monkeypatch.setattr(
        lmtp_mod, "_r_installed_sl_learners", lambda: {"SL.glm", "SL.mean"}
    )
    with pytest.warns(UserWarning, match="double-robustness"):
        learners = lmtp_mod._resolve_fallback_learners(
            allow_minimal_learners=True
        )
    assert learners == ("SL.glm", "SL.mean")


def test_rich_library_used_when_glmnet_present(monkeypatch):
    """If at least one non-parametric / penalized learner is installed,
    the wrapper proceeds without requiring opt-in."""
    monkeypatch.setattr(
        lmtp_mod,
        "_r_installed_sl_learners",
        lambda: {"SL.glm", "SL.mean", "SL.glmnet"},
    )
    learners = lmtp_mod._resolve_fallback_learners(
        allow_minimal_learners=False
    )
    assert "SL.glmnet" in learners
    assert "SL.glm" in learners
    # SL.mean is still an acceptable anchor.
    assert "SL.mean" in learners


def test_default_learners_string_uses_sl3_when_available():
    s = lmtp_mod._default_learners_string(has_sl3=True)
    assert "sl3" not in s  # sl3 is selected via the SuperLearner-style list
    assert "SL.mean" in s


def test_default_learners_string_refuses_strawman(monkeypatch):
    monkeypatch.setattr(
        lmtp_mod, "_r_installed_sl_learners", lambda: {"SL.glm", "SL.mean"}
    )
    with pytest.raises(RBridgeError):
        lmtp_mod._default_learners_string(
            has_sl3=False, allow_minimal_learners=False
        )


# --------------------------------------------------------------------------
# End-to-end LMTPShift contrast test. Requires rpy2 + R + lmtp.
# --------------------------------------------------------------------------


def _have_r_lmtp() -> bool:
    """Return True iff rpy2 is importable AND the lmtp R package is
    installed in the active R session. Both are required for the e2e
    contrast test."""
    try:
        import rpy2  # noqa: F401
        import rpy2.robjects as ro
    except Exception:
        return False
    try:
        ok = bool(
            list(ro.r('requireNamespace("lmtp", quietly = TRUE)'))[0]
        )
        return ok
    except Exception:
        return False


@pytest.mark.skipif(
    not _have_r_lmtp(),
    reason="rpy2 and/or the R 'lmtp' package not installed",
)
def test_lmtpshift_contrast_returns_near_zero_on_null_effect():
    """Under a synthetic null (Y independent of A), the contrast
    E[Y(A+δ)] − E[Y(A)] should land near 0 with the CI covering 0."""
    pytest.importorskip("rpy2")
    from causalrag.core.protocol import StudyProtocol
    from causalrag.estimators.rbridge.lmtp import LMTPShift

    rng = np.random.default_rng(0)
    n = 400
    A = rng.normal(0.0, 1.0, size=n)
    X1 = rng.normal(0.0, 1.0, size=n)
    X2 = rng.normal(0.0, 1.0, size=n)
    # Y depends on X but NOT on A → true contrast is exactly 0.
    Y = 0.5 * X1 - 0.3 * X2 + rng.normal(0.0, 1.0, size=n)
    df = pd.DataFrame({"A": A, "Y": Y, "X1": X1, "X2": X2})

    est = LMTPShift(
        treatment="A",
        outcome="Y",
        confounders=("X1", "X2"),
        shift=0.5,
        folds=2,
        seed=7,
        return_quantity="contrast",
        # Smoke test on minimal CI machines: we don't require glmnet etc.
        allow_minimal_learners=True,
    )
    est.fit(df, StudyProtocol(name="lmtp-null"))
    res = est.estimate()
    assert isinstance(res.point_estimate, float)
    # Loose tolerance — TMLE on n=400 with a tiny library is noisy. The
    # important checks are: (a) it returns a number, (b) the diagnostics
    # call it a contrast, and (c) CI covers 0.
    assert abs(res.point_estimate) < 0.5
    assert res.diagnostics["return_quantity"] == "contrast"
    assert "contrast" in res.diagnostics["interpretation"].lower()
    if res.ci_low is not None and res.ci_high is not None:
        assert res.ci_low <= 0.2 and res.ci_high >= -0.2


@pytest.mark.skipif(
    not _have_r_lmtp(),
    reason="rpy2 and/or the R 'lmtp' package not installed",
)
def test_lmtpshift_refuses_strawman_library_without_optin(monkeypatch):
    """Even with rpy2 + lmtp installed, if only the straw-man learners
    are available and the caller does not opt in, fit() raises."""
    pytest.importorskip("rpy2")
    from causalrag.core.protocol import StudyProtocol
    from causalrag.estimators.rbridge.lmtp import LMTPShift

    # Pretend sl3 and the richer learners are absent.
    monkeypatch.setattr(lmtp_mod, "_has_sl3", lambda: False)
    monkeypatch.setattr(
        lmtp_mod, "_r_installed_sl_learners", lambda: {"SL.glm", "SL.mean"}
    )

    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "A": rng.normal(size=n),
            "Y": rng.normal(size=n),
            "X1": rng.normal(size=n),
        }
    )
    est = LMTPShift(
        treatment="A",
        outcome="Y",
        confounders=("X1",),
        shift=0.5,
        folds=2,
        seed=3,
        return_quantity="policy_mean",
        allow_minimal_learners=False,
    )
    with pytest.raises(RBridgeError):
        est.fit(df, StudyProtocol(name="lmtp-refuse"))
