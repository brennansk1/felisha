"""Proximal causal inference via two-stage regression (Liu et al. AmJEpi 2024).

Background
----------
Classical adjustment can fail when there is an *unmeasured* confounder U of
treatment T and outcome Y. Proximal causal inference (PCI) repairs the
non-identification by leveraging two *proxies* of U:

  * a **negative-control exposure** (NCE) — call it ``W`` — that is associated
    with U but is not caused by T and does not directly cause Y given U,
  * a **negative-control outcome** (NCO) — call it ``Z`` — that is associated
    with U but is not directly affected by T given U.

The two proxies pin down enough of U's signal that the ATE becomes point-
identified. The original Tchetgen-Tchetgen ``proximal g-formula`` requires
solving a Fredholm integral equation, which is fragile in finite samples.

Liu, Mealli, Pacini and Tchetgen Tchetgen (American Journal of Epidemiology,
2024) showed that under a *linear* bridge ansatz one can recover the same
estimand with a tractable **two-stage regression** that mirrors 2SLS:

  Stage 1.  Regress the NCE proxy basis on (T, X, NCO) — i.e. project NCE
            onto the instrument-like NCO conditional on (T, X). This gives
            fitted values ``W_hat`` that capture only the part of NCE that
            is *predictable* from the instrument NCO and exogenous (T, X),
            stripping out direct dependence on Y.
  Stage 2.  Fit ``Y ~ T + X + W_hat`` (the bridge function ``h``). The
            counterfactual contrast ``E[h(1, X, W) - h(0, X, W)]`` is the
            ATE.

This file implements that estimator under the standard ``CausalEstimator``
protocol. Bias is further reduced via K-fold **cross-fitting**, and standard
errors come from a non-parametric **row bootstrap**.

Edge cases handled
------------------
* ``negative_control_exposure`` / ``negative_control_outcome`` *must* be
  supplied — if either is missing we raise a clear ``ValueError`` at
  construction or fit time.
* The classifier-style binary treatment is supported but the implementation
  is agnostic to whether T is binary or continuous; for continuous T we
  report ``E[h(1, X, W) - h(0, X, W)]`` for the canonical 0→1 contrast
  (which matches the binary-T ATE under linearity).
* ``min_sample_size = 300`` mirrors the §33 sprint plan — below that the
  two-stage regression is too noisy to be honest about.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import EstimatorEntry, register
from causalrag.core.result import EstimationResult


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _design(
    t: np.ndarray,
    x: np.ndarray | None,
    extra: np.ndarray | None,
) -> np.ndarray:
    """Stack (intercept, T, X, extra) into a single design matrix.

    ``x`` and ``extra`` may be ``None``; in that case they're skipped. The
    leading column is always a 1-vector so the downstream linear models can
    treat the result as a flat regressor matrix without re-adding an
    intercept.
    """
    n = t.shape[0]
    blocks: list[np.ndarray] = [np.ones((n, 1), dtype=np.float64), t.reshape(-1, 1)]
    if x is not None and x.size:
        blocks.append(x)
    if extra is not None and extra.size:
        blocks.append(extra)
    return np.concatenate(blocks, axis=1)


def _fit_stage1(
    t: np.ndarray,
    x: np.ndarray | None,
    z: np.ndarray,  # NCO (instrument)
    w: np.ndarray,  # NCE (endogenous proxy)
    ridge_alpha: float,
) -> Ridge:
    """Stage-1 model: regress NCE on (T, X, NCO).

    A small Ridge penalty keeps the projection stable when ``NCO`` is weakly
    informative or collinear with X. Returns the fitted scikit-learn model;
    callers use ``predict(_design(...))`` to obtain the projected ``W_hat``.
    """
    design = _design(t, x, z.reshape(-1, 1))
    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(design, w)
    return model


def _fit_stage2(
    t: np.ndarray,
    x: np.ndarray | None,
    w_hat: np.ndarray,
    y: np.ndarray,
    ridge_alpha: float,
) -> Ridge:
    """Stage-2 bridge model: Y on (T, X, W_hat).

    Under the Liu et al. linear bridge ansatz the coefficient block on T
    (combined with X-interactions, if any) recovers the ATE.
    """
    design = _design(t, x, w_hat.reshape(-1, 1))
    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(design, y)
    return model


def _two_stage_ate(
    y: np.ndarray,
    t: np.ndarray,
    x: np.ndarray | None,
    z: np.ndarray,
    w: np.ndarray,
    ridge_alpha: float,
) -> float:
    """Single-fit two-stage ATE.

    Used both by the cross-fit driver (per fold) and by the bootstrap. The
    contrast is the average of ``h(1, X_i, W_i) - h(0, X_i, W_i)`` over the
    sample passed in — i.e. the ATE on that sample.
    """
    stage1 = _fit_stage1(t, x, z, w, ridge_alpha)
    w_hat = stage1.predict(_design(t, x, z.reshape(-1, 1)))
    stage2 = _fit_stage2(t, x, w_hat, y, ridge_alpha)

    n = t.shape[0]
    t1 = np.ones(n, dtype=np.float64)
    t0 = np.zeros(n, dtype=np.float64)
    # Stage-2 takes W_hat (the projected NCE), so contrast holds W_hat fixed.
    d1 = _design(t1, x, w_hat.reshape(-1, 1))
    d0 = _design(t0, x, w_hat.reshape(-1, 1))
    return float(np.mean(stage2.predict(d1) - stage2.predict(d0)))


def _cross_fit_ate(
    y: np.ndarray,
    t: np.ndarray,
    x: np.ndarray | None,
    z: np.ndarray,
    w: np.ndarray,
    *,
    n_folds: int,
    seed: int,
    ridge_alpha: float,
) -> tuple[float, np.ndarray]:
    """K-fold cross-fit ATE.

    For each fold, stage-1 and stage-2 are fit on the K-1 training folds and
    used to predict the bridge contrast on the *held-out* fold. The
    fold-level contrasts are averaged; this is the Kennedy-style DR bias
    reduction (here applied to a linear bridge, which is the most useful
    regime in practice).

    Returns the cross-fit point estimate plus the per-row contrasts (useful
    for downstream IF-style SE or for residual diagnostics).
    """
    n = t.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    contrasts = np.empty(n, dtype=np.float64)
    for train_idx, test_idx in kf.split(np.arange(n)):
        x_tr = x[train_idx] if x is not None else None
        x_te = x[test_idx] if x is not None else None
        stage1 = _fit_stage1(t[train_idx], x_tr, z[train_idx], w[train_idx], ridge_alpha)
        # Project NCE on the held-out rows using the train-fold stage-1 model.
        w_hat_tr = stage1.predict(_design(t[train_idx], x_tr, z[train_idx].reshape(-1, 1)))
        w_hat_te = stage1.predict(_design(t[test_idx], x_te, z[test_idx].reshape(-1, 1)))
        stage2 = _fit_stage2(t[train_idx], x_tr, w_hat_tr, y[train_idx], ridge_alpha)
        n_te = test_idx.shape[0]
        t1 = np.ones(n_te, dtype=np.float64)
        t0 = np.zeros(n_te, dtype=np.float64)
        d1 = _design(t1, x_te, w_hat_te.reshape(-1, 1))
        d0 = _design(t0, x_te, w_hat_te.reshape(-1, 1))
        contrasts[test_idx] = stage2.predict(d1) - stage2.predict(d0)
    return float(np.mean(contrasts)), contrasts


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------

class ProximalRegressionEstimator:
    """Two-stage proximal-CI regression (Liu et al. AmJEpi 2024).

    See module docstring for the identification logic. The estimator
    requires two extra columns beyond the usual ``(T, Y, X)``: a negative-
    control exposure (NCE) and a negative-control outcome (NCO). It returns
    the ATE under the linear bridge ansatz, with cross-fit bias reduction
    and row-bootstrap SE / CI.
    """

    id: str = "python.proximal.regression"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE",)
    # PROXIMAL_PAIR_AVAILABLE isn't in the DataFlag enum today; leave empty
    # so flag-routing doesn't accidentally lock the estimator out. The
    # required (NCE, NCO) inputs are enforced at fit() time instead.
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 300
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...] = (),
        *,
        negative_control_exposure: str,
        negative_control_outcome: str,
        n_folds: int = 5,
        bootstrap_iterations: int = 200,
        seed: int = 42,
        ridge_alpha: float = 1e-4,
    ) -> None:
        if not negative_control_exposure or not negative_control_outcome:
            raise ValueError(
                "ProximalRegressionEstimator requires both a "
                "negative_control_exposure (NCE) and a "
                "negative_control_outcome (NCO). Got "
                f"NCE={negative_control_exposure!r}, NCO={negative_control_outcome!r}."
            )
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = tuple(confounders)
        self.negative_control_exposure = negative_control_exposure
        self.negative_control_outcome = negative_control_outcome
        self.n_folds = int(n_folds)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.seed = int(seed)
        self.ridge_alpha = float(ridge_alpha)

        self._y: np.ndarray | None = None
        self._t: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._z: np.ndarray | None = None  # NCO
        self._w: np.ndarray | None = None  # NCE
        self._n_used: int = 0
        self._point: float | None = None
        self._point_single_fit: float | None = None
        self._contrasts: np.ndarray | None = None
        self._fit_seconds: float | None = None
        self._backend_version: str | None = None
        self._cross_fit_used: bool = False

    # ------------------------------------------------------------------
    # fit / estimate
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> ProximalRegressionEstimator:
        needed = [
            self.outcome,
            self.treatment,
            self.negative_control_exposure,
            self.negative_control_outcome,
            *self.confounders,
        ]
        for c in needed:
            if c not in data.columns:
                raise ValueError(f"Column not in data: {c!r}")
        df = data[needed].dropna()
        self._n_used = int(len(df))
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"ProximalRegressionEstimator requires at least {self.min_sample_size} "
                f"rows after dropna; got {self._n_used}."
            )
        self._y = df[self.outcome].to_numpy(dtype=np.float64)
        self._t = df[self.treatment].to_numpy(dtype=np.float64)
        self._w = df[self.negative_control_exposure].to_numpy(dtype=np.float64)
        self._z = df[self.negative_control_outcome].to_numpy(dtype=np.float64)
        if self.confounders:
            self._x = df[list(self.confounders)].to_numpy(dtype=np.float64)
        else:
            self._x = None

        start = time.perf_counter()
        # Cross-fit point estimate when we have enough rows per fold; fall back
        # to a single fit otherwise.
        use_cv = self._n_used >= self.n_folds * 10 and self.n_folds >= 2
        if use_cv:
            point, contrasts = _cross_fit_ate(
                self._y,
                self._t,
                self._x,
                self._z,
                self._w,
                n_folds=self.n_folds,
                seed=self.seed,
                ridge_alpha=self.ridge_alpha,
            )
            self._point = point
            self._contrasts = contrasts
            self._cross_fit_used = True
        else:
            point = _two_stage_ate(
                self._y, self._t, self._x, self._z, self._w, self.ridge_alpha
            )
            self._point = point
            self._contrasts = None
            self._cross_fit_used = False

        # Always also compute the single-fit estimate for diagnostics.
        self._point_single_fit = _two_stage_ate(
            self._y, self._t, self._x, self._z, self._w, self.ridge_alpha
        )

        self._fit_seconds = time.perf_counter() - start

        import sklearn  # local import to keep top-of-module light
        self._backend_version = f"scikit-learn {sklearn.__version__}"
        return self

    def _bootstrap(
        self, alpha: float = 0.05
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Row-bootstrap CI / SE / p-value around the cross-fit point.

        For speed each replicate uses a *single* (not cross-fit) two-stage
        regression on the resampled rows. This is the standard practice for
        Kennedy-style estimators: the cross-fit is used for the point
        estimate's bias reduction, and the bootstrap reflects sampling
        variability around it.
        """
        assert self._y is not None and self._t is not None
        assert self._w is not None and self._z is not None
        B = max(1, int(self.bootstrap_iterations))
        rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        n = self._n_used
        reps: list[float] = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            try:
                est = _two_stage_ate(
                    self._y[idx],
                    self._t[idx],
                    self._x[idx] if self._x is not None else None,
                    self._z[idx],
                    self._w[idx],
                    self.ridge_alpha,
                )
                reps.append(est)
            except Exception:
                continue
        if len(reps) < 2:
            return None, None, None, None
        arr = np.asarray(reps, dtype=np.float64)
        lo = float(np.quantile(arr, alpha / 2))
        hi = float(np.quantile(arr, 1 - alpha / 2))
        se = float(np.std(arr, ddof=1))
        if se > 0:
            from math import erfc, sqrt
            z = abs(float(np.mean(arr))) / se
            p = float(erfc(z / sqrt(2.0)))
        else:
            p = None
        return lo, hi, se, p

    def estimate(self) -> EstimationResult:
        if self._point is None:
            raise RuntimeError("Call fit() before estimate().")
        lo, hi, se, p = self._bootstrap(alpha=0.05)

        diagnostics: dict[str, Any] = {
            "cross_fit_used": self._cross_fit_used,
            "n_folds": self.n_folds if self._cross_fit_used else 0,
            "bootstrap_iterations": self.bootstrap_iterations,
            "point_single_fit": self._point_single_fit,
            "ridge_alpha": self.ridge_alpha,
            "negative_control_exposure": self.negative_control_exposure,
            "negative_control_outcome": self.negative_control_outcome,
            "n_confounders": len(self.confounders),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATE",
            point_estimate=float(self._point),
            se=se,
            ci_low=lo,
            ci_high=hi,
            p_value=p,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=self._backend_version,
            fit_seconds=self._fit_seconds,
        )

    # ------------------------------------------------------------------
    # diagnostics / refutation
    # ------------------------------------------------------------------
    def diagnose(self) -> dict[str, Any]:
        """Light diagnostics — proxy strength + bridge fit quality.

        ``nce_partial_r2`` is a Pearson-style measure of how well NCO and the
        exogenous regressors predict NCE in stage 1; if it's near zero the
        proxy pair is too weak and the estimator should be treated as
        unreliable (analogous to a weak-instrument warning in IV).
        """
        out: dict[str, Any] = {
            "fitted": self._point is not None,
            "n_used": self._n_used,
            "cross_fit_used": self._cross_fit_used,
        }
        if self._y is None or self._t is None or self._w is None or self._z is None:
            return out

        stage1 = _fit_stage1(self._t, self._x, self._z, self._w, self.ridge_alpha)
        w_hat = stage1.predict(_design(self._t, self._x, self._z.reshape(-1, 1)))
        ss_res = float(np.sum((self._w - w_hat) ** 2))
        ss_tot = float(np.sum((self._w - self._w.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out["nce_stage1_r2"] = r2
        # Heuristic: <0.02 is "weak proxy" by analogy with the
        # Staiger-Stock F<10 weak-instrument threshold.
        out["weak_proxy"] = bool(r2 < 0.02)
        return out

    def refute(self) -> dict[str, Any]:
        """Refutation: re-fit with the NCE/NCO roles swapped.

        If the proxies really are negative controls (NCE not caused by T;
        NCO not directly causing Y given U), swapping them should not give
        a wildly different ATE. A large discrepancy is a red flag — either
        the swap is invalid (the proxies are genuinely asymmetric, which is
        expected) or the bridge ansatz is misspecified. We surface both
        numbers; downstream sensitivity logic decides what to do with them.
        """
        if (
            self._y is None
            or self._t is None
            or self._w is None
            or self._z is None
            or self._point is None
        ):
            return {}
        swapped = _two_stage_ate(
            self._y, self._t, self._x, self._w, self._z, self.ridge_alpha
        )
        return {
            "ate_original_proxies": float(self._point),
            "ate_swapped_proxies": float(swapped),
            "swap_delta": float(swapped - self._point),
        }


# ---------------------------------------------------------------------------
# Registry registration (import-time side effect)
# ---------------------------------------------------------------------------

def _register() -> None:
    try:
        register(
            EstimatorEntry(
                id=ProximalRegressionEstimator.id,
                factory=ProximalRegressionEstimator,
                backend=ProximalRegressionEstimator.backend,
                supported_estimands=frozenset(
                    ProximalRegressionEstimator.supported_estimands
                ),
                required_flags=ProximalRegressionEstimator.required_flags,
                excluded_flags=ProximalRegressionEstimator.excluded_flags,
                min_sample_size=ProximalRegressionEstimator.min_sample_size,
                produces_cate=ProximalRegressionEstimator.produces_cate,
                produces_full_counterfactual=(
                    ProximalRegressionEstimator.produces_full_counterfactual
                ),
                propensity_required=ProximalRegressionEstimator.propensity_required,
            )
        )
    except ValueError:
        # Already registered (re-import in tests). Ignore.
        pass


_register()
