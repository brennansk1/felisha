"""Pearl front-door estimator (PDD §29 / Sprint 6.6).

The front-door criterion (Pearl 1995, Causality §3.3.2) identifies the
causal effect of T on Y even in the presence of an unobserved confounder
U of (T, Y), provided we observe a mediator M that:

1. Is on every directed path from T to Y (T → M → Y mediates fully).
2. Is unconfounded with Y given T (no direct U → M edge).
3. Has no unblocked back-door path from T to M.

When those conditions hold, the do-calculus identification is::

    P(Y=y | do(T=t)) = ∑_m P(M=m | T=t) · ∑_{t'} P(Y=y | T=t', M=m) · P(T=t')

and, in expectation form (which is what we estimate for the ATE)::

    E[Y | do(T=t)] = E_M[ E_{T'}[ Y | T=T', M=M, X ] | T=t, X ] (avg over X)

with measured covariates ``X`` entering both component regressions and
the marginalisation. This file implements that expression empirically
via three sklearn component models — outcome regression, mediator model,
and a (marginal, X-only) treatment model used as the T' distribution —
followed by an honest row-bootstrap for SE and 95% CI.

v1.0 supports binary T (so the T'-marginal is just P(T=0), P(T=1)) with
continuous or binary M and continuous Y. Continuous-T support is parked
for v1.1 (would require a kernel- or model-based mediator conditional
density rather than a Gaussian residual draw, and Monte-Carlo M sampling
for the inner expectation).
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import EstimatorEntry, register
from causalrag.core.result import EstimationResult


class FrontDoorEstimator:
    """Pearl front-door estimator with g-formula + bootstrap CI.

    Three component models, all sklearn:

    1. ``P(M | T, X)`` — mediator regressor (mean only; the M-conditional
       distribution used in the inner expectation is the *empirical*
       distribution of M predictions at each ``(t, X_i)``, not a fitted
       Gaussian — this matches the standard plug-in estimator and avoids
       a parametric residual assumption).
    2. ``P(Y | T, M, X)`` — outcome regressor (used for the inner
       ``E_{T'}[ Y | T=T', M=m, X ]`` marginalisation).
    3. ``P(T | X)`` — propensity model giving the T'-marginal P(T=t' | X).
       For binary T this collapses to two values per row.

    ATE construction (binary T)::

        For each row i with covariate vector X_i:
          # predicted mediator under each treatment arm
          mhat_1 = E[M | T=1, X_i]
          mhat_0 = E[M | T=0, X_i]
          # marginalise the outcome model over T' ~ P(T | X_i)
          pi_1   = P(T=1 | X_i),  pi_0 = 1 - pi_1
          mu(t, m, x) := pi_0 * E[Y | T=0, M=m, X=x]
                       + pi_1 * E[Y | T=1, M=m, X=x]
          # outer expectation over M | T=t, X_i
          Y_do1_i = mu_T_marginal(mhat_1, X_i)
          Y_do0_i = mu_T_marginal(mhat_0, X_i)
        ATE = mean_i (Y_do1_i - Y_do0_i)

    Bootstrap row-resamples refit the three component models and recompute
    the ATE B times; the empirical 2.5 / 97.5 quantiles give the 95% CI.
    """

    id: str = "python.frontdoor"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE",)
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.BINARY_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.TIME_VARYING_TREATMENT,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 200
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        confounders: tuple[str, ...] = (),
        *,
        bootstrap_iterations: int = 200,
        seed: int = 42,
        outcome_learner: Literal["gbm", "linear"] = "gbm",
        mediator_learner: Literal["gbm", "linear"] = "gbm",
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.mediator = mediator
        self.confounders = tuple(confounders)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.seed = int(seed)
        self.outcome_learner = outcome_learner
        self.mediator_learner = mediator_learner

        # Filled in during fit():
        self._n_used: int = 0
        self._fit_seconds: float | None = None
        self._point_estimate: float | None = None
        self._models: tuple[Any, Any, Any] | None = None  # (m_model, y_model, t_model)
        self._arrays: dict[str, np.ndarray] | None = None
        self._diagnostics_extra: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_mediator_model(self, random_state: int) -> Any:
        if self.mediator_learner == "linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=random_state)

    def _make_outcome_model(self, random_state: int) -> Any:
        if self.outcome_learner == "linear":
            from sklearn.linear_model import LinearRegression

            return LinearRegression()
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(random_state=random_state)

    def _make_treatment_model(self, random_state: int) -> Any:
        from sklearn.linear_model import LogisticRegression

        # Liblinear is robust at small n and accepts no-X (intercept-only)
        # designs cleanly. Lbfgs would work too; either is fine.
        return LogisticRegression(max_iter=1000, random_state=random_state)

    def _stack_tx(self, t: np.ndarray, x: np.ndarray | None) -> np.ndarray:
        t_col = t.reshape(-1, 1).astype(np.float64)
        if x is None or x.size == 0:
            return t_col
        return np.concatenate([t_col, x], axis=1)

    def _stack_tmx(
        self, t: np.ndarray, m: np.ndarray, x: np.ndarray | None
    ) -> np.ndarray:
        t_col = t.reshape(-1, 1).astype(np.float64)
        m_col = m.reshape(-1, 1).astype(np.float64)
        if x is None or x.size == 0:
            return np.concatenate([t_col, m_col], axis=1)
        return np.concatenate([t_col, m_col, x], axis=1)

    def _x_or_intercept(self, x: np.ndarray | None, n: int) -> np.ndarray:
        """Return X for fitting P(T | X); when X is absent use an intercept-
        only column so the LogisticRegression model still has something to
        fit. The fitted coefficient on the constant column is meaningless
        (sklearn always fits an intercept too) but the predict_proba output
        is the unconditional empirical rate, which is what we want."""
        if x is None or x.size == 0:
            return np.zeros((n, 1))
        return x

    def _fit_models(
        self,
        y: np.ndarray,
        t: np.ndarray,
        m: np.ndarray,
        x: np.ndarray | None,
        random_state: int,
    ) -> tuple[Any, Any, Any]:
        m_model = self._make_mediator_model(random_state)
        y_model = self._make_outcome_model(random_state)
        t_model = self._make_treatment_model(random_state)

        m_model.fit(self._stack_tx(t, x), m)
        y_model.fit(self._stack_tmx(t, m, x), y)
        t_model.fit(self._x_or_intercept(x, len(t)), t.astype(int))
        return m_model, y_model, t_model

    def _ate_from_models(
        self,
        m_model: Any,
        y_model: Any,
        t_model: Any,
        x: np.ndarray | None,
        n: int,
    ) -> float:
        ones = np.ones(n, dtype=np.float64)
        zeros = np.zeros(n, dtype=np.float64)

        # Predicted mediator under each arm.
        mhat_1 = np.asarray(m_model.predict(self._stack_tx(ones, x)), dtype=np.float64)
        mhat_0 = np.asarray(m_model.predict(self._stack_tx(zeros, x)), dtype=np.float64)

        # P(T = 1 | X). sklearn classes_ ordering can vary; pull the col
        # whose class is 1 explicitly.
        proba = t_model.predict_proba(self._x_or_intercept(x, n))
        classes = list(getattr(t_model, "classes_", [0, 1]))
        col1 = classes.index(1) if 1 in classes else (proba.shape[1] - 1)
        pi_1 = proba[:, col1]
        pi_0 = 1.0 - pi_1

        # T'-marginal of the outcome model, evaluated at (T=t*, M=mhat_t, X).
        # For each candidate treatment t in {0, 1}:
        #   mu_marginal(mhat_t, X) = pi_0(X) * E[Y | T=0, M=mhat_t, X]
        #                          + pi_1(X) * E[Y | T=1, M=mhat_t, X]
        y_do1 = pi_0 * y_model.predict(self._stack_tmx(zeros, mhat_1, x)) + pi_1 * y_model.predict(
            self._stack_tmx(ones, mhat_1, x)
        )
        y_do0 = pi_0 * y_model.predict(self._stack_tmx(zeros, mhat_0, x)) + pi_1 * y_model.predict(
            self._stack_tmx(ones, mhat_0, x)
        )
        return float(np.mean(y_do1 - y_do0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> FrontDoorEstimator:
        needed = [self.outcome, self.treatment, self.mediator, *self.confounders]
        for c in needed:
            if c not in data.columns:
                raise ValueError(f"Column not in data: {c!r}")
        df = data[needed].dropna()
        n = len(df)
        if n < self.min_sample_size:
            raise ValueError(
                f"FrontDoorEstimator requires ≥ {self.min_sample_size} rows after "
                f"dropna; got {n}."
            )
        y = df[self.outcome].to_numpy(dtype=np.float64)
        t = df[self.treatment].to_numpy(dtype=np.float64)
        m = df[self.mediator].to_numpy(dtype=np.float64)
        unique_t = set(np.unique(t).tolist())
        if not unique_t.issubset({0.0, 1.0}):
            raise ValueError(
                f"FrontDoorEstimator v1.0 requires binary {{0, 1}} treatment; "
                f"got {unique_t}."
            )
        if len(unique_t) < 2:
            raise ValueError(
                "FrontDoorEstimator requires both treatment arms present in the data."
            )
        x = (
            df[list(self.confounders)].to_numpy(dtype=np.float64)
            if self.confounders
            else None
        )

        start = time.perf_counter()
        m_model, y_model, t_model = self._fit_models(y, t, m, x, self.seed)
        self._point_estimate = self._ate_from_models(m_model, y_model, t_model, x, n)
        self._fit_seconds = time.perf_counter() - start

        self._models = (m_model, y_model, t_model)
        self._arrays = {"y": y, "t": t, "m": m}
        if x is not None:
            self._arrays["x"] = x
        self._n_used = n
        import sklearn

        self._backend_version = f"sklearn {sklearn.__version__}"
        return self

    def estimate(self) -> EstimationResult:
        if self._point_estimate is None or self._arrays is None:
            raise RuntimeError("Call fit() before estimate().")
        point = float(self._point_estimate)
        lo, hi, se, p = self._bootstrap_ci(alpha=0.05)
        diagnostics: dict[str, Any] = {
            "n_confounders": len(self.confounders),
            "bootstrap_iterations": self.bootstrap_iterations,
            "outcome_learner": self.outcome_learner,
            "mediator_learner": self.mediator_learner,
        }
        diagnostics.update(self._diagnostics_extra)
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATE",
            point_estimate=point,
            se=se,
            ci_low=lo,
            ci_high=hi,
            p_value=p,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=getattr(self, "_backend_version", None),
            fit_seconds=self._fit_seconds,
        )

    def _bootstrap_ci(
        self, alpha: float = 0.05
    ) -> tuple[float | None, float | None, float | None, float | None]:
        assert self._arrays is not None
        y = self._arrays["y"]
        t = self._arrays["t"]
        m = self._arrays["m"]
        x = self._arrays.get("x")
        n = self._n_used
        B = max(1, int(self.bootstrap_iterations))
        rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        replicates: list[float] = []
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            t_b = t[idx]
            if len(np.unique(t_b)) < 2:
                continue
            y_b = y[idx]
            m_b = m[idx]
            x_b = x[idx] if x is not None else None
            try:
                m_mod, y_mod, t_mod = self._fit_models(
                    y_b, t_b, m_b, x_b, random_state=self.seed + b + 1
                )
                replicates.append(
                    self._ate_from_models(m_mod, y_mod, t_mod, x_b, len(t_b))
                )
            except Exception:  # pragma: no cover - defensive
                continue
        if len(replicates) < 2:
            return None, None, None, None
        arr = np.asarray(replicates, dtype=np.float64)
        lo = float(np.quantile(arr, alpha / 2))
        hi = float(np.quantile(arr, 1 - alpha / 2))
        se = float(np.std(arr, ddof=1))
        if se > 0 and self._point_estimate is not None:
            from math import erfc, sqrt

            z = abs(float(self._point_estimate)) / se
            p = float(erfc(z / sqrt(2.0)))
        else:
            p = None
        return lo, hi, se, p

    # ------------------------------------------------------------------
    # Diagnostics + refutations
    # ------------------------------------------------------------------
    def diagnose(self) -> dict[str, Any]:
        """Report front-door-relevant diagnostics.

        - ``fitted``: did fit() complete?
        - ``n_used``: rows after dropna.
        - ``mediator_correlation_T``: |corr(T, M)| — must be non-trivial,
          else the mediator carries no treatment signal.
        - ``mediator_correlation_Y_given_T``: residual partial correlation
          of M with Y after removing T — front-door requires M to predict
          Y beyond T.
        """
        out: dict[str, Any] = {
            "fitted": self._models is not None,
            "n_used": self._n_used,
        }
        if self._arrays is None:
            return out
        t = self._arrays["t"]
        m = self._arrays["m"]
        y = self._arrays["y"]
        if np.std(t) > 0 and np.std(m) > 0:
            out["mediator_correlation_T"] = float(
                abs(np.corrcoef(t, m)[0, 1])
            )
        # Residualise M and Y on T, then correlate residuals.
        try:
            t_centered = t - t.mean()
            denom = float((t_centered**2).sum())
            if denom > 0:
                bM = float((t_centered * (m - m.mean())).sum() / denom)
                bY = float((t_centered * (y - y.mean())).sum() / denom)
                m_resid = (m - m.mean()) - bM * t_centered
                y_resid = (y - y.mean()) - bY * t_centered
                if np.std(m_resid) > 0 and np.std(y_resid) > 0:
                    out["mediator_correlation_Y_given_T"] = float(
                        abs(np.corrcoef(m_resid, y_resid)[0, 1])
                    )
        except Exception:  # pragma: no cover - defensive
            pass
        return out

    def refute(self) -> dict[str, Any]:
        """Refutation: front-door fails when the mediator is itself
        confounded with Y by U. We can't observe U, but we can test the
        weaker observable implication that M predicts Y beyond T (front-
        door requires this). Flag a warning when that signal is near zero;
        the actual bias from a U → M edge is *not* directly observable from
        the data alone.

        Returns a dict with ``status`` ∈ {``ok``, ``weak_mediator``,
        ``no_mediator_signal``} and the supporting partial-correlation
        statistic. This mirrors the contract used by dowhy refuters: a
        diagnostic warning rather than an alternative estimate.
        """
        d = self.diagnose()
        partial = d.get("mediator_correlation_Y_given_T")
        if partial is None:
            return {"status": "no_mediator_signal", "partial_corr_M_Y_given_T": None}
        if partial < 0.05:
            return {
                "status": "weak_mediator",
                "partial_corr_M_Y_given_T": partial,
                "note": (
                    "Mediator carries little independent signal for Y after "
                    "conditioning on T. Front-door identification is fragile "
                    "in this regime; consider re-examining the proposed "
                    "T → M → Y path or whether M is itself U-confounded."
                ),
            }
        return {"status": "ok", "partial_corr_M_Y_given_T": partial}


def _register() -> None:
    register(
        EstimatorEntry(
            id=FrontDoorEstimator.id,
            factory=FrontDoorEstimator,
            backend=FrontDoorEstimator.backend,
            supported_estimands=frozenset(FrontDoorEstimator.supported_estimands),
            required_flags=FrontDoorEstimator.required_flags,
            excluded_flags=FrontDoorEstimator.excluded_flags,
            min_sample_size=FrontDoorEstimator.min_sample_size,
            produces_cate=FrontDoorEstimator.produces_cate,
            produces_full_counterfactual=FrontDoorEstimator.produces_full_counterfactual,
            propensity_required=FrontDoorEstimator.propensity_required,
        )
    )


_register()
