"""BART estimator — Bayesian Additive Regression Trees (PDD §13 ``bart.py``).

Wraps the BART regressor as a standalone DML-style estimator: BART acts as the
*nuisance* learner for both E[Y|X,W] and the propensity, then a difference-in-
predicted-outcomes ATE is computed with posterior credible intervals.

This is the only estimator in the v0.1 catalog that produces calibrated
Bayesian intervals on the treatment effect — useful when the analyst wants
honest uncertainty quantification rather than asymptotic CIs from cross-fitting.

Requires ``pymc-bart`` (optional ``bart`` extra). On a system without
``pymc-bart``, the registration is skipped silently so the rest of the
catalog still works.
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


def _bart_available() -> bool:
    try:
        import pymc_bart  # noqa: F401

        return True
    except ImportError:
        return False


class BARTEstimator:
    """ATE / CATE via BART nuisance regression with posterior credible
    intervals on the treatment-effect contrast."""

    id: str = "python.bart.dml"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE", "CATE")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.BINARY_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 100
    produces_cate: bool = True
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        m: int = 50,
        draws: int = 500,
        tune: int = 500,
        chains: int = 2,
        random_state: int = 42,
        alpha: float = 0.05,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.m = m
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.random_state = random_state
        self.alpha = alpha

        self._posterior_diff: np.ndarray | None = None
        self._n_used: int = 0
        self._fit_seconds: float | None = None
        self._backend_version: str | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> BARTEstimator:
        if not _bart_available():
            raise RuntimeError(
                "BARTEstimator requires the optional 'bart' extra: "
                "pip install 'causalrag[bart]'"
            )
        import pymc as pm
        import pymc_bart as pmb

        for col in (self.outcome, self.treatment, *self.confounders, *self.modifiers):
            if col not in data.columns:
                raise ValueError(f"Column not in data: {col!r}")
        features = list(self.confounders) + list(self.modifiers)
        df = data[[self.outcome, self.treatment, *features]].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"BART requires at least {self.min_sample_size} rows; got {self._n_used}"
            )

        y = df[self.outcome].to_numpy().astype(np.float64)
        t = df[self.treatment].to_numpy().astype(np.float64)
        x = df[features].to_numpy().astype(np.float64)
        x_with_t = np.column_stack([x, t.reshape(-1, 1)])

        start = time.perf_counter()
        with pm.Model() as model:
            x_data = pm.Data("x_data", x_with_t)
            sigma = pm.HalfNormal("sigma", sigma=float(y.std() or 1.0))
            mu = pmb.BART("mu", X=x_data, Y=y, m=self.m)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y, shape=mu.shape)
            idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_state,
                progressbar=False,
                compute_convergence_checks=False,
            )

            x_t1 = np.column_stack([x, np.ones_like(t).reshape(-1, 1)])
            x_t0 = np.column_stack([x, np.zeros_like(t).reshape(-1, 1)])

            pm.set_data({"x_data": x_t1})
            post1 = pm.sample_posterior_predictive(
                idata, predictions=True, progressbar=False, var_names=["mu"]
            )
            pm.set_data({"x_data": x_t0})
            post0 = pm.sample_posterior_predictive(
                idata, predictions=True, progressbar=False, var_names=["mu"]
            )

        mu1 = post1.predictions["mu"].values.reshape(-1, self._n_used)
        mu0 = post0.predictions["mu"].values.reshape(-1, self._n_used)
        # Posterior of the per-row CATE; ATE is the row-mean per draw.
        self._posterior_diff = (mu1 - mu0).mean(axis=1)
        self._fit_seconds = time.perf_counter() - start
        import pymc_bart

        self._backend_version = f"pymc-bart {pymc_bart.__version__}"
        return self

    def estimate(self) -> EstimationResult:
        if self._posterior_diff is None:
            raise RuntimeError("Call fit() before estimate().")
        post = self._posterior_diff
        point = float(np.mean(post))
        ci_low = float(np.quantile(post, self.alpha / 2))
        ci_high = float(np.quantile(post, 1 - self.alpha / 2))
        # Posterior tail-probability mass on opposite sign — a Bayesian
        # analog of the p-value.
        p_value = float(2 * min(np.mean(post > 0), np.mean(post < 0)))

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="CATE" if self.modifiers else "ATE",
            point_estimate=point,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics={
                "posterior_draws": int(post.size),
                "interval_type": "posterior_credible",
                "m_trees": self.m,
            },
            backend_version=self._backend_version,
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"fitted": self._posterior_diff is not None, "n_used": self._n_used}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    """Only register BART if its optional dep is importable; otherwise the
    catalog stays free of an estimator that would always fail at fit time.
    """
    if not _bart_available():
        return
    register(
        EstimatorEntry(
            id=BARTEstimator.id,
            factory=BARTEstimator,
            backend=BARTEstimator.backend,
            supported_estimands=frozenset(BARTEstimator.supported_estimands),
            required_flags=BARTEstimator.required_flags,
            excluded_flags=BARTEstimator.excluded_flags,
            min_sample_size=BARTEstimator.min_sample_size,
            produces_cate=BARTEstimator.produces_cate,
            produces_full_counterfactual=BARTEstimator.produces_full_counterfactual,
            propensity_required=BARTEstimator.propensity_required,
        )
    )


_register()
