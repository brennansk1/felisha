"""Plain OLS estimator — the honest fallback at very small sample sizes.

DML and SuperLearner-stacked nuisance estimators need n ≥ ~100 for stable
cross-fitting; below that they overfit and produce CIs that are too
narrow. At n = 5–80 the right answer is the boring one: an OLS
regression of outcome on treatment + adjustment set, with
heteroskedasticity-robust (HC3) standard errors. The auto-selector falls
back to this when ``SMALL_SAMPLE`` is the binding constraint.

This estimator is intentionally simple — it's a way to give the analyst
a defensible point estimate + CI at sample sizes where the more
sophisticated methods cannot deliver. The report will show the OLS
choice in the analyst-decision ledger so reviewers can see why.
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


class OLSEstimator:
    """OLS regression of Y on T + W with HC3 robust SE.

    Treatment coefficient is the ATE under the linearity + ignorability
    assumptions. CI uses statsmodels' ``cov_type='HC3'`` for finite-sample
    robustness; this is the textbook small-sample standard.
    """

    id: str = "python.linear.ols"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE",)
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 5
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        alpha: float = 0.05,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.alpha = alpha
        self._fit: Any = None
        self._n_used = 0
        self._fit_seconds: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> OLSEstimator:
        from statsmodels.api import OLS, add_constant

        needed = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        for c in needed:
            if c not in data.columns:
                raise ValueError(f"Column not in data: {c!r}")
        df = data[needed].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"OLSEstimator needs ≥ {self.min_sample_size} rows after dropna; got {self._n_used}"
            )
        # Refuse a saturated fit. With n rows and p covariates (incl. intercept)
        # we need ≥ 2 residual degrees of freedom for a meaningful SE.
        p = 1 + 1 + len(self.confounders) + len(self.modifiers)  # intercept + T + W + X
        if self._n_used - p < 2:
            raise ValueError(
                f"OLS would be saturated: n={self._n_used}, p={p}, dof_resid="
                f"{self._n_used - p}. Drop confounders to leave at least 2 residual "
                f"degrees of freedom."
            )
        y = df[self.outcome].astype(float).to_numpy()
        x_cols = [self.treatment, *self.confounders, *self.modifiers]
        x = add_constant(df[x_cols].astype(float).to_numpy())
        start = time.perf_counter()
        # HC3 = MacKinnon-White (1985) — the canonical small-sample-friendly
        # robust SE estimator.
        self._fit = OLS(y, x).fit(cov_type="HC3")
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        if self._fit is None:
            raise RuntimeError("Call fit() before estimate().")
        # Index 0 is the intercept; treatment is index 1 in the design matrix.
        point = float(self._fit.params[1])
        se = float(self._fit.bse[1])
        ci_low, ci_high = self._fit.conf_int(alpha=self.alpha)[1]
        p = float(self._fit.pvalues[1])
        import statsmodels

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATE",
            point_estimate=point,
            se=se,
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            p_value=p,
            n_used=self._n_used,
            diagnostics={
                "cov_type": "HC3",
                "r_squared": float(self._fit.rsquared),
                "adj_r_squared": float(self._fit.rsquared_adj),
                "df_resid": int(self._fit.df_resid),
            },
            backend_version=f"statsmodels {statsmodels.__version__}",
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"fitted": self._fit is not None, "n_used": self._n_used}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    register(
        EstimatorEntry(
            id=OLSEstimator.id,
            factory=OLSEstimator,
            backend=OLSEstimator.backend,
            supported_estimands=frozenset(OLSEstimator.supported_estimands),
            required_flags=OLSEstimator.required_flags,
            excluded_flags=OLSEstimator.excluded_flags,
            min_sample_size=OLSEstimator.min_sample_size,
            produces_cate=OLSEstimator.produces_cate,
            produces_full_counterfactual=OLSEstimator.produces_full_counterfactual,
            propensity_required=OLSEstimator.propensity_required,
        )
    )


_register()
