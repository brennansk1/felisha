"""``WeightIt`` + ``cobalt`` wrapper — propensity-score weighting with
balance diagnostics.

WeightIt unifies a wide range of propensity-weighting methods under one
API:

- **glm** — logistic regression PS, IPW
- **gbm** — generalized boosted trees PS
- **cbps** — Covariate Balancing PS (Imai-Ratkovic 2014)
- **ebal** — entropy balancing (Hainmueller 2012)
- **bart** — BART-fitted PS
- **super** — SuperLearner-fitted PS
- **energy** — energy balancing (Huling-Mak 2022)

After weighting, ATE estimation is a simple weighted regression of Y on
T (or T + covariates for additional adjustment). ``cobalt::bal.tab``
produces balance diagnostics (standardized mean differences, variance
ratios) for the report.

Use when the analyst wants weighting (vs matching). Strong choice when
POSITIVITY_VIOLATION is mild — overlap can be improved without
truncating samples.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import EstimatorEntry, register
from causalrag.core.result import EstimationResult
from causalrag.estimators.rbridge._r import (
    converter,
    r_session,
    r_session_metadata,
    require,
)


METHOD_NAMES = ("glm", "gbm", "cbps", "ebal", "bart", "super", "energy")
ESTIMAND_NAMES = ("ATE", "ATT", "ATC", "ATO", "ATM")


class WeightItEstimator:
    """Propensity-weighting ATE/ATT/ATC via WeightIt + weighted OLS."""

    id: str = "rbridge.weightit"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATE", "ATT", "ATC", "ATO", "ATM")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.BINARY_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT}
    )
    min_sample_size: int = 50
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        method: str = "ebal",
        estimand: str = "ATE",
        stabilize: bool = True,
        trim_quantile: float | None = 0.99,
    ) -> None:
        # Default method bumped from "glm" → "ebal" (entropy balancing) per
        # Zhao 2017 *Entropy Balancing Is Doubly Robust* (Journal of Causal
        # Inference). EBAL exactly balances the specified moments and is
        # doubly robust when the outcome model is correctly specified —
        # strictly stronger guarantees than logistic-regression propensity
        # weights. "glm" remains available via explicit `method="glm"`.
        if method not in METHOD_NAMES:
            raise ValueError(f"method must be one of {METHOD_NAMES}; got {method!r}")
        if estimand not in ESTIMAND_NAMES:
            raise ValueError(f"estimand must be one of {ESTIMAND_NAMES}; got {estimand!r}")
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.method = method
        self.estimand_choice = estimand
        self.stabilize = stabilize
        self.trim_quantile = trim_quantile
        self._n_used: int = 0
        self._fit_seconds: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "WeightItEstimator":
        require("WeightIt")
        require("cobalt")
        require("marginaleffects")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna().copy()
        df[self.treatment] = df[self.treatment].astype(int)
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(f"WeightIt needs ≥ {self.min_sample_size}; got {self._n_used}")
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)
        formula = f"{self.treatment} ~ " + " + ".join(self.confounders)
        stabilize_r = "TRUE" if self.stabilize else "FALSE"

        start = time.perf_counter()
        ro.r(
            f'w_ <- WeightIt::weightit({formula}, data = df_, '
            f'method = "{self.method}", estimand = "{self.estimand_choice}", '
            f"stabilize = {stabilize_r})"
        )
        # Optional trimming of extreme weights
        if self.trim_quantile is not None:
            ro.r(
                f"w_ <- WeightIt::trim(w_, at = {self.trim_quantile}, "
                f"lower = TRUE)"
            )
        ro.r("df_$.w_ <- w_$weights")
        # Weighted outcome regression for the ATE; marginaleffects handles
        # the standard-error machinery (cluster-robust on subclass-id if
        # needed).
        outcome_formula = (
            f"{self.outcome} ~ {self.treatment} * (" + " + ".join(self.confounders) + ")"
        )
        ro.r(f"omod_ <- lm({outcome_formula}, data = df_, weights = .w_)")
        ro.r(
            f'mfx_ <- marginaleffects::avg_comparisons(omod_, '
            f'variables = "{self.treatment}", wts = ".w_", vcov = "HC3")'
        )
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        ro = r_session()
        ate = float(list(ro.r("mfx_$estimate"))[0])
        se = float(list(ro.r("mfx_$std.error"))[0])
        ci_low = float(list(ro.r("mfx_$conf.low"))[0])
        ci_high = float(list(ro.r("mfx_$conf.high"))[0])
        p = float(list(ro.r("mfx_$p.value"))[0])
        # Balance diagnostics (max post-weighting SMD)
        try:
            max_smd = float(
                list(
                    ro.r(
                        "max(abs(cobalt::bal.tab(w_, m.threshold = 0.1)$Balance$Diff.Adj), na.rm = TRUE)"
                    )
                )[0]
            )
        except Exception:
            max_smd = float("nan")
        # Effective sample size after weighting
        try:
            ess = float(list(ro.r("WeightIt::ESS(w_$weights)"))[0])
        except Exception:
            ess = float("nan")

        return EstimationResult(
            estimator_id=self.id,
            estimand_class=self.estimand_choice,
            point_estimate=ate,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p,
            n_used=self._n_used,
            diagnostics={
                "method": self.method,
                "stabilize": self.stabilize,
                "trim_quantile": self.trim_quantile,
                "effective_n": ess,
                "max_post_weighting_smd": max_smd,
            },
            backend_version=r_session_metadata().get("packages", {}).get("WeightIt", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"n_used": self._n_used, "method": self.method, "estimand": self.estimand_choice}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    register(
        EstimatorEntry(
            id=WeightItEstimator.id,
            factory=WeightItEstimator,
            backend=WeightItEstimator.backend,
            supported_estimands=frozenset(WeightItEstimator.supported_estimands),
            required_flags=WeightItEstimator.required_flags,
            excluded_flags=WeightItEstimator.excluded_flags,
            min_sample_size=WeightItEstimator.min_sample_size,
            produces_cate=WeightItEstimator.produces_cate,
            produces_full_counterfactual=WeightItEstimator.produces_full_counterfactual,
            propensity_required=WeightItEstimator.propensity_required,
        )
    )


_register()
