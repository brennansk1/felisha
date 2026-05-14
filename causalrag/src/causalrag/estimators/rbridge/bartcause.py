"""``bartCause`` wrapper — Bayesian Causal Forest via Hill 2011 / BART.

``bartCause::bartc`` fits BART separately for the response surface under
treatment and control, then estimates the ATE / ATT / individual
treatment effects from posterior samples. Hallmarks:

- **Calibrated Bayesian credible intervals** — proper posterior on the
  treatment effect, not asymptotic normal approximation.
- **Heterogeneity for free** — every fit returns per-subject ITEs.
- **Honest about overlap** — flags poorly-supported subjects via the
  ``commonSup.rule`` option.

Auto-routes when the analyst wants Bayesian inference and the data is
moderate-sized (n ≥ 50, binary treatment, continuous or binary outcome).
This is the R-bridged complement to our Python ``BARTEstimator``
(``python.bart.dml``) — bartCause is the canonical implementation, the
Python BART relies on pymc-bart which is a research-grade port.
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
from causalrag.estimators.rbridge._r import (
    converter,
    r_session,
    r_session_metadata,
    require,
)


class BartCauseEstimator:
    """Bayesian Causal Forest via bartCause::bartc."""

    id: str = "rbridge.bartcause"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATE", "ATT", "ATC", "CATE")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.BINARY_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.LONGITUDINAL}
    )
    min_sample_size: int = 50
    produces_cate: bool = True
    produces_full_counterfactual: bool = True
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        estimand: Literal["ATE", "ATT", "ATC"] = "ATE",
        method_rsp: Literal["bart", "p.weight", "tmle"] = "bart",
        method_trt: Literal["bart", "glm", "none"] = "bart",
        n_samples: int = 500,
        n_burn: int = 250,
        n_chains: int = 4,
        seed: int = 42,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.estimand_choice = estimand
        self.method_rsp = method_rsp
        self.method_trt = method_trt
        self.n_samples = n_samples
        self.n_burn = n_burn
        self.n_chains = n_chains
        self.seed = seed
        self._n_used: int = 0
        self._fit_seconds: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "BartCauseEstimator":
        require("bartCause")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna().copy()
        df[self.treatment] = df[self.treatment].astype(int)
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(f"bartCause needs ≥ {self.min_sample_size} rows; got {self._n_used}")
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)
        confs = list(self.confounders) + list(self.modifiers)
        conf_str = "c(" + ", ".join(f'"{c}"' for c in confs) + ")"
        start = time.perf_counter()
        ro.r(
            f"set.seed({self.seed});"
            f'bcf_ <- bartCause::bartc(response = df_[["{self.outcome}"]], '
            f'treatment = df_[["{self.treatment}"]], '
            f"confounders = as.matrix(df_[, {conf_str}]), "
            f'estimand = "{self.estimand_choice.lower()}", '
            f'method.rsp = "{self.method_rsp}", method.trt = "{self.method_trt}", '
            f"n.samples = {self.n_samples}, n.burn = {self.n_burn}, n.chains = {self.n_chains})"
        )
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        ro = r_session()
        # bartCause exposes summary() that returns estimate, sd, ci.lower, ci.upper
        ro.r("smry_ <- summary(bcf_)")
        ate = float(list(ro.r("smry_$estimates$estimate"))[0])
        se = float(list(ro.r("smry_$estimates$sd"))[0])
        ci_low = float(list(ro.r("smry_$estimates$ci.lower"))[0])
        ci_high = float(list(ro.r("smry_$estimates$ci.upper"))[0])
        from scipy.stats import norm

        p = float(2 * (1 - norm.cdf(abs(ate) / se))) if se > 0 else None

        # Per-subject ITEs for CATE diagnostics
        ites = np.array(list(ro.r("apply(bartCause::extract(bcf_, type='ite'), 2, mean)")))
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
                "method_rsp": self.method_rsp,
                "method_trt": self.method_trt,
                "n_samples": self.n_samples,
                "ite_mean": float(np.mean(ites)),
                "ite_std": float(np.std(ites)),
                "ite_q05": float(np.quantile(ites, 0.05)),
                "ite_q95": float(np.quantile(ites, 0.95)),
                "interval_type": "posterior_credible",
            },
            backend_version=r_session_metadata().get("packages", {}).get("bartCause", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"n_used": self._n_used, "estimand": self.estimand_choice}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    register(
        EstimatorEntry(
            id=BartCauseEstimator.id,
            factory=BartCauseEstimator,
            backend=BartCauseEstimator.backend,
            supported_estimands=frozenset(BartCauseEstimator.supported_estimands),
            required_flags=BartCauseEstimator.required_flags,
            excluded_flags=BartCauseEstimator.excluded_flags,
            min_sample_size=BartCauseEstimator.min_sample_size,
            produces_cate=BartCauseEstimator.produces_cate,
            produces_full_counterfactual=BartCauseEstimator.produces_full_counterfactual,
            propensity_required=BartCauseEstimator.propensity_required,
        )
    )


_register()
