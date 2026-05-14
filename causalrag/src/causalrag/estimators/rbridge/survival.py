"""``survRM2`` wrapper — Restricted Mean Survival Time contrast.

The conventional way to summarize a binary-treatment survival comparison
when proportional hazards is unreliable. Reports E[min(T, τ) | A=1] −
E[min(T, τ) | A=0] for a chosen restriction time τ.

Auto-routes when ``RIGHT_CENSORED_OUTCOME`` is flagged AND the analyst
asked for an RMST estimand (vs the survival forest CATE path).
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


class SurvRM2Estimator:
    id: str = "rbridge.survrm2"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("RMST_CONTRAST",)
    required_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.BINARY_TREATMENT, DataFlag.RIGHT_CENSORED_OUTCOME}
    )
    excluded_flags: frozenset[DataFlag] = frozenset({DataFlag.TIME_VARYING_TREATMENT})
    min_sample_size: int = 50
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...] = (),
        modifiers: tuple[str, ...] = (),
        *,
        event: str = "event",
        tau: float | None = None,
        alpha: float = 0.05,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.event = event
        self.confounders = confounders
        self.modifiers = modifiers
        self.tau = tau
        self.alpha = alpha
        self._n_used = 0
        self._tau_used: float | None = None
        self._fit_seconds: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "SurvRM2Estimator":
        require("survRM2")
        ro = r_session()
        cols = [self.outcome, self.event, self.treatment, *self.confounders]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(f"survRM2 needs ≥ {self.min_sample_size}; got {self._n_used}")
        tau = self.tau if self.tau is not None else float(np.quantile(df[self.outcome], 0.75))
        self._tau_used = tau
        with converter():
            ro.globalenv["time_"] = ro.FloatVector(df[self.outcome].astype(float).to_numpy())
            ro.globalenv["status_"] = ro.IntVector(df[self.event].astype(int).to_numpy())
            ro.globalenv["arm_"] = ro.IntVector(df[self.treatment].astype(int).to_numpy())
        start = time.perf_counter()
        ro.r(f"res_ <- survRM2::rmst2(time_, status_, arm_, tau = {tau}, alpha = {self.alpha})")
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        ro = r_session()
        # res_$unadjusted.result is a 3x4 matrix: rows = RMST diff, ratio, ratio of restricted mean lost time
        row = list(ro.r("res_$unadjusted.result[1,]"))  # difference row: est, lo, hi, p
        ate, ci_low, ci_high, p = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        se = (ci_high - ci_low) / (2 * 1.959963984540054) if ci_high > ci_low else None
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="RMST_CONTRAST",
            point_estimate=ate,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p,
            n_used=self._n_used,
            diagnostics={"tau": self._tau_used, "alpha": self.alpha},
            backend_version=r_session_metadata().get("packages", {}).get("survRM2", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"n_used": self._n_used, "tau": self._tau_used}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    register(
        EstimatorEntry(
            id=SurvRM2Estimator.id,
            factory=SurvRM2Estimator,
            backend=SurvRM2Estimator.backend,
            supported_estimands=frozenset(SurvRM2Estimator.supported_estimands),
            required_flags=SurvRM2Estimator.required_flags,
            excluded_flags=SurvRM2Estimator.excluded_flags,
            min_sample_size=SurvRM2Estimator.min_sample_size,
            produces_cate=SurvRM2Estimator.produces_cate,
            produces_full_counterfactual=SurvRM2Estimator.produces_full_counterfactual,
            propensity_required=SurvRM2Estimator.propensity_required,
        )
    )


_register()
