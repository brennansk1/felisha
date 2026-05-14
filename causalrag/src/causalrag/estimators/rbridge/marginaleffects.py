"""``marginaleffects`` standalone wrapper — continuous-treatment marginal
slopes and counterfactual predictions.

``marginaleffects::avg_slopes`` computes the average ∂Y/∂T marginal
slope from a fitted model — exactly the right primitive for continuous
treatments where the analyst wants "the average effect of a one-unit
increase in T". Cleaner than lmtp's policy framing when the question
is a slope rather than a counterfactual mean.

We fit a parsimonious outcome model (OLS with treatment × confounders
interactions for HTE) and let ``avg_slopes`` do the math with HC3
robust covariance.

Routes when ``CONTINUOUS_TREATMENT`` is flagged and the question is
phrased as a slope (vs a dose-response curve from lmtp).
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


class MarginalSlopesEstimator:
    """Average marginal slope ∂Y/∂T via marginaleffects::avg_slopes."""

    id: str = "rbridge.marginaleffects.slopes"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATE", "CATE")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.CONTINUOUS_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT}
    )
    min_sample_size: int = 30
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
        by: tuple[str, ...] = (),
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.by = by  # group-wise slopes (per stratum)
        self._n_used: int = 0
        self._fit_seconds: float | None = None
        self._by_results: list[dict[str, Any]] = []

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "MarginalSlopesEstimator":
        require("marginaleffects")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers, *self.by]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"MarginalSlopes needs ≥ {self.min_sample_size}; got {self._n_used}"
            )
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)
        rhs = [self.treatment]
        rhs.extend(self.confounders)
        rhs.extend(self.modifiers)
        # treatment * modifiers for HTE
        if self.modifiers:
            mod_terms = " + ".join(
                f"{self.treatment}:{m}" for m in self.modifiers
            )
            formula = f"{self.outcome} ~ {' + '.join(set(rhs))} + {mod_terms}"
        else:
            formula = f"{self.outcome} ~ {' + '.join(set(rhs))}"
        ro.r(f"omod_ <- lm({formula}, data = df_)")
        start = time.perf_counter()
        by_arg = ""
        if self.by:
            by_str = "c(" + ", ".join(f'"{b}"' for b in self.by) + ")"
            by_arg = f", by = {by_str}"
        ro.r(
            f'sl_ <- marginaleffects::avg_slopes(omod_, variables = "{self.treatment}"'
            f', vcov = "HC3"{by_arg})'
        )
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        ro = r_session()
        ate = float(list(ro.r("sl_$estimate"))[0])
        se = float(list(ro.r("sl_$std.error"))[0])
        ci_low = float(list(ro.r("sl_$conf.low"))[0])
        ci_high = float(list(ro.r("sl_$conf.high"))[0])
        p = float(list(ro.r("sl_$p.value"))[0])
        # If `by=`, capture per-stratum slopes
        by_rows: list[dict[str, Any]] = []
        if self.by:
            n_rows = int(list(ro.r("nrow(sl_)"))[0])
            for i in range(1, n_rows + 1):
                by_rows.append(
                    {
                        "estimate": float(list(ro.r(f"sl_$estimate[{i}]"))[0]),
                        "se": float(list(ro.r(f"sl_$std.error[{i}]"))[0]),
                    }
                )
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="CATE" if (self.modifiers or self.by) else "ATE",
            point_estimate=ate,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p,
            n_used=self._n_used,
            diagnostics={
                "interpretation": "Average marginal slope ∂Y/∂T",
                "by": list(self.by),
                "per_stratum": by_rows,
            },
            backend_version=r_session_metadata().get("packages", {}).get("marginaleffects", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"n_used": self._n_used, "by": list(self.by)}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    register(
        EstimatorEntry(
            id=MarginalSlopesEstimator.id,
            factory=MarginalSlopesEstimator,
            backend=MarginalSlopesEstimator.backend,
            supported_estimands=frozenset(MarginalSlopesEstimator.supported_estimands),
            required_flags=MarginalSlopesEstimator.required_flags,
            excluded_flags=MarginalSlopesEstimator.excluded_flags,
            min_sample_size=MarginalSlopesEstimator.min_sample_size,
            produces_cate=MarginalSlopesEstimator.produces_cate,
            produces_full_counterfactual=MarginalSlopesEstimator.produces_full_counterfactual,
            propensity_required=MarginalSlopesEstimator.propensity_required,
        )
    )


_register()
