"""``MatchIt`` + ``marginaleffects`` wrapper — propensity score matching
plus post-matching g-computation.

MatchIt is the most-used causal-inference R package. It handles nearest-
neighbor matching, full matching, optimal matching, genetic matching, and
exact matching with caliper/replacement/sub-classification options.

After matching we use ``marginaleffects::avg_comparisons`` to compute the
treatment-effect estimate on the matched sample with cluster-robust SE
(clustered by matched-set). This is the canonical workflow recommended
by the MatchIt vignette ("MatchIt with Survey Weights and Robust Standard
Errors").

Auto-routes to this estimator when:
- BINARY_TREATMENT is flagged
- The analyst explicitly requests matching via ``prefer='matching'``
- POSITIVITY_VIOLATION is flagged (matching trims the support)
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


class MatchItEstimator:
    """Propensity score matching + g-computation."""

    id: str = "rbridge.matchit"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATE", "ATT", "ATC")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.BINARY_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT}
    )
    min_sample_size: int = 50
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    METHODS = ("nearest", "optimal", "full", "genetic", "exact", "cem")

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        method: str = "nearest",
        caliper: float | None = 0.2,
        ratio: int = 1,
        replace: bool = False,
        estimand: str = "ATT",
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        if estimand not in ("ATE", "ATT", "ATC"):
            raise ValueError("estimand must be one of ATE / ATT / ATC")
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.method = method
        self.caliper = caliper
        self.ratio = ratio
        self.replace = replace
        self.estimand_choice = estimand
        self._n_used: int = 0
        self._n_matched: int = 0
        self._fit_seconds: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "MatchItEstimator":
        require("MatchIt")
        require("marginaleffects")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"MatchIt needs ≥ {self.min_sample_size} rows; got {self._n_used}"
            )
        # MatchIt requires the treatment column to be 0/1 numeric.
        df = df.copy()
        df[self.treatment] = df[self.treatment].astype(int)
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)

        formula = f"{self.treatment} ~ " + " + ".join(self.confounders)
        caliper_arg = f", caliper = {self.caliper}" if self.caliper is not None else ""
        replace_r = "TRUE" if self.replace else "FALSE"
        start = time.perf_counter()
        ro.r(
            f'match_ <- MatchIt::matchit({formula}, data = df_, '
            f'method = "{self.method}", estimand = "{self.estimand_choice}", '
            f"ratio = {self.ratio}, replace = {replace_r}{caliper_arg})"
        )
        # Get matched data and fit outcome model on it
        ro.r("matched_ <- MatchIt::match.data(match_)")
        outcome_formula = (
            f"{self.outcome} ~ {self.treatment} * (" + " + ".join(self.confounders) + ")"
        )
        ro.r(f"omod_ <- lm({outcome_formula}, data = matched_, weights = weights)")
        # g-computation via marginaleffects
        ro.r(
            f'mfx_ <- marginaleffects::avg_comparisons(omod_, '
            f'variables = "{self.treatment}", vcov = ~subclass, '
            f'newdata = subset(matched_, {self.treatment} == 1), '
            f'wts = "weights")'
        )
        self._fit_seconds = time.perf_counter() - start
        self._n_matched = int(list(ro.r("nrow(matched_)"))[0])
        return self

    def estimate(self) -> EstimationResult:
        ro = r_session()
        ate = float(list(ro.r("mfx_$estimate"))[0])
        se = float(list(ro.r("mfx_$std.error"))[0])
        ci_low = float(list(ro.r("mfx_$conf.low"))[0])
        ci_high = float(list(ro.r("mfx_$conf.high"))[0])
        p = float(list(ro.r("mfx_$p.value"))[0])
        # Balance + diagnostics
        try:
            sumr = ro.r("summary(match_, standardize = TRUE)$sum.matched")
            max_smd = float(list(ro.r("max(abs(summary(match_, standardize = TRUE)$sum.matched[,3]))"))[0])
        except Exception:
            max_smd = float("nan")
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
                "caliper": self.caliper,
                "ratio": self.ratio,
                "replace": self.replace,
                "n_matched": self._n_matched,
                "max_post_match_smd": max_smd,
            },
            backend_version=r_session_metadata().get("packages", {}).get("MatchIt", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"n_used": self._n_used, "n_matched": self._n_matched, "method": self.method}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    register(
        EstimatorEntry(
            id=MatchItEstimator.id,
            factory=MatchItEstimator,
            backend=MatchItEstimator.backend,
            supported_estimands=frozenset(MatchItEstimator.supported_estimands),
            required_flags=MatchItEstimator.required_flags,
            excluded_flags=MatchItEstimator.excluded_flags,
            min_sample_size=MatchItEstimator.min_sample_size,
            produces_cate=MatchItEstimator.produces_cate,
            produces_full_counterfactual=MatchItEstimator.produces_full_counterfactual,
            propensity_required=MatchItEstimator.propensity_required,
        )
    )


_register()
