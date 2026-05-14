"""Regression Discontinuity via the Calonico-Cattaneo-Titiunik (CCT) R stack.

This module bridges to four R packages that together implement the modern,
consensus toolbox for regression-discontinuity (RD) inference:

- ``rdrobust`` — local-polynomial RD point estimate at the cutoff with
  MSE-optimal bandwidth selection and bias-corrected robust confidence
  intervals (Calonico-Cattaneo-Titiunik 2014, ECMA).
- ``rdbwselect`` — standalone bandwidth selectors (``mserd``, ``msetwo``,
  ``msesum``, ``certwo``, ``cersum``). Exposed via ``rdrobust(bwselect=...)``.
- ``rdmulti`` — multi-cutoff / score RD (single-call meta-analysis across
  cutoffs); reserved for future routing.
- ``rddensity`` — Cattaneo-Jansson-Ma manipulation test (the modern
  successor to McCrary's density discontinuity test).

The wrapper handles both **sharp** RD (treatment is a deterministic
function of the running variable crossing the cutoff) and **fuzzy** RD
(running variable shifts the *probability* of treatment, with the actual
treatment indicator passed via the ``fuzzy_treatment`` argument).

Headline result conventions (consistent with the CCT papers):

- ``point_estimate`` / ``se`` report the **conventional** local-poly
  estimate at the optimal bandwidth — this is the point most practitioners
  cite.
- ``ci_low`` / ``ci_high`` report the **robust bias-corrected** CI from
  ``rdrobust`` (this is the recommended inference object — Calonico,
  Cattaneo, Titiunik 2014 show the conventional CI under-covers).
- ``p_value`` is the robust BC p-value.
- Both the conventional and bias-corrected variants are kept in
  ``diagnostics`` so a downstream report can show all three side-by-side
  exactly as ``summary(rdrobust(...))`` does.

The manipulation test p-value (``rddensity``) is surfaced in diagnostics
as a falsification signal; under no sorting around the cutoff, this
should be ≥ 0.10.
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


# ``RD_CANDIDATE`` is not (yet) present in the DataFlag enum. Use the
# closest existing flag if defined, else an empty required-flag set —
# the caller can still target this estimator by id.
_RD_CANDIDATE_FLAG = getattr(DataFlag, "RD_CANDIDATE", None)


_VALID_BW = ("mserd", "msetwo", "msesum", "certwo", "cersum")


class RDRobustEstimator:
    """Sharp / fuzzy regression discontinuity via ``rdrobust`` (R).

    Parameters
    ----------
    running_variable
        Column name of the score / forcing variable. Treatment assignment
        is determined by whether this crosses ``cutoff``.
    cutoff
        Threshold value c. Sharp RD treats X >= c as treated.
    outcome
        Column name of the outcome.
    fuzzy_treatment
        Optional column name carrying the **realised** binary treatment
        indicator. If supplied, switches to fuzzy RD via
        ``rdrobust(y, x, fuzzy = T)``; reports the local Wald (LATE) at
        the cutoff.
    bandwidth_method
        One of ``mserd`` / ``msetwo`` / ``msesum`` / ``certwo`` /
        ``cersum`` (Calonico-Cattaneo-Titiunik bandwidth selectors).
        Default ``mserd``.
    p_order
        Polynomial order for the local regression (default 1 — local
        linear, the CCT recommendation).
    kernel
        Kernel for the local-poly fit. ``triangular`` (default),
        ``epanechnikov``, or ``uniform``.
    """

    id: str = "rbridge.rd.rdrobust"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("LATE_AT_CUTOFF", "ATE")
    required_flags: frozenset[DataFlag] = (
        frozenset({_RD_CANDIDATE_FLAG}) if _RD_CANDIDATE_FLAG is not None else frozenset()
    )
    excluded_flags: frozenset[DataFlag] = frozenset()
    min_sample_size: int = 200
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        running_variable: str,
        cutoff: float,
        outcome: str,
        *,
        fuzzy_treatment: str | None = None,
        bandwidth_method: Literal[
            "mserd", "msetwo", "msesum", "certwo", "cersum"
        ] = "mserd",
        p_order: int = 1,
        kernel: Literal["triangular", "epanechnikov", "uniform"] = "triangular",
    ) -> None:
        if bandwidth_method not in _VALID_BW:
            raise ValueError(
                f"bandwidth_method must be one of {_VALID_BW}; got {bandwidth_method!r}"
            )
        self.running_variable = running_variable
        self.cutoff = float(cutoff)
        self.outcome = outcome
        self.fuzzy_treatment = fuzzy_treatment
        self.bandwidth_method = bandwidth_method
        self.p_order = int(p_order)
        self.kernel = kernel
        self._result: Any = None
        self._density_p: float | None = None
        self._n_used: int = 0
        self._n_h: int = 0
        self._fit_seconds: float | None = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, protocol: StudyProtocol | None = None) -> "RDRobustEstimator":
        require("rdrobust")
        # rddensity is optional — if it's missing we still return the
        # point + CI but log a None for the manipulation-test p.
        cols = [self.outcome, self.running_variable]
        if self.fuzzy_treatment:
            cols.append(self.fuzzy_treatment)
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"RDRobustEstimator needs >= {self.min_sample_size} rows; got {self._n_used}"
            )

        ro = r_session()
        with converter():
            ro.globalenv["Y_"] = ro.FloatVector(
                df[self.outcome].astype(float).to_numpy()
            )
            ro.globalenv["X_"] = ro.FloatVector(
                df[self.running_variable].astype(float).to_numpy()
            )
            if self.fuzzy_treatment:
                ro.globalenv["FZ_"] = ro.FloatVector(
                    df[self.fuzzy_treatment].astype(float).to_numpy()
                )

        fuzzy_arg = ", fuzzy = FZ_" if self.fuzzy_treatment else ""
        start = time.perf_counter()
        ro.r(
            f"rd_ <- rdrobust::rdrobust("
            f"y = Y_, x = X_, c = {self.cutoff}, "
            f'p = {self.p_order}, kernel = "{self.kernel}", '
            f'bwselect = "{self.bandwidth_method}"'
            f"{fuzzy_arg})"
        )
        self._fit_seconds = time.perf_counter() - start
        # rdrobust stores its result in ``rd_`` in the R global env; we
        # only need a sentinel here so estimate() knows fit() ran.
        self._result = True

        # Effective sample size (N_h left + N_h right).
        try:
            n_h = list(ro.r("sum(rd_$N_h)"))
            self._n_h = int(n_h[0]) if n_h else 0
        except Exception:
            self._n_h = 0

        # Manipulation test (Cattaneo-Jansson-Ma). Best-effort: if the
        # package isn't installed, we surface a None.
        try:
            require("rddensity")
            ro.r(
                f"rdd_ <- rddensity::rddensity(X = X_, c = {self.cutoff})"
            )
            self._density_p = float(list(ro.r("rdd_$test$p_jk"))[0])
        except Exception:
            self._density_p = None
        return self

    # ------------------------------------------------------------------
    # Estimate
    # ------------------------------------------------------------------
    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()

        # rdrobust stores three variants in length-3 vectors named
        # 'Conventional', 'Bias-Corrected', 'Robust' on $coef, $se, $pv, $ci.
        coef = [float(v) for v in list(ro.r("as.numeric(rd_$coef)"))]
        se = [float(v) for v in list(ro.r("as.numeric(rd_$se)"))]
        pv = [float(v) for v in list(ro.r("as.numeric(rd_$pv)"))]
        # ci is a 3x2 matrix in rdrobust: rows = (conv, bc, robust), cols = (lo, hi)
        ci_lo = [float(v) for v in list(ro.r("as.numeric(rd_$ci[,1])"))]
        ci_hi = [float(v) for v in list(ro.r("as.numeric(rd_$ci[,2])"))]

        # Bandwidths: rd_$bws is a 2x2 (left/right rows? actually rows=
        # (h, b), cols = (left, right) in modern rdrobust). Either way,
        # the headline used in the conventional poly is row "h", the
        # bias-corr is row "b". Report the left value (they are
        # typically symmetric under mserd).
        try:
            h_val = float(list(ro.r("as.numeric(rd_$bws[1, 1])"))[0])
        except Exception:
            h_val = float("nan")
        try:
            b_val = float(list(ro.r("as.numeric(rd_$bws[2, 1])"))[0])
        except Exception:
            b_val = float("nan")

        conventional_point = coef[0]
        bias_corrected_point = coef[1] if len(coef) > 1 else conventional_point
        robust_point = coef[2] if len(coef) > 2 else conventional_point
        conventional_se = se[0]
        bias_corrected_se = se[1] if len(se) > 1 else conventional_se
        robust_se = se[2] if len(se) > 2 else conventional_se
        robust_p = pv[2] if len(pv) > 2 else pv[0]

        conventional_ci = [ci_lo[0], ci_hi[0]] if ci_lo else [float("nan"), float("nan")]
        robust_ci = (
            [ci_lo[2], ci_hi[2]]
            if len(ci_lo) > 2
            else conventional_ci
        )

        rd_design = "fuzzy" if self.fuzzy_treatment else "sharp"
        estimand_class = "LATE_AT_CUTOFF"

        diagnostics: dict[str, Any] = {
            "bandwidth_h": h_val,
            "bandwidth_b": b_val,
            "bandwidth_method": self.bandwidth_method,
            "rd_design": rd_design,
            "manipulation_test_pvalue": self._density_p,
            "conventional_point": conventional_point,
            "conventional_se": conventional_se,
            "conventional_ci": conventional_ci,
            "bias_corrected_point": bias_corrected_point,
            "bias_corrected_se": bias_corrected_se,
            "robust_point": robust_point,
            "robust_se": robust_se,
            "robust_ci": robust_ci,
            "polynomial_order": self.p_order,
            "kernel": self.kernel,
            "cutoff": self.cutoff,
            "running_variable": self.running_variable,
            "n_h_effective": self._n_h,
            "r_session": r_session_metadata(),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class=estimand_class,
            point_estimate=conventional_point,
            se=conventional_se,
            ci_low=robust_ci[0],
            ci_high=robust_ci[1],
            p_value=robust_p,
            n_used=self._n_h or self._n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata().get("packages", {}).get("rdrobust", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._result is not None,
            "n_used": self._n_used,
            "n_h_effective": self._n_h,
            "rd_design": "fuzzy" if self.fuzzy_treatment else "sharp",
            "manipulation_test_pvalue": self._density_p,
        }

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    register(
        EstimatorEntry(
            id=RDRobustEstimator.id,
            factory=RDRobustEstimator,
            backend=RDRobustEstimator.backend,
            supported_estimands=frozenset(RDRobustEstimator.supported_estimands),
            required_flags=RDRobustEstimator.required_flags,
            excluded_flags=RDRobustEstimator.excluded_flags,
            min_sample_size=RDRobustEstimator.min_sample_size,
            produces_cate=RDRobustEstimator.produces_cate,
            produces_full_counterfactual=RDRobustEstimator.produces_full_counterfactual,
            propensity_required=RDRobustEstimator.propensity_required,
        )
    )


_register()


__all__ = ["RDRobustEstimator"]
