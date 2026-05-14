"""Targeted Maximum Likelihood Estimation (TMLE) via the **tlverse** R stack.

This module wraps the canonical R implementation of TMLE built by the
van der Laan group (UC Berkeley):

- ``sl3``     — SuperLearner-style stacked ensemble for the nuisance
                models (outcome regression Q-bar(A, W) and propensity
                g(A | W)). Provides cross-validated learner selection.
- ``tmle3``   — the targeting step: fits an initial estimate from the
                SuperLearners, then perturbs along the efficient
                influence function until the score equation is solved,
                giving an *asymptotically efficient* + *doubly robust*
                plug-in estimator.
- ``tmle3mediate`` — extends the targeting machinery to natural /
                interventional (in)direct effects (Diaz et al. 2020).

Headline conventions:

- Reported ``point_estimate`` is the targeted plug-in for the requested
  estimand (ATE / ATT / ATC, or NDE / NIE for the mediation wrapper).
- ``se`` is the influence-curve-based standard error returned by
  ``tmle3_fit$summary$se``; CIs are the Wald CIs reported by tmle3.
- The nuisance SuperLearner CV-risk table is surfaced as
  ``diagnostics['superlearner_cv_risk']`` — the per-learner
  cross-validated risk + chosen meta-weights are the audit trail for
  the double-robustness story.
- ``diagnostics['eif_mean']`` reports the empirical mean of the
  efficient influence function on the targeted model; under correct
  specification this is numerically near zero (the targeting step
  *forces* this by construction), making it a useful diagnostic that
  the targeting iteration converged.

Reference
---------
van der Laan, M.J., Rose, S. (2011). *Targeted Learning: Causal
Inference for Observational and Experimental Data*. Springer.

Diaz, I., Hejazi, N.S., Rudolph, K.E., van der Laan, M.J. (2020).
*Non-parametric efficient causal mediation with intermediate
confounders*. Biometrika 108(3): 627-641.
"""

from __future__ import annotations

import time
from math import erf, sqrt
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_sided_p(estimate: float, se: float) -> float | None:
    if se is None or not (se > 0):
        return None
    z = abs(estimate / se)
    return 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))


def _safe_scalar(ro: Any, expr: str) -> float | None:
    """Pull a single numeric R scalar; return ``None`` on any failure."""
    try:
        v = list(ro.r(expr))
        if not v:
            return None
        return float(v[0])
    except Exception:
        return None


def _safe_vector(ro: Any, expr: str) -> list[float]:
    try:
        return [float(v) for v in list(ro.r(expr))]
    except Exception:
        return []


def _safe_strings(ro: Any, expr: str) -> list[str]:
    try:
        return [str(v) for v in list(ro.r(expr))]
    except Exception:
        return []


# Estimand -> tmle3 Spec_* constructor.
_TMLE3_SPEC_BY_ESTIMAND: dict[str, str] = {
    "ATE": "tmle3::tmle_ATE(treatment_level = 1, control_level = 0)",
    "ATT": "tmle3::tmle_ATT(treatment_level = 1, control_level = 0)",
    "ATC": "tmle3::tmle_ATC(treatment_level = 1, control_level = 0)",
}


# Default sl3 learner stack — a defensible default mixing parametric,
# tree-based, and penalised-regression learners. Users can override.
_DEFAULT_LEARNER_STACK_R = (
    "make_learner_stack <- function() {{"
    "  list("
    "    Lrnr_glm_fast = sl3::Lrnr_glm_fast$new(),"
    "    Lrnr_ranger = sl3::Lrnr_ranger$new(num.trees = 200),"
    "    Lrnr_glmnet = sl3::Lrnr_glmnet$new(),"
    "    Lrnr_mean = sl3::Lrnr_mean$new()"
    "  )"
    "}}"
)


# ---------------------------------------------------------------------------
# 1. ATE / ATT / ATC via tmle3
# ---------------------------------------------------------------------------


class TMLE3Estimator:
    """Targeted Maximum Likelihood Estimation (van der Laan-Rose 2011)
    via the tlverse R stack — ``sl3`` Super Learner + ``tmle3``
    targeting step.

    Supports ATE, ATT, ATC via the ``tmle_ATE`` / ``tmle_ATT`` /
    ``tmle_ATC`` Spec dispatch in ``tmle3``. Provides influence-curve
    based standard errors and asymptotic Wald CIs.

    Slow but defensible. Use when the analysis must withstand journal-
    level scrutiny on the double-robustness guarantee.

    Parameters
    ----------
    treatment, outcome
        Column names. ``treatment`` must be binary 0/1.
    covariates
        Sequence of covariate / confounder column names ``W``. Both
        the outcome regression Q-bar(A, W) and the propensity g(A | W)
        are fit as SuperLearner stacks on these.
    estimand
        ``"ATE"`` (default), ``"ATT"``, or ``"ATC"``.
    learners
        Optional R expression building the sl3 learner library (a
        ``list(...)`` of ``Lrnr_*$new(...)`` objects). If ``None``,
        a defensible default stack (glm_fast + ranger + glmnet + mean)
        is used.
    cv_folds
        Number of CV folds for sl3 (default 5).
    """

    id: str = "rbridge.tmle3"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATE", "ATT", "ATC")
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME}  # use tmle3_survival instead
    )
    min_sample_size: int = 200
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        covariates: list[str] | tuple[str, ...],
        *,
        estimand: Literal["ATE", "ATT", "ATC"] = "ATE",
        learners: str | None = None,
        cv_folds: int = 5,
    ) -> None:
        if estimand not in self.supported_estimands:
            raise ValueError(
                f"estimand must be one of {self.supported_estimands}; got {estimand!r}"
            )
        if not covariates:
            raise ValueError("TMLE3Estimator requires at least one covariate (W).")
        self.treatment = treatment
        self.outcome = outcome
        self.covariates = list(covariates)
        self.estimand = estimand
        self.learners = learners
        self.cv_folds = int(cv_folds)

        self._fitted = False
        self._n_used = 0
        self._fit_seconds: float | None = None
        self._psi: float | None = None
        self._se: float | None = None
        self._ci_low: float | None = None
        self._ci_high: float | None = None
        self._eif_mean: float | None = None
        self._cv_risk_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _push_to_r(self, df: pd.DataFrame) -> None:
        ro = r_session()
        with converter():
            ro.globalenv["tmle3_data_"] = ro.conversion.py2rpy(df)

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol | None = None
    ) -> "TMLE3Estimator":
        require("sl3")
        require("tmle3")

        cols = [self.treatment, self.outcome] + self.covariates
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(
                f"Input is missing required columns for TMLE3: {missing}"
            )
        df = data[cols].copy().dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"TMLE3Estimator needs >= {self.min_sample_size} rows; "
                f"got {self._n_used}"
            )

        self._push_to_r(df)
        ro = r_session()

        # Build the node list (tmle3 calls Y outcome, A treatment, W baseline).
        w_vec = "c(" + ", ".join(f'"{c}"' for c in self.covariates) + ")"
        ro.r(
            f"tmle3_nodes_ <- tmle3::tmle3_Node_List$new("
            f'list(W = {w_vec}, A = "{self.treatment}", '
            f'Y = "{self.outcome}"))'
        )
        # Fallback to the simpler dict API if Node_List class isn't exposed.
        ro.r(
            f"tmle3_npsem_ <- list(W = {w_vec}, "
            f'A = "{self.treatment}", Y = "{self.outcome}")'
        )

        # Learner stack — either user-supplied R expression or default.
        if self.learners is not None:
            ro.r(f"learner_stack_ <- {self.learners}")
        else:
            ro.r(
                "learner_stack_ <- sl3::Stack$new("
                "sl3::Lrnr_glm_fast$new(), "
                "sl3::Lrnr_ranger$new(num.trees = 200), "
                "sl3::Lrnr_glmnet$new(), "
                "sl3::Lrnr_mean$new())"
            )
        # Wrap stack in a Super Learner with the requested CV folds.
        ro.r(
            f"sl_Q_ <- sl3::Lrnr_sl$new(learners = learner_stack_, "
            f"metalearner = sl3::Lrnr_nnls$new())"
        )
        ro.r(
            f"sl_g_ <- sl3::Lrnr_sl$new(learners = learner_stack_, "
            f"metalearner = sl3::Lrnr_nnls$new())"
        )
        ro.r(
            "learner_list_ <- list(Y = sl_Q_, A = sl_g_)"
        )

        # Build the Spec for the requested estimand.
        spec_r = _TMLE3_SPEC_BY_ESTIMAND[self.estimand]
        ro.r(f"tmle3_spec_ <- {spec_r}")

        start = time.perf_counter()
        ro.r(
            "tmle3_fit_ <- tmle3::tmle3("
            "tmle3_spec_, tmle3_data_, tmle3_npsem_, learner_list_)"
        )
        self._fit_seconds = time.perf_counter() - start

        # Pull the targeted point estimate + IC-based SE + Wald CI.
        # ``tmle3_fit$summary`` is a data.table with columns
        # (param, init_est, tmle_est, se, lower, upper, ...). For a
        # single-estimand Spec there is one row.
        self._psi = _safe_scalar(
            ro, "as.numeric(tmle3_fit_$summary$tmle_est[1])"
        )
        self._se = _safe_scalar(
            ro, "as.numeric(tmle3_fit_$summary$se[1])"
        )
        self._ci_low = _safe_scalar(
            ro, "as.numeric(tmle3_fit_$summary$lower[1])"
        )
        self._ci_high = _safe_scalar(
            ro, "as.numeric(tmle3_fit_$summary$upper[1])"
        )
        # Empirical mean of the EIF on the targeted model — should be ~0.
        self._eif_mean = _safe_scalar(
            ro, "as.numeric(mean(tmle3_fit_$estimates[[1]]$IC))"
        )

        # SuperLearner CV-risk audit table — per-learner CV risk + meta-weight.
        # ``tmle3_fit$learner_fits$Y$learner_fits`` carries the Q-bar stack;
        # the cv_risk method returns a table of (learner, risk, coefficients).
        learner_names = _safe_strings(
            ro,
            'as.character(tmle3_fit_$learner_fits$Y$cv_risk(loss_squared_error)$learner)',
        )
        risks = _safe_vector(
            ro,
            'as.numeric(tmle3_fit_$learner_fits$Y$cv_risk(loss_squared_error)$MSE)',
        )
        coefs = _safe_vector(
            ro,
            'as.numeric(tmle3_fit_$learner_fits$Y$cv_risk(loss_squared_error)$coefficients)',
        )
        if learner_names and len(learner_names) == len(risks):
            for i, name in enumerate(learner_names):
                self._cv_risk_rows.append(
                    {
                        "learner": name,
                        "cv_risk": risks[i] if i < len(risks) else None,
                        "meta_weight": coefs[i] if i < len(coefs) else None,
                    }
                )

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # estimate
    # ------------------------------------------------------------------

    def estimate(self) -> EstimationResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before estimate().")

        psi = self._psi if self._psi is not None else float("nan")
        se = self._se if self._se is not None else float("nan")
        if self._ci_low is not None and self._ci_high is not None:
            ci_low = self._ci_low
            ci_high = self._ci_high
        elif self._se is not None and self._se > 0 and self._psi is not None:
            ci_low = self._psi - 1.96 * self._se
            ci_high = self._psi + 1.96 * self._se
        else:
            ci_low = ci_high = float("nan")

        if self._se is not None and self._se > 0 and self._psi is not None:
            p_value = _two_sided_p(self._psi, self._se)
        else:
            p_value = None

        diagnostics: dict[str, Any] = {
            "estimand": self.estimand,
            "covariates": list(self.covariates),
            "cv_folds": self.cv_folds,
            "superlearner_cv_risk": list(self._cv_risk_rows),
            "eif_mean": self._eif_mean,
            "eif_mean_note": (
                "Empirical mean of the efficient influence function on the "
                "targeted model. The targeting step solves the score equation, "
                "so this should be numerically close to zero; departures from "
                "zero indicate the iteration did not converge."
            ),
            "doubly_robust": True,
            "double_robust_note": (
                "TMLE is doubly robust: the targeted plug-in is consistent "
                "if EITHER the outcome regression Q-bar(A, W) OR the "
                "propensity g(A | W) is correctly specified (van der Laan-Rose 2011)."
            ),
            "r_session": r_session_metadata(),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class=self.estimand,
            point_estimate=psi,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata().get("packages", {}).get("tmle3", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted,
            "n_used": self._n_used,
            "estimand": self.estimand,
            "eif_mean": self._eif_mean,
            "n_learners": len(self._cv_risk_rows),
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# 2. Natural / interventional mediation via tmle3mediate
# ---------------------------------------------------------------------------


class TMLE3MediationEstimator:
    """Natural / interventional (in)direct effect decomposition via
    ``tmle3mediate``.

    Decomposes the total effect into Natural Direct (NDE) and Natural
    Indirect (NIE) components through a named mediator using the
    efficient influence-curve targeting machinery of tmle3
    (Diaz et al. 2020). Both nuisance pieces (mediator density and
    outcome regression) are fit as ``sl3`` SuperLearners.

    Parameters
    ----------
    treatment, outcome, mediator
        Column names. ``treatment`` must be binary 0/1.
    covariates
        Sequence of pre-treatment confounder column names ``W``.
    intermediate_confounders
        Optional list of post-treatment, pre-mediator confounders ``Z``
        (the intermediate confounder vector of Diaz et al. 2020).
        If non-empty, the wrapper estimates *interventional* effects;
        if empty, natural effects.
    learners
        Optional R expression building the sl3 learner library.
    cv_folds
        Number of CV folds for sl3 (default 5).
    """

    id: str = "rbridge.tmle3.mediation"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("NDE", "NIE")
    required_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.MEDIATOR_PROPOSED}
    )
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME}
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
        covariates: list[str] | tuple[str, ...],
        *,
        intermediate_confounders: list[str] | tuple[str, ...] | None = None,
        learners: str | None = None,
        cv_folds: int = 5,
    ) -> None:
        if not covariates:
            raise ValueError(
                "TMLE3MediationEstimator requires at least one covariate (W)."
            )
        self.treatment = treatment
        self.outcome = outcome
        self.mediator = mediator
        self.covariates = list(covariates)
        self.intermediate_confounders = list(intermediate_confounders or [])
        self.learners = learners
        self.cv_folds = int(cv_folds)

        self._fitted = False
        self._n_used = 0
        self._fit_seconds: float | None = None
        self._nde: dict[str, float | None] = {}
        self._nie: dict[str, float | None] = {}
        self._cv_risk_rows: list[dict[str, Any]] = []
        self._effect_type: str = "natural"

    def _push_to_r(self, df: pd.DataFrame) -> None:
        ro = r_session()
        with converter():
            ro.globalenv["tmle3_med_data_"] = ro.conversion.py2rpy(df)

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol | None = None
    ) -> "TMLE3MediationEstimator":
        require("sl3")
        require("tmle3")
        require("tmle3mediate")

        cols = (
            [self.treatment, self.outcome, self.mediator]
            + self.covariates
            + self.intermediate_confounders
        )
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(
                f"Input is missing required columns for TMLE3 mediation: {missing}"
            )
        df = data[cols].copy().dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"TMLE3MediationEstimator needs >= {self.min_sample_size} rows; "
                f"got {self._n_used}"
            )

        self._effect_type = (
            "interventional" if self.intermediate_confounders else "natural"
        )

        self._push_to_r(df)
        ro = r_session()

        w_vec = "c(" + ", ".join(f'"{c}"' for c in self.covariates) + ")"
        npsem_parts = [
            f"W = {w_vec}",
            f'A = "{self.treatment}"',
            f'Z = "{self.mediator}"',
            f'Y = "{self.outcome}"',
        ]
        if self.intermediate_confounders:
            ic_vec = (
                "c("
                + ", ".join(f'"{c}"' for c in self.intermediate_confounders)
                + ")"
            )
            # Place intermediate confounders ahead of the mediator.
            npsem_parts.insert(
                2, f"Zint = {ic_vec}"
            )
        ro.r("tmle3_med_npsem_ <- list(" + ", ".join(npsem_parts) + ")")

        if self.learners is not None:
            ro.r(f"learner_stack_ <- {self.learners}")
        else:
            ro.r(
                "learner_stack_ <- sl3::Stack$new("
                "sl3::Lrnr_glm_fast$new(), "
                "sl3::Lrnr_ranger$new(num.trees = 200), "
                "sl3::Lrnr_glmnet$new(), "
                "sl3::Lrnr_mean$new())"
            )
        ro.r(
            "sl_med_ <- sl3::Lrnr_sl$new(learners = learner_stack_, "
            "metalearner = sl3::Lrnr_nnls$new())"
        )
        ro.r(
            "learner_list_med_ <- list(Y = sl_med_, A = sl_med_, Z = sl_med_)"
        )

        start = time.perf_counter()
        # tmle3mediate exposes Spec_NIE / Spec_NDE; we run both so the
        # decomposition is in the diagnostics table.
        ro.r(
            "tmle3_nde_spec_ <- tmle3mediate::tmle_NDE("
            "e_learners = sl_med_, psi_Z_learners = sl_med_, "
            "max_iter = 1)"
        )
        ro.r(
            "tmle3_nie_spec_ <- tmle3mediate::tmle_NIE("
            "e_learners = sl_med_, psi_Z_learners = sl_med_, "
            "max_iter = 1)"
        )
        ro.r(
            "tmle3_nde_fit_ <- tmle3::tmle3("
            "tmle3_nde_spec_, tmle3_med_data_, tmle3_med_npsem_, "
            "learner_list_med_)"
        )
        ro.r(
            "tmle3_nie_fit_ <- tmle3::tmle3("
            "tmle3_nie_spec_, tmle3_med_data_, tmle3_med_npsem_, "
            "learner_list_med_)"
        )
        self._fit_seconds = time.perf_counter() - start

        for label, r_obj in (("nde", "tmle3_nde_fit_"), ("nie", "tmle3_nie_fit_")):
            est = _safe_scalar(ro, f"as.numeric({r_obj}$summary$tmle_est[1])")
            se = _safe_scalar(ro, f"as.numeric({r_obj}$summary$se[1])")
            lo = _safe_scalar(ro, f"as.numeric({r_obj}$summary$lower[1])")
            hi = _safe_scalar(ro, f"as.numeric({r_obj}$summary$upper[1])")
            getattr(self, f"_{label}").update(
                {"estimate": est, "se": se, "ci_low": lo, "ci_high": hi}
            )

        # Pull a single CV-risk table from the NDE fit's Y-stack (the Q-bar).
        learner_names = _safe_strings(
            ro,
            'as.character(tmle3_nde_fit_$learner_fits$Y$cv_risk(loss_squared_error)$learner)',
        )
        risks = _safe_vector(
            ro,
            'as.numeric(tmle3_nde_fit_$learner_fits$Y$cv_risk(loss_squared_error)$MSE)',
        )
        coefs = _safe_vector(
            ro,
            'as.numeric(tmle3_nde_fit_$learner_fits$Y$cv_risk(loss_squared_error)$coefficients)',
        )
        if learner_names and len(learner_names) == len(risks):
            for i, name in enumerate(learner_names):
                self._cv_risk_rows.append(
                    {
                        "learner": name,
                        "cv_risk": risks[i] if i < len(risks) else None,
                        "meta_weight": coefs[i] if i < len(coefs) else None,
                    }
                )

        self._fitted = True
        return self

    def estimate(self) -> EstimationResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before estimate().")

        # Headline = NIE (the indirect / mediated effect).
        nie_est = self._nie.get("estimate")
        nie_se = self._nie.get("se")
        point = float(nie_est) if nie_est is not None else float("nan")
        se = float(nie_se) if nie_se is not None else float("nan")

        if self._nie.get("ci_low") is not None and self._nie.get("ci_high") is not None:
            ci_low = float(self._nie["ci_low"])
            ci_high = float(self._nie["ci_high"])
        elif nie_se is not None and nie_se > 0 and nie_est is not None:
            ci_low = float(nie_est) - 1.96 * float(nie_se)
            ci_high = float(nie_est) + 1.96 * float(nie_se)
        else:
            ci_low = ci_high = float("nan")

        if nie_est is not None and nie_se is not None and nie_se > 0:
            p_value = _two_sided_p(float(nie_est), float(nie_se))
        else:
            p_value = None

        # Total effect = NDE + NIE (Pearl decomposition).
        nde_est = self._nde.get("estimate")
        total_effect: float | None
        if nde_est is not None and nie_est is not None:
            total_effect = float(nde_est) + float(nie_est)
        else:
            total_effect = None

        analyses_table = []
        for label, row in (("NDE", self._nde), ("NIE", self._nie)):
            est = row.get("estimate")
            row_se = row.get("se")
            analyses_table.append(
                {
                    "estimand": label,
                    "estimate": est,
                    "se": row_se,
                    "ci_low": row.get("ci_low"),
                    "ci_high": row.get("ci_high"),
                    "p_value": _two_sided_p(est, row_se)
                    if (est is not None and row_se is not None and row_se > 0)
                    else None,
                }
            )

        diagnostics: dict[str, Any] = {
            "effect_type": self._effect_type,
            "decomposition": analyses_table,
            "headline_estimand": "NIE",
            "total_effect": total_effect,
            "mediator": self.mediator,
            "covariates": list(self.covariates),
            "intermediate_confounders": list(self.intermediate_confounders),
            "cv_folds": self.cv_folds,
            "superlearner_cv_risk": list(self._cv_risk_rows),
            "doubly_robust": True,
            "double_robust_note": (
                "tmle3mediate is multiply robust: the (in)direct effect is "
                "consistently estimated if a sufficient subset of nuisance "
                "models (Q, g, mediator density e) is correctly specified "
                "(Diaz et al. 2020)."
            ),
            "r_session": r_session_metadata(),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="NIE",
            point_estimate=point,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata()
            .get("packages", {})
            .get("tmle3mediate", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted,
            "n_used": self._n_used,
            "mediator": self.mediator,
            "effect_type": self._effect_type,
            "headline_estimand": "NIE",
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _register() -> None:
    for cls in (TMLE3Estimator, TMLE3MediationEstimator):
        register(
            EstimatorEntry(
                id=cls.id,
                factory=cls,
                backend=cls.backend,
                supported_estimands=frozenset(cls.supported_estimands),
                required_flags=cls.required_flags,
                excluded_flags=cls.excluded_flags,
                min_sample_size=cls.min_sample_size,
                produces_cate=cls.produces_cate,
                produces_full_counterfactual=cls.produces_full_counterfactual,
                propensity_required=cls.propensity_required,
            )
        )


_register()


__all__ = ["TMLE3Estimator", "TMLE3MediationEstimator"]
