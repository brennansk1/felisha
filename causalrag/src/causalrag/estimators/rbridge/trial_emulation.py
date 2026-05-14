"""Clinical target-trial emulation via the R ``TrialEmulation`` package.

This wrapper implements the Rezvani et al. (2024) target-trial-emulation
workflow on observational longitudinal data. The pipeline emulates a
hypothetical randomised trial from per-(subject, period) records by
constructing three sister analyses on the same long-format input:

- **Intention-to-treat (ITT)** — hazard ratio under the baseline-assigned
  treatment strategy, ignoring later deviations. This is what a real
  RCT's ITT analysis reports.
- **Per-protocol (PP)** — hazard ratio under *sustained adherence* to
  the assigned strategy. Implemented via cloning-censoring-weighting
  (CCW): each eligible subject is cloned across treatment strategies,
  clones are artificially censored when they deviate from their assigned
  strategy, and inverse-probability-of-censoring weights (IPCW) restore
  baseline-conditional exchangeability.
- **As-treated** — hazard ratio under the observed (time-varying)
  treatment trajectory, IPTW-weighted by a time-varying propensity model.
  Diagnostic only; vulnerable to time-varying confounding bias.

The R package ``TrialEmulation`` (Hernán & Robins lineage; CRAN) exposes
``data_preparation`` to expand the long panel into the trial-emulation
data structure and ``trial_msm`` (marginal structural model) to fit the
pooled-logistic / Cox MSM with IPTW + IPCW weights. The wrapper drives
both, then surfaces ITT + PP + as-treated as a *triple-row diagnostic*
under one ``EstimationResult`` (the headline ``point_estimate`` is the
**PP log-hazard contrast**, i.e. the protocol-relevant estimand).

Reference
---------
Rezvani, R., et al. (2024). *Target trial emulation using
observational data: a practical guide and an R implementation
(TrialEmulation)*. Journal of Clinical Epidemiology.

Notes
-----
The implementation is best-effort against the public ``TrialEmulation``
API; numeric pulls are wrapped in ``try`` blocks so the diagnostics dict
degrades gracefully on minor signature drift (the package's public API
is stable but several optional fields have moved between versions).
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


_PANEL = getattr(DataFlag, "PANEL_STRUCTURE", None)
_REQUIRED_FLAGS: frozenset[DataFlag] = (
    frozenset({_PANEL}) if _PANEL is not None else frozenset()
)


def _two_sided_p(estimate: float, se: float) -> float | None:
    """Two-sided z-test p-value; ``None`` if SE is non-positive."""
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


class TrialEmulationEstimator:
    """Clinical target-trial emulation with PP weighting (Rezvani 2024).

    Input shape (long format, per (subject, period)):
      - ``subject_id``, ``period``, ``eligible`` (1 at the baseline
        period; 0 otherwise), ``treatment_strategy_id``,
        ``treatment_received_this_period`` (0/1), ``outcome`` (0/1
        event indicator within the period), ``censoring_indicator``
        (1 at administrative censor), plus optional time-varying
        confounders.

    Three sister analyses are estimated from the same long panel:

      - **ITT** — pooled-logistic / Cox hazard ratio under the
        baseline-assigned strategy, no per-protocol censoring.
      - **PP** — hazard ratio under sustained adherence, via
        cloning-censoring-weighting. IPCW weights are fit by
        ``TrialEmulation`` from the deviation indicator on the
        time-varying confounders supplied in ``switch_n_cov`` /
        ``switch_d_cov``.
      - **As-treated** — hazard ratio under the observed
        time-varying treatment, IPTW-weighted.

    The headline ``point_estimate`` is the **PP log-hazard contrast**;
    ITT and as-treated are surfaced in ``diagnostics`` as a
    triple-row table.

    Parameters
    ----------
    subject_id, period, eligible, treatment_strategy_id,
    treatment_received_this_period, outcome, censoring_indicator
        Column names in the long input frame.
    time_varying_confounders
        Optional list of time-varying confounder column names. Used by
        ``TrialEmulation`` to fit the IPCW (and IPTW) models.
    baseline_covariates
        Optional list of baseline covariate column names included as
        adjustment terms in the marginal structural model.
    estimand
        ``"hazard_ratio"`` (default) or ``"RMST_CONTRAST"`` for an RMST
        contrast computed via the MSM survival curves.
    """

    id: str = "rbridge.trial_emulation"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("RMST_CONTRAST", "ATE", "ATT")
    required_flags: frozenset[DataFlag] = _REQUIRED_FLAGS
    excluded_flags: frozenset[DataFlag] = frozenset()
    min_sample_size: int = 500
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    def __init__(
        self,
        subject_id: str,
        period: str,
        eligible: str,
        treatment_strategy_id: str,
        treatment_received_this_period: str,
        outcome: str,
        censoring_indicator: str,
        *,
        time_varying_confounders: list[str] | None = None,
        baseline_covariates: list[str] | None = None,
        estimand: Literal["hazard_ratio", "RMST_CONTRAST"] = "hazard_ratio",
    ) -> None:
        if estimand not in ("hazard_ratio", "RMST_CONTRAST"):
            raise ValueError(
                f"estimand must be 'hazard_ratio' or 'RMST_CONTRAST'; got {estimand!r}"
            )
        self.subject_id = subject_id
        self.period = period
        self.eligible = eligible
        self.treatment_strategy_id = treatment_strategy_id
        self.treatment_received_this_period = treatment_received_this_period
        self.outcome = outcome
        self.censoring_indicator = censoring_indicator
        self.time_varying_confounders = list(time_varying_confounders or [])
        self.baseline_covariates = list(baseline_covariates or [])
        self.estimand = estimand

        self._fitted = False
        self._n_used = 0
        self._n_subjects = 0
        self._fit_seconds: float | None = None
        # Cached estimates for estimate() to read back out of R.
        self._itt: dict[str, float | None] = {}
        self._pp: dict[str, float | None] = {}
        self._as_treated: dict[str, float | None] = {}

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _push_to_r(self, df: pd.DataFrame) -> None:
        ro = r_session()
        with converter():
            ro.globalenv["te_panel_"] = ro.conversion.py2rpy(df)

    def _required_cols(self) -> list[str]:
        return [
            self.subject_id,
            self.period,
            self.eligible,
            self.treatment_strategy_id,
            self.treatment_received_this_period,
            self.outcome,
            self.censoring_indicator,
        ]

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol | None = None
    ) -> "TrialEmulationEstimator":
        require("TrialEmulation")

        cols = self._required_cols() + list(self.time_varying_confounders) + list(
            self.baseline_covariates
        )
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(
                f"Input is missing required columns for TrialEmulation: {missing}"
            )

        df = data[cols].copy().dropna(
            subset=[
                self.subject_id,
                self.period,
                self.outcome,
                self.censoring_indicator,
            ]
        )
        self._n_used = len(df)
        self._n_subjects = int(df[self.subject_id].nunique())
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"TrialEmulationEstimator needs >= {self.min_sample_size} rows; "
                f"got {self._n_used}"
            )

        self._push_to_r(df)
        ro = r_session()

        # Build covariate R-formula fragments.
        switch_cov = (
            "c(" + ", ".join(f'"{c}"' for c in self.time_varying_confounders) + ")"
            if self.time_varying_confounders
            else "NULL"
        )
        outcome_cov = (
            "c("
            + ", ".join(
                f'"{c}"'
                for c in (self.baseline_covariates + self.time_varying_confounders)
            )
            + ")"
            if (self.baseline_covariates or self.time_varying_confounders)
            else "NULL"
        )

        start = time.perf_counter()

        # Per-protocol analysis with cloning-censoring-weighting.
        ro.r(
            f"te_prep_pp_ <- TrialEmulation::data_preparation("
            f"data = te_panel_, "
            f'id = "{self.subject_id}", '
            f'period = "{self.period}", '
            f'eligible = "{self.eligible}", '
            f'treatment = "{self.treatment_received_this_period}", '
            f'outcome = "{self.outcome}", '
            f'censored = "{self.censoring_indicator}", '
            f'estimand_type = "PP", '
            f"switch_n_cov = {switch_cov}, "
            f"switch_d_cov = {switch_cov}, "
            f"outcome_cov = {outcome_cov})"
        )
        ro.r(
            "te_pp_ <- TrialEmulation::trial_msm("
            "data = te_prep_pp_, "
            'estimand_type = "PP", '
            "use_weight = TRUE, "
            "include_followup_time = TRUE, "
            "include_trial_period = TRUE)"
        )

        # Intention-to-treat — no per-protocol censoring; weight only for
        # administrative censoring (if any).
        ro.r(
            f"te_prep_itt_ <- TrialEmulation::data_preparation("
            f"data = te_panel_, "
            f'id = "{self.subject_id}", '
            f'period = "{self.period}", '
            f'eligible = "{self.eligible}", '
            f'treatment = "{self.treatment_received_this_period}", '
            f'outcome = "{self.outcome}", '
            f'censored = "{self.censoring_indicator}", '
            f'estimand_type = "ITT", '
            f"outcome_cov = {outcome_cov})"
        )
        ro.r(
            "te_itt_ <- TrialEmulation::trial_msm("
            "data = te_prep_itt_, "
            'estimand_type = "ITT", '
            "use_weight = FALSE, "
            "include_followup_time = TRUE, "
            "include_trial_period = TRUE)"
        )

        # As-treated — observed time-varying treatment, IPTW-weighted.
        ro.r(
            f"te_prep_at_ <- TrialEmulation::data_preparation("
            f"data = te_panel_, "
            f'id = "{self.subject_id}", '
            f'period = "{self.period}", '
            f'eligible = "{self.eligible}", '
            f'treatment = "{self.treatment_received_this_period}", '
            f'outcome = "{self.outcome}", '
            f'censored = "{self.censoring_indicator}", '
            f'estimand_type = "As-Treated", '
            f"switch_n_cov = {switch_cov}, "
            f"switch_d_cov = {switch_cov}, "
            f"outcome_cov = {outcome_cov})"
        )
        ro.r(
            "te_at_ <- TrialEmulation::trial_msm("
            "data = te_prep_at_, "
            'estimand_type = "As-Treated", '
            "use_weight = TRUE, "
            "include_followup_time = TRUE, "
            "include_trial_period = TRUE)"
        )
        self._fit_seconds = time.perf_counter() - start

        # Pull treatment coefficients for each MSM. The treatment column
        # name in the fitted MSM is the same string as `treatment_received_this_period`.
        for label, r_obj in (
            ("itt", "te_itt_"),
            ("pp", "te_pp_"),
            ("as_treated", "te_at_"),
        ):
            est = _safe_scalar(
                ro,
                f'as.numeric({r_obj}$robust$summary$estimate[{r_obj}$robust$summary$term == "{self.treatment_received_this_period}"])',
            )
            se = _safe_scalar(
                ro,
                f'as.numeric({r_obj}$robust$summary$robust_se[{r_obj}$robust$summary$term == "{self.treatment_received_this_period}"])',
            )
            # Fallback to non-robust SE if robust pull failed.
            if se is None:
                se = _safe_scalar(
                    ro,
                    f'as.numeric({r_obj}$summary$std.error[{r_obj}$summary$term == "{self.treatment_received_this_period}"])',
                )
            if est is None:
                est = _safe_scalar(
                    ro,
                    f'as.numeric({r_obj}$summary$estimate[{r_obj}$summary$term == "{self.treatment_received_this_period}"])',
                )
            getattr(self, f"_{label}").update({"estimate": est, "se": se})

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # estimate
    # ------------------------------------------------------------------

    def estimate(self) -> EstimationResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before estimate().")

        # Headline = per-protocol log-hazard contrast.
        pp_est = self._pp.get("estimate")
        pp_se = self._pp.get("se")
        point = float(pp_est) if pp_est is not None else float("nan")
        se = float(pp_se) if pp_se is not None else float("nan")

        if se is not None and se > 0 and pp_est is not None:
            ci_low = point - 1.96 * se
            ci_high = point + 1.96 * se
            p_value = _two_sided_p(point, se)
        else:
            ci_low = ci_high = float("nan")
            p_value = None

        analyses_table = []
        for label, row in (
            ("ITT", self._itt),
            ("PP", self._pp),
            ("as_treated", self._as_treated),
        ):
            est = row.get("estimate")
            row_se = row.get("se")
            entry: dict[str, Any] = {
                "analysis": label,
                "log_hazard_ratio": est,
                "se": row_se,
                "hazard_ratio": (
                    float(__import__("math").exp(est)) if est is not None else None
                ),
                "p_value": _two_sided_p(est, row_se)
                if (est is not None and row_se is not None and row_se > 0)
                else None,
            }
            analyses_table.append(entry)

        diagnostics: dict[str, Any] = {
            "analyses": analyses_table,
            "headline_analysis": "PP",
            "n_subjects": self._n_subjects,
            "n_subject_periods": self._n_used,
            "time_varying_confounders": list(self.time_varying_confounders),
            "baseline_covariates": list(self.baseline_covariates),
            "estimand_target": self.estimand,
            "weighting_scheme": "IPCW (cloning-censoring-weighting) for PP; "
            "IPTW for as-treated; unweighted for ITT.",
            "rezvani_2024_ref": "Rezvani et al. 2024 (J Clin Epidemiol)",
            "r_session": r_session_metadata(),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATE",
            point_estimate=point,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata()
            .get("packages", {})
            .get("TrialEmulation", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted,
            "n_used": self._n_used,
            "n_subjects": self._n_subjects,
            "headline_analysis": "PP",
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _register() -> None:
    register(
        EstimatorEntry(
            id=TrialEmulationEstimator.id,
            factory=TrialEmulationEstimator,
            backend=TrialEmulationEstimator.backend,
            supported_estimands=frozenset(TrialEmulationEstimator.supported_estimands),
            required_flags=TrialEmulationEstimator.required_flags,
            excluded_flags=TrialEmulationEstimator.excluded_flags,
            min_sample_size=TrialEmulationEstimator.min_sample_size,
            produces_cate=TrialEmulationEstimator.produces_cate,
            produces_full_counterfactual=TrialEmulationEstimator.produces_full_counterfactual,
            propensity_required=TrialEmulationEstimator.propensity_required,
        )
    )


_register()


__all__ = ["TrialEmulationEstimator"]
