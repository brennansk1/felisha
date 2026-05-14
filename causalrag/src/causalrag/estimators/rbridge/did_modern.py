"""Modern Difference-in-Differences stack via the R bridge.

This module wraps four state-of-the-art DiD packages reflecting the
post-TWFE consensus surveyed in Roth-Sant'Anna 2025's *Practitioner's
Guide to DiD*:

- ``did::att_gt`` (Callaway-Sant'Anna 2021) — group-time average
  treatment effects under staggered adoption, plus the ``aggte``
  dynamic / overall aggregation.
- ``didimputation::did_imputation`` (Borusyak-Jaravel-Spiess 2024) —
  imputation estimator that uses **only never-treated controls** and is
  efficient under homogeneity.
- ``DIDmultiplegt::did_multiplegt_dyn`` (de Chaisemartin-
  D'Haultfoeuille 2023) — dynamic effects when treatment can switch on
  *and off*; surfaces the negative-weight share as a TWFE diagnostic.
- ``HonestDiD::createSensitivityResults_relativeMagnitudes``
  (Rambachan-Roth 2023) — a *sensitivity* wrapper (not a primary
  estimator) that bounds the post-treatment ATT under partial-identifying
  restrictions on parallel-trends violations, returning the "robust
  CI" contour over the smoothness parameter M-bar.

Headline conventions follow Roth-Sant'Anna 2025:

- The reported ``point_estimate`` for the three primary estimators is
  the **overall average post-treatment ATT** (the dynamic-effect
  aggregation evaluated at "overall" / weighted across event time).
- Pretests on the pre-period coefficients are surfaced **only as a
  diagnostic** (``parallel_trends_pretest_pvalue``). Roth-Sant'Anna and
  Rambachan-Roth both argue against using these as null-hypothesis
  tests; the recommended path is HonestDiD sensitivity. The diagnostic
  is annotated with that caveat in ``pretest_caveat``.
- ``did_design`` is a coarse label of the data structure the wrapper
  detected. Values: ``"staggered_never_treated"``,
  ``"staggered_last_treated"``, ``"two_period"``,
  ``"continuous_staggered"``.
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


# Flag set — STAGGERED_ADOPTION (required), SINGLE_TREATED_UNIT (excluded).
_STAGGERED = getattr(DataFlag, "STAGGERED_ADOPTION", None)
_SINGLE_UNIT = getattr(DataFlag, "SINGLE_TREATED_UNIT", None)

_REQUIRED_FLAGS: frozenset[DataFlag] = (
    frozenset({_STAGGERED}) if _STAGGERED is not None else frozenset()
)
_EXCLUDED_FLAGS: frozenset[DataFlag] = (
    frozenset({_SINGLE_UNIT}) if _SINGLE_UNIT is not None else frozenset()
)


_PRETEST_CAVEAT = (
    "Parallel-trends pretests are under-powered against meaningful "
    "violations; do NOT use this p-value as evidence FOR identification. "
    "Use HonestDiD sensitivity bounds (Rambachan-Roth 2023) instead."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _classify_did_design(
    df: pd.DataFrame,
    onset_col: str,
    subject_col: str,
    time_col: str,
) -> str:
    """Return a coarse design label for diagnostics."""
    onsets = df[onset_col].dropna().unique()
    times = sorted(df[time_col].unique())
    has_never = df[onset_col].isna().any()
    n_cohorts = len(onsets)
    if len(times) == 2:
        return "two_period"
    if n_cohorts <= 1:
        return "two_period" if len(times) <= 2 else "staggered_never_treated"
    if has_never:
        return "staggered_never_treated"
    return "staggered_last_treated"


def _push_panel_to_r(
    df: pd.DataFrame,
    *,
    subject: str,
    time: str,
    onset: str,
    outcome: str,
    covariates: list[str] | None = None,
    r_name: str = "panel_",
) -> None:
    """Push a long-form panel into R as a data.frame ``r_name``.

    Treatment-onset NA is converted to a sentinel of 0 (the convention
    most modern DiD R packages use for "never-treated").
    """
    ro = r_session()
    cols = [subject, time, onset, outcome]
    if covariates:
        cols += list(covariates)
    sub = df[cols].copy()
    # never-treated -> 0 sentinel (CSA, didimputation, DIDmultiplegt
    # all interpret 0/NA equivalently for never-treated).
    sub[onset] = sub[onset].fillna(0).astype(float)
    with converter():
        ro.globalenv[r_name] = ro.conversion.py2rpy(sub)


# ---------------------------------------------------------------------------
# 1. Callaway-Sant'Anna group-time ATTs
# ---------------------------------------------------------------------------


class CallawaySantAnnaDiDEstimator:
    """Callaway-Sant'Anna 2021 staggered DiD via ``did::att_gt`` + ``aggte``.

    Reports the **dynamic / event-study aggregation** of group-time
    ATTs as the headline, with the per-(g, t) ATTs preserved in
    diagnostics (``group_time_atts``) and dynamic effects in
    ``event_study_dynamic_effects``.

    Parameters
    ----------
    subject_id, time, treatment_onset_time, outcome
        Column names in the long panel. ``treatment_onset_time`` is NA
        (or 0) for never-treated units.
    covariates
        Optional list of column names passed to ``xformla``.
    control_group
        ``"nevertreated"`` (default) or ``"notyettreated"``. The latter
        is required if there are no never-treated units.
    anticipation
        Periods of anticipation; default 0.
    """

    id: str = "rbridge.did.callaway_santanna"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATT", "DYNAMIC_ATT")
    required_flags: frozenset[DataFlag] = _REQUIRED_FLAGS
    excluded_flags: frozenset[DataFlag] = _EXCLUDED_FLAGS
    min_sample_size: int = 100
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        subject_id: str,
        time: str,
        treatment_onset_time: str,
        outcome: str,
        *,
        covariates: list[str] | None = None,
        control_group: Literal["nevertreated", "notyettreated"] = "nevertreated",
        anticipation: int = 0,
    ) -> None:
        if control_group not in ("nevertreated", "notyettreated"):
            raise ValueError(
                f"control_group must be 'nevertreated' or 'notyettreated'; got {control_group!r}"
            )
        self.subject_id = subject_id
        self.time = time
        self.treatment_onset_time = treatment_onset_time
        self.outcome = outcome
        self.covariates = list(covariates) if covariates else []
        self.control_group = control_group
        self.anticipation = int(anticipation)
        self._fitted = False
        self._n_used = 0
        self._design: str = "staggered_never_treated"
        self._fit_seconds: float | None = None

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol | None = None
    ) -> "CallawaySantAnnaDiDEstimator":
        require("did")
        keep_cols = [
            self.subject_id,
            self.time,
            self.treatment_onset_time,
            self.outcome,
        ] + self.covariates
        # Don't drop rows where treatment_onset_time is NA — those are
        # the never-treated controls.
        df = data[keep_cols].copy()
        df = df.dropna(subset=[self.subject_id, self.time, self.outcome])
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"CallawaySantAnnaDiDEstimator needs >= {self.min_sample_size} rows; got {self._n_used}"
            )
        self._design = _classify_did_design(
            df, self.treatment_onset_time, self.subject_id, self.time
        )

        _push_panel_to_r(
            df,
            subject=self.subject_id,
            time=self.time,
            onset=self.treatment_onset_time,
            outcome=self.outcome,
            covariates=self.covariates,
            r_name="panel_",
        )

        ro = r_session()
        xformla = (
            f"~{' + '.join(self.covariates)}" if self.covariates else "~1"
        )
        start = time.perf_counter()
        ro.r(
            f"att_ <- did::att_gt("
            f'yname = "{self.outcome}", '
            f'tname = "{self.time}", '
            f'idname = "{self.subject_id}", '
            f'gname = "{self.treatment_onset_time}", '
            f"xformla = {xformla}, "
            f'control_group = "{self.control_group}", '
            f"anticipation = {self.anticipation}, "
            f"data = panel_, allow_unbalanced_panel = TRUE)"
        )
        # Dynamic aggregation (event-study).
        ro.r('agg_dyn_ <- did::aggte(att_, type = "dynamic", na.rm = TRUE)')
        # Overall (single number summary).
        ro.r('agg_simple_ <- did::aggte(att_, type = "simple", na.rm = TRUE)')
        self._fit_seconds = time.perf_counter() - start
        self._fitted = True
        return self

    def estimate(self) -> EstimationResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()

        overall_att = float(list(ro.r("as.numeric(agg_simple_$overall.att)"))[0])
        overall_se = float(list(ro.r("as.numeric(agg_simple_$overall.se)"))[0])
        # 95% CI on the simple aggregate.
        ci_low = overall_att - 1.96 * overall_se
        ci_high = overall_att + 1.96 * overall_se
        # Two-sided z p-value (approximation; agg_simple_ doesn't ship pv directly).
        try:
            from math import erf, sqrt

            z = abs(overall_att / overall_se) if overall_se > 0 else float("inf")
            p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
        except Exception:
            p_value = None

        # Group-time ATTs.
        try:
            gt_g = [float(v) for v in list(ro.r("as.numeric(att_$group)"))]
            gt_t = [float(v) for v in list(ro.r("as.numeric(att_$t)"))]
            gt_att = [float(v) for v in list(ro.r("as.numeric(att_$att)"))]
            gt_se = [float(v) for v in list(ro.r("as.numeric(att_$se)"))]
            group_time_atts = [
                {"group": g, "time": t, "att": a, "se": s}
                for g, t, a, s in zip(gt_g, gt_t, gt_att, gt_se, strict=False)
            ]
        except Exception:
            group_time_atts = []

        # Dynamic effects (event-study).
        try:
            ev = [float(v) for v in list(ro.r("as.numeric(agg_dyn_$egt)"))]
            ev_att = [
                float(v) for v in list(ro.r("as.numeric(agg_dyn_$att.egt)"))
            ]
            ev_se = [
                float(v) for v in list(ro.r("as.numeric(agg_dyn_$se.egt)"))
            ]
            event_study = [
                {"event_time": e, "att": a, "se": s}
                for e, a, s in zip(ev, ev_att, ev_se, strict=False)
            ]
        except Exception:
            event_study = []

        # Pre-trends pretest: did::att_gt returns Wpre in $Wpval (p-value of
        # the joint test of zero pre-treatment ATT(g, t)).
        try:
            wpre = list(ro.r("as.numeric(att_$Wpval)"))
            pretest_p = float(wpre[0]) if wpre else None
        except Exception:
            pretest_p = None

        diagnostics: dict[str, Any] = {
            "did_design": self._design,
            "group_time_atts": group_time_atts,
            "event_study_dynamic_effects": event_study,
            "parallel_trends_pretest_pvalue": pretest_p,
            "pretest_caveat": _PRETEST_CAVEAT,
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "covariates": list(self.covariates),
            "n_groups": len({a["group"] for a in group_time_atts})
            if group_time_atts
            else 0,
            "r_session": r_session_metadata(),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATT",
            point_estimate=overall_att,
            se=overall_se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata().get("packages", {}).get("did", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted,
            "n_used": self._n_used,
            "did_design": self._design,
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# 2. Borusyak-Jaravel-Spiess imputation
# ---------------------------------------------------------------------------


class BJSImputationDiDEstimator:
    """Borusyak-Jaravel-Spiess 2024 imputation DiD via ``didimputation``.

    Efficient under homogeneity; requires never-treated controls.

    Parameters
    ----------
    subject_id, time, treatment_onset_time, outcome
        Column names. ``treatment_onset_time`` MUST contain at least
        one NA (never-treated) value.
    horizon
        Maximum event-time horizon for the dynamic study (default
        ``True`` = all horizons).
    pretrends
        Pre-trend coefficients to compute (default ``True`` = all).
    """

    id: str = "rbridge.did.bjs_imputation"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATT", "DYNAMIC_ATT")
    required_flags: frozenset[DataFlag] = _REQUIRED_FLAGS
    excluded_flags: frozenset[DataFlag] = _EXCLUDED_FLAGS
    min_sample_size: int = 100
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        subject_id: str,
        time: str,
        treatment_onset_time: str,
        outcome: str,
        *,
        horizon: bool = True,
        pretrends: bool = True,
    ) -> None:
        self.subject_id = subject_id
        self.time = time
        self.treatment_onset_time = treatment_onset_time
        self.outcome = outcome
        self.horizon = bool(horizon)
        self.pretrends = bool(pretrends)
        self._fitted = False
        self._n_used = 0
        self._design = "staggered_never_treated"
        self._fit_seconds: float | None = None

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol | None = None
    ) -> "BJSImputationDiDEstimator":
        require("didimputation")
        df = data[
            [self.subject_id, self.time, self.treatment_onset_time, self.outcome]
        ].copy()
        df = df.dropna(subset=[self.subject_id, self.time, self.outcome])
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"BJSImputationDiDEstimator needs >= {self.min_sample_size} rows; got {self._n_used}"
            )
        if not df[self.treatment_onset_time].isna().any():
            # didimputation REQUIRES never-treated controls.
            raise ValueError(
                "BJSImputationDiDEstimator requires never-treated units "
                "(NAs in treatment_onset_time); found none."
            )
        self._design = _classify_did_design(
            df, self.treatment_onset_time, self.subject_id, self.time
        )

        _push_panel_to_r(
            df,
            subject=self.subject_id,
            time=self.time,
            onset=self.treatment_onset_time,
            outcome=self.outcome,
            r_name="panel_",
        )

        ro = r_session()
        horizon_arg = "TRUE" if self.horizon else "FALSE"
        pretrends_arg = "TRUE" if self.pretrends else "FALSE"
        start = time.perf_counter()
        ro.r(
            f"bjs_ <- didimputation::did_imputation("
            f"data = panel_, "
            f'yname = "{self.outcome}", '
            f'gname = "{self.treatment_onset_time}", '
            f'tname = "{self.time}", '
            f'idname = "{self.subject_id}", '
            f"horizon = {horizon_arg}, "
            f"pretrends = {pretrends_arg})"
        )
        self._fit_seconds = time.perf_counter() - start
        self._fitted = True
        return self

    def estimate(self) -> EstimationResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()

        # didimputation returns a long data.frame with cols
        # (term, estimate, std.error, lhs (outcome)). The "overall"
        # row term="treat" carries the aggregate ATT, but the package
        # gives a horizon-wise table by default; we aggregate
        # post-treatment rows for the headline.
        try:
            terms = [str(v) for v in list(ro.r('as.character(bjs_$term)'))]
            est = [float(v) for v in list(ro.r("as.numeric(bjs_$estimate)"))]
            sed = [float(v) for v in list(ro.r("as.numeric(bjs_$std.error)"))]
        except Exception:
            terms, est, sed = [], [], []

        event_study = [
            {"event_time": t, "att": a, "se": s}
            for t, a, s in zip(terms, est, sed, strict=False)
        ]

        # Split post-treatment vs pre-treatment.
        post = [(t, a, s) for t, a, s in zip(terms, est, sed, strict=False)
                if _is_numlike(t) and float(t) >= 0]
        pre = [(t, a, s) for t, a, s in zip(terms, est, sed, strict=False)
               if _is_numlike(t) and float(t) < 0]

        if post:
            atts = np.array([a for _, a, _ in post])
            ses = np.array([s for _, _, s in post])
            # Equal-weighted average of horizon ATTs as headline.
            overall_att = float(atts.mean())
            # Conservative SE: sqrt of mean of variances divided by sqrt(k)
            overall_se = float(np.sqrt((ses**2).mean()) / np.sqrt(len(atts)))
        else:
            overall_att = float("nan")
            overall_se = float("nan")

        if overall_se > 0:
            ci_low = overall_att - 1.96 * overall_se
            ci_high = overall_att + 1.96 * overall_se
            from math import erf, sqrt

            z = abs(overall_att / overall_se)
            p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
        else:
            ci_low = ci_high = float("nan")
            p_value = None

        # Pre-period pretest: chi-square style joint test on pre-period
        # coefficients (we surface a single combined p-value via a
        # simple wald-on-the-mean approximation when SEs are present).
        if pre:
            pre_atts = np.array([a for _, a, _ in pre])
            pre_ses = np.array([s for _, _, s in pre])
            mean_pre = pre_atts.mean()
            mean_pre_se = float(np.sqrt((pre_ses**2).mean()) / np.sqrt(len(pre_atts)))
            if mean_pre_se > 0:
                from math import erf, sqrt

                z_pre = abs(mean_pre / mean_pre_se)
                pretest_p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z_pre / sqrt(2.0))))
            else:
                pretest_p = None
        else:
            pretest_p = None

        diagnostics: dict[str, Any] = {
            "did_design": self._design,
            "event_study_dynamic_effects": event_study,
            "parallel_trends_pretest_pvalue": pretest_p,
            "pretest_caveat": _PRETEST_CAVEAT,
            "horizon": self.horizon,
            "pretrends": self.pretrends,
            "r_session": r_session_metadata(),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATT",
            point_estimate=overall_att,
            se=overall_se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata()
            .get("packages", {})
            .get("didimputation", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted,
            "n_used": self._n_used,
            "did_design": self._design,
        }

    def refute(self) -> dict[str, Any]:
        return {}


def _is_numlike(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 3. de Chaisemartin-D'Haultfoeuille (DIDmultiplegt)
# ---------------------------------------------------------------------------


class DIDMultipleGTEstimator:
    """de Chaisemartin-D'Haultfoeuille 2023 dynamic DiD via ``DIDmultiplegt``.

    Recommended when treatment can **reverse** (switch on and off).
    Reports the negative-weight share — the diagnostic that motivates
    moving off TWFE in the first place.

    Parameters
    ----------
    subject_id, time, treatment, outcome
        Column names. Note: this estimator takes the *contemporaneous
        treatment indicator* (D_it) rather than an onset time.
    effects
        Number of dynamic post-treatment effects to estimate (default 5).
    placebo
        Number of pre-treatment placebos (default 2).
    """

    id: str = "rbridge.did.dch_multiplegt"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATT", "DYNAMIC_ATT")
    required_flags: frozenset[DataFlag] = _REQUIRED_FLAGS
    excluded_flags: frozenset[DataFlag] = _EXCLUDED_FLAGS
    min_sample_size: int = 100
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        subject_id: str,
        time: str,
        treatment: str,
        outcome: str,
        *,
        effects: int = 5,
        placebo: int = 2,
    ) -> None:
        self.subject_id = subject_id
        self.time = time
        self.treatment = treatment
        self.outcome = outcome
        self.effects = int(effects)
        self.placebo = int(placebo)
        self._fitted = False
        self._n_used = 0
        self._design = "continuous_staggered"
        self._fit_seconds: float | None = None

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol | None = None
    ) -> "DIDMultipleGTEstimator":
        require("DIDmultiplegt")
        df = data[[self.subject_id, self.time, self.treatment, self.outcome]].copy()
        df = df.dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"DIDMultipleGTEstimator needs >= {self.min_sample_size} rows; got {self._n_used}"
            )
        # Treatment can be 0/1 each period; this is the "continuous staggered"
        # design where treatment can reverse.
        self._design = "continuous_staggered"

        ro = r_session()
        with converter():
            ro.globalenv["panel_"] = ro.conversion.py2rpy(df)
        start = time.perf_counter()
        ro.r(
            f"dch_ <- DIDmultiplegt::did_multiplegt_dyn("
            f"df = panel_, "
            f'outcome = "{self.outcome}", '
            f'group = "{self.subject_id}", '
            f'time = "{self.time}", '
            f'treatment = "{self.treatment}", '
            f"effects = {self.effects}, "
            f"placebo = {self.placebo}, "
            f"graph_off = TRUE)"
        )
        self._fit_seconds = time.perf_counter() - start
        self._fitted = True
        return self

    def estimate(self) -> EstimationResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()

        # Average post-treatment effect.
        try:
            overall_att = float(list(ro.r("as.numeric(dch_$results$ATE$Estimate)"))[0])
            overall_se = float(list(ro.r("as.numeric(dch_$results$ATE$SE)"))[0])
        except Exception:
            try:
                # Newer versions expose `$Av_tot_eff$Estimate`.
                overall_att = float(
                    list(ro.r("as.numeric(dch_$results$Av_tot_eff$Estimate)"))[0]
                )
                overall_se = float(
                    list(ro.r("as.numeric(dch_$results$Av_tot_eff$SE)"))[0]
                )
            except Exception:
                overall_att = float("nan")
                overall_se = float("nan")

        # Negative-weight share — the headline diagnostic.
        try:
            neg_share = float(list(ro.r("as.numeric(dch_$weights$neg_share)"))[0])
        except Exception:
            try:
                neg_share = float(
                    list(ro.r("as.numeric(dch_$negative_weight_share)"))[0]
                )
            except Exception:
                neg_share = float("nan")

        # Placebo / pretest p-value (joint p of placebo == 0).
        try:
            pretest_p = float(list(ro.r("as.numeric(dch_$results$placebo_pval)"))[0])
        except Exception:
            pretest_p = None

        # Event-study horizon table.
        try:
            ev = [
                float(v) for v in list(ro.r("as.numeric(dch_$results$Effects$Estimate)"))
            ]
            ev_se = [
                float(v) for v in list(ro.r("as.numeric(dch_$results$Effects$SE)"))
            ]
            event_study = [
                {"event_time": i + 1, "att": a, "se": s}
                for i, (a, s) in enumerate(zip(ev, ev_se, strict=False))
            ]
        except Exception:
            event_study = []

        if overall_se and overall_se > 0:
            ci_low = overall_att - 1.96 * overall_se
            ci_high = overall_att + 1.96 * overall_se
            from math import erf, sqrt

            z = abs(overall_att / overall_se)
            p_value = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
        else:
            ci_low = ci_high = float("nan")
            p_value = None

        diagnostics: dict[str, Any] = {
            "did_design": self._design,
            "negative_weight_share": neg_share,
            "event_study_dynamic_effects": event_study,
            "parallel_trends_pretest_pvalue": pretest_p,
            "pretest_caveat": _PRETEST_CAVEAT,
            "effects": self.effects,
            "placebo": self.placebo,
            "r_session": r_session_metadata(),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATT",
            point_estimate=overall_att,
            se=overall_se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata()
            .get("packages", {})
            .get("DIDmultiplegt", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted,
            "n_used": self._n_used,
            "did_design": self._design,
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# 4. Rambachan-Roth HonestDiD sensitivity
# ---------------------------------------------------------------------------


class HonestDiDSensitivity:
    """Rambachan-Roth 2023 sensitivity bounds via ``HonestDiD``.

    *Not a primary estimator* — a wrapper that takes the per-period
    event-study coefficients from a Callaway-Sant'Anna fit and bounds
    the post-treatment ATT under partial-identifying restrictions on
    parallel-trends violations. The "robust CI" widens as M-bar grows;
    the **breakdown M-bar** is the smallest M-bar at which the CI
    covers zero (and the result becomes non-significant).

    Construct by passing a *fitted* ``CallawaySantAnnaDiDEstimator``;
    ``estimate()`` returns a ``EstimationResult`` whose
    ``point_estimate`` echoes the underlying CSA aggregate but whose
    CI is the **robust** Rambachan-Roth bound at the smallest tested
    M-bar (i.e., the tightest defensible CI).

    Parameters
    ----------
    csa
        A *fitted* ``CallawaySantAnnaDiDEstimator``.
    m_bar_grid
        Sequence of relative-magnitude bound parameters M-bar to scan.
        Default ``(0.0, 0.5, 1.0, 1.5, 2.0)``.
    horizons
        Sequence of post-treatment event-times to summarise (default
        ``(0,)`` = average ATT(0)). Passed to HonestDiD as ``l_vec``.
    """

    id: str = "rbridge.did.honest_did"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATT", "DYNAMIC_ATT")
    required_flags: frozenset[DataFlag] = _REQUIRED_FLAGS
    excluded_flags: frozenset[DataFlag] = _EXCLUDED_FLAGS
    min_sample_size: int = 100
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        csa: "CallawaySantAnnaDiDEstimator",
        *,
        m_bar_grid: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
        horizons: tuple[int, ...] = (0,),
    ) -> None:
        self.csa = csa
        self.m_bar_grid = tuple(float(m) for m in m_bar_grid)
        self.horizons = tuple(int(h) for h in horizons)
        self._fitted = False
        self._fit_seconds: float | None = None
        self._sens_rows: list[dict[str, float]] = []
        self._breakdown_M_bar: float | None = None

    def fit(
        self, data: pd.DataFrame | None = None, protocol: StudyProtocol | None = None
    ) -> "HonestDiDSensitivity":
        require("HonestDiD")
        if not self.csa._fitted:  # noqa: SLF001
            raise RuntimeError(
                "HonestDiDSensitivity requires a fitted CallawaySantAnnaDiDEstimator."
            )
        ro = r_session()
        # Build the event-study vector (betahat) + vcov from the fitted CSA.
        # ``did::aggte(..., type='dynamic')`` exposes ``$att.egt`` and ``$V``
        # (inference object). HonestDiD's helper extracts these.
        start = time.perf_counter()
        m_bar_r = "c(" + ", ".join(f"{m}" for m in self.m_bar_grid) + ")"
        l_vec_r = (
            "c(" + ", ".join("1" for _ in self.horizons) + f") / {len(self.horizons)}"
        )
        ro.r(
            f"hd_ <- HonestDiD::createSensitivityResults_relativeMagnitudes("
            f"betahat = agg_dyn_$att.egt, sigma = agg_dyn_$V_analytical, "
            f"numPrePeriods = sum(agg_dyn_$egt < 0), "
            f"numPostPeriods = sum(agg_dyn_$egt >= 0), "
            f"Mbarvec = {m_bar_r}, l_vec = {l_vec_r})"
        )
        self._fit_seconds = time.perf_counter() - start

        # Pull the sensitivity table.
        try:
            mbar = [float(v) for v in list(ro.r("as.numeric(hd_$Mbar)"))]
            lo = [float(v) for v in list(ro.r("as.numeric(hd_$lb)"))]
            hi = [float(v) for v in list(ro.r("as.numeric(hd_$ub)"))]
        except Exception:
            mbar, lo, hi = [], [], []
        self._sens_rows = [
            {"M_bar": m, "ci_low": ll, "ci_high": hh}
            for m, ll, hh in zip(mbar, lo, hi, strict=False)
        ]
        # Breakdown M-bar = smallest M-bar at which CI covers 0.
        self._breakdown_M_bar = None
        for row in self._sens_rows:
            if row["ci_low"] <= 0.0 <= row["ci_high"]:
                self._breakdown_M_bar = row["M_bar"]
                break
        self._fitted = True
        return self

    def estimate(self) -> EstimationResult:
        if not self._fitted:
            raise RuntimeError("Call fit() before estimate().")
        # Echo the CSA point + use the tightest (M-bar = 0) robust CI.
        underlying = self.csa.estimate()
        if self._sens_rows:
            tightest = self._sens_rows[0]
            ci_low = tightest["ci_low"]
            ci_high = tightest["ci_high"]
        else:
            ci_low, ci_high = underlying.ci_low, underlying.ci_high

        diagnostics: dict[str, Any] = dict(underlying.diagnostics)
        diagnostics.update(
            {
                "honest_did_sensitivity_grid": list(self._sens_rows),
                "honest_did_breakdown_M_bar": self._breakdown_M_bar,
                "m_bar_grid": list(self.m_bar_grid),
                "horizons": list(self.horizons),
                "honest_did_note": (
                    "CI shown is the Rambachan-Roth robust CI at the smallest "
                    "tested M-bar. The 'breakdown' M-bar is the smallest M-bar "
                    "at which the result is no longer significant; treat as a "
                    "tipping-point sensitivity headline."
                ),
                "r_session": r_session_metadata(),
            }
        )
        return EstimationResult(
            estimator_id=self.id,
            estimand_class=underlying.estimand_class,
            point_estimate=underlying.point_estimate,
            se=underlying.se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=underlying.p_value,
            n_used=underlying.n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata()
            .get("packages", {})
            .get("HonestDiD", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted,
            "breakdown_M_bar": self._breakdown_M_bar,
            "n_grid": len(self.m_bar_grid),
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _register() -> None:
    for cls in (
        CallawaySantAnnaDiDEstimator,
        BJSImputationDiDEstimator,
        DIDMultipleGTEstimator,
        HonestDiDSensitivity,
    ):
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


__all__ = [
    "BJSImputationDiDEstimator",
    "CallawaySantAnnaDiDEstimator",
    "DIDMultipleGTEstimator",
    "HonestDiDSensitivity",
]
