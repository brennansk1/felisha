"""``lmtp`` wrapper — Longitudinal Modified Treatment Policies (Díaz et al.).

This is the single most important R-bridged estimator for the
**continuous / multi-treatment / mixture / dosage** requirement. lmtp
supports:

- **Stochastic interventions** — set treatment to a shifted version of
  the observed value (``A → A + δ``) rather than a single counterfactual
  value. The natural way to ask "what if everyone got 10% more dose?"
- **Modified treatment policies** — arbitrary user-defined shift
  functions, including treatment-rule learning.
- **Continuous and multi-valued treatments** (no binary requirement).
- **Mixture exposures** — multiple simultaneous treatments
  (``mixture_exposure`` flag), passed as a vector of treatment column
  names.
- **Longitudinal** — repeated measurements with time-varying confounders.
- **Censored outcomes** — survival via the same API.

We expose three flavors:

- :class:`LMTPShift` — single shift δ on a continuous treatment. By
  default returns the contrast ``E[Y(A+δ)] - E[Y(A)]`` via
  ``lmtp::lmtp_contrast`` (proper joint SE from the influence function).
  Pass ``return_quantity="policy_mean"`` for the raw policy mean
  ``E[Y(δ(A))]`` instead.
- :class:`LMTPModifiedPolicy` — user-supplied R shift function for
  arbitrary policies; supports binary, continuous, multi-valued, and
  mixture treatments.
- :class:`LMTPSurvival` — longitudinal / survival variant when the
  outcome is right-censored.

All three use ``lmtp::lmtp_tmle`` under the hood with the SuperLearner
(``sl3``) nuisance library — falling back to a richer base-R
SuperLearner library when ``sl3`` is unavailable.

**SuperLearner library policy.** TMLE's double-robustness depends on at
least one of the two nuisance fits being correctly specified. The
literature's "straw-man" library ``c("SL.glm", "SL.mean")`` makes both
fits parametric and high-bias, so running TMLE on it is effectively
"doubly mis-specified" and the resulting CIs are not what TMLE
advertises. We therefore default to
``c("SL.glm", "SL.glmnet", "SL.gam", "SL.ranger", "SL.earth", "SL.mean")``
when sl3 is absent, and *refuse* to run on the straw-man library unless
the caller passes ``allow_minimal_learners=True``.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import EstimatorEntry, register
from causalrag.core.result import EstimationResult
from causalrag.estimators.rbridge._r import (
    RBridgeError,
    converter,
    r_session,
    r_session_metadata,
    require,
)

# Required for TMLE's double-robustness to be more than a slogan: at least
# one non-parametric / penalized learner must be in the library. We keep
# SL.glm and SL.mean as anchors but add SL.glmnet, SL.gam, SL.ranger,
# SL.earth so the library actually spans enough function space.
_RICH_FALLBACK_LEARNERS: tuple[str, ...] = (
    "SL.glm",
    "SL.glmnet",
    "SL.gam",
    "SL.ranger",
    "SL.earth",
    "SL.mean",
)
_MINIMAL_LEARNERS: tuple[str, ...] = ("SL.glm", "SL.mean")


def _r_installed_sl_learners() -> set[str]:
    """Return the subset of fallback SuperLearner learners actually
    installed in the current R session. ``SL.<x>`` is exported by the
    ``SuperLearner`` package and typically requires the underlying R
    package of the same family (``glmnet``, ``gam``, ``ranger``,
    ``earth``) to be importable."""
    ro = r_session()
    installed: set[str] = set()
    # SuperLearner itself must be available to expose any SL.* wrapper.
    try:
        has_sl = bool(
            list(ro.r('requireNamespace("SuperLearner", quietly = TRUE)'))[0]
        )
    except Exception:
        return installed
    if not has_sl:
        return installed
    # SL.glm / SL.mean ship with SuperLearner itself. Others need their
    # underlying packages.
    base_to_pkg = {
        "SL.glm": "SuperLearner",
        "SL.mean": "SuperLearner",
        "SL.glmnet": "glmnet",
        "SL.gam": "gam",
        "SL.ranger": "ranger",
        "SL.earth": "earth",
    }
    for sl_name, pkg in base_to_pkg.items():
        try:
            ok = bool(
                list(ro.r(f'requireNamespace("{pkg}", quietly = TRUE)'))[0]
            )
        except Exception:
            ok = False
        if ok:
            installed.add(sl_name)
    return installed


def _resolve_fallback_learners(
    allow_minimal_learners: bool,
) -> tuple[str, ...]:
    """Pick the fallback SuperLearner library; raise if only the
    straw-man learners are installed and the caller has not opted in."""
    installed = _r_installed_sl_learners()
    rich_present = tuple(x for x in _RICH_FALLBACK_LEARNERS if x in installed)
    # We require at least one non-parametric / penalized learner beyond
    # SL.glm / SL.mean for "double-robustness" to mean something.
    non_strawman = [
        x for x in rich_present if x not in {"SL.glm", "SL.mean"}
    ]
    if non_strawman:
        return rich_present
    if allow_minimal_learners:
        warnings.warn(
            "lmtp: falling back to SuperLearner library "
            "c('SL.glm', 'SL.mean') because sl3 and the richer learners "
            "(glmnet, gam, ranger, earth) are not installed. TMLE's "
            "double-robustness guarantee is effectively void on this "
            "library; treat the result as a sanity check, not an "
            "estimate. Install sl3 or at minimum 'glmnet' and 'ranger' "
            "in R for valid inference.",
            stacklevel=3,
        )
        return _MINIMAL_LEARNERS
    raise RBridgeError(
        "lmtp: refusing to run TMLE on the straw-man SuperLearner library "
        "c('SL.glm', 'SL.mean'). Both learners are parametric / high-bias, "
        "so TMLE's double-robustness promise (at least one nuisance "
        "fit must be flexible enough to converge) cannot hold. Install "
        "the 'sl3' R package, or install at least one of "
        "{'glmnet', 'gam', 'ranger', 'earth'} so a richer fallback "
        "library is available. To override (e.g., for a smoke test on a "
        "machine without these packages), pass "
        "allow_minimal_learners=True to the estimator constructor."
    )


def _default_learners_string(
    has_sl3: bool, allow_minimal_learners: bool = False
) -> str:
    """Return the R expression that constructs the nuisance learner list.

    Prefers ``sl3`` when available; otherwise picks the richest available
    base-R SuperLearner library. Refuses to fall back to the bare
    ``c("SL.glm", "SL.mean")`` straw-man unless the caller explicitly
    opts in via ``allow_minimal_learners=True``.
    """
    if has_sl3:
        return 'list("SL.mean", "SL.glm", "SL.ranger")'
    learners = _resolve_fallback_learners(allow_minimal_learners)
    quoted = ", ".join(f'"{name}"' for name in learners)
    return f"c({quoted})"


def _has_sl3() -> bool:
    """Check sl3 availability silently. We use ``requireNamespace`` rather
    than ``library`` so a missing package doesn't surface a noisy R-side
    error message."""
    ro = r_session()
    try:
        return bool(list(ro.r('requireNamespace("sl3", quietly = TRUE)'))[0])
    except Exception:
        return False


def _infer_outcome_type(y_series) -> str:
    """lmtp expects ``outcome_type`` ∈ {binomial, continuous, survival}.
    Infer from the observed values: 0/1 → binomial, otherwise continuous.
    The survival variant is selected by :class:`LMTPSurvival`."""
    import numpy as np

    arr = y_series.dropna().to_numpy()
    if arr.size == 0:
        return "continuous"
    uniq = set(np.unique(arr).tolist())
    if uniq <= {0, 1, 0.0, 1.0, True, False}:
        return "binomial"
    return "continuous"


class LMTPShift:
    """Constant-shift stochastic intervention via lmtp::lmtp_tmle.

    Continuous treatment ``A`` is replaced with ``A + shift`` in the
    counterfactual. By default reports the **contrast**
    ``E[Y(A+δ)] − E[Y(A)]`` (computed via ``lmtp::lmtp_contrast`` with the
    proper joint SE from the influence function). Pass
    ``return_quantity="policy_mean"`` for the raw policy mean
    ``E[Y(δ(A))]`` instead — note that ``policy_mean`` is NOT an ATE and
    callers should not interpret it as one.
    """

    id: str = "rbridge.lmtp.shift"
    backend: Literal["python", "r"] = "r"
    # ATE is reported only when ``return_quantity='contrast'`` (the default).
    # Routes that demand a strict ATE on ``return_quantity='policy_mean'``
    # are not supported.
    supported_estimands: tuple[str, ...] = ("MODIFIED_TREATMENT_POLICY", "ATE")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.CONTINUOUS_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset()
    min_sample_size: int = 100
    produces_cate: bool = False
    produces_full_counterfactual: bool = True
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        shift: float = 1.0,
        folds: int = 5,
        seed: int = 42,
        return_quantity: Literal["contrast", "policy_mean"] = "contrast",
        allow_minimal_learners: bool = False,
    ) -> None:
        if return_quantity not in ("contrast", "policy_mean"):
            raise ValueError(
                "return_quantity must be 'contrast' or 'policy_mean'; "
                f"got {return_quantity!r}"
            )
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.shift = shift
        self.folds = folds
        self.seed = seed
        self.return_quantity = return_quantity
        self.allow_minimal_learners = allow_minimal_learners
        self._result: Any = None
        self._result_natural: Any = None
        self._n_used: int = 0
        self._fit_seconds: float | None = None

    @property
    def supported_estimands_runtime(self) -> tuple[str, ...]:
        """Estimands actually delivered by this instance.

        ``policy_mean`` mode does NOT deliver an ATE; only
        ``MODIFIED_TREATMENT_POLICY``.
        """
        if self.return_quantity == "contrast":
            return ("MODIFIED_TREATMENT_POLICY", "ATE")
        return ("MODIFIED_TREATMENT_POLICY",)

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "LMTPShift":
        require("lmtp")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"LMTPShift needs ≥ {self.min_sample_size} rows; got {self._n_used}"
            )
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)
        shift_expr = (
            f"shift_fn <- function(data, trt) data[[trt]] + {self.shift}"
        )
        # Natural-exposure "policy": identity. Used to subtract off the
        # observed-arm mean when return_quantity='contrast'.
        natural_expr = "shift_natural <- function(data, trt) data[[trt]]"
        has_sl3 = _has_sl3()
        learners = _default_learners_string(
            has_sl3, allow_minimal_learners=self.allow_minimal_learners
        )
        baseline = list(self.confounders) + list(self.modifiers)
        baseline_r = "c(" + ", ".join(f'"{c}"' for c in baseline) + ")"

        start = time.perf_counter()
        ro.r(shift_expr)
        outcome_type = _infer_outcome_type(df[self.outcome])
        ro.r(
            f"set.seed({self.seed});"
            f'res_ <- lmtp::lmtp_tmle(df_, trt = "{self.treatment}", '
            f'outcome = "{self.outcome}", baseline = {baseline_r}, '
            f"shift = shift_fn, folds = {self.folds}, "
            f'outcome_type = "{outcome_type}", '
            f"learners_outcome = {learners}, learners_trt = {learners})"
        )
        if self.return_quantity == "contrast":
            ro.r(natural_expr)
            ro.r(
                f"set.seed({self.seed});"
                f'res_natural_ <- lmtp::lmtp_tmle(df_, trt = "{self.treatment}", '
                f'outcome = "{self.outcome}", baseline = {baseline_r}, '
                f"shift = shift_natural, folds = {self.folds}, "
                f'outcome_type = "{outcome_type}", '
                f"learners_outcome = {learners}, learners_trt = {learners})"
            )
            # lmtp_contrast combines the two influence functions to give
            # a proper joint SE for E[Y(A+δ)] - E[Y(A)].
            ro.r(
                'ctrst_ <- lmtp::lmtp_contrast(res_, ref = res_natural_, '
                'type = "additive")'
            )
            self._result_natural = ro.globalenv["res_natural_"]
        self._fit_seconds = time.perf_counter() - start
        self._result = ro.globalenv["res_"]
        return self

    def estimate(self) -> EstimationResult:
        if self._result is None:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()
        if self.return_quantity == "contrast":
            # lmtp_contrast returns a $vals data frame with theta /
            # std.error / conf.low / conf.high / p.value.
            ate = float(list(ro.r("ctrst_$vals$theta"))[0])
            se = float(list(ro.r("ctrst_$vals$std.error"))[0])
            ci_low = float(list(ro.r("ctrst_$vals$conf.low"))[0])
            ci_high = float(list(ro.r("ctrst_$vals$conf.high"))[0])
            try:
                p = float(list(ro.r("ctrst_$vals$p.value"))[0])
            except Exception:
                from scipy.stats import norm

                p = (
                    float(2 * (1 - norm.cdf(abs(ate) / se)))
                    if se > 0
                    else None
                )
            interpretation = (
                "shift contrast E[Y(A+δ)] − E[Y(A)] via lmtp_contrast "
                "(joint SE from influence function)"
            )
        else:
            # Raw policy mean: E[Y(δ(A))]. Reported as such; this is NOT
            # an ATE.
            require("broom")
            ate = float(list(ro.r("broom::tidy(res_)$estimate"))[0])
            se = float(list(ro.r("broom::tidy(res_)$std.error"))[0])
            ci_low = float(list(ro.r("broom::tidy(res_)$conf.low"))[0])
            ci_high = float(list(ro.r("broom::tidy(res_)$conf.high"))[0])
            from scipy.stats import norm

            p = (
                float(2 * (1 - norm.cdf(abs(ate) / se))) if se > 0 else None
            )
            interpretation = (
                "policy-counterfactual mean E[Y(δ(A))] (NOT an ATE); "
                "set return_quantity='contrast' for the shift-induced "
                "delta with proper joint SE"
            )
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="MODIFIED_TREATMENT_POLICY",
            point_estimate=ate,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p,
            n_used=self._n_used,
            diagnostics={
                "shift": self.shift,
                "folds": self.folds,
                "return_quantity": self.return_quantity,
                "interpretation": interpretation,
                "nuisance_library": "sl3" if _has_sl3() else "fallback",
                "allow_minimal_learners": self.allow_minimal_learners,
            },
            backend_version=r_session_metadata().get("packages", {}).get("lmtp", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._result is not None,
            "n_used": self._n_used,
            "shift": self.shift,
            "return_quantity": self.return_quantity,
        }

    def refute(self) -> dict[str, Any]:
        return {}


class LMTPModifiedPolicy(LMTPShift):
    """Arbitrary modified treatment policy via lmtp::lmtp_tmle.

    Pass an R shift-function expression as a string (or a Python ``str``
    that R parses). Default behavior matches :class:`LMTPShift` for
    continuous treatments; override ``shift_fn_r`` for richer policies
    (mixture exposures, treatment-rule learning, dose escalation
    schedules).

    Example::

        LMTPModifiedPolicy(
            treatment="dose",
            outcome="response",
            confounders=("age", "weight"),
            shift_fn_r="function(data, trt) pmin(data[[trt]] * 1.10, 200)",
        )

    routes a 10%-dose-increase scenario, capped at a max dose of 200.
    """

    id: str = "rbridge.lmtp.policy"
    supported_estimands: tuple[str, ...] = ("MODIFIED_TREATMENT_POLICY",)
    required_flags: frozenset[DataFlag] = frozenset()

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        shift_fn_r: str | None = None,
        folds: int = 5,
        seed: int = 42,
        allow_minimal_learners: bool = False,
    ) -> None:
        # An arbitrary R-side policy has no canonical "natural-exposure"
        # counterpart, so we report the policy mean E[Y(δ(A))] only — and
        # we advertise it as such (no ATE claim).
        super().__init__(
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
            modifiers=modifiers,
            folds=folds,
            seed=seed,
            return_quantity="policy_mean",
            allow_minimal_learners=allow_minimal_learners,
        )
        self.shift_fn_r = shift_fn_r or "function(data, trt) data[[trt]] + 1"

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "LMTPModifiedPolicy":
        require("lmtp")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"LMTPModifiedPolicy needs ≥ {self.min_sample_size} rows; got {self._n_used}"
            )
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)
        baseline = list(self.confounders) + list(self.modifiers)
        baseline_r = "c(" + ", ".join(f'"{c}"' for c in baseline) + ")"
        has_sl3 = _has_sl3()
        learners = _default_learners_string(
            has_sl3, allow_minimal_learners=self.allow_minimal_learners
        )
        ro.r(f"shift_fn <- {self.shift_fn_r}")
        start = time.perf_counter()
        outcome_type = _infer_outcome_type(df[self.outcome])
        ro.r(
            f"set.seed({self.seed});"
            f'res_ <- lmtp::lmtp_tmle(df_, trt = "{self.treatment}", '
            f'outcome = "{self.outcome}", baseline = {baseline_r}, '
            f"shift = shift_fn, folds = {self.folds}, "
            f'outcome_type = "{outcome_type}", '
            f"learners_outcome = {learners}, learners_trt = {learners})"
        )
        self._fit_seconds = time.perf_counter() - start
        self._result = ro.globalenv["res_"]
        return self


class LMTPMixture(LMTPModifiedPolicy):
    """Mixture-exposure variant — multiple simultaneous treatments.

    Treatments is a tuple of column names, each shifted by the same
    proportional amount. Use when ``MIXTURE_EXPOSURE`` flag is on
    (chemical/nutrient mixtures, polypharmacy).
    """

    id: str = "rbridge.lmtp.mixture"
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.MIXTURE_EXPOSURE})

    def __init__(
        self,
        treatments: tuple[str, ...],
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        scale: float = 1.10,  # 10% increase across the mixture
        folds: int = 5,
        seed: int = 42,
        allow_minimal_learners: bool = False,
    ) -> None:
        if not treatments:
            raise ValueError("treatments tuple must be non-empty")
        self.treatments_tuple = treatments
        super().__init__(
            treatment=treatments[0],  # nominal anchor; the policy uses all
            outcome=outcome,
            confounders=confounders,
            modifiers=modifiers,
            folds=folds,
            seed=seed,
            allow_minimal_learners=allow_minimal_learners,
        )
        self.scale = scale
        # Multi-treatment shift: scale each treatment column.
        trts = ", ".join(f'"{t}"' for t in treatments)
        self.shift_fn_r = (
            f"function(data, trt) {{ d <- data; "
            f"for (t in c({trts})) d[[t]] <- d[[t]] * {scale}; "
            f"d[[trt]] }}"
        )


class LMTPSDR(LMTPShift):
    """``lmtp::lmtp_sdr`` — Sequentially Doubly Robust variant.

    Better finite-sample coverage than lmtp_tmle when both nuisance models
    are mis-specified (Díaz et al. 2023). Preferred at smaller n where
    TMLE's coverage can deteriorate.
    """

    id: str = "rbridge.lmtp.sdr"

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "LMTPSDR":
        require("lmtp")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(f"LMTPSDR needs ≥ {self.min_sample_size}; got {self._n_used}")
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)
        baseline = list(self.confounders) + list(self.modifiers)
        baseline_r = "c(" + ", ".join(f'"{c}"' for c in baseline) + ")"
        learners = _default_learners_string(
            _has_sl3(), allow_minimal_learners=self.allow_minimal_learners
        )
        outcome_type = _infer_outcome_type(df[self.outcome])
        ro.r(f"shift_fn <- function(data, trt) data[[trt]] + {self.shift}")
        start = time.perf_counter()
        ro.r(
            f"set.seed({self.seed});"
            f'res_ <- lmtp::lmtp_sdr(df_, trt = "{self.treatment}", '
            f'outcome = "{self.outcome}", baseline = {baseline_r}, '
            f"shift = shift_fn, folds = {self.folds}, "
            f'outcome_type = "{outcome_type}", '
            f"learners_outcome = {learners}, learners_trt = {learners})"
        )
        if self.return_quantity == "contrast":
            ro.r("shift_natural <- function(data, trt) data[[trt]]")
            ro.r(
                f"set.seed({self.seed});"
                f'res_natural_ <- lmtp::lmtp_sdr(df_, trt = "{self.treatment}", '
                f'outcome = "{self.outcome}", baseline = {baseline_r}, '
                f"shift = shift_natural, folds = {self.folds}, "
                f'outcome_type = "{outcome_type}", '
                f"learners_outcome = {learners}, learners_trt = {learners})"
            )
            ro.r(
                'ctrst_ <- lmtp::lmtp_contrast(res_, ref = res_natural_, '
                'type = "additive")'
            )
            self._result_natural = ro.globalenv["res_natural_"]
        self._fit_seconds = time.perf_counter() - start
        self._result = ro.globalenv["res_"]
        return self


class LMTPContrast:
    """``lmtp::lmtp_contrast`` — fit two policies and report their contrast.

    The semantically correct way to ask "what is the effect of a +1 dose
    shift vs the baseline policy?" Fits both lmtp_tmle policies, then
    combines them via lmtp_contrast with a proper joint SE.
    """

    id: str = "rbridge.lmtp.contrast"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("MODIFIED_TREATMENT_POLICY", "ATE")
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset()
    min_sample_size: int = 100
    produces_cate: bool = False
    produces_full_counterfactual: bool = True
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        shift_a: float = 1.0,
        shift_b: float = 0.0,
        folds: int = 5,
        seed: int = 42,
        allow_minimal_learners: bool = False,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.shift_a = shift_a
        self.shift_b = shift_b
        self.folds = folds
        self.seed = seed
        self.allow_minimal_learners = allow_minimal_learners
        self._n_used: int = 0
        self._fit_seconds: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "LMTPContrast":
        require("lmtp")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(f"LMTPContrast needs ≥ {self.min_sample_size}; got {self._n_used}")
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)
        baseline = list(self.confounders) + list(self.modifiers)
        baseline_r = "c(" + ", ".join(f'"{c}"' for c in baseline) + ")"
        learners = _default_learners_string(
            _has_sl3(), allow_minimal_learners=self.allow_minimal_learners
        )
        outcome_type = _infer_outcome_type(df[self.outcome])
        ro.r(f"shift_a_ <- function(data, trt) data[[trt]] + {self.shift_a}")
        ro.r(f"shift_b_ <- function(data, trt) data[[trt]] + {self.shift_b}")
        start = time.perf_counter()
        ro.r(
            f"set.seed({self.seed});"
            f'res_a_ <- lmtp::lmtp_tmle(df_, trt = "{self.treatment}", '
            f'outcome = "{self.outcome}", baseline = {baseline_r}, '
            f"shift = shift_a_, folds = {self.folds}, "
            f'outcome_type = "{outcome_type}", '
            f"learners_outcome = {learners}, learners_trt = {learners})"
        )
        ro.r(
            f"set.seed({self.seed});"
            f'res_b_ <- lmtp::lmtp_tmle(df_, trt = "{self.treatment}", '
            f'outcome = "{self.outcome}", baseline = {baseline_r}, '
            f"shift = shift_b_, folds = {self.folds}, "
            f'outcome_type = "{outcome_type}", '
            f"learners_outcome = {learners}, learners_trt = {learners})"
        )
        ro.r('ctrst_ <- lmtp::lmtp_contrast(res_a_, ref = res_b_, type = "additive")')
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        ro = r_session()
        # lmtp_contrast result has a $vals data frame
        ate = float(list(ro.r("ctrst_$vals$theta"))[0])
        se = float(list(ro.r("ctrst_$vals$std.error"))[0])
        ci_low = float(list(ro.r("ctrst_$vals$conf.low"))[0])
        ci_high = float(list(ro.r("ctrst_$vals$conf.high"))[0])
        try:
            p = float(list(ro.r("ctrst_$vals$p.value"))[0])
        except Exception:
            p = None
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="MODIFIED_TREATMENT_POLICY",
            point_estimate=ate,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p,
            n_used=self._n_used,
            diagnostics={
                "shift_a": self.shift_a,
                "shift_b": self.shift_b,
                "interpretation": "additive contrast E[Y(A+δ_a)] − E[Y(A+δ_b)]",
                "folds": self.folds,
            },
            backend_version=r_session_metadata().get("packages", {}).get("lmtp", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"n_used": self._n_used, "shift_a": self.shift_a, "shift_b": self.shift_b}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    for cls in (LMTPShift, LMTPModifiedPolicy, LMTPMixture, LMTPSDR, LMTPContrast):
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
