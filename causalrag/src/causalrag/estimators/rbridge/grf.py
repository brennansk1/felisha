"""``grf`` wrappers — Athey-Wager Causal Forest + Causal Survival Forest.

Two estimators self-register here:

- ``rbridge.grf.causal_forest`` — non-linear CATE for continuous outcomes
  via :func:`grf::causal_forest`. Honest splitting; valid CI by default.
  Routed to when CATE is requested with ≥3 modifiers and a continuous
  outcome (alternative to EconML's ``CausalForestDML`` — grf is the
  reference implementation).

- ``rbridge.grf.causal_survival_forest`` — Cui-Athey-Tibshirani 2023
  censored-outcome forest. **The headline reason for the R bridge** —
  no production Python equivalent. Auto-selected when
  ``RIGHT_CENSORED_OUTCOME`` is flagged.

Both wrappers honour the standard :class:`CausalEstimator` Protocol and
emit a fully-typed :class:`EstimationResult` carrying R session
provenance.
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
    RPackageMissing,
    converter,
    r_session,
    r_session_metadata,
    require,
)


def _partial_first_stage_F(
    T: np.ndarray,
    Z: np.ndarray,
    X: np.ndarray | None = None,
) -> tuple[float, int, int]:
    """Compute the partial first-stage F-statistic of T on Z given X.

    Regresses T on [Z, X, 1] (unrestricted) and T on [X, 1] (restricted),
    then returns ``F = ((RSS_r - RSS_u) / q) / (RSS_u / (n - k_u))`` where
    ``q`` is the number of instruments (columns of Z) and ``k_u`` is the
    number of regressors in the unrestricted model (including the intercept).

    Parameters
    ----------
    T : (n,) array
        Endogenous treatment.
    Z : (n,) or (n, q) array
        Instrument(s).
    X : (n, p) array or None
        Exogenous controls (confounders). May be None / empty.

    Returns
    -------
    F : float
        Partial first-stage F-statistic.
    q : int
        Number of instruments.
    n_used : int
        Number of complete rows used.
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    n = T.shape[0]
    if Z.shape[0] != n:
        raise ValueError("Z and T have mismatched lengths")
    if X is None:
        X_mat = np.empty((n, 0), dtype=float)
    else:
        X_mat = np.asarray(X, dtype=float)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(-1, 1)
        if X_mat.shape[0] != n:
            raise ValueError("X and T have mismatched lengths")

    q = Z.shape[1]
    intercept = np.ones((n, 1), dtype=float)

    # Restricted: T ~ X + 1
    R_r = np.hstack([X_mat, intercept])
    # Unrestricted: T ~ Z + X + 1
    R_u = np.hstack([Z, X_mat, intercept])

    # OLS via lstsq
    beta_r, *_ = np.linalg.lstsq(R_r, T, rcond=None)
    resid_r = T - R_r @ beta_r
    rss_r = float(resid_r @ resid_r)

    beta_u, *_ = np.linalg.lstsq(R_u, T, rcond=None)
    resid_u = T - R_u @ beta_u
    rss_u = float(resid_u @ resid_u)

    k_u = R_u.shape[1]
    dof = n - k_u
    if dof <= 0:
        raise ValueError(f"Insufficient degrees of freedom (n={n}, k_u={k_u})")
    if rss_u <= 0:
        return float("inf"), q, n
    F = ((rss_r - rss_u) / q) / (rss_u / dof)
    return float(F), q, n


def _iv_relevance_verdict(F: float) -> str:
    """Classify partial first-stage F by standard weak-IV thresholds.

    - ``strong``   : F >= 23.1 (Olea-Pflueger 5% relative bias, single IV)
    - ``adequate`` : 10 <= F < 23.1 (Staiger-Stock 1997 rule of thumb)
    - ``weak``     : F < 10
    """
    if F >= 23.1:
        return "strong"
    if F >= 10.0:
        return "adequate"
    return "weak"


_IV_WEAK_WARNING = (
    "Partial first-stage F-statistic < 10 (Staiger-Stock 1997 rule of thumb). "
    "The instrument may be weak: LATE point estimates can be biased toward "
    "OLS and confidence-interval coverage can be poor. Consider stronger "
    "instruments or weak-IV-robust inference (e.g. Anderson-Rubin)."
)


def _prep_matrices(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...],
    modifiers: tuple[str, ...],
    event: str | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    cols = [outcome, treatment, *confounders, *modifiers]
    if event:
        cols.append(event)
    df = data[cols].dropna()
    y = df[outcome].astype(float).to_numpy()
    t = df[treatment].astype(float).to_numpy()
    # X passed to grf is the joint set of confounders + modifiers; the forest
    # decides which split where.
    feature_cols = list(confounders) + list(modifiers)
    x = df[feature_cols].astype(float).to_numpy()
    d = df[event].astype(float).to_numpy() if event else None
    return df, y, t, x, d


class GRFCausalForest:
    """Causal Forest via grf::causal_forest (R)."""

    id: str = "rbridge.grf.causal_forest"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATE", "CATE")
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT}
    )
    min_sample_size: int = 200
    produces_cate: bool = True
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        num_trees: int = 2000,
        honest_split: bool = True,
        seed: int = 42,
        alpha: float = 0.05,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.num_trees = num_trees
        self.honest_split = honest_split
        self.seed = seed
        self.alpha = alpha
        self._forest: Any = None
        self._x: np.ndarray | None = None
        self._n_used: int = 0
        self._fit_seconds: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "GRFCausalForest":
        require("grf")
        ro = r_session()
        df, y, t, x, _ = _prep_matrices(data, self.treatment, self.outcome, self.confounders, self.modifiers)
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"GRFCausalForest needs ≥ {self.min_sample_size} rows; got {self._n_used}"
            )
        with converter():
            ro.globalenv["X_"] = ro.conversion.py2rpy(pd.DataFrame(x))
            ro.globalenv["Y_"] = ro.FloatVector(y)
            ro.globalenv["W_"] = ro.FloatVector(t)
        start = time.perf_counter()
        ro.r(
            f"forest_ <- grf::causal_forest("
            f"X = as.matrix(X_), Y = Y_, W = W_, "
            f"num.trees = {self.num_trees}, honesty = {str(self.honest_split).upper()}, "
            f"seed = {self.seed})"
        )
        self._fit_seconds = time.perf_counter() - start
        self._forest = ro.globalenv["forest_"]
        self._x = x
        return self

    def estimate(self) -> EstimationResult:
        if self._forest is None:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()
        # ATE via grf::average_treatment_effect
        ate_row = ro.r("grf::average_treatment_effect(forest_)")
        # rpy2 returns a named numeric vector; positions 0,1 = estimate, std.err
        ate = float(list(ate_row)[0])
        se = float(list(ate_row)[1])
        z = 1.959963984540054
        ci_low = ate - z * se
        ci_high = ate + z * se
        # Two-sided z-test p-value
        from scipy.stats import norm

        p = float(2 * (1 - norm.cdf(abs(ate) / se))) if se > 0 else None

        # Per-row CATE for CATE estimand
        diagnostics: dict[str, Any] = {
            "num_trees": self.num_trees,
            "honest_split": self.honest_split,
            "r_session": r_session_metadata(),
        }
        if self.modifiers:
            tau_hat = np.array(list(ro.r("predict(forest_)$predictions")))
            diagnostics["cate_mean"] = float(np.mean(tau_hat))
            diagnostics["cate_std"] = float(np.std(tau_hat))
            diagnostics["cate_q05"] = float(np.quantile(tau_hat, 0.05))
            diagnostics["cate_q95"] = float(np.quantile(tau_hat, 0.95))

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="CATE" if self.modifiers else "ATE",
            point_estimate=ate,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=r_session_metadata().get("packages", {}).get("grf", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"fitted": self._forest is not None, "n_used": self._n_used}

    def refute(self) -> dict[str, Any]:
        return {}


class GRFCausalSurvivalForest:
    """Causal Survival Forest via grf::causal_survival_forest (R).

    Cui-Athey-Tibshirani 2023. Handles right-censored outcomes natively;
    this is the gold-standard method for survival CATE. Auto-routed when
    ``RIGHT_CENSORED_OUTCOME`` is flagged.

    Required columns:
      - ``outcome`` — observed time (min of event time and censoring time)
      - ``event`` — 1 if event observed, 0 if censored
      - ``treatment`` — binary
      - confounders + modifiers

    Reports the RMST-contrast point estimate at the chosen horizon.
    """

    id: str = "rbridge.grf.causal_survival_forest"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("RMST_CONTRAST", "ATE")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.RIGHT_CENSORED_OUTCOME})
    excluded_flags: frozenset[DataFlag] = frozenset({DataFlag.TIME_VARYING_TREATMENT})
    min_sample_size: int = 200
    produces_cate: bool = True
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        event: str = "event",
        horizon: float | None = None,
        num_trees: int = 2000,
        seed: int = 42,
        alpha: float = 0.05,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.event = event
        self.confounders = confounders
        self.modifiers = modifiers
        self.horizon = horizon
        self.num_trees = num_trees
        self.seed = seed
        self.alpha = alpha
        self._forest: Any = None
        self._n_used: int = 0
        self._fit_seconds: float | None = None
        self._horizon_used: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "GRFCausalSurvivalForest":
        require("grf")
        ro = r_session()
        df, y, t, x, d = _prep_matrices(
            data, self.treatment, self.outcome, self.confounders, self.modifiers, event=self.event
        )
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"CausalSurvivalForest needs ≥ {self.min_sample_size} rows; got {self._n_used}"
            )
        horizon = self.horizon if self.horizon is not None else float(np.quantile(y, 0.75))
        self._horizon_used = horizon
        with converter():
            ro.globalenv["X_"] = ro.conversion.py2rpy(pd.DataFrame(x))
            ro.globalenv["Y_"] = ro.FloatVector(y)
            ro.globalenv["W_"] = ro.FloatVector(t)
            ro.globalenv["D_"] = ro.FloatVector(d)
        start = time.perf_counter()
        ro.r(
            f"csf_ <- grf::causal_survival_forest("
            f"X = as.matrix(X_), Y = Y_, W = W_, D = D_, "
            f"horizon = {horizon}, num.trees = {self.num_trees}, seed = {self.seed})"
        )
        self._fit_seconds = time.perf_counter() - start
        self._forest = ro.globalenv["csf_"]
        return self

    def estimate(self) -> EstimationResult:
        if self._forest is None:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()
        ate_row = ro.r("grf::average_treatment_effect(csf_)")
        ate = float(list(ate_row)[0])
        se = float(list(ate_row)[1])
        z = 1.959963984540054
        from scipy.stats import norm

        p = float(2 * (1 - norm.cdf(abs(ate) / se))) if se > 0 else None
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="RMST_CONTRAST",
            point_estimate=ate,
            se=se,
            ci_low=ate - z * se,
            ci_high=ate + z * se,
            p_value=p,
            n_used=self._n_used,
            diagnostics={
                "horizon": self._horizon_used,
                "num_trees": self.num_trees,
            },
            backend_version=r_session_metadata().get("packages", {}).get("grf", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"fitted": self._forest is not None, "n_used": self._n_used, "horizon": self._horizon_used}

    def refute(self) -> dict[str, Any]:
        return {}


class GRFInstrumentalForest:
    """grf::instrumental_forest — IV-CATE.

    Routes when ``INSTRUMENTAL_CANDIDATE_PRESENT`` is flagged. Reports
    the LATE under the IV relevance + exclusion restriction; relevance
    is empirically checked (and surfaced) by the bridge; exclusion is
    untestable and logged as an analyst-asserted assumption.
    """

    id: str = "rbridge.grf.instrumental_forest"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("LATE", "CATE")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT}
    )
    min_sample_size: int = 500
    produces_cate: bool = True
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        instrument: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        num_trees: int = 2000,
        seed: int = 42,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.instrument = instrument
        self.confounders = confounders
        self.modifiers = modifiers
        self.num_trees = num_trees
        self.seed = seed
        self._forest: Any = None
        self._n_used: int = 0
        self._fit_seconds: float | None = None
        self._iv_relevance_p: float | None = None
        self._iv_first_stage_F: float | None = None
        self._iv_instruments: list[str] = []
        self._iv_relevance_verdict: str | None = None
        self._iv_warning: str | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "GRFInstrumentalForest":
        require("grf")
        ro = r_session()
        cols = [self.outcome, self.treatment, self.instrument, *self.confounders, *self.modifiers]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"InstrumentalForest needs ≥ {self.min_sample_size} rows; got {self._n_used}"
            )
        # Secondary (demoted) rank-correlation signal — kept for back-compat.
        try:
            from scipy.stats import kendalltau
            _, p_rel = kendalltau(df[self.instrument], df[self.treatment])
            self._iv_relevance_p = float(p_rel)
        except Exception:
            pass
        y = df[self.outcome].astype(float).to_numpy()
        t = df[self.treatment].astype(float).to_numpy()
        z = df[self.instrument].astype(float).to_numpy()
        feature_cols = list(self.confounders) + list(self.modifiers)
        x = df[feature_cols].astype(float).to_numpy()
        # Primary weak-IV diagnostic: partial first-stage F of T on Z given X.
        self._iv_instruments = [self.instrument]
        try:
            F, _q, _n = _partial_first_stage_F(t, z, x if x.size else None)
            self._iv_first_stage_F = F
            self._iv_relevance_verdict = _iv_relevance_verdict(F)
            if self._iv_relevance_verdict == "weak":
                self._iv_warning = _IV_WEAK_WARNING
        except Exception:
            self._iv_first_stage_F = None
            self._iv_relevance_verdict = None
        with converter():
            ro.globalenv["X_"] = ro.conversion.py2rpy(pd.DataFrame(x))
            ro.globalenv["Y_"] = ro.FloatVector(y)
            ro.globalenv["W_"] = ro.FloatVector(t)
            ro.globalenv["Z_"] = ro.FloatVector(z)
        start = time.perf_counter()
        ro.r(
            f"ivf_ <- grf::instrumental_forest("
            f"X = as.matrix(X_), Y = Y_, W = W_, Z = Z_, "
            f"num.trees = {self.num_trees}, seed = {self.seed})"
        )
        self._fit_seconds = time.perf_counter() - start
        self._forest = ro.globalenv["ivf_"]
        return self

    def estimate(self) -> EstimationResult:
        if self._forest is None:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()
        ate_row = ro.r("grf::average_treatment_effect(ivf_)")
        ate = float(list(ate_row)[0])
        se = float(list(ate_row)[1])
        z = 1.959963984540054
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(ate) / se))) if se > 0 else None
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="LATE",
            point_estimate=ate,
            se=se,
            ci_low=ate - z * se,
            ci_high=ate + z * se,
            p_value=p,
            n_used=self._n_used,
            diagnostics={
                "instrument": self.instrument,
                "iv_instruments": list(self._iv_instruments),
                "iv_first_stage_F": self._iv_first_stage_F,
                "iv_relevance_verdict": self._iv_relevance_verdict,
                **({"iv_warning": self._iv_warning} if self._iv_warning else {}),
                "iv_kendall_p": self._iv_relevance_p,
                "exclusion_restriction": "analyst-asserted (untestable)",
                "num_trees": self.num_trees,
            },
            backend_version=r_session_metadata().get("packages", {}).get("grf", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_used": self._n_used,
            "iv_instruments": list(self._iv_instruments),
            "iv_first_stage_F": self._iv_first_stage_F,
            "iv_relevance_verdict": self._iv_relevance_verdict,
            "iv_kendall_p": self._iv_relevance_p,
            "fitted": self._forest is not None,
        }
        if self._iv_warning:
            out["iv_warning"] = self._iv_warning
        return out

    def refute(self) -> dict[str, Any]:
        return {}


class GRFMultiArmForest:
    """grf::multi_arm_causal_forest — multi-arm treatment CATE.

    Routes when ``CATEGORICAL_TREATMENT`` is flagged with ≥3 levels.
    Reports the headline contrast vs the chosen baseline level + every
    pairwise contrast in diagnostics.
    """

    id: str = "rbridge.grf.multi_arm_causal_forest"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("ATE", "CATE")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.CATEGORICAL_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.TIME_VARYING_TREATMENT}
    )
    min_sample_size: int = 500
    produces_cate: bool = True
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        num_trees: int = 2000,
        seed: int = 42,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.num_trees = num_trees
        self.seed = seed
        self._forest: Any = None
        self._n_used: int = 0
        self._fit_seconds: float | None = None
        self._levels: list[str] = []

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "GRFMultiArmForest":
        require("grf")
        ro = r_session()
        cols = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna().copy()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(f"MultiArmForest needs ≥ {self.min_sample_size}; got {self._n_used}")
        levels = sorted(df[self.treatment].unique().tolist())
        self._levels = [str(l) for l in levels]
        y = df[self.outcome].astype(float).to_numpy()
        feature_cols = list(self.confounders) + list(self.modifiers)
        x = df[feature_cols].astype(float).to_numpy()
        with converter():
            ro.globalenv["X_"] = ro.conversion.py2rpy(pd.DataFrame(x))
            ro.globalenv["Y_"] = ro.FloatVector(y)
            ro.globalenv["W_"] = ro.StrVector([str(v) for v in df[self.treatment]])
        levels_r = "c(" + ", ".join(repr(l) for l in self._levels) + ")"
        ro.r(f"W_factor_ <- factor(W_, levels = {levels_r})")
        start = time.perf_counter()
        ro.r(
            f"mac_ <- grf::multi_arm_causal_forest("
            f"X = as.matrix(X_), Y = Y_, W = W_factor_, "
            f"num.trees = {self.num_trees}, seed = {self.seed})"
        )
        self._fit_seconds = time.perf_counter() - start
        self._forest = ro.globalenv["mac_"]
        return self

    def estimate(self) -> EstimationResult:
        if self._forest is None:
            raise RuntimeError("Call fit() before estimate().")
        ro = r_session()
        # multi-arm output is a data frame: contrast | estimate | std.err
        contrasts_df_r = ro.r("as.data.frame(grf::average_treatment_effect(mac_))")
        n_contrasts = int(list(ro.r("nrow(grf::average_treatment_effect(mac_))"))[0])
        contrasts: list[dict[str, Any]] = []
        for i in range(1, n_contrasts + 1):
            contrasts.append(
                {
                    "contrast": str(list(ro.r(f"grf::average_treatment_effect(mac_)[{i}, 1]"))[0]),
                    "estimate": float(list(ro.r(f"grf::average_treatment_effect(mac_)[{i}, 2]"))[0]),
                    "se": float(list(ro.r(f"grf::average_treatment_effect(mac_)[{i}, 3]"))[0]),
                }
            )
        headline = contrasts[0] if contrasts else {"estimate": 0.0, "se": 0.0, "contrast": "—"}
        ate, se = headline["estimate"], headline["se"]
        z_quantile = 1.959963984540054
        from scipy.stats import norm
        p = float(2 * (1 - norm.cdf(abs(ate) / se))) if se > 0 else None
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATE",
            point_estimate=ate,
            se=se,
            ci_low=ate - z_quantile * se,
            ci_high=ate + z_quantile * se,
            p_value=p,
            n_used=self._n_used,
            diagnostics={
                "levels": self._levels,
                "headline_contrast": headline["contrast"],
                "all_contrasts": contrasts,
                "num_trees": self.num_trees,
            },
            backend_version=r_session_metadata().get("packages", {}).get("grf", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"n_used": self._n_used, "levels": self._levels, "fitted": self._forest is not None}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    for cls in (
        GRFCausalForest,
        GRFCausalSurvivalForest,
        GRFInstrumentalForest,
        GRFMultiArmForest,
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
