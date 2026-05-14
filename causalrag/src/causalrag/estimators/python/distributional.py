"""Distributional / quantile-regression estimators (Sprint 7.7).

Three estimators for distributional treatment effects beyond the mean:

1. :class:`FirpoRIFQuantileEstimator` — Firpo (2007) unconditional-quantile
   partial effect (UQPE). For a marginal target quantile ``tau`` of Y, the
   recentered influence function (RIF) is

       RIF(Y; q_tau) = q_tau + (tau - 1{Y <= q_tau}) / f_Y(q_tau)

   where ``q_tau`` is the empirical tau-quantile of Y and ``f_Y(q_tau)`` is a
   kernel-density estimate at that quantile. Regressing RIF on (T, X) by OLS
   recovers the unconditional-quantile partial effect of T — i.e. how a small
   shift in T moves the marginal tau-quantile, holding the covariate
   distribution fixed. Reported as the OLS coefficient on T.

2. :class:`CFVCounterfactualDistribution` — Chernozhukov-Fernandez-Val-Melly
   (2013) counterfactual distributions via distribution regression. For a
   grid of outcome thresholds ``y_g`` we fit logistic regressions of
   ``1{Y <= y_g}`` on (T, X) and integrate out X to get the counterfactual
   CDF ``F_{Y(t)}(y_g) = E_X[ P(Y <= y_g | T = t, X) ]``. Inverting the CDF
   at a user grid of probability levels ``tau`` yields the counterfactual
   quantiles and the QTE curve ``tau -> F^{-1}_{Y(1)}(tau) - F^{-1}_{Y(0)}(tau)``.

3. :class:`DiNardoFortinLemieuxReweighting` — DFL (1996) distributional
   decomposition. Reweights the untreated subsample so its covariate
   composition matches the treated subsample, then contrasts quantiles of
   the reweighted "counterfactual" distribution against the observed
   treated distribution. Weights are propensity-score odds-ratios
   ``e(X) / (1 - e(X)) * (1 - p_T) / p_T``.

All three estimators are pure Python (sklearn for nuisance, numpy for the
quantile / CDF arithmetic) and register themselves on import.
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


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prepare(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...],
    min_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    cols = [outcome, treatment, *confounders]
    for c in cols:
        if c not in data.columns:
            raise ValueError(f"Column not in data: {c!r}")
    df = data[cols].dropna()
    n = len(df)
    if n < min_n:
        raise ValueError(
            f"Distributional estimator requires at least {min_n} rows after "
            f"dropna; got {n}"
        )
    y = df[outcome].to_numpy().astype(np.float64)
    t = df[treatment].to_numpy().astype(np.float64)
    unique_t = set(np.unique(t).tolist())
    if not unique_t.issubset({0.0, 1.0}):
        raise ValueError(
            f"Distributional estimators require binary {{0,1}} treatment; got "
            f"{unique_t}"
        )
    x = (
        df[list(confounders)].to_numpy().astype(np.float64)
        if confounders
        else None
    )
    return y, t, x


def _gaussian_kde_density(values: np.ndarray, at: float) -> float:
    """Plain Gaussian KDE evaluated at one point with Silverman's bandwidth.

    Returns a strictly-positive float (clamped to a tiny lower bound to
    avoid division-by-zero when the queried point is in a sparse tail).
    """
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n < 2:
        return 1e-6
    std = float(np.std(v, ddof=1))
    iqr = float(np.subtract(*np.percentile(v, [75, 25])))
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    if scale <= 0:
        scale = 1.0
    h = 0.9 * scale * n ** (-1.0 / 5.0)
    h = max(h, 1e-6)
    z = (v - at) / h
    dens = float(np.mean(np.exp(-0.5 * z * z) / (np.sqrt(2.0 * np.pi) * h)))
    return max(dens, 1e-6)


def _empirical_quantile(values: np.ndarray, tau: float) -> float:
    return float(np.quantile(values, tau))


def _invert_cdf(cdf_values: np.ndarray, thresholds: np.ndarray, tau: float) -> float:
    """Invert a monotone-increasing tabulated CDF at probability level ``tau``.

    ``cdf_values`` is assumed sorted by ``thresholds`` ascending. We enforce
    monotonicity defensively (CFM-style rearrangement) and linearly
    interpolate to the requested quantile.
    """
    # Rearrange: enforce monotone non-decreasing
    cdf_mono = np.maximum.accumulate(np.asarray(cdf_values, dtype=np.float64))
    cdf_mono = np.clip(cdf_mono, 0.0, 1.0)
    thr = np.asarray(thresholds, dtype=np.float64)
    # If tau falls below the minimum reachable CDF value, return the lowest
    # threshold; if above max, return the highest. Otherwise interpolate.
    if tau <= cdf_mono[0]:
        return float(thr[0])
    if tau >= cdf_mono[-1]:
        return float(thr[-1])
    # np.interp requires xp non-decreasing — which cdf_mono is.
    return float(np.interp(tau, cdf_mono, thr))


_Z_975 = 1.959963984540054


# ---------------------------------------------------------------------------
# 1. Firpo (2007) unconditional-quantile RIF regression
# ---------------------------------------------------------------------------

class FirpoRIFQuantileEstimator:
    """Unconditional-quantile partial effect via RIF regression (Firpo 2007).

    Estimates ``d F^{-1}_Y(tau) / d T`` at a user-specified quantile ``tau``
    by OLS-regressing the recentered influence function

        RIF_i = q_tau + (tau - 1{Y_i <= q_tau}) / f_Y(q_tau)

    on ``[T, X]``. The coefficient on ``T`` is the unconditional-quantile
    partial effect (UQPE). Standard errors come from the classical OLS
    sandwich with finite-sample HC1 correction.
    """

    id: str = "python.firpo.rif_quantile"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("QUANTILE_TREATMENT_EFFECT",)
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.BINARY_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.TIME_VARYING_TREATMENT,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 100
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
        tau: float = 0.5,
        random_state: int = 42,
    ) -> None:
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0, 1); got {tau}")
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = tuple(confounders)
        # Modifiers are folded into the covariate set for adjustment; this
        # estimator does not report per-row CATE.
        self.modifiers = tuple(modifiers)
        self.tau = float(tau)
        self.random_state = int(random_state)
        self._fitted: dict[str, Any] | None = None
        self._fit_seconds: float | None = None
        self._n_used: int = 0

    def _all_covariates(self) -> tuple[str, ...]:
        return self.confounders + self.modifiers

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "FirpoRIFQuantileEstimator":
        y, t, x = _prepare(
            data,
            self.treatment,
            self.outcome,
            self._all_covariates(),
            self.min_sample_size,
        )
        n = y.size
        self._n_used = n

        q_tau = _empirical_quantile(y, self.tau)
        f_y = _gaussian_kde_density(y, q_tau)
        rif = q_tau + (self.tau - (y <= q_tau).astype(np.float64)) / f_y

        # Design matrix: intercept + T + X
        cols = [np.ones(n), t]
        if x is not None and x.size:
            cols.extend(x.T)
        X_design = np.column_stack(cols)
        # OLS via lstsq
        start = time.perf_counter()
        beta, *_ = np.linalg.lstsq(X_design, rif, rcond=None)
        resid = rif - X_design @ beta
        # HC1 sandwich
        k = X_design.shape[1]
        XtX_inv = np.linalg.pinv(X_design.T @ X_design)
        meat = (X_design * resid[:, None]).T @ (X_design * resid[:, None])
        cov = XtX_inv @ meat @ XtX_inv
        # HC1 small-sample correction
        if n > k:
            cov *= n / (n - k)
        self._fit_seconds = time.perf_counter() - start

        self._fitted = {
            "beta": beta,
            "cov": cov,
            "q_tau": q_tau,
            "f_y": f_y,
            "rif": rif,
            "n": n,
            "k": k,
        }
        return self

    def estimate(self) -> EstimationResult:
        if self._fitted is None:
            raise RuntimeError("Call fit() before estimate().")
        beta = self._fitted["beta"]
        cov = self._fitted["cov"]
        # Coefficient on T is index 1 (after intercept).
        point = float(beta[1])
        se = float(np.sqrt(max(cov[1, 1], 0.0)))
        ci_low = point - _Z_975 * se
        ci_high = point + _Z_975 * se
        # Two-sided p-value via normal approximation.
        if se > 0:
            from math import erfc, sqrt

            z = abs(point) / se
            p_value: float | None = float(erfc(z / sqrt(2.0)))
        else:
            p_value = None

        diagnostics: dict[str, Any] = {
            "tau": self.tau,
            "q_tau": float(self._fitted["q_tau"]),
            "f_y_at_q_tau": float(self._fitted["f_y"]),
            "n_covariates": len(self._all_covariates()),
        }
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="QUANTILE_TREATMENT_EFFECT",
            point_estimate=point,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version="numpy",
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted is not None,
            "tau": self.tau,
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# 2. CFM 2013 counterfactual distributions
# ---------------------------------------------------------------------------

class CFVCounterfactualDistribution:
    """CFM (2013) counterfactual distribution + inverse-CDF differences.

    Returns the QTE curve over a user-specified probability grid. The single
    ``point_estimate`` reported by :meth:`estimate` is the QTE at the median
    of the requested ``tau_grid``; the full curve is exposed via
    :meth:`qte_curve` and surfaced in ``diagnostics``.
    """

    id: str = "python.cfvm.counterfactual_dist"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = (
        "QUANTILE_TREATMENT_EFFECT",
        "COUNTERFACTUAL_DISTRIBUTION",
    )
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.BINARY_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.TIME_VARYING_TREATMENT,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 200
    produces_cate: bool = False
    produces_full_counterfactual: bool = True
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...] = (),
        modifiers: tuple[str, ...] = (),
        *,
        tau_grid: tuple[float, ...] | np.ndarray = (
            0.1, 0.25, 0.5, 0.75, 0.9,
        ),
        n_thresholds: int = 40,
        random_state: int = 42,
    ) -> None:
        tg = np.asarray(tau_grid, dtype=np.float64)
        if tg.size == 0 or np.any(tg <= 0) or np.any(tg >= 1):
            raise ValueError("tau_grid must be non-empty and contained in (0, 1)")
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = tuple(confounders)
        self.modifiers = tuple(modifiers)
        self.tau_grid = tg
        self.n_thresholds = int(n_thresholds)
        self.random_state = int(random_state)
        self._fitted: dict[str, Any] | None = None
        self._fit_seconds: float | None = None
        self._n_used: int = 0

    def _all_covariates(self) -> tuple[str, ...]:
        return self.confounders + self.modifiers

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol
    ) -> "CFVCounterfactualDistribution":
        y, t, x = _prepare(
            data,
            self.treatment,
            self.outcome,
            self._all_covariates(),
            self.min_sample_size,
        )
        n = y.size
        self._n_used = n
        # Threshold grid spans the inner range of observed Y so logistic
        # regressions never face all-zero or all-one targets.
        lo, hi = np.quantile(y, [0.02, 0.98])
        thresholds = np.linspace(lo, hi, self.n_thresholds)

        from sklearn.linear_model import LogisticRegression

        # Build the (T, X) design used to fit each threshold-level logit.
        if x is not None and x.size:
            feat = np.column_stack([t, x])
        else:
            feat = t.reshape(-1, 1)

        # For counterfactual t' in {0, 1} we evaluate at (T=t', X_i) for
        # every i and average — i.e. the g-formula over the empirical X.
        if x is not None and x.size:
            feat_t1 = np.column_stack([np.ones(n), x])
            feat_t0 = np.column_stack([np.zeros(n), x])
        else:
            feat_t1 = np.ones((n, 1))
            feat_t0 = np.zeros((n, 1))

        cdf_t1 = np.empty(self.n_thresholds)
        cdf_t0 = np.empty(self.n_thresholds)

        start = time.perf_counter()
        for j, thr in enumerate(thresholds):
            target = (y <= thr).astype(int)
            if target.sum() == 0:
                cdf_t1[j] = 0.0
                cdf_t0[j] = 0.0
                continue
            if target.sum() == n:
                cdf_t1[j] = 1.0
                cdf_t0[j] = 1.0
                continue
            clf = LogisticRegression(
                solver="liblinear",
                max_iter=200,
                random_state=self.random_state,
            )
            clf.fit(feat, target)
            p1 = clf.predict_proba(feat_t1)[:, 1]
            p0 = clf.predict_proba(feat_t0)[:, 1]
            cdf_t1[j] = float(np.mean(p1))
            cdf_t0[j] = float(np.mean(p0))
        self._fit_seconds = time.perf_counter() - start

        # Build the QTE curve by inverting both counterfactual CDFs.
        q_t1 = np.array(
            [_invert_cdf(cdf_t1, thresholds, float(tau)) for tau in self.tau_grid]
        )
        q_t0 = np.array(
            [_invert_cdf(cdf_t0, thresholds, float(tau)) for tau in self.tau_grid]
        )
        qte = q_t1 - q_t0

        self._fitted = {
            "thresholds": thresholds,
            "cdf_t1": cdf_t1,
            "cdf_t0": cdf_t0,
            "q_t1": q_t1,
            "q_t0": q_t0,
            "qte": qte,
        }
        return self

    def qte_curve(self) -> dict[str, np.ndarray]:
        if self._fitted is None:
            raise RuntimeError("Call fit() before qte_curve().")
        return {
            "tau": self.tau_grid.copy(),
            "q_treated": self._fitted["q_t1"].copy(),
            "q_control": self._fitted["q_t0"].copy(),
            "qte": self._fitted["qte"].copy(),
        }

    def estimate(self) -> EstimationResult:
        if self._fitted is None:
            raise RuntimeError("Call fit() before estimate().")
        qte = self._fitted["qte"]
        # Report the QTE nearest tau = 0.5 as the scalar point estimate.
        idx_median = int(np.argmin(np.abs(self.tau_grid - 0.5)))
        point = float(qte[idx_median])

        diagnostics: dict[str, Any] = {
            "tau_grid": self.tau_grid.tolist(),
            "qte_curve": [float(v) for v in qte],
            "q_treated": [float(v) for v in self._fitted["q_t1"]],
            "q_control": [float(v) for v in self._fitted["q_t0"]],
            "n_thresholds": self.n_thresholds,
            "reported_tau": float(self.tau_grid[idx_median]),
            "n_covariates": len(self._all_covariates()),
        }
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="COUNTERFACTUAL_DISTRIBUTION",
            point_estimate=point,
            se=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version="sklearn+numpy",
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted is not None,
            "n_thresholds": self.n_thresholds,
            "tau_grid": self.tau_grid.tolist(),
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# 3. DiNardo-Fortin-Lemieux reweighting
# ---------------------------------------------------------------------------

class DiNardoFortinLemieuxReweighting:
    """DFL (1996) distributional reweighting.

    Reweights the untreated subsample so its covariate distribution matches
    the treated subsample. The reweighting factor is

        psi(X) = [ Pr(T=1 | X) / Pr(T=0 | X) ] * [ Pr(T=0) / Pr(T=1) ]

    applied to T=0 observations. Quantiles of the reweighted untreated
    distribution estimate the counterfactual F^{-1}_{Y(0) | T=1}(tau)
    under the treated covariate composition; contrasting against the
    observed treated quantiles gives a distributional decomposition.

    Reports the QTE at a user-specified tau as the scalar point estimate
    and exposes the full curve through :meth:`qte_curve`.
    """

    id: str = "python.dfl.reweighting"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = (
        "QUANTILE_TREATMENT_EFFECT",
        "COUNTERFACTUAL_DISTRIBUTION",
    )
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.BINARY_TREATMENT})
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.TIME_VARYING_TREATMENT,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 100
    produces_cate: bool = False
    produces_full_counterfactual: bool = True
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...] = (),
        modifiers: tuple[str, ...] = (),
        *,
        tau: float = 0.5,
        tau_grid: tuple[float, ...] | np.ndarray = (
            0.1, 0.25, 0.5, 0.75, 0.9,
        ),
        trim: float = 0.01,
        random_state: int = 42,
    ) -> None:
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0, 1); got {tau}")
        if not (0.0 <= trim < 0.5):
            raise ValueError(f"trim must be in [0, 0.5); got {trim}")
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = tuple(confounders)
        self.modifiers = tuple(modifiers)
        self.tau = float(tau)
        self.tau_grid = np.asarray(tau_grid, dtype=np.float64)
        self.trim = float(trim)
        self.random_state = int(random_state)
        self._fitted: dict[str, Any] | None = None
        self._fit_seconds: float | None = None
        self._n_used: int = 0

    def _all_covariates(self) -> tuple[str, ...]:
        return self.confounders + self.modifiers

    def _weighted_quantile(
        self, values: np.ndarray, weights: np.ndarray, q: float
    ) -> float:
        order = np.argsort(values)
        v = values[order]
        w = weights[order]
        cw = np.cumsum(w)
        total = cw[-1]
        if total <= 0:
            return float("nan")
        target = q * total
        idx = int(np.searchsorted(cw, target, side="left"))
        idx = min(idx, v.size - 1)
        return float(v[idx])

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol
    ) -> "DiNardoFortinLemieuxReweighting":
        y, t, x = _prepare(
            data,
            self.treatment,
            self.outcome,
            self._all_covariates(),
            self.min_sample_size,
        )
        n = y.size
        self._n_used = n
        if x is None or x.size == 0:
            # No covariates ⇒ reweighting reduces to identity. We still
            # compute quantile contrasts; the propensity is the marginal.
            p_hat = np.full(n, float(np.mean(t)))
        else:
            from sklearn.linear_model import LogisticRegression

            clf = LogisticRegression(
                solver="liblinear",
                max_iter=200,
                random_state=self.random_state,
            )
            clf.fit(x, t.astype(int))
            p_hat = clf.predict_proba(x)[:, 1]

        # Trim propensities to avoid blow-up in the odds ratio.
        eps = max(self.trim, 1e-6)
        p_hat = np.clip(p_hat, eps, 1 - eps)
        p_t = float(np.mean(t))
        p_t = float(np.clip(p_t, eps, 1 - eps))

        # DFL weights on T=0 observations:
        #     psi(X) = e(X) / (1 - e(X)) * (1 - p_T) / p_T
        start = time.perf_counter()
        is_treated = t == 1
        is_control = ~is_treated
        psi = np.zeros(n)
        psi[is_control] = (p_hat[is_control] / (1 - p_hat[is_control])) * (
            (1 - p_t) / p_t
        )

        # Observed treated quantiles (no reweighting)
        y_t1 = y[is_treated]
        # Reweighted control distribution: quantiles of y_t0 weighted by psi
        y_t0 = y[is_control]
        w_t0 = psi[is_control]
        # Trim weights very aggressively at the top to stabilise
        # finite-sample quantile estimates.
        wcap = float(np.quantile(w_t0, 0.99)) if w_t0.size > 0 else 1.0
        w_t0 = np.minimum(w_t0, wcap)

        q_t1 = np.array(
            [float(np.quantile(y_t1, float(tau))) for tau in self.tau_grid]
        )
        q_t0_cf = np.array(
            [self._weighted_quantile(y_t0, w_t0, float(tau)) for tau in self.tau_grid]
        )
        qte = q_t1 - q_t0_cf
        self._fit_seconds = time.perf_counter() - start

        # Also build a fine-grained curve for downstream consumers.
        fine_grid = np.linspace(0.05, 0.95, 19)
        q_t1_fine = np.array(
            [float(np.quantile(y_t1, float(tau))) for tau in fine_grid]
        )
        q_t0_fine = np.array(
            [self._weighted_quantile(y_t0, w_t0, float(tau)) for tau in fine_grid]
        )

        self._fitted = {
            "p_hat": p_hat,
            "psi": psi,
            "q_treated": q_t1,
            "q_control_cf": q_t0_cf,
            "qte": qte,
            "fine_grid": fine_grid,
            "q_treated_fine": q_t1_fine,
            "q_control_cf_fine": q_t0_fine,
            "qte_fine": q_t1_fine - q_t0_fine,
        }
        return self

    def qte_curve(self) -> dict[str, np.ndarray]:
        if self._fitted is None:
            raise RuntimeError("Call fit() before qte_curve().")
        return {
            "tau": self._fitted["fine_grid"].copy(),
            "q_treated": self._fitted["q_treated_fine"].copy(),
            "q_control": self._fitted["q_control_cf_fine"].copy(),
            "qte": self._fitted["qte_fine"].copy(),
        }

    def estimate(self) -> EstimationResult:
        if self._fitted is None:
            raise RuntimeError("Call fit() before estimate().")
        # Pick the qte at self.tau (closest grid point).
        idx = int(np.argmin(np.abs(self.tau_grid - self.tau)))
        point = float(self._fitted["qte"][idx])
        diagnostics: dict[str, Any] = {
            "tau": self.tau,
            "tau_grid": self.tau_grid.tolist(),
            "qte_curve": [float(v) for v in self._fitted["qte"]],
            "q_treated": [float(v) for v in self._fitted["q_treated"]],
            "q_control_cf": [float(v) for v in self._fitted["q_control_cf"]],
            "trim": self.trim,
            "n_covariates": len(self._all_covariates()),
        }
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="COUNTERFACTUAL_DISTRIBUTION",
            point_estimate=point,
            se=None,
            ci_low=None,
            ci_high=None,
            p_value=None,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version="sklearn+numpy",
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted is not None,
            "tau": self.tau,
            "trim": self.trim,
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _register() -> None:
    for cls in (
        FirpoRIFQuantileEstimator,
        CFVCounterfactualDistribution,
        DiNardoFortinLemieuxReweighting,
    ):
        try:
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
        except ValueError:
            # Already registered (re-import) — registry forbids duplicates.
            pass


_register()
