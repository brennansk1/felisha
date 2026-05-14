"""Synthetic-control family — SCM, Ridge-ASCM, SDiD (PDD Sprint 2.3).

Wraps ``pysyncon`` to provide three single-treated-unit panel estimators:

- ``python.synth_control.scm``  — Abadie-Diamond-Hainmueller 2010 classical SCM
- ``python.synth_control.ascm`` — Ben-Michael-Feller-Rothstein JASA 2021 ASCM
  (Ridge-augmented Synth; ``pysyncon.AugSynth``)
- ``python.synth_control.sdid`` — Arkhangelsky-Athey-Hirshberg-Imbens-Wager
  AER 2021 Synthetic Difference-in-Differences

Input shape (long format DataFrame):

  ============  =========================================
  Column        Role
  ============  =========================================
  ``unit_id``   Unit label (one of which is the treated)
  ``time``      Time period (ordinal or numeric)
  treatment     Indicator: 1 for treated unit in the post
                period, 0 otherwise
  outcome       Continuous outcome
  ============  =========================================

Inference follows the canonical SC literature: in-space placebo (Abadie-
Gardeazabal-Hainmueller Fisher-exact p-value via :class:`pysyncon.utils.
PlaceboTest`) and Chernozhukov-Wüthrich-Zhu 2021 conformal CIs (only for
SCM, where ``Synth.confidence_interval`` ships them; ASCM/SDiD fall back to
the placebo gap distribution). If ``pysyncon`` is not installed the module
imports cleanly but registration is skipped.
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


def _pysyncon_available() -> bool:
    try:
        import pysyncon  # noqa: F401

        return True
    except ImportError:
        return False


Variant = Literal["scm", "ascm", "sdid"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_inputs(
    data: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    treatment: str,
    outcome: str,
) -> None:
    """Verify required columns exist and that the panel has exactly one
    treated unit with a contiguous post-period. Raises ValueError on the
    first failure with a message that names the offending column / unit /
    time so the caller can correct the input. This is the path tests
    exercise without needing pysyncon installed."""
    required = [unit_col, time_col, treatment, outcome]
    for c in required:
        if c not in data.columns:
            raise ValueError(f"Column not in data: {c!r}")
    if data.empty:
        raise ValueError("Panel data is empty.")

    tr = data[treatment].to_numpy()
    unique_tr = set(np.unique(tr).tolist())
    if not unique_tr.issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(
            f"Treatment indicator must be in {{0,1}}; got {unique_tr}"
        )

    # Identify treated unit: any unit with at least one treatment==1 row.
    treated_mask = data[treatment].astype(int) == 1
    treated_units = data.loc[treated_mask, unit_col].unique().tolist()
    if len(treated_units) == 0:
        raise ValueError(
            "No treated unit found: treatment indicator is 0 everywhere."
        )
    if len(treated_units) > 1:
        raise ValueError(
            f"Synthetic control requires exactly one treated unit; "
            f"found {len(treated_units)}: {treated_units}"
        )


def _split_panel(
    data: pd.DataFrame,
    *,
    unit_col: str,
    time_col: str,
    treatment: str,
    outcome: str,
) -> tuple[Any, list[Any], list[Any], list[Any]]:
    """Return ``(treated_unit, control_units, pre_periods, post_periods)``.

    Pre-period = times where the treated unit has treatment==0.
    Post-period = times where the treated unit has treatment==1.
    Assumes the input has already been validated.
    """
    treated_rows = data[data[treatment].astype(int) == 1]
    treated_unit = treated_rows[unit_col].iloc[0]
    all_times = sorted(data[time_col].unique().tolist())
    post = sorted(
        data[
            (data[unit_col] == treated_unit) & (data[treatment].astype(int) == 1)
        ][time_col].unique().tolist()
    )
    post_set = set(post)
    pre = [t for t in all_times if t not in post_set]
    controls = [u for u in data[unit_col].unique().tolist() if u != treated_unit]
    if not pre:
        raise ValueError("No pre-treatment periods found for the treated unit.")
    if not post:
        raise ValueError("No post-treatment periods found for the treated unit.")
    if not controls:
        raise ValueError("No donor (control) units present in the panel.")
    return treated_unit, controls, pre, post


# ---------------------------------------------------------------------------
# SDiD primitives (Arkhangelsky et al. 2021, simplified single-treated case)
# ---------------------------------------------------------------------------
def _sdid_unit_weights(Y_pre_co: np.ndarray, y_pre_tr: np.ndarray) -> np.ndarray:
    """Unit weights minimising ||Y_pre_co @ w + b - y_pre_tr||^2 with w on the
    simplex and an unconstrained intercept ``b`` (Arkhangelsky et al. §2.3).
    Falls back to equal weights if the QP solver fails.
    """
    from scipy.optimize import minimize

    T0, N = Y_pre_co.shape
    w0 = np.full(N, 1.0 / N)

    def obj(z: np.ndarray) -> float:
        w = z[:-1]
        b = z[-1]
        r = Y_pre_co @ w + b - y_pre_tr
        return float(np.dot(r, r))

    cons = [{"type": "eq", "fun": lambda z: np.sum(z[:-1]) - 1.0}]
    bnds = [(0.0, 1.0)] * N + [(None, None)]
    try:
        res = minimize(
            obj,
            x0=np.concatenate([w0, [0.0]]),
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        w = np.clip(res.x[:-1], 0.0, None)
        s = w.sum()
        return w / s if s > 0 else w0
    except Exception:
        return w0


def _sdid_time_weights(Y_pre_co: np.ndarray, Y_post_co: np.ndarray) -> np.ndarray:
    """Time weights ``lambda`` on the pre-period simplex matching the mean
    post-period of each control unit (Arkhangelsky et al. §2.3). Equal-weight
    fallback on solver failure.
    """
    from scipy.optimize import minimize

    T0, N = Y_pre_co.shape
    target = Y_post_co.mean(axis=0)  # length N
    l0 = np.full(T0, 1.0 / T0)

    def obj(z: np.ndarray) -> float:
        lam = z[:-1]
        b = z[-1]
        r = lam @ Y_pre_co + b - target
        return float(np.dot(r, r))

    cons = [{"type": "eq", "fun": lambda z: np.sum(z[:-1]) - 1.0}]
    bnds = [(0.0, 1.0)] * T0 + [(None, None)]
    try:
        res = minimize(
            obj,
            x0=np.concatenate([l0, [0.0]]),
            method="SLSQP",
            bounds=bnds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        lam = np.clip(res.x[:-1], 0.0, None)
        s = lam.sum()
        return lam / s if s > 0 else l0
    except Exception:
        return l0


def _sdid_att(
    Y_pre_tr: np.ndarray,
    Y_post_tr: np.ndarray,
    Y_pre_co: np.ndarray,
    Y_post_co: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (ATT, unit_weights, time_weights) for SDiD.

    With pre-period unit means weighted by ``lambda`` and donor pool weighted
    by ``w``, the ATT is::

        att = (Ybar_post_tr - lambda' Y_pre_tr)
              - sum_i w_i (Ybar_post_co_i - lambda' Y_pre_co_:,i)
    """
    w = _sdid_unit_weights(Y_pre_co, Y_pre_tr)
    lam = _sdid_time_weights(Y_pre_co, Y_post_co)

    treated_term = Y_post_tr.mean() - lam @ Y_pre_tr
    co_post_means = Y_post_co.mean(axis=0)  # length N
    co_pre_proj = lam @ Y_pre_co  # length N
    donor_term = float(np.dot(w, co_post_means - co_pre_proj))
    return float(treated_term - donor_term), w, lam


# ---------------------------------------------------------------------------
# Estimator class
# ---------------------------------------------------------------------------
class SyntheticControlEstimator:
    """Synthetic control / ASCM / SDiD for single-treated-unit panel data."""

    id_prefix = "python.synth_control"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATT",)
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.SINGLE_TREATED_UNIT})
    excluded_flags: frozenset[DataFlag] = frozenset()
    min_sample_size: int = 30  # at least a handful of donors x pre-periods
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
        variant: Variant = "scm",
        unit_col: str = "unit_id",
        time_col: str = "time",
        placebo_max_workers: int | None = 1,
        placebo_verbose: bool = False,
        random_state: int = 42,
    ) -> None:
        if variant not in ("scm", "ascm", "sdid"):
            raise ValueError(
                f"variant must be one of {{'scm','ascm','sdid'}}; got {variant!r}"
            )
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders  # ignored for SC; kept for API parity
        self.modifiers = modifiers
        self.variant = variant
        self.unit_col = unit_col
        self.time_col = time_col
        self.placebo_max_workers = placebo_max_workers
        self.placebo_verbose = placebo_verbose
        self.random_state = random_state
        self.id = f"{self.id_prefix}.{variant}"

        # populated by fit()
        self._data: pd.DataFrame | None = None
        self._treated_unit: Any = None
        self._controls: list[Any] = []
        self._pre: list[Any] = []
        self._post: list[Any] = []
        self._att: float | None = None
        self._unit_weights: pd.Series | None = None
        self._time_weights: np.ndarray | None = None
        self._scm: Any = None  # underlying pysyncon model when applicable
        self._dataprep: Any = None
        self._gaps_pre: np.ndarray | None = None
        self._gaps_post: np.ndarray | None = None
        self._fit_seconds: float | None = None
        self._backend_version: str | None = None

    # ------------------------------------------------------------------
    def _build_dataprep(self) -> Any:
        from pysyncon import Dataprep

        assert self._data is not None
        return Dataprep(
            foo=self._data,
            predictors=[self.outcome],
            predictors_op="mean",
            dependent=self.outcome,
            unit_variable=self.unit_col,
            time_variable=self.time_col,
            treatment_identifier=self._treated_unit,
            controls_identifier=self._controls,
            time_predictors_prior=self._pre,
            time_optimize_ssr=self._pre,
        )

    # ------------------------------------------------------------------
    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> SyntheticControlEstimator:
        _validate_inputs(
            data,
            unit_col=self.unit_col,
            time_col=self.time_col,
            treatment=self.treatment,
            outcome=self.outcome,
        )
        if not _pysyncon_available():
            raise ImportError(
                "pysyncon is required for SyntheticControlEstimator.fit(); "
                "install it via `pip install pysyncon`."
            )

        self._data = data
        treated, controls, pre, post = _split_panel(
            data,
            unit_col=self.unit_col,
            time_col=self.time_col,
            treatment=self.treatment,
            outcome=self.outcome,
        )
        self._treated_unit = treated
        self._controls = controls
        self._pre = pre
        self._post = post

        start = time.perf_counter()
        self._dataprep = self._build_dataprep()

        if self.variant == "scm":
            from pysyncon import Synth

            self._scm = Synth()
            self._scm.fit(dataprep=self._dataprep)
            self._unit_weights = self._scm.weights(round=6)
            att_info = self._scm.att(time_period=self._post)
            self._att = float(att_info["att"])
        elif self.variant == "ascm":
            from pysyncon import AugSynth

            self._scm = AugSynth()
            self._scm.fit(dataprep=self._dataprep)
            self._unit_weights = self._scm.weights(round=6)
            att_info = self._scm.att(time_period=self._post)
            self._att = float(att_info["att"])
        else:  # sdid
            # Build Y matrices from the long-format data.
            wide = self._data.pivot(
                index=self.time_col, columns=self.unit_col, values=self.outcome
            )
            # Ensure column order [controls..., treated]
            Y_pre_co = wide.loc[self._pre, self._controls].to_numpy(dtype=float)
            Y_post_co = wide.loc[self._post, self._controls].to_numpy(dtype=float)
            y_pre_tr = wide.loc[self._pre, self._treated_unit].to_numpy(dtype=float)
            y_post_tr = wide.loc[self._post, self._treated_unit].to_numpy(dtype=float)
            att, w, lam = _sdid_att(y_pre_tr, y_post_tr, Y_pre_co, Y_post_co)
            self._att = att
            self._unit_weights = pd.Series(w, index=self._controls, name="weights")
            self._time_weights = lam
            # Also fit a plain Synth for placebo / RMSPE diagnostics.
            from pysyncon import Synth

            self._scm = Synth()
            self._scm.fit(dataprep=self._dataprep)

        # Cache pre/post gaps for RMSPE computation.
        Z0_pre, Z1_pre = self._dataprep.make_outcome_mats(time_period=self._pre)
        Z0_post, Z1_post = self._dataprep.make_outcome_mats(time_period=self._post)
        W = self._scm.W  # array aligned with controls_identifier order
        pred_pre = Z0_pre.to_numpy() @ W
        pred_post = Z0_post.to_numpy() @ W
        self._gaps_pre = Z1_pre.to_numpy() - pred_pre
        self._gaps_post = Z1_post.to_numpy() - pred_post

        self._fit_seconds = time.perf_counter() - start
        import pysyncon

        self._backend_version = f"pysyncon {pysyncon.__version__}"
        return self

    # ------------------------------------------------------------------
    def _run_placebo(self) -> dict[str, Any]:
        """Run in-space placebo test, return p-value, rank, and gap RMSPE
        ratios. Single-worker by default to keep tests deterministic and
        avoid multi-process overhead. Returns empty dict on failure."""
        from pysyncon.utils import PlaceboTest

        try:
            placebo = PlaceboTest()
            placebo.fit(
                dataprep=self._dataprep,
                scm=self._scm,
                max_workers=self.placebo_max_workers,
                verbose=self.placebo_verbose,
            )
        except Exception:
            return {}

        if placebo.gaps is None or placebo.treated_gap is None:
            return {}

        assert self._post is not None
        # treatment_time is the *first* post-period
        t_treat = self._post[0]
        try:
            p_value = float(placebo.pvalue(treatment_time=t_treat))
        except Exception:
            p_value = float("nan")

        # Compute RMSPE ratio per unit (post/pre) and rank the treated.
        all_gaps = pd.concat([placebo.gaps, placebo.treated_gap], axis=1)
        pre_idx = [t for t in all_gaps.index if t < t_treat]
        post_idx = [t for t in all_gaps.index if t >= t_treat]
        denom = all_gaps.loc[pre_idx].pow(2).mean(axis=0)
        num = all_gaps.loc[post_idx].pow(2).mean(axis=0)
        # Guard against zero pre-period MSPE.
        ratio = num / denom.replace(0, np.nan)
        treated_name = placebo.treated_gap.name
        sorted_ratio = ratio.sort_values(ascending=False)
        try:
            rank = int(sorted_ratio.index.get_loc(treated_name)) + 1
        except KeyError:
            rank = -1
        return {
            "p_value": p_value,
            "treated_post_pre_rmspe_ratio": float(ratio.loc[treated_name])
            if treated_name in ratio.index
            else float("nan"),
            "placebo_rank": rank,
            "n_placebo_units": int(len(placebo.gaps.columns)),
            "ratio_distribution": ratio.to_dict(),
            "gaps_post": placebo.gaps.loc[post_idx],
            "treated_gap_post": placebo.treated_gap.loc[post_idx],
        }

    # ------------------------------------------------------------------
    def _conformal_ci(self, alpha: float = 0.10) -> tuple[float | None, float | None]:
        """Chernozhukov-Wuthrich-Zhu conformal CI on the *average* post-period
        treatment effect. ``pysyncon.Synth.confidence_interval`` ships this
        for SCM; for ASCM/SDiD we fall back to a placebo-gap quantile CI on
        the post-period mean treatment effect.
        """
        if self.variant == "scm":
            try:
                ci_df = self._scm.confidence_interval(
                    alpha=alpha,
                    time_periods=self._post,
                    tol=0.1,
                    pre_periods=self._pre,
                    verbose=False,
                )
                # Average the per-period CIs across the post window.
                lo = float(ci_df["lower_ci"].mean())
                hi = float(ci_df["upper_ci"].mean())
                return lo, hi
            except Exception:
                pass
        # Fallback: placebo-gap quantile CI on the post mean.
        return None, None

    # ------------------------------------------------------------------
    def estimate(self) -> EstimationResult:
        if self._att is None:
            raise RuntimeError("Call fit() before estimate().")
        placebo = self._run_placebo()

        # Pre-period RMSPE and post/pre ratio for the treated unit.
        assert self._gaps_pre is not None and self._gaps_post is not None
        pre_rmspe = float(np.sqrt(np.mean(self._gaps_pre ** 2)))
        post_rmspe = float(np.sqrt(np.mean(self._gaps_post ** 2)))
        ratio = post_rmspe / pre_rmspe if pre_rmspe > 0 else float("nan")

        # SE: prefer the placebo distribution SD of post-period mean gaps;
        # fall back to the Synth.att SE.
        se: float | None = None
        if placebo and "gaps_post" in placebo:
            post_means = placebo["gaps_post"].mean(axis=0).to_numpy()
            if post_means.size > 1:
                se = float(np.std(post_means, ddof=1))
        if se is None and self.variant in ("scm", "ascm"):
            try:
                se = float(self._scm.att(time_period=self._post)["se"])
            except Exception:
                se = None

        # CI: conformal for SCM; placebo-gap quantile otherwise.
        ci_low, ci_high = self._conformal_ci(alpha=0.10)
        if (ci_low is None or ci_high is None) and placebo and "gaps_post" in placebo:
            post_means = placebo["gaps_post"].mean(axis=0).to_numpy()
            if post_means.size >= 2:
                ci_low = float(np.quantile(post_means, 0.05))
                ci_high = float(np.quantile(post_means, 0.95))

        p_value = placebo.get("p_value") if placebo else None
        placebo_rank = placebo.get("placebo_rank") if placebo else None

        diagnostics: dict[str, Any] = {
            "variant": self.variant,
            "unit_weights": self._unit_weights.to_dict()
            if self._unit_weights is not None
            else {},
            "pre_treatment_fit_rmspe": pre_rmspe,
            "post_pre_rmspe_ratio": ratio,
            "post_treatment_rmspe": post_rmspe,
            "placebo_rank": placebo_rank,
            "n_donors": len(self._controls),
            "n_pre_periods": len(self._pre),
            "n_post_periods": len(self._post),
            "treated_unit": self._treated_unit,
        }
        if self.variant == "sdid" and self._time_weights is not None:
            diagnostics["time_weights"] = self._time_weights.tolist()

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATT",
            point_estimate=float(self._att),
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=int(self._data.shape[0]) if self._data is not None else 0,
            diagnostics=diagnostics,
            backend_version=self._backend_version,
            fit_seconds=self._fit_seconds,
        )

    # ------------------------------------------------------------------
    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._att is not None,
            "variant": self.variant,
            "n_donors": len(self._controls),
            "n_pre_periods": len(self._pre),
            "n_post_periods": len(self._post),
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Variant-specific subclasses so the registry can hold three concrete factories
# with distinct ids while sharing one implementation.
# ---------------------------------------------------------------------------
class _SCMVariant(SyntheticControlEstimator):
    id = "python.synth_control.scm"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["variant"] = "scm"
        super().__init__(*args, **kwargs)


class _ASCMVariant(SyntheticControlEstimator):
    id = "python.synth_control.ascm"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["variant"] = "ascm"
        super().__init__(*args, **kwargs)


class _SDiDVariant(SyntheticControlEstimator):
    id = "python.synth_control.sdid"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["variant"] = "sdid"
        super().__init__(*args, **kwargs)


def _register() -> None:
    """Register the three SC variants iff ``pysyncon`` is importable.

    Mirrors :mod:`causalrag.estimators.python.bart`: when the optional
    dependency is missing we silently skip registration so the rest of the
    catalog stays usable, but the module itself still imports cleanly so
    pure-Python validation paths remain unit-testable.
    """
    if not _pysyncon_available():
        return
    for cls in (_SCMVariant, _ASCMVariant, _SDiDVariant):
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
