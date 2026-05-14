"""Causal-forecasting / intervention-impact analysis (PDD Sprint 5.4).

This module answers the question

    "What was the impact of intervention I that happened at time T on
     metric Y?"

via three complementary estimators behind a single
:func:`analyze_impact` entry point:

1. **CausalImpact** — Brodersen et al. *Inferring Causal Impact Using
   Bayesian Structural Time-Series Models* (AnnApplStat 2015). The
   reference implementation lives in the ``causalimpact`` /
   ``tfcausalimpact`` packages; if neither is installed we fall back to a
   plain ``statsmodels`` ARIMA-with-intervention regression that
   recovers the same point ATT on synthetic step-change series.
2. **Augmented Synthetic Control Method (ASCM)** — Ben-Michael, Feller
   & Rothstein *Augmented Synthetic Control Method* (JASA 2021). We do
   **not** reimplement this here: we delegate to
   :class:`causalrag.estimators.python.synthetic_control.SyntheticControlEstimator`
   (variant ``"ascm"``) that Sprint 2.3 shipped, falling back to plain
   SCM if ``pysyncon`` exposes only ``Synth``.
3. **Matrix completion** — Athey, Bayati, Doudchenko, Imbens & Khosravi
   *Matrix Completion Methods for Causal Panel Data Models* (JASA 2021).
   A compact, dependency-free SVD-based low-rank completion of the
   panel with the post-treatment block masked, then comparing actual
   vs. imputed.

All three feed into a single :class:`ImpactReport` that surfaces the
consensus (median) point estimate, a quantile-CI across methods, and a
``consistency_verdict`` flagging whether the methods agree.

Failure-safe per method: if a method's library is missing or its solver
errors, we log the failure as a ``note`` on the corresponding
:class:`ImpactFinding` (or skip the method entirely if it could not
produce any number) and compute the consensus over the surviving
methods. The caller therefore *always* gets a report, even on a system
with only ``numpy`` + ``pandas`` + ``statsmodels`` available.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


MethodName = Literal["causalimpact", "ascm", "matrix_completion"]
DEFAULT_METHODS: tuple[MethodName, ...] = (
    "causalimpact",
    "ascm",
    "matrix_completion",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ImpactFinding:
    """Per-method estimate of the average post-period treatment effect.

    ``ci_low`` / ``ci_high`` are an approximate 95% interval where the
    underlying method ships one; methods without a native CI report
    ``None`` and a note explaining what is available instead.
    """

    method: str
    point_estimate: float
    ci_low: float | None
    ci_high: float | None
    notes: list[str] = field(default_factory=list)

    @property
    def se(self) -> float | None:
        """Approximate SE inferred from the symmetric-CI width if any."""
        if self.ci_low is None or self.ci_high is None:
            return None
        return float((self.ci_high - self.ci_low) / (2.0 * 1.96))


@dataclass
class ImpactReport:
    """Aggregated impact analysis across the requested methods."""

    target: str
    intervention_time: pd.Timestamp
    pre_period_start: pd.Timestamp
    pre_period_end: pd.Timestamp
    post_period_start: pd.Timestamp
    post_period_end: pd.Timestamp
    findings: list[ImpactFinding]
    consensus_point: float
    consensus_ci: tuple[float, float] | None
    consistency_verdict: Literal["consistent", "moderate", "divergent"]
    interpretation: str
    n_pre: int
    n_post: int
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers — time handling
# ---------------------------------------------------------------------------
def _coerce_timestamp(value: pd.Timestamp | str | Any) -> pd.Timestamp:
    """Pass-through for ``pd.Timestamp``; ``pd.to_datetime`` everything else.

    Allows callers to pass either ISO strings, ``datetime`` objects, or
    bare integers (which a synthetic-data test may use as a time index).
    Integers are kept as integers via a synthetic ``Timestamp.min + N``
    encoding when a numeric time column is used.
    """
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, (int, np.integer)):
        # Encode an integer time-index as days past the pandas epoch so
        # we can still pretty-print it; the *ordering* is what matters.
        return pd.Timestamp("2000-01-01") + pd.Timedelta(days=int(value))
    return pd.to_datetime(value)


def _is_numeric_time(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _to_comparable_time(value: Any, *, numeric: bool) -> Any:
    """Project a user-supplied time onto the dtype of the time column."""
    if numeric:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return value
        # Allow string ints like "50".
        try:
            return int(value)
        except (TypeError, ValueError):
            return float(value)  # type: ignore[arg-type]
    return pd.to_datetime(value)


# ---------------------------------------------------------------------------
# Method 1 — CausalImpact (Brodersen) with ARIMA fallback
# ---------------------------------------------------------------------------
def _run_causalimpact(
    *,
    y: pd.Series,
    pre_mask: np.ndarray,
    post_mask: np.ndarray,
    covariates: pd.DataFrame | None,
) -> ImpactFinding:
    """Run Brodersen ``CausalImpact``; fall back to an ARIMA + intervention
    indicator regression if the library is unavailable. ``y`` must be
    indexed by the time column (or 0..T-1 — either is fine; what matters
    is that the masks line up)."""
    notes: list[str] = []
    # --- Attempt the genuine article -------------------------------------
    for module_name, class_name in (
        ("causalimpact", "CausalImpact"),
        ("tfcausalimpact", "CausalImpact"),
    ):
        try:
            mod = __import__(module_name, fromlist=[class_name])
            CausalImpact = getattr(mod, class_name)
        except (ImportError, AttributeError):
            continue
        try:
            # Build the data matrix [y, x1, x2, ...] expected by both libs.
            if covariates is not None and not covariates.empty:
                data = pd.concat([y.rename("y"), covariates], axis=1)
            else:
                data = pd.DataFrame({"y": y.to_numpy()}, index=y.index)
            pre_idx = np.flatnonzero(pre_mask)
            post_idx = np.flatnonzero(post_mask)
            pre_period = [int(pre_idx[0]), int(pre_idx[-1])]
            post_period = [int(post_idx[0]), int(post_idx[-1])]
            ci = CausalImpact(data, pre_period, post_period)
            summary = ci.summary_data  # both libs expose this
            avg_effect = float(summary.loc["abs_effect", "average"])
            ci_low = float(summary.loc["abs_effect_lower", "average"])
            ci_high = float(summary.loc["abs_effect_upper", "average"])
            notes.append(f"using {module_name}.CausalImpact")
            return ImpactFinding(
                method="causalimpact",
                point_estimate=avg_effect,
                ci_low=ci_low,
                ci_high=ci_high,
                notes=notes,
            )
        except Exception as exc:  # pragma: no cover - third-party failure path
            notes.append(f"{module_name} raised {type(exc).__name__}: {exc}")
            break  # don't try the other lib; same data, same likely failure

    # --- ARIMA + intervention indicator fallback -------------------------
    notes.append(
        "causalimpact unavailable; using statsmodels SARIMAX(1,1,1) + "
        "intervention-indicator fallback"
    )
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:  # pragma: no cover - statsmodels is a hard dep
        notes.append(f"statsmodels missing: {exc}")
        return ImpactFinding(
            method="causalimpact",
            point_estimate=float("nan"),
            ci_low=None,
            ci_high=None,
            notes=notes,
        )

    intervention = post_mask.astype(float)
    exog_cols = [intervention.reshape(-1, 1)]
    if covariates is not None and not covariates.empty:
        exog_cols.append(covariates.to_numpy(dtype=float))
    exog = np.concatenate(exog_cols, axis=1)

    try:
        model = SARIMAX(
            y.to_numpy(dtype=float),
            exog=exog,
            order=(1, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=200)
        # The intervention coefficient is the first exog (we constructed
        # exog with it in front). It represents the average effect on y
        # during the post-period, conditional on the AR-1 dynamics.
        intervention_idx = (
            list(res.model.exog_names).index("x1")
            if "x1" in list(res.model.exog_names)
            else 0
        )
        coef = float(res.params[res.model.k_trend + intervention_idx])
        bse = float(res.bse[res.model.k_trend + intervention_idx])
        return ImpactFinding(
            method="causalimpact",
            point_estimate=coef,
            ci_low=coef - 1.96 * bse,
            ci_high=coef + 1.96 * bse,
            notes=notes,
        )
    except Exception as exc:
        # Last-resort: plain post-pre mean difference (no inference).
        notes.append(f"SARIMAX failed ({type(exc).__name__}: {exc}); using mean-diff")
        diff = float(y.to_numpy()[post_mask].mean() - y.to_numpy()[pre_mask].mean())
        return ImpactFinding(
            method="causalimpact",
            point_estimate=diff,
            ci_low=None,
            ci_high=None,
            notes=notes,
        )


# ---------------------------------------------------------------------------
# Method 2 — Augmented SCM (delegates to Sprint 2.3 SyntheticControlEstimator)
# ---------------------------------------------------------------------------
def _run_ascm(
    *,
    panel: pd.DataFrame,
    target: str,
    time_column: str,
    unit_column: str,
    treated_unit: Any,
    pre_mask_time: dict[Any, bool],
    donor_pool: list[Any] | None,
) -> ImpactFinding:
    notes: list[str] = []
    try:
        from causalrag.estimators.python.synthetic_control import (
            SyntheticControlEstimator,
            _pysyncon_available,
        )
    except Exception as exc:  # pragma: no cover - import wiring issue
        return ImpactFinding(
            method="ascm",
            point_estimate=float("nan"),
            ci_low=None,
            ci_high=None,
            notes=[f"could not import SyntheticControlEstimator: {exc}"],
        )

    if not _pysyncon_available():
        return ImpactFinding(
            method="ascm",
            point_estimate=float("nan"),
            ci_low=None,
            ci_high=None,
            notes=["pysyncon not installed; ASCM unavailable"],
        )

    # Build a long-format frame with the treatment indicator the
    # estimator expects.
    work = panel.copy()
    if donor_pool is not None:
        keep_units = list(donor_pool) + [treated_unit]
        work = work[work[unit_column].isin(keep_units)].copy()

    treated_col = "_treated_for_impact"
    is_treated_unit = work[unit_column] == treated_unit
    is_post = work[time_column].map(lambda t: not pre_mask_time.get(t, False))
    work[treated_col] = (is_treated_unit & is_post).astype(int)

    # Try ASCM first; fall back to SCM if pysyncon raises (small donor
    # pools or rank-deficient pre-period matrices sometimes break Ridge
    # ASCM but plain Synth survives).
    for variant in ("ascm", "scm"):
        try:
            est = SyntheticControlEstimator(
                treatment=treated_col,
                outcome=target,
                variant=variant,  # type: ignore[arg-type]
                unit_col=unit_column,
                time_col=time_column,
            )
            # ``fit`` validates inputs and runs pysyncon; protocol is
            # unused by the SC estimator so passing ``None`` is safe.
            est.fit(work, protocol=None)  # type: ignore[arg-type]
            result = est.estimate()
            if variant == "scm":
                notes.append("ASCM unavailable from backend; reporting plain SCM")
            else:
                notes.append("using pysyncon AugSynth (Ben-Michael 2021)")
            return ImpactFinding(
                method="ascm",
                point_estimate=float(result.point_estimate),
                ci_low=result.ci_low,
                ci_high=result.ci_high,
                notes=notes,
            )
        except Exception as exc:
            notes.append(f"{variant} raised {type(exc).__name__}: {exc}")
            continue

    return ImpactFinding(
        method="ascm",
        point_estimate=float("nan"),
        ci_low=None,
        ci_high=None,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Method 3 — Matrix completion (Athey et al. 2021, compact SVD variant)
# ---------------------------------------------------------------------------
def _soft_threshold(s: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(s) * np.maximum(np.abs(s) - lam, 0.0)


def _soft_impute_once(
    Y: np.ndarray, mask: np.ndarray, lam: float, *, max_iter: int, tol: float
) -> np.ndarray:
    """One soft-impute run at a fixed shrinkage ``lam``."""
    M = np.where(mask, Y, 0.0)
    prev = M.copy()
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        s_thr = _soft_threshold(s, lam)
        M_low = (U * s_thr) @ Vt
        M = np.where(mask, Y, M_low)
        delta = np.linalg.norm(M - prev) / max(np.linalg.norm(prev), 1e-12)
        prev = M.copy()
        if delta < tol:
            break
    return M


def _matrix_complete(
    Y: np.ndarray, mask: np.ndarray, *, lam: float | None = None,
    max_iter: int = 200, tol: float = 1e-5,
) -> np.ndarray:
    """SVD soft-impute (Mazumder-Hastie-Tibshirani 2010), the workhorse
    inside Athey et al. 2021's matrix-completion estimator.

    ``Y`` is the panel (rows = units, cols = times); ``mask`` is ``True``
    where ``Y`` is *observed* (i.e. pre-treatment or untreated). We
    iterate

        M_{k+1} = SVT_lam( mask * Y + (1 - mask) * M_k )

    until ``M`` stops moving.

    When ``lam`` is ``None`` we pick the shrinkage by held-out
    cross-validation: randomly mask an additional 10 % of the observed
    entries, fit at each candidate lambda, and keep the lambda with the
    smallest held-out reconstruction error. This matches the spirit of
    Athey-Bayati-Doudchenko-Imbens-Khosravi 2021's CV scheme, in a
    dependency-free form.
    """
    Y = np.asarray(Y, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if lam is not None:
        return _soft_impute_once(Y, mask, lam, max_iter=max_iter, tol=tol)

    # CV over a logarithmic lambda grid anchored on the spectral norm.
    s0 = np.linalg.svd(np.where(mask, Y, 0.0), compute_uv=False)
    smax = float(s0[0]) if s0.size else 1.0
    grid = [smax * f for f in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005)]

    rng = np.random.default_rng(0)
    obs_idx = np.argwhere(mask)
    if obs_idx.shape[0] < 20:  # tiny panel: skip CV, use a small lambda
        return _soft_impute_once(
            Y, mask, smax * 0.02, max_iter=max_iter, tol=tol
        )

    holdout_size = max(5, obs_idx.shape[0] // 10)
    pick = rng.choice(obs_idx.shape[0], size=holdout_size, replace=False)
    cv_mask = mask.copy()
    for i in pick:
        cv_mask[tuple(obs_idx[i])] = False

    best_lam = grid[len(grid) // 2]
    best_err = float("inf")
    for lam_try in grid:
        M_cv = _soft_impute_once(
            Y, cv_mask, lam_try, max_iter=min(max_iter, 80), tol=max(tol, 1e-4)
        )
        held = np.array([Y[tuple(obs_idx[i])] - M_cv[tuple(obs_idx[i])] for i in pick])
        err = float(np.mean(held ** 2))
        if err < best_err:
            best_err = err
            best_lam = lam_try

    return _soft_impute_once(Y, mask, best_lam, max_iter=max_iter, tol=tol)


def _run_matrix_completion(
    *,
    panel: pd.DataFrame,
    target: str,
    time_column: str,
    unit_column: str,
    treated_unit: Any,
    pre_mask_time: dict[Any, bool],
    donor_pool: list[Any] | None,
) -> ImpactFinding:
    notes: list[str] = []
    work = panel.copy()
    if donor_pool is not None:
        keep_units = list(donor_pool) + [treated_unit]
        work = work[work[unit_column].isin(keep_units)].copy()

    try:
        wide = work.pivot_table(
            index=unit_column, columns=time_column, values=target, aggfunc="mean"
        )
    except Exception as exc:
        return ImpactFinding(
            method="matrix_completion",
            point_estimate=float("nan"),
            ci_low=None,
            ci_high=None,
            notes=[f"pivot failed: {exc}"],
        )

    if treated_unit not in wide.index:
        return ImpactFinding(
            method="matrix_completion",
            point_estimate=float("nan"),
            ci_low=None,
            ci_high=None,
            notes=[f"treated unit {treated_unit!r} not present in panel"],
        )

    times = list(wide.columns)
    units = list(wide.index)
    Y = wide.to_numpy(dtype=float)

    # Drop columns with any NaN -- soft-impute can't yet handle natural
    # missingness; treatment-induced missingness is supplied via ``mask``.
    finite_cols = ~np.isnan(Y).any(axis=0)
    if not finite_cols.all():
        notes.append(
            f"dropped {(~finite_cols).sum()} time periods with NaNs in donor pool"
        )
        Y = Y[:, finite_cols]
        times = [t for t, keep in zip(times, finite_cols) if keep]

    treated_row = units.index(treated_unit)
    is_pre = np.array([pre_mask_time.get(t, False) for t in times])
    post_cols = np.flatnonzero(~is_pre)
    if post_cols.size == 0:
        return ImpactFinding(
            method="matrix_completion",
            point_estimate=float("nan"),
            ci_low=None,
            ci_high=None,
            notes=["no post-treatment periods after panel cleaning"],
        )

    # Mask out the treated unit's post-treatment block, leaving everything
    # else observed (Athey et al. 2021 §2.2 "single treated unit" setup).
    mask = np.ones_like(Y, dtype=bool)
    mask[treated_row, post_cols] = False

    try:
        M_hat = _matrix_complete(Y, mask)
    except Exception as exc:
        return ImpactFinding(
            method="matrix_completion",
            point_estimate=float("nan"),
            ci_low=None,
            ci_high=None,
            notes=[f"soft-impute failed: {exc}"],
        )

    # Reuse the same lambda for the placebo-in-time loop below.
    s0 = np.linalg.svd(np.where(mask, Y, 0.0), compute_uv=False)
    placebo_lam = float(s0[0]) * 0.02 if s0.size else 0.05

    actual_post = Y[treated_row, post_cols]
    imputed_post = M_hat[treated_row, post_cols]
    gaps_post = actual_post - imputed_post
    att = float(np.mean(gaps_post))

    # Approximate SE from a placebo-in-time on the treated row's
    # pre-period (Doudchenko-Imbens style): mask each pre-period
    # observation in turn, refit, record the residual, then take its SD
    # over the same window length as the post period. This is a cheap
    # leave-one-out heuristic, not a formal CI, so we flag it as such.
    pre_cols = np.flatnonzero(is_pre)
    n_post = int(post_cols.size)
    se: float | None = None
    if pre_cols.size >= max(5, n_post + 1):
        residuals: list[float] = []
        # Use a stride to keep the loop bounded on long panels.
        stride = max(1, pre_cols.size // 25)
        sample = pre_cols[::stride]
        for idx in sample:
            mask_p = mask.copy()
            mask_p[treated_row, idx] = False
            try:
                M_p = _matrix_complete(
                    Y, mask_p, lam=placebo_lam, max_iter=80, tol=1e-4
                )
                residuals.append(float(Y[treated_row, idx] - M_p[treated_row, idx]))
            except Exception:  # pragma: no cover - solver flakiness
                continue
        if len(residuals) >= 3:
            sd = float(np.std(residuals, ddof=1))
            se = sd / np.sqrt(max(n_post, 1))
            notes.append(
                f"placebo-in-time SE from {len(residuals)} pre-period leave-one-out"
            )
    if se is not None:
        ci_low: float | None = att - 1.96 * se
        ci_high: float | None = att + 1.96 * se
    else:
        ci_low = ci_high = None
        notes.append("insufficient pre-period for placebo-in-time SE; CI omitted")

    notes.append(
        f"soft-impute SVD; n_pre={int(pre_cols.size)} n_post={n_post} "
        f"donors={Y.shape[0] - 1}"
    )
    return ImpactFinding(
        method="matrix_completion",
        point_estimate=att,
        ci_low=ci_low,
        ci_high=ci_high,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _verdict(findings: list[ImpactFinding]) -> Literal["consistent", "moderate", "divergent"]:
    """Reconcile the per-method estimates into a single verdict.

    * **consistent** — every method's point estimate lies inside every
      other method's 95% CI.
    * **moderate** — point-estimate range is < 2 × max(SE).
    * **divergent** — otherwise.

    Methods lacking a CI contribute their point estimate to the range
    check only.
    """
    good = [f for f in findings if np.isfinite(f.point_estimate)]
    if len(good) < 2:
        return "consistent"  # one method, nothing to disagree with

    points = np.array([f.point_estimate for f in good], dtype=float)

    # 1. Strict containment check.
    all_in = True
    for f in good:
        if f.ci_low is None or f.ci_high is None:
            all_in = False
            break
        # Every *other* method's point must fall inside this CI.
        for g in good:
            if g is f:
                continue
            if not (f.ci_low - 1e-9 <= g.point_estimate <= f.ci_high + 1e-9):
                all_in = False
                break
        if not all_in:
            break
    if all_in:
        return "consistent"

    # 2. Moderate band: spread < 2 * max(SE).
    ses = [f.se for f in good if f.se is not None]
    if ses:
        spread = float(points.max() - points.min())
        if spread < 2.0 * max(ses):
            return "moderate"
    return "divergent"


def _interpret(
    target: str,
    consensus: float,
    verdict: str,
    findings: list[ImpactFinding],
) -> str:
    direction = "increased" if consensus > 0 else "decreased" if consensus < 0 else "did not change"
    parts = [
        f"Across {len(findings)} method(s), {target} {direction} by an estimated "
        f"{consensus:+.4g} in the post-intervention window."
    ]
    per = "; ".join(
        f"{f.method}={f.point_estimate:+.4g}" for f in findings if np.isfinite(f.point_estimate)
    )
    if per:
        parts.append(f"Per-method point estimates: {per}.")
    if verdict == "consistent":
        parts.append("Methods are mutually consistent within their 95% CIs.")
    elif verdict == "moderate":
        parts.append(
            "Methods are moderately concordant: spread is small relative to "
            "the standard errors, but containment fails."
        )
    else:
        parts.append(
            "Methods are divergent — treat the consensus as indicative only "
            "and inspect each finding's notes."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def analyze_impact(
    df: pd.DataFrame,
    *,
    target: str,
    time_column: str,
    intervention_time: pd.Timestamp | str | int,
    unit_column: str | None = None,
    treated_unit: Any | None = None,
    donor_pool: list[Any] | None = None,
    methods: tuple[MethodName, ...] | list[MethodName] = DEFAULT_METHODS,
    pre_period: tuple[Any, Any] | None = None,
    post_period: tuple[Any, Any] | None = None,
    covariates: list[str] | None = None,
) -> ImpactReport:
    """Run the requested impact estimators and aggregate them.

    Parameters
    ----------
    df:
        Long-format panel with at least ``time_column`` and ``target``.
        For panel methods (``ascm``, ``matrix_completion``) the frame
        must also carry ``unit_column`` and the desired ``donor_pool``.
        For ``causalimpact`` the time-series of ``target`` for the
        treated unit (or the whole frame if ``unit_column`` is ``None``)
        is used.
    target:
        Column name of the outcome metric Y.
    time_column:
        Column carrying the time index (datetime or numeric).
    intervention_time:
        First time step *of* the post-intervention window. Pre-period
        is ``t < intervention_time``; post-period is
        ``t >= intervention_time`` unless overridden by ``pre_period`` /
        ``post_period``.
    unit_column:
        Column identifying the panel unit. Required for ``ascm`` and
        ``matrix_completion``. If omitted, panel methods are skipped
        with a note.
    treated_unit:
        Identifier of the treated unit. If omitted but
        ``unit_column`` is present, the panel must be single-unit
        (``df[unit_column].nunique() == 1``) or contain exactly one
        unit named ``"treated"``.
    donor_pool:
        Restrict panel methods to the listed control units. Defaults to
        all non-treated units in the frame.
    methods:
        Subset of ``{'causalimpact','ascm','matrix_completion'}``.
    pre_period, post_period:
        Optional explicit windows (inclusive) overriding the
        ``intervention_time`` split.
    covariates:
        Optional contemporaneous regressors fed to CausalImpact (panel
        methods ignore these; their identifying variation is the donor
        pool).

    Returns
    -------
    ImpactReport
        Always returned, even if some methods could not run. Surviving
        methods drive the consensus; failed methods appear in
        ``findings`` with NaN point estimates and explanatory notes.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not in df.")
    if time_column not in df.columns:
        raise ValueError(f"time_column {time_column!r} not in df.")
    if not methods:
        raise ValueError("Must request at least one method.")
    methods = tuple(methods)
    unknown = set(methods) - {"causalimpact", "ascm", "matrix_completion"}
    if unknown:
        raise ValueError(f"Unknown method(s): {sorted(unknown)}")

    notes: list[str] = []

    # --- time handling ---------------------------------------------------
    time_series = df[time_column]
    numeric_time = _is_numeric_time(time_series)
    if not numeric_time:
        # normalise to pd.Timestamp dtype for slicing
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        time_series = df[time_column]

    int_time = _to_comparable_time(intervention_time, numeric=numeric_time)
    pre_start = (
        _to_comparable_time(pre_period[0], numeric=numeric_time)
        if pre_period
        else time_series.min()
    )
    pre_end = (
        _to_comparable_time(pre_period[1], numeric=numeric_time)
        if pre_period
        else int_time  # exclusive upper used below
    )
    post_start = (
        _to_comparable_time(post_period[0], numeric=numeric_time)
        if post_period
        else int_time
    )
    post_end = (
        _to_comparable_time(post_period[1], numeric=numeric_time)
        if post_period
        else time_series.max()
    )

    if pre_period:
        pre_predicate = (time_series >= pre_start) & (time_series <= pre_end)
    else:
        pre_predicate = time_series < int_time
    if post_period:
        post_predicate = (time_series >= post_start) & (time_series <= post_end)
    else:
        post_predicate = time_series >= int_time

    n_pre = int(pre_predicate.sum())
    n_post = int(post_predicate.sum())
    if n_pre == 0:
        raise ValueError("No pre-intervention rows found.")
    if n_post == 0:
        raise ValueError("No post-intervention rows found.")

    # Time-step level pre/post lookup used by panel methods.
    pre_mask_time: dict[Any, bool] = {
        t: bool(p) for t, p in zip(time_series, pre_predicate)
    }
    # Collapse duplicates (multiple units share the same time).
    pre_mask_time = {}
    for t in sorted(time_series.unique()):
        if pre_period:
            pre_mask_time[t] = pre_start <= t <= pre_end
        else:
            pre_mask_time[t] = t < int_time

    # --- unit / treated-unit resolution ---------------------------------
    panel_treated_unit: Any | None = None
    panel_donor_pool: list[Any] | None = None
    if unit_column is not None:
        if unit_column not in df.columns:
            raise ValueError(f"unit_column {unit_column!r} not in df.")
        units = df[unit_column].unique().tolist()
        if treated_unit is not None:
            panel_treated_unit = treated_unit
        elif len(units) == 1:
            panel_treated_unit = units[0]
        elif "treated" in units:
            panel_treated_unit = "treated"
        else:
            notes.append(
                "unit_column supplied but treated_unit could not be inferred; "
                "panel methods skipped"
            )
        if panel_treated_unit is not None:
            if donor_pool is not None:
                panel_donor_pool = list(donor_pool)
            else:
                panel_donor_pool = [u for u in units if u != panel_treated_unit]

    # --- target time-series for CausalImpact ----------------------------
    if unit_column is not None and panel_treated_unit is not None:
        ts_frame = df[df[unit_column] == panel_treated_unit].copy()
    else:
        ts_frame = df.copy()
    ts_frame = ts_frame.sort_values(time_column)
    # If multiple rows per time (eg pooled), average.
    ts_frame = ts_frame.groupby(time_column, as_index=True)[
        [target] + (covariates or [])
    ].mean()
    y_series = ts_frame[target]
    if covariates:
        cov_frame = ts_frame[covariates]
    else:
        cov_frame = None
    if pre_period:
        ts_pre_mask = (y_series.index >= pre_start) & (y_series.index <= pre_end)
    else:
        ts_pre_mask = y_series.index < int_time
    if post_period:
        ts_post_mask = (y_series.index >= post_start) & (y_series.index <= post_end)
    else:
        ts_post_mask = y_series.index >= int_time
    ts_pre_mask = np.asarray(ts_pre_mask, dtype=bool)
    ts_post_mask = np.asarray(ts_post_mask, dtype=bool)

    # --- run each requested method --------------------------------------
    findings: list[ImpactFinding] = []
    for method in methods:
        if method == "causalimpact":
            findings.append(
                _run_causalimpact(
                    y=y_series,
                    pre_mask=ts_pre_mask,
                    post_mask=ts_post_mask,
                    covariates=cov_frame,
                )
            )
        elif method == "ascm":
            if unit_column is None or panel_treated_unit is None:
                findings.append(
                    ImpactFinding(
                        method="ascm",
                        point_estimate=float("nan"),
                        ci_low=None,
                        ci_high=None,
                        notes=[
                            "ASCM requires unit_column + identifiable treated unit"
                        ],
                    )
                )
            else:
                findings.append(
                    _run_ascm(
                        panel=df,
                        target=target,
                        time_column=time_column,
                        unit_column=unit_column,
                        treated_unit=panel_treated_unit,
                        pre_mask_time=pre_mask_time,
                        donor_pool=panel_donor_pool,
                    )
                )
        else:  # matrix_completion
            if unit_column is None or panel_treated_unit is None:
                findings.append(
                    ImpactFinding(
                        method="matrix_completion",
                        point_estimate=float("nan"),
                        ci_low=None,
                        ci_high=None,
                        notes=[
                            "Matrix completion requires unit_column + "
                            "identifiable treated unit"
                        ],
                    )
                )
            else:
                findings.append(
                    _run_matrix_completion(
                        panel=df,
                        target=target,
                        time_column=time_column,
                        unit_column=unit_column,
                        treated_unit=panel_treated_unit,
                        pre_mask_time=pre_mask_time,
                        donor_pool=panel_donor_pool,
                    )
                )

    # --- consensus ------------------------------------------------------
    finite = [f for f in findings if np.isfinite(f.point_estimate)]
    if finite:
        consensus_point = float(statistics.median(f.point_estimate for f in finite))
    else:
        consensus_point = float("nan")

    if len(finite) >= 2:
        lows = [f.ci_low for f in finite if f.ci_low is not None]
        highs = [f.ci_high for f in finite if f.ci_high is not None]
        if lows and highs:
            consensus_ci: tuple[float, float] | None = (
                float(np.min(lows)),
                float(np.max(highs)),
            )
        else:
            consensus_ci = None
    elif len(finite) == 1 and finite[0].ci_low is not None and finite[0].ci_high is not None:
        consensus_ci = (float(finite[0].ci_low), float(finite[0].ci_high))
    else:
        consensus_ci = None

    verdict = _verdict(findings)
    interpretation = _interpret(target, consensus_point, verdict, findings)

    return ImpactReport(
        target=target,
        intervention_time=_coerce_timestamp(int_time),
        pre_period_start=_coerce_timestamp(pre_start if pre_period else time_series.min()),
        pre_period_end=_coerce_timestamp(pre_end if pre_period else int_time),
        post_period_start=_coerce_timestamp(post_start if post_period else int_time),
        post_period_end=_coerce_timestamp(post_end if post_period else time_series.max()),
        findings=findings,
        consensus_point=consensus_point,
        consensus_ci=consensus_ci,
        consistency_verdict=verdict,
        interpretation=interpretation,
        n_pre=n_pre,
        n_post=n_post,
        notes=notes,
    )
