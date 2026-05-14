"""Meta-learner estimators — T/S/X/DR (PDD §29.1).

Each wraps the corresponding EconML metalearner under the standard
:class:`CausalEstimator` Protocol. All four are ATE/CATE estimators for binary
treatment with a single continuous outcome. The X- and DR-learners are
doubly-robust; the T- and S-learners are simpler baselines.

Implementations share a single helper that prepares Y/T/X/W matrices, fits the
chosen learner, and emits a standardized :class:`EstimationResult`.

CI handling: EconML's metalearners don't always expose ``effect_interval``.
When unavailable, we fall back to an honest non-parametric bootstrap (resample
rows with replacement, refit, take empirical quantiles) rather than silently
emitting ``None``. The ``diagnostics["bootstrap_used"]`` flag tells downstream
consumers which path was taken.

Confounders vs. modifiers: EconML's metalearners take a single ``X`` matrix
and have no separate ``W``. To preserve the W/X distinction we feed
``confounders + modifiers`` as the learner's ``X`` (so adjustment happens) but
report ATE at the marginal mean of the modifiers and expose per-row CATE only
via :meth:`cate_predictions` over modifier grids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import EstimatorEntry, register
from causalrag.core.result import EstimationResult


@dataclass
class _Prepared:
    y: np.ndarray
    t: np.ndarray
    x: np.ndarray | None  # learner-facing feature matrix (W + X concatenated)
    w: np.ndarray | None  # confounders only
    xm: np.ndarray | None  # modifiers only
    n: int


def _prepare(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...],
    modifiers: tuple[str, ...],
    min_n: int,
) -> _Prepared:
    features = tuple(confounders) + tuple(modifiers)
    cols = [outcome, treatment, *features]
    for c in cols:
        if c not in data.columns:
            raise ValueError(f"Column not in data: {c!r}")
    df = data[cols].dropna()
    n = len(df)
    if n < min_n:
        raise ValueError(
            f"Meta-learner requires at least {min_n} rows after dropna; got {n}"
        )
    y = df[outcome].to_numpy().astype(np.float64)
    t = df[treatment].to_numpy().astype(np.float64)
    unique_t = set(np.unique(t).tolist())
    if not unique_t.issubset({0.0, 1.0}):
        raise ValueError(
            f"Meta-learners require binary {{0, 1}} treatment; got {unique_t}"
        )
    x = df[list(features)].to_numpy().astype(np.float64) if features else None
    w = (
        df[list(confounders)].to_numpy().astype(np.float64)
        if confounders
        else None
    )
    xm = (
        df[list(modifiers)].to_numpy().astype(np.float64) if modifiers else None
    )
    return _Prepared(y=y, t=t.astype(int), x=x, w=w, xm=xm, n=n)


def _nuisance_models(
    random_state: int,
    library: str = "auto",
    n: int | None = None,
    heavy_missing: bool = False,
) -> tuple[Any, Any]:
    from causalrag.estimators.python.nuisance import nuisance_models

    return nuisance_models(random_state, library=library, n=n, heavy_missing=heavy_missing)  # type: ignore[arg-type]


class _MetaBase:
    id: str = ""
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE", "CATE")
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
    produces_cate: bool = True
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    _learner_name: str = ""

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        random_state: int = 42,
        nuisance_library: str = "auto",
        heavy_missing: bool = False,
        bootstrap_iterations: int = 200,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.random_state = random_state
        self.nuisance_library = nuisance_library
        self.heavy_missing = heavy_missing
        self.bootstrap_iterations = int(bootstrap_iterations)
        self._fitted: Any = None
        self._prep: _Prepared | None = None
        self._fit_seconds: float | None = None
        self._backend_version: str | None = None
        self._protocol_seed: int | None = None

    def _features(self) -> tuple[str, ...]:
        # Concatenated W + X is fed to EconML's single X argument (metalearners
        # don't expose W). The W/X distinction is preserved by reporting CATE
        # only over the modifier grid in :meth:`cate_predictions`.
        return tuple(self.confounders) + tuple(self.modifiers)

    def _seed_for_bootstrap(self) -> int:
        if self._protocol_seed is not None:
            return int(self._protocol_seed)
        return 12345

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> _MetaBase:
        prep = _prepare(
            data,
            self.treatment,
            self.outcome,
            tuple(self.confounders),
            tuple(self.modifiers),
            self.min_sample_size,
        )
        if prep.x is None:
            raise ValueError("Meta-learners need at least one covariate.")
        from causalrag.estimators.python.nuisance import resolve_library

        self._resolved_library = resolve_library(
            self.nuisance_library,  # type: ignore[arg-type]
            n=prep.n,
            heavy_missing=self.heavy_missing,
        )
        self._resolved_n = prep.n
        # Pull a seed off the protocol if it exposes one (llm.seed exists today;
        # we treat it as best-effort, falling back to 12345 in the bootstrap).
        try:
            self._protocol_seed = int(getattr(protocol.llm, "seed", 0)) or None
        except Exception:
            self._protocol_seed = None

        learner = self._build_learner()
        start = time.perf_counter()
        learner.fit(Y=prep.y, T=prep.t, X=prep.x)
        self._fit_seconds = time.perf_counter() - start
        self._fitted = learner
        self._prep = prep
        import econml

        self._backend_version = f"econml {econml.__version__}"
        return self

    def _nuisance(self) -> tuple[Any, Any]:
        lib = getattr(self, "_resolved_library", "single-gbm")
        n = getattr(self, "_resolved_n", None)
        return _nuisance_models(
            self.random_state, library=lib, n=n, heavy_missing=self.heavy_missing
        )

    # ------------------------------------------------------------------
    # Bootstrap fallback for CIs / SE / p-value
    # ------------------------------------------------------------------
    def _bootstrap_ate_ci(
        self,
        prep: _Prepared,
        alpha: float = 0.05,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        """Non-parametric row-bootstrap CI / SE / p for the ATE.

        Resamples rows with replacement B times, refits the learner, takes the
        mean CATE on the resample as the bootstrap replicate of the ATE, and
        reports empirical-quantile CI plus bootstrap SD as SE. The p-value is
        a two-sided test of H0: ATE = 0 via the bootstrap-SE z statistic.

        Returns ``(None, None, None, None)`` if fewer than two replicates
        succeed (e.g., learner construction repeatedly fails).
        """
        B = max(1, int(self.bootstrap_iterations))
        rng = np.random.default_rng(
            np.random.SeedSequence(self._seed_for_bootstrap())
        )
        n = prep.n
        replicates: list[float] = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            y_b = prep.y[idx]
            t_b = prep.t[idx]
            x_b = prep.x[idx]
            # Need both treatment arms present for the learner to fit.
            if len(np.unique(t_b)) < 2:
                continue
            try:
                learner_b = self._build_learner()
                learner_b.fit(Y=y_b, T=t_b, X=x_b)
                cate_b = learner_b.effect(x_b)
                replicates.append(float(np.mean(cate_b)))
            except Exception:
                continue
        if len(replicates) < 2:
            return None, None, None, None
        arr = np.asarray(replicates, dtype=np.float64)
        lo = float(np.quantile(arr, alpha / 2))
        hi = float(np.quantile(arr, 1 - alpha / 2))
        se = float(np.std(arr, ddof=1))
        # z-based two-sided p relative to 0 using bootstrap SE.
        if se > 0:
            from math import erfc, sqrt

            mean_b = float(np.mean(arr))
            z = abs(mean_b) / se
            p = float(erfc(z / sqrt(2.0)))
        else:
            p = None
        return lo, hi, se, p

    def estimate(self) -> EstimationResult:
        if self._fitted is None or self._prep is None:
            raise RuntimeError("Call fit() before estimate().")
        prep = self._prep
        learner = self._fitted
        cate = learner.effect(prep.x)
        point = float(np.mean(cate))

        ci_low: float | None = None
        ci_high: float | None = None
        se: float | None = None
        p_value: float | None = None
        bootstrap_used = False

        # Prefer EconML's analytic IF-based interval where available.
        try:
            interval = learner.effect_interval(prep.x, alpha=0.05)
            ci_low = float(np.mean(interval[0]))
            ci_high = float(np.mean(interval[1]))
            # Some learners also expose inference summaries; we leave SE/p None
            # rather than fabricating them from the CI half-width.
        except Exception:
            lo, hi, s, p = self._bootstrap_ate_ci(prep, alpha=0.05)
            ci_low, ci_high, se, p_value = lo, hi, s, p
            bootstrap_used = True

        diagnostics: dict[str, Any] = {
            "learner": self._learner_name,
            "cate_available": bool(self.modifiers),
            "bootstrap_used": bootstrap_used,
            "n_confounders": len(self.confounders),
            "n_modifiers": len(self.modifiers),
        }
        if bootstrap_used:
            diagnostics["bootstrap_iterations"] = self.bootstrap_iterations

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="CATE" if self.modifiers else "ATE",
            point_estimate=point,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=prep.n,
            diagnostics=diagnostics,
            backend_version=self._backend_version,
            fit_seconds=self._fit_seconds,
        )

    # ------------------------------------------------------------------
    # Per-row CATE predictions over a modifier grid
    # ------------------------------------------------------------------
    def cate_predictions(self, X_grid: np.ndarray | pd.DataFrame) -> dict[str, np.ndarray]:
        """Return per-row CATE + CI over a modifier grid.

        ``X_grid`` must have one column per modifier in ``self.modifiers``,
        in the same order. Confounders are held at their training-sample mean
        so the fed-to-EconML matrix has the right shape (``confounders +
        modifiers``).

        Returns a dict with keys ``point``, ``ci_low``, ``ci_high``, and
        ``bootstrap_used``. CIs fall back to bootstrap when EconML's per-row
        ``effect_interval`` is unavailable.
        """
        if self._fitted is None or self._prep is None:
            raise RuntimeError("Call fit() before cate_predictions().")
        if not self.modifiers:
            raise ValueError(
                "cate_predictions requires modifiers; estimator was constructed "
                "without any (ATE-only mode)."
            )
        if isinstance(X_grid, pd.DataFrame):
            xm = X_grid[list(self.modifiers)].to_numpy().astype(np.float64)
        else:
            xm = np.asarray(X_grid, dtype=np.float64)
            if xm.ndim == 1:
                xm = xm.reshape(-1, 1)
        if xm.shape[1] != len(self.modifiers):
            raise ValueError(
                f"X_grid must have {len(self.modifiers)} columns "
                f"(one per modifier); got {xm.shape[1]}"
            )

        prep = self._prep
        # Hold confounders at training means.
        if prep.w is not None and prep.w.size:
            w_mean = prep.w.mean(axis=0, keepdims=True)
            w_block = np.repeat(w_mean, repeats=xm.shape[0], axis=0)
            x_full = np.concatenate([w_block, xm], axis=1)
        else:
            x_full = xm
        point = np.asarray(self._fitted.effect(x_full), dtype=np.float64)

        ci_low: np.ndarray
        ci_high: np.ndarray
        bootstrap_used = False
        try:
            interval = self._fitted.effect_interval(x_full, alpha=0.05)
            ci_low = np.asarray(interval[0], dtype=np.float64)
            ci_high = np.asarray(interval[1], dtype=np.float64)
        except Exception:
            ci_low, ci_high = self._bootstrap_cate_intervals(prep, x_full, alpha=0.05)
            bootstrap_used = True

        return {
            "point": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "bootstrap_used": np.array(bootstrap_used),
        }

    def _bootstrap_cate_intervals(
        self,
        prep: _Prepared,
        x_eval: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray]:
        B = max(1, int(self.bootstrap_iterations))
        rng = np.random.default_rng(
            np.random.SeedSequence(self._seed_for_bootstrap())
        )
        n = prep.n
        m = x_eval.shape[0]
        reps: list[np.ndarray] = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            y_b = prep.y[idx]
            t_b = prep.t[idx]
            x_b = prep.x[idx]
            if len(np.unique(t_b)) < 2:
                continue
            try:
                lb = self._build_learner()
                lb.fit(Y=y_b, T=t_b, X=x_b)
                reps.append(np.asarray(lb.effect(x_eval), dtype=np.float64))
            except Exception:
                continue
        if len(reps) < 2:
            return np.full(m, np.nan), np.full(m, np.nan)
        stack = np.stack(reps, axis=0)  # (B', m)
        lo = np.quantile(stack, alpha / 2, axis=0)
        hi = np.quantile(stack, 1 - alpha / 2, axis=0)
        return lo, hi

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._fitted is not None,
            "learner": self._learner_name,
            "cate_available": bool(self.modifiers),
        }

    def refute(self) -> dict[str, Any]:
        return {}

    def _build_learner(self) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError


class TLearnerEstimator(_MetaBase):
    id = "python.meta.t_learner"
    _learner_name = "T-learner"

    def _build_learner(self) -> Any:
        from econml.metalearners import TLearner

        reg, _ = self._nuisance()
        return TLearner(models=reg)


class SLearnerEstimator(_MetaBase):
    id = "python.meta.s_learner"
    _learner_name = "S-learner"

    def _build_learner(self) -> Any:
        from econml.metalearners import SLearner

        reg, _ = self._nuisance()
        return SLearner(overall_model=reg)


class XLearnerEstimator(_MetaBase):
    id = "python.meta.x_learner"
    _learner_name = "X-learner"

    def _build_learner(self) -> Any:
        from econml.metalearners import XLearner

        reg, clf = self._nuisance()
        return XLearner(models=reg, propensity_model=clf)


class DRLearnerEstimator(_MetaBase):
    id = "python.dr.dr_learner"
    _learner_name = "DR-learner"

    def _build_learner(self) -> Any:
        from econml.dr import DRLearner

        reg, clf = self._nuisance()
        return DRLearner(
            model_regression=reg,
            model_propensity=clf,
            model_final=reg,
            cv=5,
            random_state=self.random_state,
        )


def _register() -> None:
    for cls in (
        TLearnerEstimator,
        SLearnerEstimator,
        XLearnerEstimator,
        DRLearnerEstimator,
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
