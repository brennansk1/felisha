"""LinearDML wrapper — first Python-native estimator (PDD §29.1).

Ported from the predecessor CausalRAG (TCGA-BRCA) ``causalrag.causal.dml_engine``,
generalized to drop the oncology-specific column resolution. The architectural
choices preserved from the original:

- GradientBoosting nuisance models (regressor for outcome, classifier for binary
  treatment propensity). PDD §28.A flags GBM as the v0.1 default; the original
  poster experiment used 100-estimator, depth-4, lr=0.1.
- Cross-fitted residualization via EconML's ``LinearDML(cv=k)``.
- p-value from ``effect_inference().pvalue()`` with a normal-approximation
  fallback derived from the 95% CI when inference is unavailable.

The wrapper is registered in the global estimator registry at import time.
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

_Z_975 = 1.959963984540054


class LinearDMLEstimator:
    """LinearDML for binary or continuous treatment, continuous outcome.

    Targets ATE under conditional ignorability with a linear final-stage model.
    CATE is supported when ``modifiers`` are supplied to the constructor.
    """

    id: str = "python.dml.linear"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE", "CATE")
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.TIME_VARYING_TREATMENT,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 100
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
        discrete_treatment: bool | None = None,
        n_splits: int = 5,
        random_state: int = 42,
        alpha: float = 0.05,
        nuisance_library: str = "auto",
        heavy_missing: bool = False,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.discrete_treatment = discrete_treatment
        self.n_splits = n_splits
        self.random_state = random_state
        self.alpha = alpha
        self.nuisance_library = nuisance_library
        self.heavy_missing = heavy_missing

        self._fitted_model: Any = None
        self._x_used: np.ndarray | None = None
        self._n_used: int = 0
        self._fit_seconds: float | None = None
        self._backend_version: str | None = None
        self._effective_nuisance_library: str | None = None

    # --- Protocol methods --------------------------------------------------------

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> LinearDMLEstimator:
        try:
            import econml
            from econml.dml import LinearDML

            from causalrag.estimators.python.nuisance import (
                super_learner_classifier,
                super_learner_regressor,
            )
        except ImportError as e:
            raise RuntimeError(
                "LinearDMLEstimator requires the optional 'estimators' extra: "
                "pip install 'causalrag[estimators]'"
            ) from e

        for col in (self.outcome, self.treatment, *self.confounders, *self.modifiers):
            if col not in data.columns:
                raise ValueError(f"Column not in data: {col!r}")

        needed = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[needed].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"LinearDML requires at least {self.min_sample_size} rows after dropna; "
                f"got {self._n_used}"
            )

        y = df[self.outcome].to_numpy().astype(np.float64)
        t = df[self.treatment].to_numpy().astype(np.float64)
        x = df[list(self.modifiers)].to_numpy().astype(np.float64) if self.modifiers else None
        w = df[list(self.confounders)].to_numpy().astype(np.float64) if self.confounders else None

        unique_t = set(np.unique(t).tolist())
        if self.discrete_treatment is None:
            is_binary = unique_t.issubset({0.0, 1.0}) and len(unique_t) <= 2
        else:
            is_binary = bool(self.discrete_treatment)
            if is_binary and not unique_t.issubset({0.0, 1.0}):
                raise ValueError(
                    f"discrete_treatment=True requires binary {{0, 1}} treatment; got {unique_t}"
                )

        # SuperLearner stacks add an inner CV that multiplies with EconML's
        # outer CV. The ``auto`` resolver downgrades to a single GBM when n
        # is too small for stable stacking, and prefers HistGBM when the
        # heavy-missingness flag is set so we keep native NaN handling.
        from causalrag.estimators.python.nuisance import resolve_library

        resolved = resolve_library(
            self.nuisance_library,  # type: ignore[arg-type]
            n=self._n_used,
            heavy_missing=self.heavy_missing,
        )
        self._effective_nuisance_library = resolved

        model_y = super_learner_regressor(
            self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
        )
        model_t: Any
        if is_binary:
            model_t = super_learner_classifier(
                self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
            )
        else:
            model_t = super_learner_regressor(
                self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
            )

        model = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=is_binary,
            cv=self.n_splits,
            random_state=self.random_state,
        )

        start = time.perf_counter()
        model.fit(Y=y, T=t, X=x, W=w)
        self._fit_seconds = time.perf_counter() - start

        self._fitted_model = model
        self._x_used = x
        self._backend_version = f"econml {econml.__version__}"
        return self

    def estimate(self) -> EstimationResult:
        if self._fitted_model is None:
            raise RuntimeError("Call fit() before estimate().")
        model = self._fitted_model
        x = self._x_used

        ate_value = model.ate(X=x)
        ate_low, ate_high = model.ate_interval(X=x, alpha=self.alpha)
        point = float(np.atleast_1d(ate_value).mean())
        ci_low = float(np.atleast_1d(ate_low).mean())
        ci_high = float(np.atleast_1d(ate_high).mean())
        se = (ci_high - ci_low) / (2 * _Z_975) if ci_high > ci_low else None

        p_value = self._compute_p_value(model, x, point, ci_low, ci_high)

        diagnostics: dict[str, Any] = {
            "discrete_treatment": bool(getattr(model, "discrete_treatment", False)),
            "n_splits": self.n_splits,
            "has_modifiers": bool(self.modifiers),
        }

        if self.modifiers:
            cate = model.effect(X=x)
            cate_low, cate_high = model.effect_interval(X=x, alpha=self.alpha)
            diagnostics["cate_mean"] = float(np.mean(cate))
            diagnostics["cate_ci_low"] = float(np.mean(cate_low))
            diagnostics["cate_ci_high"] = float(np.mean(cate_high))

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="CATE" if self.modifiers else "ATE",
            point_estimate=point,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=self._n_used,
            diagnostics=diagnostics,
            backend_version=self._backend_version,
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"fitted": self._fitted_model is not None, "n_used": self._n_used}

    def refute(self) -> dict[str, Any]:
        # Placebo / subset / unobserved refuters land in Week 3 (§33.110).
        return {}

    # --- Helpers ----------------------------------------------------------------

    def _compute_p_value(
        self,
        model: Any,
        x: np.ndarray | None,
        point: float,
        ci_low: float,
        ci_high: float,
    ) -> float | None:
        try:
            inference = model.effect_inference(X=x)
            p_values = inference.pvalue()
            return float(np.mean(p_values))
        except Exception:
            if ci_high <= ci_low:
                return None
            se = (ci_high - ci_low) / (2 * _Z_975)
            if se <= 0:
                return None
            try:
                from scipy.stats import norm

                z_stat = abs(point) / se
                return float(2 * (1 - norm.cdf(z_stat)))
            except ImportError:
                return None


class CausalForestDMLEstimator(LinearDMLEstimator):
    """Causal Forest DML — non-linear CATE via random forests.

    Inherits the LinearDML fit/estimate plumbing; the only difference is the
    EconML class used at construction time. Required to cover non-linear
    treatment-effect heterogeneity per PDD §29.1.
    """

    id: str = "python.dml.causal_forest"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE", "CATE")
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.TIME_VARYING_TREATMENT,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 200  # forests need more rows than linear DML
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
        n_trees: int = 200,
        discrete_treatment: bool | None = None,
        n_splits: int = 5,
        random_state: int = 42,
        alpha: float = 0.05,
        nuisance_library: str = "auto",
        heavy_missing: bool = False,
    ) -> None:
        super().__init__(
            treatment=treatment,
            outcome=outcome,
            confounders=confounders,
            modifiers=modifiers,
            discrete_treatment=discrete_treatment,
            n_splits=n_splits,
            random_state=random_state,
            alpha=alpha,
            nuisance_library=nuisance_library,
            heavy_missing=heavy_missing,
        )
        self.n_trees = n_trees

    def fit(self, data, protocol):  # type: ignore[override]
        try:
            import econml
            from econml.dml import CausalForestDML

            from causalrag.estimators.python.nuisance import (
                super_learner_classifier,
                super_learner_regressor,
            )
        except ImportError as e:
            raise RuntimeError(
                "CausalForestDMLEstimator requires the optional 'estimators' extra: "
                "pip install 'causalrag[estimators]'"
            ) from e

        for col in (self.outcome, self.treatment, *self.confounders, *self.modifiers):
            if col not in data.columns:
                raise ValueError(f"Column not in data: {col!r}")

        needed = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[needed].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"CausalForestDML requires at least {self.min_sample_size} rows after dropna; "
                f"got {self._n_used}"
            )

        y = df[self.outcome].to_numpy().astype(np.float64)
        t = df[self.treatment].to_numpy().astype(np.float64)
        x = df[list(self.modifiers)].to_numpy().astype(np.float64) if self.modifiers else None
        w = df[list(self.confounders)].to_numpy().astype(np.float64) if self.confounders else None

        unique_t = set(np.unique(t).tolist())
        is_binary = (
            unique_t.issubset({0.0, 1.0}) and len(unique_t) <= 2
            if self.discrete_treatment is None
            else bool(self.discrete_treatment)
        )

        from causalrag.estimators.python.nuisance import resolve_library

        resolved = resolve_library(
            self.nuisance_library,  # type: ignore[arg-type]
            n=self._n_used,
            heavy_missing=self.heavy_missing,
        )
        self._effective_nuisance_library = resolved

        model_y = super_learner_regressor(
            self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
        )
        if is_binary:
            model_t: object = super_learner_classifier(
                self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
            )
        else:
            model_t = super_learner_regressor(
                self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
            )

        model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=is_binary,
            cv=self.n_splits,
            n_estimators=self.n_trees,
            random_state=self.random_state,
        )

        import time

        start = time.perf_counter()
        model.fit(Y=y, T=t, X=x, W=w)
        self._fit_seconds = time.perf_counter() - start
        self._fitted_model = model
        self._x_used = x
        self._backend_version = f"econml {econml.__version__}"
        return self


class SparseLinearDMLEstimator(LinearDMLEstimator):
    """SparseLinearDML — Lasso-regularized final-stage CATE.

    Preferred when the ``HIGH_DIMENSIONAL`` flag is on (p > sqrt(n) after
    one-hot expansion). The nuisance models are still SuperLearner-stacked;
    the difference is the final-stage Lasso shrinkage which regularizes the
    CATE model rather than fitting a dense linear function in the modifiers.
    """

    id: str = "python.dml.sparse_linear"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE", "CATE")
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.TIME_VARYING_TREATMENT,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 100

    def fit(self, data, protocol):  # type: ignore[override]
        try:
            import econml
            from econml.dml import SparseLinearDML

            from causalrag.estimators.python.nuisance import (
                super_learner_classifier,
                super_learner_regressor,
            )
        except ImportError as e:
            raise RuntimeError(
                "SparseLinearDMLEstimator requires the optional 'estimators' extra."
            ) from e

        for col in (self.outcome, self.treatment, *self.confounders, *self.modifiers):
            if col not in data.columns:
                raise ValueError(f"Column not in data: {col!r}")

        needed = [self.outcome, self.treatment, *self.confounders, *self.modifiers]
        df = data[needed].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(
                f"SparseLinearDML requires at least {self.min_sample_size} rows; "
                f"got {self._n_used}"
            )

        y = df[self.outcome].to_numpy().astype(np.float64)
        t = df[self.treatment].to_numpy().astype(np.float64)
        x = df[list(self.modifiers)].to_numpy().astype(np.float64) if self.modifiers else None
        w = df[list(self.confounders)].to_numpy().astype(np.float64) if self.confounders else None

        unique_t = set(np.unique(t).tolist())
        is_binary = (
            unique_t.issubset({0.0, 1.0}) and len(unique_t) <= 2
            if self.discrete_treatment is None
            else bool(self.discrete_treatment)
        )

        from causalrag.estimators.python.nuisance import resolve_library

        resolved = resolve_library(
            self.nuisance_library,  # type: ignore[arg-type]
            n=self._n_used,
            heavy_missing=self.heavy_missing,
        )
        self._effective_nuisance_library = resolved

        model_y = super_learner_regressor(
            self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
        )
        if is_binary:
            model_t: Any = super_learner_classifier(
                self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
            )
        else:
            model_t = super_learner_regressor(
                self.random_state, library=resolved, n=self._n_used, heavy_missing=self.heavy_missing
            )

        model = SparseLinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=is_binary,
            cv=self.n_splits,
            random_state=self.random_state,
        )

        import time

        start = time.perf_counter()
        model.fit(Y=y, T=t, X=x, W=w)
        self._fit_seconds = time.perf_counter() - start
        self._fitted_model = model
        self._x_used = x
        self._backend_version = f"econml {econml.__version__}"
        return self


def _register() -> None:
    for cls in (LinearDMLEstimator, CausalForestDMLEstimator, SparseLinearDMLEstimator):
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
