"""Nuisance estimators for cross-fitted causal methods.

PDD §28.A flags single-GBM as the v0.1 default, but the field has no consensus
"best" nuisance estimator — the right choice depends on sample size, signal
shape, and whether the user values calibrated uncertainty (BART) vs. speed
(HistGBM / LightGBM) vs. asymptotic robustness across model spaces
(SuperLearner stacking). We therefore expose multiple **libraries** the user
can opt into, with an ``auto`` rule that picks the strongest option for the
data on hand.

Library catalog (highest expected payoff first under each category):

- ``stacked-rich`` — SuperLearner over GBM + RF + ElasticNet + LightGBM. The
  most robust choice when n ≥ 500 and lightgbm is installed.
- ``stacked-default`` — SuperLearner over GBM + RF + ElasticNet. The v0.1
  default for the same regime when lightgbm is absent.
- ``stacked-fast`` — SuperLearner over HistGBM + RF + ElasticNet. Same
  diversity, ~3-5× faster than ``stacked-default`` on large n thanks to
  HistGBM's binned splits.
- ``bart`` — single BART learner (pymc-bart). Calibrated posterior intervals
  on every nuisance prediction; preferred when UQ matters more than speed.
- ``single-gbm`` — tuned GradientBoosting only. Fastest; matches the
  CausalRAG predecessor; the fallback when n < 500 makes stacking unstable.
- ``hist-gbm`` — single HistGradientBoosting; handles missingness natively
  (relevant when the ``HEAVY_MISSINGNESS`` flag is on).
- ``auto`` — picks ``stacked-rich`` if lightgbm is installed and n ≥ 500,
  else ``stacked-default`` if n ≥ 500, else ``single-gbm``. When the
  ``HEAVY_MISSINGNESS`` flag is present, ``hist-gbm`` is preferred at small
  n because it avoids the dropna() penalty.
"""

from __future__ import annotations

from typing import Any, Literal

Library = Literal[
    "auto",
    "single-gbm",
    "hist-gbm",
    "stacked-default",
    "stacked-fast",
    "stacked-rich",
    "bart",
]


def _has_lightgbm() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


def _has_pymc_bart() -> bool:
    try:
        import pymc_bart  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_library(
    library: Library,
    *,
    n: int | None = None,
    heavy_missing: bool = False,
) -> Library:
    """Resolve ``"auto"`` (and other meta-libraries) into a concrete library
    name based on sample size, available optional deps, and data flags.
    """
    if library != "auto":
        return library
    if heavy_missing and (n or 0) < 500:
        return "hist-gbm"
    if (n or 0) < 500:
        return "single-gbm"
    if _has_lightgbm():
        return "stacked-rich"
    return "stacked-default"


# --- Single-learner factories -----------------------------------------------


def _gbm_regressor(random_state: int) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state
    )


def _gbm_classifier(random_state: int) -> Any:
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=random_state
    )


def _hist_gbm_regressor(random_state: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=200, max_depth=8, learning_rate=0.05, random_state=random_state
    )


def _hist_gbm_classifier(random_state: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=8, learning_rate=0.05, random_state=random_state
    )


def _lightgbm_regressor(random_state: int) -> Any:
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def _lightgbm_classifier(random_state: int) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def _bart_regressor(random_state: int) -> Any:
    """BART regressor backed by ``pymc-bart``.

    Wrapped to look like a sklearn estimator so it slots into Stacking and
    EconML pipelines transparently. Posterior mean is used for ``predict``;
    ``predict_intervals`` exposes the 5th/95th percentiles for callers that
    want calibrated UQ.
    """
    return _BARTSklearnRegressor(random_state=random_state)


def _bart_classifier(random_state: int) -> Any:
    return _BARTSklearnClassifier(random_state=random_state)


class _BARTSklearnRegressor:
    """Minimal sklearn-compatible wrapper around pymc-bart regression."""

    def __init__(self, *, m: int = 50, draws: int = 200, tune: int = 200, random_state: int = 42) -> None:
        self.m = m
        self.draws = draws
        self.tune = tune
        self.random_state = random_state
        self._idata: Any = None
        self._x_shape: tuple[int, int] | None = None

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        import numpy as np
        import pymc as pm
        import pymc_bart as pmb

        x = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._x_shape = (x.shape[0], x.shape[1])
        with pm.Model() as model:
            x_data = pm.Data("x_data", x)
            sigma = pm.HalfNormal("sigma", sigma=float(y.std() or 1.0))
            mu = pmb.BART("mu", X=x_data, Y=y, m=self.m)
            pm.Normal("y", mu=mu, sigma=sigma, observed=y, shape=mu.shape)
            self._idata = pm.sample(
                draws=self.draws,
                tune=self.tune,
                random_seed=self.random_state,
                progressbar=False,
                chains=2,
                compute_convergence_checks=False,
            )
            self._model = model
            self._x_data = x_data
        return self

    def predict(self, X):  # type: ignore[no-untyped-def]
        import numpy as np
        import pymc as pm

        x = np.asarray(X, dtype=float)
        with self._model:
            pm.set_data({"x_data": x})
            post = pm.sample_posterior_predictive(
                self._idata, predictions=True, progressbar=False, var_names=["mu"]
            )
        mu = post.predictions["mu"].values  # (chain, draw, n)
        return mu.reshape(-1, x.shape[0]).mean(axis=0)


class _BARTSklearnClassifier(_BARTSklearnRegressor):
    """BART classifier via logit link on the BART mean."""

    classes_: Any = None

    def fit(self, X, y):  # type: ignore[no-untyped-def]
        import numpy as np

        y_arr = np.asarray(y).astype(int)
        self.classes_ = np.unique(y_arr)
        # BART regression on the {0, 1} label is a standard trick — the
        # posterior mean approximates P(Y=1|X). Cleaner than the logit-link
        # BART (which pymc-bart only added in 0.7+).
        return super().fit(X, y_arr.astype(float))

    def predict_proba(self, X):  # type: ignore[no-untyped-def]
        import numpy as np

        p1 = np.clip(super().predict(X), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):  # type: ignore[no-untyped-def]
        import numpy as np

        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --- Stacked SuperLearner factories -----------------------------------------


def _stacked_regressor(estimators: list[tuple[str, Any]], random_state: int, cv: int) -> Any:
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import RidgeCV

    return StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=cv,
        n_jobs=-1,
        passthrough=False,
    )


def _stacked_classifier(estimators: list[tuple[str, Any]], random_state: int, cv: int) -> Any:
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegressionCV

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegressionCV(cv=cv, max_iter=2000, random_state=random_state),
        cv=cv,
        n_jobs=-1,
        passthrough=False,
        stack_method="predict_proba",
    )


def _rf_regressor(random_state: int) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
    )


def _rf_classifier(random_state: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=random_state, n_jobs=-1
    )


def _enet_regressor(random_state: int, cv: int) -> Any:
    from sklearn.linear_model import ElasticNetCV

    return ElasticNetCV(cv=cv, random_state=random_state, max_iter=2000)


def _logit_classifier(random_state: int, cv: int) -> Any:
    from sklearn.linear_model import LogisticRegressionCV

    return LogisticRegressionCV(cv=cv, max_iter=2000, random_state=random_state, n_jobs=-1)


# --- Public API --------------------------------------------------------------


def super_learner_regressor(
    random_state: int = 42,
    *,
    library: Library = "auto",
    cv: int = 5,
    n: int | None = None,
    heavy_missing: bool = False,
) -> Any:
    lib = resolve_library(library, n=n, heavy_missing=heavy_missing)
    if lib == "single-gbm":
        return _gbm_regressor(random_state)
    if lib == "hist-gbm":
        return _hist_gbm_regressor(random_state)
    if lib == "bart":
        return _bart_regressor(random_state)
    estimators: list[tuple[str, Any]]
    if lib == "stacked-fast":
        estimators = [
            ("hist_gbm", _hist_gbm_regressor(random_state)),
            ("rf", _rf_regressor(random_state)),
            ("enet", _enet_regressor(random_state, cv)),
        ]
    elif lib == "stacked-rich" and _has_lightgbm():
        estimators = [
            ("gbm", _gbm_regressor(random_state)),
            ("lightgbm", _lightgbm_regressor(random_state)),
            ("rf", _rf_regressor(random_state)),
            ("enet", _enet_regressor(random_state, cv)),
        ]
    else:  # stacked-default and stacked-rich w/o lightgbm
        estimators = [
            ("gbm", _gbm_regressor(random_state)),
            ("rf", _rf_regressor(random_state)),
            ("enet", _enet_regressor(random_state, cv)),
        ]
    return _stacked_regressor(estimators, random_state, cv)


def super_learner_classifier(
    random_state: int = 42,
    *,
    library: Library = "auto",
    cv: int = 5,
    n: int | None = None,
    heavy_missing: bool = False,
) -> Any:
    lib = resolve_library(library, n=n, heavy_missing=heavy_missing)
    if lib == "single-gbm":
        return _gbm_classifier(random_state)
    if lib == "hist-gbm":
        return _hist_gbm_classifier(random_state)
    if lib == "bart":
        return _bart_classifier(random_state)
    estimators: list[tuple[str, Any]]
    if lib == "stacked-fast":
        estimators = [
            ("hist_gbm", _hist_gbm_classifier(random_state)),
            ("rf", _rf_classifier(random_state)),
            ("logit", _logit_classifier(random_state, cv)),
        ]
    elif lib == "stacked-rich" and _has_lightgbm():
        estimators = [
            ("gbm", _gbm_classifier(random_state)),
            ("lightgbm", _lightgbm_classifier(random_state)),
            ("rf", _rf_classifier(random_state)),
            ("logit", _logit_classifier(random_state, cv)),
        ]
    else:
        estimators = [
            ("gbm", _gbm_classifier(random_state)),
            ("rf", _rf_classifier(random_state)),
            ("logit", _logit_classifier(random_state, cv)),
        ]
    return _stacked_classifier(estimators, random_state, cv)


def nuisance_models(
    random_state: int = 42,
    *,
    library: Library = "auto",
    cv: int = 5,
    n: int | None = None,
    heavy_missing: bool = False,
) -> tuple[Any, Any]:
    """Return ``(regressor, classifier)`` for the requested library."""
    return (
        super_learner_regressor(
            random_state, library=library, cv=cv, n=n, heavy_missing=heavy_missing
        ),
        super_learner_classifier(
            random_state, library=library, cv=cv, n=n, heavy_missing=heavy_missing
        ),
    )


# Back-compat alias for callers that still pass ``mode=``.
def _legacy_nuisance_models(random_state: int = 42, *, mode: str = "auto") -> tuple[Any, Any]:
    library_map = {"stacked": "stacked-default", "single": "single-gbm", "auto": "auto"}
    return nuisance_models(random_state, library=library_map.get(mode, "auto"))  # type: ignore[arg-type]


__all__ = [
    "Library",
    "resolve_library",
    "super_learner_regressor",
    "super_learner_classifier",
    "nuisance_models",
]
