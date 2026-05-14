"""Estimator catalog. Importing this package registers all bundled estimators."""

from causalrag.estimators.base import CausalEstimator

# Side-effect import — registers entries with the global registry.
from causalrag.estimators.python import dml as _dml  # noqa: F401
from causalrag.estimators.python import meta as _meta  # noqa: F401
from causalrag.estimators.python import ols as _ols  # noqa: F401
from causalrag.estimators.python import bart as _bart  # noqa: F401  # self-skips w/o pymc-bart

# R-bridged estimators self-skip registration when rpy2 / the underlying R
# package isn't available — importing the module is always safe.
def _try_import_rbridge() -> None:
    """Best-effort import of R-bridged wrappers. Any ImportError / R bridge
    error is swallowed so the catalog stays available even without R."""
    for mod in (
        "grf",
        "lmtp",
        "matchit",
        "survival",
        "mediation",
        "bartcause",
        "weighting",
        "marginaleffects",
    ):
        try:
            __import__(f"causalrag.estimators.rbridge.{mod}")
        except Exception:
            pass


_try_import_rbridge()

__all__ = ["CausalEstimator"]
