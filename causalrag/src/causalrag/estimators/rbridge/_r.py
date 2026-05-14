"""Singleton R session + type marshalling for the R bridge (PDD §17).

One R session per Python process, lazily started on first use. All R-bridged
estimators import :func:`r_session` and run their R code through it. The
session caches `library(pkg)` loads and the converted (pandas ↔ R DataFrame)
contexts so we don't re-import on every call.

Design choices anchored to PDD §17:

- **Lazy isolation** — ``r_session()`` only initializes when called. Importing
  ``causalrag.estimators.rbridge`` does NOT incur the R startup cost.
- **Singleton** — multiple concurrent estimators share one R session. Thread
  safety is left to the user (the pipeline runs estimators serially).
- **Type marshalling** — DataFrames flow via ``rpy2.robjects.conversion`` +
  ``rpy2.robjects.pandas2ri`` (auto-converts numeric columns; categorical
  columns become R factors). Numeric arrays flow via ``ro.FloatVector`` /
  ``ro.IntVector``.
- **Error handling** — R errors are caught and re-raised as Python
  ``RBridgeError`` with the R traceback embedded; missing-package errors are
  surfaced as ``RPackageMissing`` with the exact ``install.packages()``
  command to fix.
- **Reproducibility** — :func:`r_session_metadata` returns R version + per-
  package versions; every R-bridged ``EstimationResult`` records these in
  ``r_session_metadata``.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


class RBridgeError(RuntimeError):
    """A raised-by-R error, re-raised on the Python side with traceback."""


class RPackageMissing(RuntimeError):
    """An R package is not installed. Includes the install command."""

    def __init__(self, pkg: str) -> None:
        super().__init__(
            f"R package {pkg!r} is not installed. Install with:\n"
            f"  R -e 'install.packages(\"{pkg}\")'"
        )
        self.pkg = pkg


@dataclass
class RSessionInfo:
    r_version: str
    r_home: str
    rpy2_version: str
    loaded_packages: tuple[str, ...] = ()


_LOCK = threading.Lock()
_R = None
_INFO: RSessionInfo | None = None
_LOADED: set[str] = set()
_CONVERTER = None


def r_session():
    """Return the active rpy2 ``robjects`` module, starting R if needed."""
    global _R, _INFO, _CONVERTER
    if _R is not None:
        return _R
    with _LOCK:
        if _R is not None:
            return _R
        try:
            import rpy2
            import rpy2.robjects as ro
        except ImportError as e:
            raise RBridgeError(
                "rpy2 not installed. Install with: pip install 'causalrag[rbridge]'"
            ) from e
        # Build the pandas↔R converter once and keep a handle.
        from rpy2.robjects import default_converter, pandas2ri
        from rpy2.robjects.conversion import localconverter

        _CONVERTER = default_converter + pandas2ri.converter
        _R = ro
        _INFO = RSessionInfo(
            r_version=list(ro.r("R.version.string"))[0],
            r_home=list(ro.r("R.home()"))[0],
            rpy2_version=getattr(rpy2, "__version__", "?"),
        )
    return _R


def r_session_metadata() -> dict[str, Any]:
    """Return reproducibility metadata for the current R session."""
    if _INFO is None:
        r_session()  # initialize so _INFO is populated
    assert _INFO is not None
    versions: dict[str, str] = {}
    if _R is not None:
        for pkg in sorted(_LOADED):
            try:
                v = list(_R.r(f'as.character(packageVersion("{pkg}"))'))[0]
                versions[pkg] = v
            except Exception:
                versions[pkg] = "?"
    return {
        "r_version": _INFO.r_version,
        "r_home": _INFO.r_home,
        "rpy2_version": _INFO.rpy2_version,
        "packages": versions,
    }


def require(pkg: str) -> None:
    """Load an R package; raise :class:`RPackageMissing` if not installed."""
    if pkg in _LOADED:
        return
    ro = r_session()
    try:
        ro.r(f'suppressPackageStartupMessages(library({pkg}))')
        _LOADED.add(pkg)
    except Exception as e:  # rpy2.rinterface_lib.embedded.RRuntimeError
        msg = str(e)
        if "there is no package called" in msg or "no such file" in msg.lower():
            raise RPackageMissing(pkg) from e
        raise RBridgeError(f"Failed to load R package {pkg!r}: {msg}") from e


def converter():
    """Return the pandas↔R conversion context. Use as
    ``with rbridge.converter(): ro.conversion.py2rpy(df)``."""
    if _CONVERTER is None:
        r_session()
    from rpy2.robjects.conversion import localconverter

    return localconverter(_CONVERTER)


def r_call(fn: str, *args, **kwargs):
    """Convenience: call an R function with pandas-typed args.

    Use for one-off invocations where you don't need a dedicated wrapper.
    Returns the raw rpy2 object — the caller decides how to convert.
    """
    ro = r_session()
    with converter():
        return ro.r[fn](*args, **kwargs)


__all__ = [
    "RBridgeError",
    "RPackageMissing",
    "RSessionInfo",
    "converter",
    "r_call",
    "r_session",
    "r_session_metadata",
    "require",
]
