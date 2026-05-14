"""Conditional-independence test layer with auto-routed backends.

Sprint 1.3 (CausalRoadmap v1.0). Wraps the ``causal-learn`` CI module
behind a uniform :class:`CITester` API so the discovery layer can pick
RCIT / KCI / fisher-z as a function of (data type, n, p, time-series)
without each call site re-implementing the routing.

Failure-safe by design: when ``causal-learn`` isn't installed (it's an
optional dependency for the v1.0 bootstrap) every backend gracefully
degrades to a pure-NumPy Fisher-z partial-correlation test. The
fallback's job is to keep the discovery pipeline running, not to win
benchmarks — non-linear / mixed-type CI is only available with the
optional backend.

Backend routing (``backend='auto'``):

* All-numeric, n < 2000      → KCI (most powerful, kernel CI)
* All-numeric, 2000 ≤ n < 5000 → RCIT (fast non-linear approximation)
* Mixed types                → CCIT shim (classifier CI; unavailable
  upstream in causal-learn today, falls back to fisher-z with a note)
* Time-series flagged        → CMIknn-mixed shim (unavailable, falls
  back to fisher-z with a note pointing at Tigramite)
* Otherwise                  → fisher-z partial correlation

Forced backends (``backend='kci' | 'rcit' | 'ccit' | 'cmiknn' | 'fast'``)
bypass routing but still degrade to fisher-z when the chosen backend is
unavailable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger("causalrag.discovery.ci_backend")


# Backend keys understood by ``CITester.__init__``.
_VALID_BACKENDS = frozenset(
    {"auto", "fast", "fisher_z", "kci", "kcit", "rcit", "rcot", "ccit", "cmiknn"}
)


def _causallearn_available() -> bool:
    """Return True iff ``causal-learn`` is importable.

    Cached at module load via a try/except so the routing logic is a
    cheap attribute read on the hot path.
    """
    try:
        import causallearn.utils.cit  # noqa: F401

        return True
    except Exception:  # pragma: no cover - exercised only when the lib is missing
        return False


_HAS_CAUSALLEARN = _causallearn_available()


@dataclass
class CITestResult:
    """Result of one conditional-independence test.

    Attributes
    ----------
    p_value:
        Two-sided p-value of the H0 ``X ⊥ Y | Z``. Higher = more
        evidence for conditional independence.
    test_stat:
        Backend-specific statistic when one is naturally defined
        (``z`` for fisher-z, otherwise ``None`` — causal-learn's KCI
        returns only a p-value).
    backend:
        Which backend actually ran (``'fisher_z'``, ``'kci'``,
        ``'rcit'``, etc.) — may differ from the requested backend if
        causal-learn was unavailable and we fell back.
    df:
        Degrees of freedom of the underlying test when applicable
        (fisher-z partial correlation), else ``None``.
    notes:
        Free-form diagnostics — fallback warnings, kernel choices,
        etc. Carried through to the discovery report.
    """

    p_value: float
    test_stat: float | None
    backend: str
    df: int | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for run.lock.json round-tripping."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CITestResult:
        """Inverse of :meth:`to_dict` — reconstruct from JSON-loadable dict."""
        return cls(
            p_value=float(payload["p_value"]),
            test_stat=(
                None if payload.get("test_stat") is None else float(payload["test_stat"])
            ),
            backend=str(payload["backend"]),
            df=(None if payload.get("df") is None else int(payload["df"])),
            notes=list(payload.get("notes", [])),
        )


class CITester:
    """Conditional-independence test layer with auto-routed backends.

    Parameters
    ----------
    backend:
        Either ``'auto'`` for size/type-driven routing, ``'fast'`` /
        ``'fisher_z'`` for the pure-NumPy partial-correlation path, or
        one of ``'kci'`` / ``'rcit'`` / ``'ccit'`` / ``'cmiknn'`` to
        force a specific causal-learn backend.
    n, p:
        Hints used by ``'auto'`` routing — sample size and column count
        of the working frame. Both optional; missing values fall back
        to conservative defaults (treat as small).
    has_mixed_types:
        Set by the discovery agent when the working frame mixes
        numeric and categorical columns. Routes to CCIT (when
        available) instead of a Gaussian-only test.
    time_series:
        Set when the data is panel / longitudinal with temporal
        ordering. Routes to a CMIknn-mixed-style backend (currently a
        fisher-z shim with a note — Sprint 6.5 will wire Tigramite).

    Notes
    -----
    Construction is deliberately cheap — backend selection is a
    string-comparison ladder, the actual test runs in :meth:`test`.
    Resolved backend is recorded in :attr:`resolved_backend` so the
    caller can log it alongside the discovery report.
    """

    def __init__(
        self,
        backend: str = "auto",
        *,
        n: int | None = None,
        p: int | None = None,
        has_mixed_types: bool = False,
        time_series: bool = False,
    ) -> None:
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of {sorted(_VALID_BACKENDS)}"
            )
        # Normalise spelling variants (kcit/rcot are PDD names; causal-learn calls
        # them kci/rcit).
        canon = {"kcit": "kci", "rcot": "rcit", "fisher_z": "fast"}.get(backend, backend)
        self.requested_backend = canon
        self.n_hint = n
        self.p_hint = p
        self.has_mixed_types = has_mixed_types
        self.time_series = time_series
        self.notes: list[str] = []
        self.resolved_backend = self._resolve(
            canon,
            n=n,
            has_mixed_types=has_mixed_types,
            time_series=time_series,
        )

    # ─── routing ────────────────────────────────────────────────────────

    def _resolve(
        self,
        backend: str,
        *,
        n: int | None,
        has_mixed_types: bool,
        time_series: bool,
    ) -> str:
        """Map (backend, hints) → concrete backend name, recording fallbacks.

        Any time we'd like to route to a causal-learn-backed backend but
        the library is missing, we degrade to fisher-z and stamp a note
        so the caller can surface it in the discovery report.
        """
        if backend == "fast":
            return "fisher_z"

        if backend == "auto":
            if time_series:
                target = "cmiknn"
            elif has_mixed_types:
                target = "ccit"
            elif n is not None and n < 2000:
                target = "kci"
            elif n is not None and n < 5000:
                target = "rcit"
            else:
                # n is None (no hint) or n ≥ 5000 — fisher-z is the only
                # backend that scales to large n without surprise. Auto
                # routing deliberately doesn't blow up runtime on n=10⁵.
                target = "fisher_z"
        else:
            target = backend

        if target == "fisher_z":
            return "fisher_z"

        if not _HAS_CAUSALLEARN:
            self.notes.append(
                f"causal-learn unavailable; backend {target!r} downgraded to "
                "fisher_z (install causal-learn for non-linear / mixed-type CI)."
            )
            logger.warning(
                "causal-learn unavailable; CI backend %r → fisher_z fallback", target
            )
            return "fisher_z"

        # CCIT / CMIknn-mixed are not exposed by causal-learn at v1.0 of the
        # roadmap (they're typically delegated to Tigramite or a custom
        # classifier-CI). Stamp a note and degrade to fisher-z so the
        # pipeline keeps moving.
        if target == "ccit":
            self.notes.append(
                "CCIT (classifier CI for mixed types) not yet wired; using "
                "fisher_z. Mixed-type independence may be under-detected."
            )
            return "fisher_z"
        if target == "cmiknn":
            self.notes.append(
                "CMIknnMixed (Tigramite TS CI) not yet wired; using fisher_z. "
                "Temporal lag structure is ignored."
            )
            return "fisher_z"

        return target  # 'kci' | 'rcit'

    # ─── public API ─────────────────────────────────────────────────────

    def test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None = None,
    ) -> CITestResult:
        """Test ``X ⊥ Y | Z``.

        ``x`` and ``y`` must be 1-D arrays of equal length. ``z`` may
        be ``None`` (marginal independence), a 1-D array (single
        conditioning variable), or a 2-D ``(n, k)`` array (conditioning
        set of size ``k``). Returns a :class:`CITestResult` whose
        ``backend`` field records which backend actually ran.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x and y must have the same length; got {x.shape[0]} vs {y.shape[0]}"
            )
        z2: np.ndarray | None = None
        if z is not None:
            z2 = np.asarray(z, dtype=float)
            if z2.ndim == 1:
                z2 = z2.reshape(-1, 1)
            if z2.shape[0] != x.shape[0]:
                # Allow the (k, n) orientation seen in the IAMB caller.
                if z2.shape[1] == x.shape[0]:
                    z2 = z2.T
                else:
                    raise ValueError(
                        f"z first dim {z2.shape[0]} does not match n={x.shape[0]}"
                    )

        if self.resolved_backend == "fisher_z":
            return self._fisher_z(x, y, z2)
        if self.resolved_backend == "kci":
            return self._kci(x, y, z2)
        if self.resolved_backend == "rcit":
            return self._rcit(x, y, z2)
        # Defensive — should never hit because _resolve only emits the
        # backends handled above.
        return self._fisher_z(x, y, z2)  # pragma: no cover

    # ─── backend implementations ────────────────────────────────────────

    def _fisher_z(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None
    ) -> CITestResult:
        """Pure-NumPy Fisher-z partial-correlation test.

        Mirrors the implementation in ``markov_boundary._partial_correlation_pvalue``
        but returns the structured :class:`CITestResult` and exposes the
        z-statistic + degrees of freedom for downstream logging.
        """
        n = x.shape[0]
        if z is None or z.size == 0:
            if x.std() < 1e-12 or y.std() < 1e-12:
                return CITestResult(1.0, 0.0, "fisher_z", max(n - 3, 1), list(self.notes))
            r = float(np.corrcoef(x, y)[0, 1])
            if math.isnan(r) or abs(r) >= 1.0:
                return CITestResult(1.0, 0.0, "fisher_z", max(n - 3, 1), list(self.notes))
            df = max(n - 3, 1)
            z_stat = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(df)
            p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            return CITestResult(float(p), float(z_stat), "fisher_z", df, list(self.notes))

        Z = np.column_stack([np.ones(n), z])
        try:
            bx, *_ = np.linalg.lstsq(Z, x, rcond=None)
            by, *_ = np.linalg.lstsq(Z, y, rcond=None)
            rx = x - Z @ bx
            ry = y - Z @ by
        except np.linalg.LinAlgError:
            return CITestResult(
                1.0, None, "fisher_z", None, list(self.notes) + ["lstsq failed"]
            )
        if rx.std() < 1e-12 or ry.std() < 1e-12:
            return CITestResult(1.0, 0.0, "fisher_z", None, list(self.notes))
        r = float(np.corrcoef(rx, ry)[0, 1])
        if math.isnan(r) or abs(r) >= 1.0:
            return CITestResult(1.0, 0.0, "fisher_z", None, list(self.notes))
        df = max(n - z.shape[1] - 3, 1)
        z_stat = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(df)
        p = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        return CITestResult(float(p), float(z_stat), "fisher_z", df, list(self.notes))

    def _causallearn_pvalue(
        self,
        cit_kind: str,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray | None,
    ) -> float:
        """Run causal-learn's :class:`CIT` and return the p-value.

        Centralised so KCI / RCIT share data-marshalling: causal-learn
        wants a single (n, p) matrix plus column indices, not three
        separate arrays.
        """
        from causallearn.utils.cit import CIT

        n = x.shape[0]
        if z is None or z.size == 0:
            data = np.column_stack([x, y]).astype(float)
            cit = CIT(data, cit_kind)
            return float(cit(0, 1, []))
        z2 = z if z.ndim == 2 else z.reshape(-1, 1)
        data = np.column_stack([x, y, z2]).astype(float)
        cit = CIT(data, cit_kind)
        cond_idx = list(range(2, 2 + z2.shape[1]))
        return float(cit(0, 1, cond_idx))

    def _kci(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> CITestResult:
        """Kernel CI (Zhang et al. 2011) via causal-learn's :class:`CIT`."""
        try:
            p = self._causallearn_pvalue("kci", x, y, z)
        except Exception as e:
            logger.warning("KCI failed (%s); falling back to fisher_z", e)
            result = self._fisher_z(x, y, z)
            result.notes.append(f"kci failed: {type(e).__name__}; used fisher_z")
            return result
        df = None if z is None else int(z.shape[1])
        return CITestResult(p, None, "kci", df, list(self.notes))

    def _rcit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> CITestResult:
        """Randomised CI Test (Strobl-Visweswaran-Spirtes 2019) via causal-learn."""
        try:
            p = self._causallearn_pvalue("rcit", x, y, z)
        except Exception as e:
            logger.warning("RCIT failed (%s); falling back to fisher_z", e)
            result = self._fisher_z(x, y, z)
            result.notes.append(f"rcit failed: {type(e).__name__}; used fisher_z")
            return result
        df = None if z is None else int(z.shape[1])
        return CITestResult(p, None, "rcit", df, list(self.notes))

    # The CCIT / RCoT / CMIknn methods named in the PDD signature are
    # provided as aliases so the sprint-plan vocabulary works at the
    # public surface even though _resolve degrades them today.
    def _rcot(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> CITestResult:
        """Alias for :meth:`_rcit` — the PDD uses the older name 'RCoT'."""
        return self._rcit(x, y, z)

    def _kcit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> CITestResult:
        """Alias for :meth:`_kci` — the PDD uses the older name 'KCIT'."""
        return self._kci(x, y, z)

    def _ccit(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None) -> CITestResult:
        """Placeholder for CCIT (Sen et al. 2017) — degrades to fisher-z.

        Wired here so the public surface matches the sprint-plan
        signature; a future ticket replaces the body with a real
        classifier-CI implementation.
        """
        result = self._fisher_z(x, y, z)
        result.notes.append("ccit not yet wired; used fisher_z")
        return result


__all__ = ["CITester", "CITestResult"]
