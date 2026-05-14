"""Cost-aware budget tracker for the master loop.

Sprint 3.4 (PDD §33). Sibling module to :mod:`causalrag.loop_observability`'s
postmortem and circuit breaker — provides a structured budget specification,
a live tracker for tokens / wallclock / RAM / R-bridge time, and a small
``TimerContext`` helper.

The tracker is intentionally side-effect free with respect to the master
loop: callers invoke ``record_*`` and ``check()`` at decision points; this
module never edits :mod:`causalrag.master_loop` directly. A future sprint
will subscribe to :class:`LoopEvent` and route counts in automatically.

Parsing convention (see :meth:`BudgetSpec.parse`)::

    --budget tokens=200k,wall=15min,ram=4G,rbridge=5min,experiments=10

Suffixes:

* tokens: ``k`` = 10^3, ``m`` = 10^6 (case-insensitive).
* wallclock / r-bridge: ``s``/``sec``/``seconds``, ``m``/``min``/``minutes``,
  ``h``/``hr``/``hours``. Bare numbers are seconds.
* ram: ``K`` = 2^10, ``M`` = 2^20, ``G`` = 2^30, ``T`` = 2^40
  (binary units, case-insensitive).
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any


# ─────────── Spec parsing ────────────────────────────────────────────────────


class BudgetSpecError(ValueError):
    """Raised when a ``--budget`` string fails to parse."""


_TOKEN_SUFFIX = {"": 1, "k": 1_000, "m": 1_000_000}
_TIME_SUFFIX = {
    "": 1.0,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}
_BYTE_SUFFIX = {
    "": 1,
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
    "t": 1024**4,
    "tb": 1024**4,
}

_NUM_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*$")


def _split_number_suffix(raw: str) -> tuple[float, str]:
    m = _NUM_RE.match(raw)
    if not m:
        raise BudgetSpecError(f"cannot parse number/suffix from {raw!r}")
    return float(m.group(1)), m.group(2).lower()


def _parse_tokens(raw: str) -> int:
    n, suffix = _split_number_suffix(raw)
    if suffix not in _TOKEN_SUFFIX:
        raise BudgetSpecError(
            f"unknown token suffix {suffix!r} (expected k or m)"
        )
    return int(n * _TOKEN_SUFFIX[suffix])


def _parse_seconds(raw: str) -> float:
    n, suffix = _split_number_suffix(raw)
    if suffix not in _TIME_SUFFIX:
        raise BudgetSpecError(
            f"unknown time suffix {suffix!r} (expected s/min/h)"
        )
    return n * _TIME_SUFFIX[suffix]


def _parse_bytes(raw: str) -> int:
    n, suffix = _split_number_suffix(raw)
    if suffix not in _BYTE_SUFFIX:
        raise BudgetSpecError(
            f"unknown byte suffix {suffix!r} (expected K/M/G/T)"
        )
    return int(n * _BYTE_SUFFIX[suffix])


def _parse_int(raw: str) -> int:
    try:
        return int(raw.strip())
    except ValueError as e:
        raise BudgetSpecError(f"cannot parse integer from {raw!r}") from e


@dataclass
class BudgetSpec:
    """User-supplied limits parsed from ``--budget tokens=200k,wall=15min,ram=4G``.

    Any field left as ``None`` means "unbounded".
    """

    max_tokens: int | None = None
    max_wallclock_seconds: float | None = None
    max_peak_ram_bytes: int | None = None
    max_r_bridge_seconds: float | None = None
    max_experiments: int | None = None  # alias of LoopConfig.n_experiments

    @classmethod
    def parse(cls, spec: str) -> "BudgetSpec":
        """Parse ``tokens=200k,wall=15min,ram=4G`` into a :class:`BudgetSpec`.

        Empty string or whitespace returns an unbounded spec.
        """
        out = cls()
        if spec is None:
            return out
        spec = spec.strip()
        if not spec:
            return out

        for raw_part in spec.split(","):
            part = raw_part.strip()
            if not part:
                continue
            if "=" not in part:
                raise BudgetSpecError(
                    f"budget clause {part!r} must be 'key=value'"
                )
            key, value = part.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                raise BudgetSpecError(f"budget clause {part!r} has empty value")

            if key in ("tokens", "token"):
                out.max_tokens = _parse_tokens(value)
            elif key in ("wall", "wallclock", "time"):
                out.max_wallclock_seconds = _parse_seconds(value)
            elif key in ("ram", "memory", "mem"):
                out.max_peak_ram_bytes = _parse_bytes(value)
            elif key in ("rbridge", "r_bridge", "r-bridge", "r"):
                out.max_r_bridge_seconds = _parse_seconds(value)
            elif key in ("experiments", "n_experiments", "n"):
                out.max_experiments = _parse_int(value)
            else:
                raise BudgetSpecError(
                    f"unknown budget key {key!r} "
                    "(expected: tokens, wall, ram, rbridge, experiments)"
                )
        return out


# ─────────── Live tracker ────────────────────────────────────────────────────


@dataclass
class BudgetTracker:
    """Live tracker for tokens, wallclock, peak RAM, R-bridge time.

    Records via ``record_*()`` during the run; ``check()`` at decision points
    returns ``(continue, reason)``. ``continue`` is ``True`` if every
    configured budget still has headroom, ``False`` (with ``reason``) if any
    budget is exhausted.
    """

    spec: BudgetSpec
    tokens_used: int = 0
    tokens_by_model: dict[str, int] = field(default_factory=dict)
    wallclock_seconds: float = 0.0
    r_bridge_seconds: float = 0.0
    experiments_completed: int = 0
    peak_ram_bytes: int = 0
    _start_perf: float | None = None

    def __init__(self, spec: BudgetSpec) -> None:
        self.spec = spec
        self.tokens_used = 0
        self.tokens_by_model = {}
        self.wallclock_seconds = 0.0
        self.r_bridge_seconds = 0.0
        self.experiments_completed = 0
        self.peak_ram_bytes = 0
        self._start_perf = time.perf_counter()

    # --- record_* ----------------------------------------------------------

    def record_tokens(self, n: int, *, model: str = "") -> None:
        if n < 0:
            raise ValueError(f"record_tokens got negative count {n}")
        self.tokens_used += n
        if model:
            self.tokens_by_model[model] = (
                self.tokens_by_model.get(model, 0) + n
            )

    def record_wallclock(self, seconds: float) -> None:
        """Override the running wallclock total (absolute value).

        Pass the elapsed time since tracker start; this lets callers feed in
        either ``perf_counter()`` deltas they measured themselves or values
        from :class:`TimerContext`.
        """
        if seconds < 0:
            raise ValueError(f"record_wallclock got negative seconds {seconds}")
        self.wallclock_seconds = seconds

    def record_r_bridge_seconds(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError(
                f"record_r_bridge_seconds got negative seconds {seconds}"
            )
        self.r_bridge_seconds += seconds

    def record_experiment_complete(self) -> None:
        self.experiments_completed += 1

    def peak_ram_snapshot(self) -> None:
        """Record current process RSS, updating :attr:`peak_ram_bytes`.

        Prefers :mod:`psutil` (already a project dependency); falls back to
        :func:`resource.getrusage`. On Linux ``ru_maxrss`` is kilobytes; on
        macOS / *BSD it is bytes — we normalize to bytes.
        """
        rss: int | None = None
        try:
            import psutil  # type: ignore[import-not-found]

            rss = int(psutil.Process().memory_info().rss)
        except Exception:
            try:
                import resource

                ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if sys.platform == "darwin":
                    rss = int(ru)
                else:
                    rss = int(ru) * 1024
            except Exception:
                rss = None
        if rss is not None and rss > self.peak_ram_bytes:
            self.peak_ram_bytes = rss

    # --- check / summary --------------------------------------------------

    def check(self) -> tuple[bool, str | None]:
        """Return ``(should_continue, reason_if_exhausted)``."""
        s = self.spec
        if s.max_tokens is not None and self.tokens_used >= s.max_tokens:
            return (
                False,
                f"token budget exhausted: {self.tokens_used} >= {s.max_tokens}",
            )
        if (
            s.max_wallclock_seconds is not None
            and self.wallclock_seconds >= s.max_wallclock_seconds
        ):
            return (
                False,
                f"wallclock budget exhausted: "
                f"{self.wallclock_seconds:.2f}s >= {s.max_wallclock_seconds:.2f}s",
            )
        if (
            s.max_peak_ram_bytes is not None
            and self.peak_ram_bytes >= s.max_peak_ram_bytes
        ):
            return (
                False,
                f"RAM budget exhausted: "
                f"{self.peak_ram_bytes} >= {s.max_peak_ram_bytes} bytes",
            )
        if (
            s.max_r_bridge_seconds is not None
            and self.r_bridge_seconds >= s.max_r_bridge_seconds
        ):
            return (
                False,
                f"R-bridge budget exhausted: "
                f"{self.r_bridge_seconds:.2f}s >= {s.max_r_bridge_seconds:.2f}s",
            )
        if (
            s.max_experiments is not None
            and self.experiments_completed >= s.max_experiments
        ):
            return (
                False,
                f"experiment count budget exhausted: "
                f"{self.experiments_completed} >= {s.max_experiments}",
            )
        return True, None

    def summary(self) -> dict[str, Any]:
        """JSON-serializable snapshot of usage and configured limits."""
        return {
            "limits": {
                "max_tokens": self.spec.max_tokens,
                "max_wallclock_seconds": self.spec.max_wallclock_seconds,
                "max_peak_ram_bytes": self.spec.max_peak_ram_bytes,
                "max_r_bridge_seconds": self.spec.max_r_bridge_seconds,
                "max_experiments": self.spec.max_experiments,
            },
            "usage": {
                "tokens_used": self.tokens_used,
                "tokens_by_model": dict(self.tokens_by_model),
                "wallclock_seconds": self.wallclock_seconds,
                "r_bridge_seconds": self.r_bridge_seconds,
                "experiments_completed": self.experiments_completed,
                "peak_ram_bytes": self.peak_ram_bytes,
            },
        }


# ─────────── Timer helper ────────────────────────────────────────────────────


class TimerContext:
    """Context manager that measures wallclock via :func:`time.perf_counter`.

    Usage::

        with TimerContext() as t:
            do_stuff()
        tracker.record_r_bridge_seconds(t.elapsed_seconds)
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._end: float | None = None

    def __enter__(self) -> "TimerContext":
        self._start = time.perf_counter()
        self._end = None
        return self

    def __exit__(self, *exc: Any) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start


__all__ = [
    "BudgetSpec",
    "BudgetSpecError",
    "BudgetTracker",
    "TimerContext",
]
