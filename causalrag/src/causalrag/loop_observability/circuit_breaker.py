"""Per-estimator-family circuit breaker (Sprint 1.8).

When an estimator family fails ``threshold`` times in a row, the loop
should stop wasting compute on it. This module implements the small
finite-state machine that callers consult before invoking an estimator
and that the loop updates after each attempt.

Semantics
---------
* "Family" is any caller-defined string — typically the catalog prefix
  (e.g., ``"python.dml"`` or ``"r.matchit"``).
* The breaker is *consecutive*: a single success on the same family
  resets the failure count back to zero (the classic CLOSED state).
* Reset is explicit. ``reset(None)`` clears every family; ``reset(fam)``
  clears one. The master loop is expected to call ``reset(None)`` on
  every new chain root, so a family that flopped on one root can try
  again on the next.
* Once OPEN, ``record_failure`` keeps counting (useful for postmortem
  attribution), but ``is_open`` stays ``True`` until reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _FamilyState:
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    is_open: bool = False
    last_reason: str | None = None
    failure_reasons: list[str] = field(default_factory=list)


class EstimatorCircuitBreaker:
    """Track per-family failure counts and trip open at ``threshold``.

    Parameters
    ----------
    threshold:
        Number of *consecutive* failures that flip the breaker to OPEN.
        Defaults to 3 (matches ``LoopConfig.max_consecutive_failures``).
    """

    def __init__(self, threshold: int = 3) -> None:
        if threshold < 1:
            raise ValueError("threshold must be >= 1")
        self.threshold = threshold
        self._state: dict[str, _FamilyState] = {}

    # ─── private ────────────────────────────────────────────────────────

    def _get(self, family: str) -> _FamilyState:
        st = self._state.get(family)
        if st is None:
            st = _FamilyState()
            self._state[family] = st
        return st

    # ─── public API ─────────────────────────────────────────────────────

    def record_failure(self, family: str, reason: str) -> None:
        """Note one failure for ``family``; flip OPEN at the threshold."""
        st = self._get(family)
        st.consecutive_failures += 1
        st.total_failures += 1
        st.last_reason = reason
        st.failure_reasons.append(reason)
        if st.consecutive_failures >= self.threshold:
            st.is_open = True

    def record_success(self, family: str) -> None:
        """Note one success for ``family``; resets the consecutive count.

        Also closes the breaker if it was open — a fresh success is
        evidence the family is workable again.
        """
        st = self._get(family)
        st.total_successes += 1
        st.consecutive_failures = 0
        st.is_open = False

    def is_open(self, family: str) -> bool:
        """Return ``True`` when ``family`` should be skipped."""
        st = self._state.get(family)
        return bool(st and st.is_open)

    def reset(self, family: str | None = None) -> None:
        """Reset one family (or all of them when ``family`` is None).

        Resetting wipes the consecutive counter *and* closes the breaker;
        the cumulative totals (``total_failures`` / ``total_successes``)
        are preserved for postmortem attribution.
        """
        if family is None:
            for st in self._state.values():
                st.consecutive_failures = 0
                st.is_open = False
            return
        st = self._state.get(family)
        if st is not None:
            st.consecutive_failures = 0
            st.is_open = False

    def summary(self) -> dict[str, dict[str, Any]]:
        """Snapshot the current state — handy for postmortems + tests."""
        return {
            fam: {
                "consecutive_failures": st.consecutive_failures,
                "total_failures": st.total_failures,
                "total_successes": st.total_successes,
                "is_open": st.is_open,
                "last_reason": st.last_reason,
            }
            for fam, st in self._state.items()
        }
