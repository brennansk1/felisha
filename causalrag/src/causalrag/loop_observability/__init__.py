"""Loop observability — structured postmortems + per-family circuit breakers.

Sprint 1.8 (PDD §33). Sibling module to `causalrag.master_loop`; never edits it.

Two public concerns live here:

* :class:`PostmortemRecord` and :func:`build_postmortem` — a structured,
  serializable autopsy of a master-loop run that explains *why* the loop
  stopped, what failed, and what a human or downstream agent should do
  next.
* :class:`EstimatorCircuitBreaker` — a small finite-state machine that
  tracks per-estimator-family failure streaks and trips open once a
  family fails ``threshold`` times in a row, so callers can stop wasting
  budget on a known-broken family until the breaker is reset (e.g., on
  a new chain root).
"""

from causalrag.loop_observability.budget import (
    BudgetSpec,
    BudgetSpecError,
    BudgetTracker,
    TimerContext,
)
from causalrag.loop_observability.circuit_breaker import EstimatorCircuitBreaker
from causalrag.loop_observability.postmortem import (
    PostmortemRecord,
    build_postmortem,
    save_postmortem,
)

__all__ = [
    "BudgetSpec",
    "BudgetSpecError",
    "BudgetTracker",
    "EstimatorCircuitBreaker",
    "PostmortemRecord",
    "TimerContext",
    "build_postmortem",
    "save_postmortem",
]
