"""Structured postmortem record for a master-loop run (Sprint 1.8).

The master loop produces a stream of :class:`LoopEvent` and a list of
:class:`RoadmapWalk` instances; downstream agents need a compact,
serializable summary that explains *why* the run terminated and what
went wrong. :class:`PostmortemRecord` is that summary.

Design notes
------------
* Read-only: this module never mutates loop state.
* Schema-locked (``extra="forbid"``): any drift will fail loudly at
  validation time rather than silently dropping fields.
* ``history`` is the list of history-row dicts produced by
  ``master_loop._run_one_experiment`` (see master_loop.py ~L1340).
  A row's ``estimator_id`` is ``None`` and ``sensitivity_verdict`` is
  ``"errored"`` (or absent) when the walk failed.
* ``chains`` is a list of :class:`ChainState`-like objects; we duck-type
  on ``chain_id`` / ``depth`` / ``last_verdict`` rather than importing
  the dataclass, so this module stays decoupled from master_loop and
  callers can pass dicts in tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TerminationKind = Literal[
    "completed",
    "max_consecutive_failures",
    "budget_exhausted",
    "user_cancel",
    "fatal_error",
]


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PostmortemRecord(BaseModel):
    """Structured autopsy of one master-loop run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    termination_kind: TerminationKind
    n_experiments_completed: int
    n_experiments_planned: int
    n_failures: int
    failure_modes: dict[str, int] = Field(default_factory=dict)
    failed_estimators: dict[str, int] = Field(default_factory=dict)
    chains_summary: list[dict[str, Any]] = Field(default_factory=list)
    last_error: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    diagnostic_hints: list[str] = Field(default_factory=list)


# ─────────── helpers ──────────────────────────────────────────────────────


def _classify_failure_reason(reason: str | None) -> str:
    """Bucket a free-text failure_reason into a coarse mode.

    Mirrors the small set of failure verbs the master loop uses internally
    (unknown estimand class, unidentifiable, estimator fit failure, …).
    """
    if not reason:
        return "unknown"
    r = reason.lower()
    if "unidentifi" in r or "no adjustment" in r or "no instrument" in r:
        return "unidentifiable"
    if "unknown estimand" in r or "estimand class" in r:
        return "invalid_estimand"
    if "fit" in r or "converge" in r or "singular" in r:
        return "fit_error"
    if "data" in r or "column" in r or "missing" in r:
        return "data_error"
    if "timeout" in r or "killed" in r:
        return "timeout"
    return "other"


def _row_failed(row: dict[str, Any]) -> bool:
    """A history row is a failure iff there is no estimator + no point estimate.

    The successful path in ``_run_one_experiment`` always writes an
    ``estimator_id``; the failure path leaves it ``None`` (and sets the
    ``failure_reason`` on the walk itself, but the walk is also recorded
    elsewhere in the caller).
    """
    if row.get("estimator_id"):
        return False
    if row.get("point_estimate") is not None:
        return False
    return True


def _chain_field(chain: Any, name: str, default: Any = None) -> Any:
    """Pull ``name`` off a ChainState dataclass OR a plain dict."""
    if isinstance(chain, dict):
        return chain.get(name, default)
    return getattr(chain, name, default)


def _hints_for(
    termination_kind: TerminationKind,
    failure_modes: dict[str, int],
    failed_estimators: dict[str, int],
) -> list[str]:
    hints: list[str] = []
    if termination_kind == "max_consecutive_failures":
        hints.append(
            "The loop hit max_consecutive_failures — inspect the most "
            "recent failure_reason; consider relaxing the estimator "
            "swap policy or widening the candidate queue."
        )
    elif termination_kind == "budget_exhausted":
        hints.append(
            "Experiment / foundation budget was exhausted before the "
            "loop chose to stop. If results look promising, re-run with "
            "a larger n_experiments or max_foundation_iterations."
        )
    elif termination_kind == "fatal_error":
        hints.append(
            "A fatal error terminated the run before normal completion. "
            "Check last_error and the matching traceback in the event log."
        )
    elif termination_kind == "user_cancel":
        hints.append(
            "Run was cancelled by the user; partial results are preserved "
            "and the run can be resumed from the last completed walk."
        )
    elif termination_kind == "completed":
        hints.append(
            "Run completed all planned experiments. Review the chain "
            "summaries and consider whether any null/red verdicts warrant "
            "follow-up hypotheses."
        )

    if failure_modes.get("unidentifiable", 0) >= 2:
        hints.append(
            "Multiple unidentifiable estimands — revisit the discovery "
            "brief; some candidates likely need a different identification "
            "strategy (IV, front-door, sensitivity) or a richer adjustment set."
        )
    if failure_modes.get("fit_error", 0) >= 2:
        hints.append(
            "Repeated estimator fit errors — consider switching estimator "
            "family (e.g., DML→matchit) or auditing sample sizes / overlap."
        )
    if failure_modes.get("data_error", 0) >= 1:
        hints.append(
            "Data-shape errors detected — re-check the dataset card and "
            "column-role assignments before re-running."
        )
    if failed_estimators:
        worst = max(failed_estimators.items(), key=lambda kv: kv[1])
        if worst[1] >= 2:
            hints.append(
                f"Estimator '{worst[0]}' failed {worst[1]}× — likely a "
                "bad family fit for this dataset; trip its circuit breaker "
                "on the next run."
            )
    return hints


# ─────────── public API ───────────────────────────────────────────────────


def build_postmortem(
    *,
    run_id: str,
    history: list[dict[str, Any]],
    chains: list[Any],
    completed: list[Any],
    target: int,
    termination_kind: TerminationKind,
    last_error: str | None = None,
) -> PostmortemRecord:
    """Assemble a :class:`PostmortemRecord` from master-loop run state.

    Parameters
    ----------
    run_id:
        Stable identifier for the run (e.g., protocol name + timestamp).
    history:
        History rows as produced by ``master_loop._run_one_experiment``.
    chains:
        Iterable of :class:`ChainState` dataclasses (or dicts with the
        same field names) — one per active chain.
    completed:
        :class:`RoadmapWalk` instances that finished successfully.
    target:
        Planned number of experiments (``LoopConfig.n_experiments``).
    termination_kind:
        Why the loop stopped — one of the literal values on
        :class:`PostmortemRecord`.
    last_error:
        Optional final error string (when ``termination_kind`` is
        ``"fatal_error"``).
    """
    failure_modes: dict[str, int] = {}
    failed_estimators: dict[str, int] = {}
    n_failures = 0
    for row in history:
        if not _row_failed(row):
            continue
        n_failures += 1
        reason = row.get("failure_reason") or row.get("error")
        bucket = _classify_failure_reason(reason)
        failure_modes[bucket] = failure_modes.get(bucket, 0) + 1
        # An estimator may be recorded under estimator_id (success) or
        # under estimator_attempts (tried-and-failed list).
        for est in row.get("estimator_attempts") or []:
            if est:
                failed_estimators[est] = failed_estimators.get(est, 0) + 1

    # Also fold in failure_reasons that live on the completed walks
    # themselves (some failure paths record the walk but not a row).
    for walk in completed:
        reason = getattr(walk, "failure_reason", None)
        if reason:
            n_failures += 1
            bucket = _classify_failure_reason(reason)
            failure_modes[bucket] = failure_modes.get(bucket, 0) + 1

    chains_summary: list[dict[str, Any]] = []
    for c in chains:
        chains_summary.append(
            {
                "chain_id": _chain_field(c, "chain_id"),
                "depth": _chain_field(c, "depth", 0),
                "last_verdict": _chain_field(c, "last_verdict"),
            }
        )

    # n_experiments_completed: count successful walks (those with at
    # least one estimation result, or with no failure_reason set).
    n_completed = 0
    for walk in completed:
        if getattr(walk, "failure_reason", None):
            continue
        estimates = getattr(walk, "q7_estimates", None)
        if estimates is None or len(estimates) > 0:
            n_completed += 1

    hints = _hints_for(termination_kind, failure_modes, failed_estimators)

    return PostmortemRecord(
        run_id=run_id,
        termination_kind=termination_kind,
        n_experiments_completed=n_completed,
        n_experiments_planned=target,
        n_failures=n_failures,
        failure_modes=failure_modes,
        failed_estimators=failed_estimators,
        chains_summary=chains_summary,
        last_error=last_error,
        diagnostic_hints=hints,
    )


def save_postmortem(record: PostmortemRecord, path: Path) -> None:
    """Write ``record`` to ``path`` as pretty-printed JSON.

    The on-disk format is the pydantic ``model_dump`` of the record;
    re-load via ``PostmortemRecord.model_validate_json(path.read_text())``.
    Parent directories are created on demand.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2))


def load_postmortem(path: Path) -> PostmortemRecord:
    """Round-trip helper for tests + downstream consumers."""
    return PostmortemRecord.model_validate(json.loads(Path(path).read_text()))
