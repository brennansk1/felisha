"""Tests for causalrag.loop_observability.postmortem (Sprint 1.8)."""

from __future__ import annotations

import json
from pathlib import Path

from causalrag.loop_observability.postmortem import (
    PostmortemRecord,
    build_postmortem,
    load_postmortem,
    save_postmortem,
)


def _success_row(idx: int, estimator: str = "python.dml.linear") -> dict:
    return {
        "id": f"H{idx}",
        "chain_id": None,
        "treatment": "T",
        "outcome": "Y",
        "estimand_class": "ATE",
        "estimator_id": estimator,
        "estimator_attempts": [estimator],
        "point_estimate": 0.12,
        "se": 0.04,
        "sensitivity_verdict": "green",
    }


def _failure_row(idx: int, reason: str, attempts: list[str]) -> dict:
    return {
        "id": f"H{idx}",
        "chain_id": None,
        "treatment": "T",
        "outcome": "Y",
        "estimand_class": "ATE",
        "estimator_id": None,
        "estimator_attempts": attempts,
        "point_estimate": None,
        "se": None,
        "failure_reason": reason,
        "sensitivity_verdict": "errored",
    }


class _FakeWalk:
    def __init__(self, hypothesis_id: str, failure_reason: str | None = None):
        self.hypothesis_id = hypothesis_id
        self.failure_reason = failure_reason
        self.q7_estimates = () if failure_reason else (object(),)


class _FakeChain:
    def __init__(self, chain_id: str, depth: int, last_verdict: str | None):
        self.chain_id = chain_id
        self.depth = depth
        self.last_verdict = last_verdict


def test_postmortem_max_consecutive_failures_counts_modes() -> None:
    history = [
        _success_row(1),
        _success_row(2),
        _success_row(3),
        _failure_row(4, "unidentifiable: no instrument", ["python.iv.tsls"]),
        _failure_row(5, "estimator fit error: singular matrix", ["python.dml.linear"]),
    ]
    completed = [
        _FakeWalk("H1"),
        _FakeWalk("H2"),
        _FakeWalk("H3"),
    ]
    chains = [_FakeChain("c1", depth=2, last_verdict="green")]

    record = build_postmortem(
        run_id="run-001",
        history=history,
        chains=chains,
        completed=completed,
        target=5,
        termination_kind="max_consecutive_failures",
    )

    assert record.termination_kind == "max_consecutive_failures"
    assert record.n_experiments_completed == 3
    assert record.n_experiments_planned == 5
    assert record.n_failures == 2
    assert record.failure_modes.get("unidentifiable") == 1
    assert record.failure_modes.get("fit_error") == 1
    assert record.failed_estimators.get("python.iv.tsls") == 1
    assert record.failed_estimators.get("python.dml.linear") == 1
    assert record.chains_summary == [
        {"chain_id": "c1", "depth": 2, "last_verdict": "green"},
    ]


def test_postmortem_round_trips_via_json(tmp_path: Path) -> None:
    record = build_postmortem(
        run_id="run-002",
        history=[_success_row(1), _failure_row(2, "unidentifiable", ["python.iv.tsls"])],
        chains=[_FakeChain("c1", 1, "yellow")],
        completed=[_FakeWalk("H1")],
        target=2,
        termination_kind="completed",
        last_error=None,
    )
    path = tmp_path / "post" / "record.json"
    save_postmortem(record, path)

    assert path.exists()
    raw = json.loads(path.read_text())
    assert raw["run_id"] == "run-002"
    assert raw["termination_kind"] == "completed"

    reloaded = load_postmortem(path)
    assert isinstance(reloaded, PostmortemRecord)
    assert reloaded.model_dump(mode="json") == record.model_dump(mode="json")


def test_diagnostic_hints_per_termination_kind() -> None:
    base_kwargs = dict(
        run_id="run-hints",
        history=[],
        chains=[],
        completed=[],
        target=1,
    )

    for kind, needle in [
        ("max_consecutive_failures", "max_consecutive_failures"),
        ("budget_exhausted", "budget"),
        ("fatal_error", "fatal"),
        ("user_cancel", "cancel"),
        ("completed", "completed"),
    ]:
        rec = build_postmortem(termination_kind=kind, **base_kwargs)  # type: ignore[arg-type]
        assert rec.diagnostic_hints, f"no hints for {kind}"
        assert any(needle in h.lower() for h in rec.diagnostic_hints), (
            f"expected '{needle}' hint for {kind}, got {rec.diagnostic_hints}"
        )


def test_diagnostic_hints_for_failure_patterns() -> None:
    history = [
        _failure_row(1, "unidentifiable: no adjustment set", ["python.dml.linear"]),
        _failure_row(2, "unidentifiable: no instrument", ["python.iv.tsls"]),
        _failure_row(3, "estimator fit error", ["python.dml.linear"]),
        _failure_row(4, "estimator fit failed to converge", ["python.dml.linear"]),
    ]
    rec = build_postmortem(
        run_id="r",
        history=history,
        chains=[],
        completed=[],
        target=4,
        termination_kind="max_consecutive_failures",
    )
    hint_blob = " | ".join(rec.diagnostic_hints).lower()
    assert "unidentifiable" in hint_blob
    assert "fit error" in hint_blob or "estimator family" in hint_blob
    # repeated estimator should be flagged
    assert "python.dml.linear" in hint_blob


def test_walk_failure_reason_counted_when_no_row() -> None:
    rec = build_postmortem(
        run_id="r",
        history=[],
        chains=[],
        completed=[_FakeWalk("H1", failure_reason="unidentifiable")],
        target=1,
        termination_kind="completed",
    )
    assert rec.n_failures == 1
    assert rec.failure_modes.get("unidentifiable") == 1
    assert rec.n_experiments_completed == 0
