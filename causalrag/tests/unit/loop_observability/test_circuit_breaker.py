"""Tests for causalrag.loop_observability.circuit_breaker (Sprint 1.8)."""

from __future__ import annotations

import pytest

from causalrag.loop_observability.circuit_breaker import EstimatorCircuitBreaker


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        EstimatorCircuitBreaker(threshold=0)


def test_opens_after_threshold_consecutive_failures() -> None:
    cb = EstimatorCircuitBreaker(threshold=3)
    fam = "python.dml"

    assert not cb.is_open(fam)
    cb.record_failure(fam, "fit error 1")
    assert not cb.is_open(fam)
    cb.record_failure(fam, "fit error 2")
    assert not cb.is_open(fam)
    cb.record_failure(fam, "fit error 3")
    assert cb.is_open(fam)

    # Other families remain unaffected.
    assert not cb.is_open("r.matchit")


def test_reset_specific_family_closes_breaker() -> None:
    cb = EstimatorCircuitBreaker(threshold=2)
    cb.record_failure("python.dml", "boom")
    cb.record_failure("python.dml", "boom2")
    cb.record_failure("r.matchit", "x")
    cb.record_failure("r.matchit", "y")
    assert cb.is_open("python.dml")
    assert cb.is_open("r.matchit")

    cb.reset("python.dml")
    assert not cb.is_open("python.dml")
    assert cb.is_open("r.matchit")

    # Cumulative totals preserved across reset.
    summary = cb.summary()
    assert summary["python.dml"]["total_failures"] == 2
    assert summary["python.dml"]["consecutive_failures"] == 0


def test_reset_all_closes_every_breaker() -> None:
    cb = EstimatorCircuitBreaker(threshold=2)
    cb.record_failure("a", "x")
    cb.record_failure("a", "y")
    cb.record_failure("b", "x")
    cb.record_failure("b", "y")
    assert cb.is_open("a") and cb.is_open("b")

    cb.reset()
    assert not cb.is_open("a")
    assert not cb.is_open("b")


def test_record_success_resets_consecutive_count_and_closes() -> None:
    cb = EstimatorCircuitBreaker(threshold=3)
    fam = "python.dml"
    cb.record_failure(fam, "f1")
    cb.record_failure(fam, "f2")
    cb.record_success(fam)

    summary = cb.summary()
    assert summary[fam]["consecutive_failures"] == 0
    assert summary[fam]["total_failures"] == 2
    assert summary[fam]["total_successes"] == 1
    assert not cb.is_open(fam)

    # A fresh streak of failures still needs `threshold` in a row.
    cb.record_failure(fam, "f3")
    cb.record_failure(fam, "f4")
    assert not cb.is_open(fam)
    cb.record_failure(fam, "f5")
    assert cb.is_open(fam)


def test_success_after_open_closes_breaker() -> None:
    cb = EstimatorCircuitBreaker(threshold=2)
    fam = "python.dml"
    cb.record_failure(fam, "x")
    cb.record_failure(fam, "y")
    assert cb.is_open(fam)
    cb.record_success(fam)
    assert not cb.is_open(fam)


def test_summary_shape() -> None:
    cb = EstimatorCircuitBreaker(threshold=2)
    cb.record_failure("fam", "reason-1")
    cb.record_failure("fam", "reason-2")
    s = cb.summary()
    assert set(s.keys()) == {"fam"}
    entry = s["fam"]
    assert entry["consecutive_failures"] == 2
    assert entry["total_failures"] == 2
    assert entry["total_successes"] == 0
    assert entry["is_open"] is True
    assert entry["last_reason"] == "reason-2"
