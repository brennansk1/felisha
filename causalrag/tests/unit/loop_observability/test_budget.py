"""Tests for the cost-aware budget tracker (Sprint 3.4)."""

from __future__ import annotations

import json
import time

import pytest

from causalrag.loop_observability.budget import (
    BudgetSpec,
    BudgetSpecError,
    BudgetTracker,
    TimerContext,
)


# ─────────── BudgetSpec.parse ─────────────────────────────────────────────


def test_parse_full_spec_k_suffixes():
    spec = BudgetSpec.parse("tokens=200k,wall=15min,ram=4G")
    assert spec.max_tokens == 200_000
    assert spec.max_wallclock_seconds == pytest.approx(15 * 60)
    assert spec.max_peak_ram_bytes == 4 * 1024**3
    assert spec.max_r_bridge_seconds is None
    assert spec.max_experiments is None


def test_parse_megaformat_tokens_and_hours():
    spec = BudgetSpec.parse("tokens=2m,wall=2h")
    assert spec.max_tokens == 2_000_000
    assert spec.max_wallclock_seconds == pytest.approx(2 * 3600)


def test_parse_empty_returns_all_none():
    spec = BudgetSpec.parse("")
    assert spec.max_tokens is None
    assert spec.max_wallclock_seconds is None
    assert spec.max_peak_ram_bytes is None
    assert spec.max_r_bridge_seconds is None
    assert spec.max_experiments is None


def test_parse_whitespace_returns_all_none():
    assert BudgetSpec.parse("   ") == BudgetSpec()


def test_parse_handles_extra_keys_and_seconds_default():
    spec = BudgetSpec.parse("rbridge=90,experiments=10,wall=45s")
    assert spec.max_r_bridge_seconds == 90.0
    assert spec.max_experiments == 10
    assert spec.max_wallclock_seconds == 45.0


def test_parse_ram_binary_units():
    assert BudgetSpec.parse("ram=512M").max_peak_ram_bytes == 512 * 1024**2
    assert BudgetSpec.parse("ram=2K").max_peak_ram_bytes == 2 * 1024


def test_parse_invalid_clause_no_equals():
    with pytest.raises(BudgetSpecError):
        BudgetSpec.parse("tokens200k")


def test_parse_invalid_token_suffix():
    with pytest.raises(BudgetSpecError):
        BudgetSpec.parse("tokens=200x")


def test_parse_unknown_key():
    with pytest.raises(BudgetSpecError):
        BudgetSpec.parse("widgets=5")


def test_parse_empty_value():
    with pytest.raises(BudgetSpecError):
        BudgetSpec.parse("tokens=")


# ─────────── BudgetTracker ─────────────────────────────────────────────────


def test_tracker_starts_under_budget():
    t = BudgetTracker(BudgetSpec(max_tokens=100))
    ok, reason = t.check()
    assert ok is True
    assert reason is None


def test_tracker_records_tokens_and_hits_limit():
    t = BudgetTracker(BudgetSpec(max_tokens=100))
    t.record_tokens(40, model="qwen3:14b")
    t.record_tokens(30, model="qwen3:14b")
    ok, _ = t.check()
    assert ok is True
    t.record_tokens(40, model="qwen3:14b")
    ok, reason = t.check()
    assert ok is False
    assert reason is not None
    assert "token" in reason.lower()
    assert t.tokens_by_model["qwen3:14b"] == 110


def test_tracker_wallclock_exhaustion():
    t = BudgetTracker(BudgetSpec(max_wallclock_seconds=1.0))
    t.record_wallclock(0.5)
    assert t.check()[0] is True
    t.record_wallclock(1.5)
    ok, reason = t.check()
    assert ok is False
    assert reason is not None
    assert "wallclock" in reason.lower()


def test_tracker_r_bridge_accumulates():
    t = BudgetTracker(BudgetSpec(max_r_bridge_seconds=2.0))
    t.record_r_bridge_seconds(0.7)
    t.record_r_bridge_seconds(0.8)
    assert t.check()[0] is True
    t.record_r_bridge_seconds(1.0)
    ok, reason = t.check()
    assert ok is False
    assert "r-bridge" in reason.lower()


def test_tracker_experiments_limit():
    t = BudgetTracker(BudgetSpec(max_experiments=2))
    t.record_experiment_complete()
    assert t.check()[0] is True
    t.record_experiment_complete()
    ok, reason = t.check()
    assert ok is False
    assert "experiment" in reason.lower()


def test_tracker_peak_ram_snapshot_records_positive_bytes():
    t = BudgetTracker(BudgetSpec())
    t.peak_ram_snapshot()
    assert t.peak_ram_bytes > 0


def test_tracker_peak_ram_budget_exhaustion():
    t = BudgetTracker(BudgetSpec(max_peak_ram_bytes=1))
    t.peak_ram_snapshot()
    ok, reason = t.check()
    assert ok is False
    assert reason is not None
    assert "ram" in reason.lower()


def test_tracker_unbounded_spec_never_trips():
    t = BudgetTracker(BudgetSpec())
    t.record_tokens(10**9)
    t.record_wallclock(10**6)
    t.record_r_bridge_seconds(10**6)
    for _ in range(1000):
        t.record_experiment_complete()
    assert t.check() == (True, None)


def test_tracker_negative_tokens_raises():
    t = BudgetTracker(BudgetSpec())
    with pytest.raises(ValueError):
        t.record_tokens(-1)


def test_summary_round_trips_through_json():
    t = BudgetTracker(BudgetSpec(max_tokens=500, max_wallclock_seconds=60.0))
    t.record_tokens(123, model="qwen3:14b")
    t.record_wallclock(12.5)
    t.record_r_bridge_seconds(3.0)
    t.record_experiment_complete()
    t.peak_ram_snapshot()

    payload = t.summary()
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)

    assert decoded["limits"]["max_tokens"] == 500
    assert decoded["limits"]["max_wallclock_seconds"] == 60.0
    assert decoded["usage"]["tokens_used"] == 123
    assert decoded["usage"]["tokens_by_model"] == {"qwen3:14b": 123}
    assert decoded["usage"]["wallclock_seconds"] == 12.5
    assert decoded["usage"]["r_bridge_seconds"] == 3.0
    assert decoded["usage"]["experiments_completed"] == 1
    assert decoded["usage"]["peak_ram_bytes"] > 0


# ─────────── TimerContext ─────────────────────────────────────────────────


def test_timer_context_measures_elapsed():
    with TimerContext() as t:
        time.sleep(0.05)
    assert t.elapsed_seconds >= 0.04
    assert t.elapsed_seconds < 1.0


def test_timer_context_elapsed_zero_before_enter():
    t = TimerContext()
    assert t.elapsed_seconds == 0.0


def test_timer_context_live_reads_inside_block():
    with TimerContext() as t:
        time.sleep(0.01)
        mid = t.elapsed_seconds
        time.sleep(0.01)
    final = t.elapsed_seconds
    assert mid > 0
    assert final >= mid
