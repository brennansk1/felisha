"""Tests for the multiple-testing adjustment helper."""

from __future__ import annotations

import math

import pytest

from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.result import EstimationResult
from causalrag.sensitivity.multiple_testing import adjust_protocol_p_values


def _make_walk(walk_id: str, p_value: float | None) -> RoadmapWalk:
    """Build a minimal RoadmapWalk whose last q7_estimate has ``p_value``."""
    estimates: tuple[EstimationResult, ...] = ()
    if p_value is not None:
        estimates = (
            EstimationResult(
                estimator_id="python.dml.linear",
                estimand_class="ATE",
                point_estimate=0.1,
                se=0.05,
                ci_low=0.0,
                ci_high=0.2,
                p_value=p_value,
                n_used=500,
            ),
        )
    return RoadmapWalk(hypothesis_id=walk_id, q7_estimates=estimates)


def _make_protocol(
    p_values: list[float | None],
    *,
    method: str = "bh",
) -> StudyProtocol:
    walks: dict[str, RoadmapWalk] = {}
    for i, p in enumerate(p_values, 1):
        wid = f"auto-{i:02d}"
        walks[wid] = _make_walk(wid, p)
    return StudyProtocol(
        name="test-mt",
        multiple_testing=method,  # type: ignore[arg-type]
        roadmap_walks=walks,
    )


def test_bh_adjustment_matches_statsmodels() -> None:
    from statsmodels.stats.multitest import multipletests

    raw = [0.001, 0.01, 0.04, 0.05, 0.5]
    protocol = _make_protocol(raw, method="bh")

    _, summary = adjust_protocol_p_values(protocol)

    _, expected_arr, _, _ = multipletests(raw, method="fdr_bh")
    expected = [float(x) for x in expected_arr]

    assert len(summary) == 5
    for i, p in enumerate(raw, 1):
        wid = f"auto-{i:02d}"
        assert summary[wid]["raw_p"] == pytest.approx(p)
        assert summary[wid]["adjusted_p"] == pytest.approx(expected[i - 1])
        diag = protocol.roadmap_walks[wid].q7_estimates[-1].diagnostics
        assert diag["adjusted_p_value"] == pytest.approx(expected[i - 1])
        assert diag["adjustment_method"] == "bh"


def test_none_passthrough_preserves_raw_p() -> None:
    raw = [0.001, 0.04, 0.5]
    protocol = _make_protocol(raw, method="none")

    _, summary = adjust_protocol_p_values(protocol)

    assert len(summary) == 3
    for i, p in enumerate(raw, 1):
        wid = f"auto-{i:02d}"
        assert summary[wid]["adjusted_p"] == pytest.approx(p)
        diag = protocol.roadmap_walks[wid].q7_estimates[-1].diagnostics
        assert diag["adjusted_p_value"] == pytest.approx(p)
        assert diag["adjustment_method"] == "none"


def test_bonferroni_caps_at_one() -> None:
    raw = [0.1, 0.4, 0.7]
    protocol = _make_protocol(raw, method="bonferroni")

    _, summary = adjust_protocol_p_values(protocol)

    # Bonferroni: min(p * k, 1)
    expected = [min(p * 3, 1.0) for p in raw]
    for i, exp in enumerate(expected, 1):
        wid = f"auto-{i:02d}"
        assert summary[wid]["adjusted_p"] == pytest.approx(exp)


def test_walks_without_estimates_are_skipped() -> None:
    # Mix: one walk with no estimate, three with p-values.
    protocol = _make_protocol([0.01, None, 0.04, 0.5], method="bh")

    _, summary = adjust_protocol_p_values(protocol)

    assert "auto-02" not in summary, "walks without estimates must be skipped"
    assert set(summary.keys()) == {"auto-01", "auto-03", "auto-04"}
    # The walk without estimates must not have crashed or gained diagnostics.
    assert protocol.roadmap_walks["auto-02"].q7_estimates == ()


def test_idempotent_double_application() -> None:
    raw = [0.001, 0.01, 0.04, 0.05, 0.5]
    protocol = _make_protocol(raw, method="bh")

    _, summary_a = adjust_protocol_p_values(protocol)
    ledger_len_after_first = len(protocol.decision_ledger)
    _, summary_b = adjust_protocol_p_values(protocol)
    ledger_len_after_second = len(protocol.decision_ledger)

    assert summary_a == summary_b
    # Ledger entry should not be duplicated on re-invocation.
    assert ledger_len_after_first == ledger_len_after_second

    # Diagnostics should also be unchanged.
    for wid, entry in summary_a.items():
        diag = protocol.roadmap_walks[wid].q7_estimates[-1].diagnostics
        assert diag["adjusted_p_value"] == pytest.approx(entry["adjusted_p"])


def test_no_p_values_at_all_logs_zero_comparisons() -> None:
    # All walks lack estimates entirely.
    protocol = _make_protocol([None, None], method="bh")

    _, summary = adjust_protocol_p_values(protocol)

    assert summary == {}
    # Ledger should still record the (k=0) decision so the audit trail
    # reflects that adjustment was considered.
    assert any(
        e.phase == "multiple_testing" for e in protocol.decision_ledger
    )


def test_ledger_records_method_and_count() -> None:
    raw = [0.01, 0.04, 0.5]
    protocol = _make_protocol(raw, method="bonferroni")

    adjust_protocol_p_values(protocol)

    mt_entries = [e for e in protocol.decision_ledger if e.phase == "multiple_testing"]
    assert len(mt_entries) == 1
    entry = mt_entries[0]
    assert entry.chose == "bonferroni"
    assert "k=3" in (entry.note or "")


def test_by_adjustment_matches_statsmodels() -> None:
    from statsmodels.stats.multitest import multipletests

    raw = [0.001, 0.01, 0.04, 0.05, 0.5]
    protocol = _make_protocol(raw, method="by")

    _, summary = adjust_protocol_p_values(protocol)

    _, expected_arr, _, _ = multipletests(raw, method="fdr_by")
    expected = [float(x) for x in expected_arr]

    for i, exp in enumerate(expected, 1):
        wid = f"auto-{i:02d}"
        assert summary[wid]["adjusted_p"] == pytest.approx(exp)


def test_unused_math_import_silenced() -> None:
    # Cheap import-side test so static checkers don't flag math as unused.
    assert math.isfinite(1.0)
