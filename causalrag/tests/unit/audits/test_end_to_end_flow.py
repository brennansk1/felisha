"""Tests for the static end-to-end pipeline flow audit (Sprint 9.5.1)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from causalrag.audits import end_to_end_flow as eef
from causalrag.audits.end_to_end_flow import (
    FlowAuditReport,
    audit_pipeline_flow,
    render_flow_audit_html,
)
from causalrag.core.flags import DataFlag
from causalrag.estimators.catalog import MethodSpec


# ─────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────


def test_audit_runs_on_real_pipeline_and_returns_a_report():
    """The audit must run end-to-end on the current pipeline without
    crashing. The severity may be red/yellow/green — what matters is
    that we get a sensible report."""
    report = audit_pipeline_flow()
    assert isinstance(report, FlowAuditReport)
    assert isinstance(report.timestamp, datetime)
    assert report.n_flags_total == len(list(DataFlag))
    assert report.n_estimators_total > 0
    assert report.n_sensitivity_panels >= 9
    assert report.severity in ("green", "yellow", "red")
    assert report.summary  # non-empty
    # Tickets list mirrors the gap lists.
    if report.severity != "green":
        assert report.actionable_tickets


def test_html_render_is_self_contained():
    """The HTML render must produce a self-contained fragment with chips
    coloured per severity and per-section lists."""
    report = audit_pipeline_flow()
    html = render_flow_audit_html(report)
    assert "<div" in html
    assert "Flow audit" in html
    # Severity chip appears.
    assert report.severity in html
    # Counts appear.
    assert str(report.n_flags_total) in html
    assert str(report.n_estimators_total) in html


# ─────────────────────────────────────────────────────────────────────────
# Synthetic-registry tests
# ─────────────────────────────────────────────────────────────────────────


def _full_future_set() -> frozenset[str]:
    """Every DataFlag name treated as 'future-reserved' — exempts the
    real-pipeline gaps so we can test isolated synthetic scenarios."""
    return frozenset(f.name for f in DataFlag)


def _all_panels_consumed_dashboard_panel_names() -> tuple[str, ...]:
    """Use an empty panel vocabulary so synthesis / report-path checks
    are vacuously green for the synthetic-empty-registry test."""
    return ()


def test_empty_registries_pass_green():
    """With every flag exempted, an empty catalog, an empty panel
    vocabulary, AND every brief field stubbed as routed, the audit must
    report severity=green and zero gaps everywhere."""
    real_brief_fields = eef._brief_field_names()
    with patch.object(eef, "_brief_field_names", return_value=()):
        report = audit_pipeline_flow(
            catalog=(),
            known_future_flags=_full_future_set(),
            explicit_only_estimators=frozenset(),
            dashboard_panel_names=_all_panels_consumed_dashboard_panel_names(),
        )
    assert real_brief_fields  # sanity: we did stub something real
    assert report.severity == "green"
    assert report.flags_emitted_no_routes == []
    assert report.flags_with_no_detector == []
    assert report.flags_with_no_router_consumer == []
    assert report.estimators_unreachable == []
    assert report.sensitivity_panels_not_in_synthesis == []
    assert report.sensitivity_panels_not_in_report == []
    assert report.brief_fields_not_routed == []


def test_fake_flag_with_no_detector_and_no_router_is_caught():
    """If we stub the emitted + routed sets so a flag is missing from
    both, the audit must list it in BOTH gap buckets and surface yellow
    tickets."""
    real_flags = {f.name for f in DataFlag}
    # Pick a flag that the real pipeline already routes (so emptying it
    # out is a meaningful synthetic mutation).
    target = "BINARY_TREATMENT"
    assert target in real_flags

    with (
        patch.object(eef, "_emitted_flag_names", return_value=real_flags - {target}),
        patch.object(eef, "_routed_flag_names", return_value=real_flags - {target}),
    ):
        report = audit_pipeline_flow(
            catalog=(),  # empty catalog: no required_flags reintroduces target
            known_future_flags=frozenset(),
            explicit_only_estimators=frozenset(),
            dashboard_panel_names=(),
        )
    assert target in report.flags_with_no_detector
    assert target in report.flags_with_no_router_consumer
    # Tickets surface both.
    assert any(
        f"DataFlag.{target}" in t and "no detector" in t
        for t in report.actionable_tickets
    )
    assert any(
        f"DataFlag.{target}" in t and "_rule_cascade" in t
        for t in report.actionable_tickets
    )


def test_flag_emitted_but_not_routed_is_red():
    """A flag any detector emits but the router ignores is the worst
    case — must be in `flags_emitted_no_routes` AND severity=red."""
    real_flags = {f.name for f in DataFlag}
    target = "BINARY_TREATMENT"

    with (
        patch.object(eef, "_emitted_flag_names", return_value=real_flags),
        # Routed set deliberately drops the target.
        patch.object(
            eef, "_routed_flag_names", return_value=real_flags - {target}
        ),
    ):
        report = audit_pipeline_flow(
            catalog=(),
            known_future_flags=frozenset(),
            explicit_only_estimators=frozenset(),
            dashboard_panel_names=(),
        )
    assert target in report.flags_emitted_no_routes
    assert report.severity == "red"


def test_fake_estimator_with_unrouted_required_flag_marked_unreachable():
    """Register a fake estimator with ``required_flags={never-emitted}``.
    The cascade obviously doesn't route to it. With no explicit-only
    exemption, the audit must mark it unreachable + red."""
    # We need a flag the cascade never picks up. Reuse a known-future
    # flag — it's in the enum and not in the cascade by design.
    never_emitted = DataFlag.NETWORK_INTERFERENCE
    fake_id = "test.fake.estimator"
    fake = MethodSpec(
        estimator_id=fake_id,
        backend="python",
        use_case="Synthetic for audit test",
        estimands=("ATE",),
        required_flags=(never_emitted,),
        excluded_flags=(),
        min_n=10,
        domain_hint="any",
        reference="—",
    )
    report = audit_pipeline_flow(
        catalog=(fake,),
        known_future_flags=_full_future_set(),  # silence noise
        explicit_only_estimators=frozenset(),
        dashboard_panel_names=(),
    )
    assert fake_id in report.estimators_unreachable
    assert fake_id not in report.estimators_only_via_explicit_id
    assert report.severity == "red"


def test_explicit_only_estimator_is_informational_not_red():
    """When the unreachable estimator is in the allow-list, it goes to
    the informational bucket and severity stays green/yellow."""
    never_emitted = DataFlag.NETWORK_INTERFERENCE
    fake_id = "test.fake.allowlisted"
    fake = MethodSpec(
        estimator_id=fake_id,
        backend="python",
        use_case="Synthetic for audit test",
        estimands=("ATE",),
        required_flags=(never_emitted,),
        excluded_flags=(),
        min_n=10,
        domain_hint="any",
        reference="—",
    )
    with patch.object(eef, "_brief_field_names", return_value=()):
        report = audit_pipeline_flow(
            catalog=(fake,),
            known_future_flags=_full_future_set(),
            explicit_only_estimators=frozenset({fake_id}),
            dashboard_panel_names=(),
        )
    assert fake_id in report.estimators_only_via_explicit_id
    assert fake_id not in report.estimators_unreachable
    assert report.severity == "green"


def test_unknown_panel_yields_yellow():
    """A panel name the synthesis prompt builder and HTML render path
    don't mention must be flagged in both buckets and produce a yellow
    severity (no red triggers)."""
    bogus_panel = "this_panel_appears_nowhere_xyzzy"
    report = audit_pipeline_flow(
        catalog=(),
        known_future_flags=_full_future_set(),
        explicit_only_estimators=frozenset(),
        dashboard_panel_names=(bogus_panel,),
    )
    assert bogus_panel in report.sensitivity_panels_not_in_synthesis
    assert bogus_panel in report.sensitivity_panels_not_in_report
    # No red triggers, so the severity ladder lands on yellow.
    assert report.severity == "yellow"
    assert any("not referenced in the synthesis prompt" in t for t in report.actionable_tickets)


# ─────────────────────────────────────────────────────────────────────────
# Helper-level tests
# ─────────────────────────────────────────────────────────────────────────


def test_emitted_flag_names_finds_known_emitters():
    """``_emitted_flag_names`` must pick up flags that ``data/flags.py``
    obviously emits (SMALL_SAMPLE, HIGH_DIMENSIONAL, ...) and flags the
    discovery brief emits (MEDIATOR_PROPOSED)."""
    emitted = eef._emitted_flag_names()
    for f in (
        "SMALL_SAMPLE",
        "HIGH_DIMENSIONAL",
        "BINARY_TREATMENT",
        "MEDIATOR_PROPOSED",
        "INSTRUMENTAL_CANDIDATE_PRESENT",
    ):
        assert f in emitted, f"expected {f!r} to be in emitted set"


def test_routed_flag_names_includes_cascade_branches_and_catalog():
    """``_routed_flag_names`` must surface flags consumed by the cascade
    (e.g., HIGH_DIMENSIONAL, RIGHT_CENSORED_OUTCOME) as well as flags
    consumed only via catalog required/excluded declarations."""
    routed = eef._routed_flag_names()
    for f in (
        "HIGH_DIMENSIONAL",
        "RIGHT_CENSORED_OUTCOME",
        "BINARY_TREATMENT",
        "CONTINUOUS_TREATMENT",
        "MEDIATOR_PROPOSED",
    ):
        assert f in routed, f"expected {f!r} to be in routed set"


def test_estimator_reachability_covers_default_routes():
    """The reachability map must include cascade-routed estimators with
    a non-empty rule label list."""
    reach = eef._estimator_reachability()
    # Default-ladder estimators that the cascade always appends.
    for eid in (
        "python.dml.linear",
        "python.dml.causal_forest",
        "python.dr.dr_learner",
        "python.meta.x_learner",
    ):
        assert reach.get(eid), f"expected {eid!r} to have at least one rule path"


# ─────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────


def test_report_timestamp_is_tz_aware():
    report = audit_pipeline_flow()
    assert report.timestamp.tzinfo is not None


def test_html_lists_red_and_yellow_chip_colors():
    """Ensure both red and yellow chip backgrounds appear in the HTML so
    the report visually distinguishes severities."""
    report = FlowAuditReport(
        timestamp=datetime.now(timezone.utc),
        n_flags_total=1,
        n_estimators_total=1,
        n_sensitivity_panels=1,
        flags_emitted_no_routes=["FOO"],
        flags_with_no_detector=["BAR"],
        severity="red",
        summary="synthetic",
    )
    html = render_flow_audit_html(report)
    assert "#c62828" in html  # red
    assert "#ed6c02" in html  # yellow
    assert "FOO" in html
    assert "BAR" in html


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])
