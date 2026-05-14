"""Static pipeline audits — Sprint 9.5.

These modules walk the pipeline *structurally* (registries, source files,
catalog tables) rather than executing it. They are intended to run in CI
as the v1.0 ship gate: regressions in producer/consumer wiring should
fail the build before they reach an analyst.
"""

from __future__ import annotations

from causalrag.audits.end_to_end_flow import (
    FlowAuditReport,
    audit_pipeline_flow,
    render_flow_audit_html,
)

__all__ = [
    "FlowAuditReport",
    "audit_pipeline_flow",
    "render_flow_audit_html",
]
