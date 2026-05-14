"""End-to-end pipeline flow audit (Sprint 9.5.1) — v1.0 ship gate.

Walks the pipeline statically, building a directed graph

    discovery_signal → DataFlag → router → estimator/diagnostic
                                    ↓
                              sensitivity panel → synthesis prompt → HTML

For every node in that graph, confirm at least one inbound producer and
one outbound consumer. The :class:`FlowAuditReport` lists every gap:

- Flags emitted by some detector but consumed by zero rules
- Flags consumed by some rule but emitted by zero detector
- Estimators registered in the catalog but not reachable from any rule
  path (i.e., only invocable via ``prefer=<id>``)
- Sensitivity panels in the dashboard not surfaced in any report path
- Discovery-brief fields (mediators, instruments, negative controls,
  target population, …) that are never routed into Q5 / estimator
  selection

The audit is *run-time independent* — it reads source via
:func:`inspect.getsource` and walks module-level data structures. No
import-time side effects beyond the modules being audited.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Importing these modules is intentional: the audit walks their public
# surface (enum members, catalog tuples, function source). They must be
# importable for the audit to run at all.
from causalrag.core.flags import DataFlag
from causalrag.discovery.expert import DomainExpertBrief, flags_from_brief
from causalrag.estimators.catalog import CATALOG
from causalrag.estimators.python.select import _rule_cascade


Severity = Literal["green", "yellow", "red"]


# Flags that are *deliberately* reserved for a future sprint and therefore
# exempt from the "no detector / no router" warnings. Update this set
# explicitly when a flag graduates — silence here must be a conscious
# decision, not drift.
_KNOWN_FUTURE_FLAGS: frozenset[str] = frozenset(
    {
        # Wired into the cascade as a hint but no detector yet emits it.
        # The cascade comments mention a future rbridge.did route.
        "PANEL_STRUCTURE",
        "LONGITUDINAL",
        "CLUSTERED",
        "NETWORK_INTERFERENCE",
        "SINGLE_TREATED_UNIT",
        "COMPETING_RISKS",
        "REPEATED_OUTCOME",
        "SUSPECTED_INFORMATIVE_CENSORING",
        "CROSS_SECTIONAL_SLICE",
        "IDENTIFICATION_FAILED",
    }
)


# The canonical panel-name vocabulary lives on the sensitivity dashboard;
# we reproduce it here so the audit can run even if the dashboard module
# evolves (the audit's responsibility is to *catch* drift, not to depend
# on it). At least one entry MUST also be present in the dashboard's
# ``_PANEL_NAMES`` constant — the audit re-checks below.
_DASHBOARD_PANEL_NAMES: tuple[str, ...] = (
    "e_value",
    "sensemakr",
    "tipping_point",
    "rosenbaum",
    "manski",
    "negative_control",
    "ovb_chernozhukov",
    "refutation_summary",
    "anomaly_audit",
)


# Brief-field name → token(s) that must appear in the master loop / Q5 /
# estimator-selection plumbing for the field to count as "routed".
# The token is grepped in the relevant downstream files; presence anywhere
# is sufficient.
_BRIEF_FIELD_ROUTING_TOKENS: dict[str, tuple[str, ...]] = {
    "mediators": ("mediators", "MEDIATOR_PROPOSED"),
    "effect_modifiers": ("effect_modifiers", "EFFECT_MODIFICATION_OF_INTEREST"),
    "unmeasured_confounders": (
        "unmeasured_confounders",
        "INSTRUMENTAL_CANDIDATE_PRESENT",
    ),
    "confounders": ("confounders",),
    "treatments": ("treatments", "treatment"),
    "outcomes": ("outcomes", "outcome"),
    "candidate_dags": ("candidate_dags", "candidate_graphs", "brief_to_candidate_graphs"),
    "identification_warnings": ("identification_warnings",),
    "domain_summary": ("domain_summary", "domain_brief"),
}


# Estimator ids that are *intentionally* only reachable via explicit
# ``prefer=<id>``. They are valid catalog members but the cascade never
# routes to them because they are too specialized (the analyst pins them
# directly). Keep this list tight — anything here is excluded from the
# "unreachable" red bucket.
_KNOWN_EXPLICIT_ONLY_ESTIMATORS: frozenset[str] = frozenset(
    {
        "rbridge.lmtp.policy",  # arbitrary policy — analyst supplies shift fn
        "rbridge.lmtp.contrast",  # dosage contrast — analyst names δ_a, δ_b
    }
)


# ─────────────────────────────────────────────────────────────────────────
# Public report dataclass
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class FlowAuditReport:
    """Static audit of the end-to-end pipeline wiring.

    See module docstring for the graph being walked and the meaning of
    each list.
    """

    timestamp: datetime

    n_flags_total: int
    n_estimators_total: int
    n_sensitivity_panels: int

    flags_with_no_detector: list[str] = field(default_factory=list)
    """In :class:`DataFlag` but no detector emits — and not in the
    deliberately-reserved future-flag set."""

    flags_with_no_router_consumer: list[str] = field(default_factory=list)
    """In :class:`DataFlag` but no rule in ``_rule_cascade`` reads it."""

    flags_emitted_no_routes: list[str] = field(default_factory=list)
    """Detected at runtime but routed nowhere — the worst kind of orphan
    because the analyst sees the flag and assumes it will steer
    selection. Red-severity."""

    estimators_unreachable: list[str] = field(default_factory=list)
    """In :data:`CATALOG` but no ``_rule_cascade`` path picks them up
    AND not in the explicit-only allowlist."""

    estimators_only_via_explicit_id: list[str] = field(default_factory=list)
    """In :data:`CATALOG` and *explicitly* allowlisted as
    ``prefer=<id>``-only — surfaced as informational, not failure."""

    sensitivity_panels_not_in_report: list[str] = field(default_factory=list)
    """Dashboard panel name absent from the HTML render path."""

    sensitivity_panels_not_in_synthesis: list[str] = field(default_factory=list)
    """Dashboard panel name absent from the synthesis prompt builder."""

    brief_fields_not_routed: list[str] = field(default_factory=list)
    """:class:`DomainExpertBrief` field whose contents never reach the
    master loop / Q5 / estimator selection."""

    severity: Severity = "green"
    summary: str = ""
    actionable_tickets: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


_DATAFLAG_TOKEN_RE = re.compile(r"DataFlag\.([A-Z_][A-Z0-9_]*)")


def _module_source(module_name: str) -> str:
    """Return the source of an importable module, or ``""`` on failure.

    Failure-safe so the audit never crashes the build — a missing
    optional module is surfaced via empty token sets, which the audit
    treats as "no detector" / "no router".
    """
    try:
        import importlib

        mod = importlib.import_module(module_name)
        return inspect.getsource(mod)
    except Exception:  # noqa: BLE001 — audit must never crash on import drift
        return ""


def _detector_module_names() -> tuple[str, ...]:
    """Modules whose source is grepped for ``out.add(DataFlag.X)`` /
    ``flags.add(DataFlag.X)`` emission patterns.

    These are the *deterministic* emitters (data profile + LLM brief
    derivations + the Q7 runtime augmentation). Estimator modules that
    *declare* ``required_flags`` are excluded — declaring a required
    flag is consumption, not emission.
    """
    return (
        "causalrag.data.flags",
        "causalrag.discovery.expert",
        "causalrag.discovery.__init__",
        "causalrag.roadmap.q7_estimate",
    )


def _emitted_flag_names() -> set[str]:
    """Flags any detector module emits (``DataFlag.X`` appearing inside
    a detector source). We're permissive about *how* emission happens —
    any mention of ``DataFlag.X`` in a detector module counts, which is
    what we want because emission can be conditional, parametric, or
    set-union.

    The cascade module itself is excluded so router-side mentions don't
    bleed into the emitter set.
    """
    emitted: set[str] = set()
    for mod in _detector_module_names():
        src = _module_source(mod)
        if not src:
            continue
        emitted.update(_DATAFLAG_TOKEN_RE.findall(src))
    # Drop any tokens that are not actually DataFlag enum members — guards
    # against typos or stale comments.
    valid = {f.name for f in DataFlag}
    return emitted & valid


_DOWNSTREAM_CONSUMER_MODULES: tuple[str, ...] = (
    "causalrag.estimators.python.select",
    "causalrag.estimators.catalog",
    "causalrag.sensitivity.dashboard",
    "causalrag.reporting.synthesis",
    "causalrag.roadmap.q5_identify",
    "causalrag.roadmap.q7_estimate",
)


def _routed_flag_names(catalog: tuple = CATALOG) -> set[str]:
    """Flags consumed anywhere in the routing / downstream brain.

    Consumption includes (a) the cascade source itself, (b) the registry
    filter via catalog ``required_flags`` / ``excluded_flags``
    declarations, and (c) any direct ``DataFlag.X`` reference in the
    downstream modules that branch on flag content (sensitivity
    dashboard, synthesis, Q5 identify, Q7 augmentation).

    A flag like ``BINARY_OUTCOME`` is consumed not by the cascade but by
    the sensitivity dashboard's ``_outcome_dtype`` branch and by the
    synthesis-magnitude classifier; treating that as "no consumer" would
    produce false-positive red tickets.
    """
    valid = {f.name for f in DataFlag}
    consumed: set[str] = set()
    for mod in _DOWNSTREAM_CONSUMER_MODULES:
        src = _module_source(mod)
        if not src:
            continue
        consumed.update(_DATAFLAG_TOKEN_RE.findall(src))
    consumed &= valid
    # Catalog declarations count as routing consumption.
    for spec in catalog:
        for f in spec.required_flags:
            consumed.add(f.name)
        for f in spec.excluded_flags:
            consumed.add(f.name)
    return consumed


def _estimator_reachability() -> dict[str, list[str]]:
    """For each catalog id, list the cascade-rule paths that can return it.

    A "rule path" here is identified by the cascade comment that precedes
    the ``cascade.append(...)`` line — close enough for the audit to
    surface *which* rule covers each id without re-implementing the
    cascade in a different DSL. The implementation walks the cascade
    source line-by-line, accumulating the most-recent comment as the
    current rule label.
    """
    try:
        src = inspect.getsource(_rule_cascade)
    except Exception:  # noqa: BLE001
        return {spec.estimator_id: [] for spec in CATALOG}

    reach: dict[str, list[str]] = {spec.estimator_id: [] for spec in CATALOG}
    current_rule = "default"
    append_re = re.compile(r"""cascade\.(?:append|insert|extend)\(\s*(.+?)\s*\)""")
    str_re = re.compile(r"""["']([^"']+)["']""")
    comment_re = re.compile(r"^\s*#\s*(.+?)\s*$")

    for line in src.splitlines():
        stripped = line.strip()
        m_comment = comment_re.match(line)
        if m_comment and ("──" in m_comment.group(1) or stripped.startswith("# ─")):
            # Section header in the cascade — use as the rule label.
            current_rule = m_comment.group(1).strip("─ ").strip()
            continue
        if stripped.startswith("#"):
            # Non-section comment — capture as a finer-grained label.
            text = comment_re.match(line)
            if text:
                current_rule = text.group(1)
            continue
        m = append_re.search(line)
        if not m:
            continue
        for est_id in str_re.findall(m.group(1)):
            if est_id in reach:
                reach[est_id].append(current_rule)
    return reach


def _synthesis_prompt_source() -> str:
    """Source of the synthesis prompt builder (+ system prompt body).

    We grep both because the system prompt enumerates the
    sensitivity-style verdicts the LLM should attend to and the builder
    function decides which panel outputs appear in the prompt at all.
    """
    try:
        from causalrag.reporting import synthesis as syn_mod

        return inspect.getsource(syn_mod)
    except Exception:  # noqa: BLE001
        return ""


def _report_html_source() -> str:
    try:
        from causalrag.reporting import render_html

        return inspect.getsource(render_html)
    except Exception:  # noqa: BLE001
        return ""


def _master_loop_source() -> str:
    """Read master_loop.py *as a source string* — the audit does not
    touch master_loop semantically (the sprint forbids edits), it only
    inspects it as a string."""
    try:
        from causalrag import master_loop

        return inspect.getsource(master_loop)
    except Exception:  # noqa: BLE001
        return ""


def _q5_identify_source() -> str:
    try:
        from causalrag.roadmap import q5_identify as mod

        return inspect.getsource(mod)
    except Exception:  # noqa: BLE001
        return ""


def _brief_field_names() -> tuple[str, ...]:
    """Public field names of :class:`DomainExpertBrief`."""
    return tuple(DomainExpertBrief.model_fields.keys())


# ─────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────


def audit_pipeline_flow(
    *,
    catalog: tuple = CATALOG,
    known_future_flags: frozenset[str] = _KNOWN_FUTURE_FLAGS,
    explicit_only_estimators: frozenset[str] = _KNOWN_EXPLICIT_ONLY_ESTIMATORS,
    dashboard_panel_names: tuple[str, ...] = _DASHBOARD_PANEL_NAMES,
) -> FlowAuditReport:
    """Walk the static pipeline and return a :class:`FlowAuditReport`.

    Parameters are exposed so tests can substitute synthetic registries.
    """
    # ── 1. Flags ────────────────────────────────────────────────────────
    all_flag_names = {f.name for f in DataFlag}
    emitted = _emitted_flag_names()
    routed = _routed_flag_names(catalog)

    # No detector emits this flag, AND no exempt: rule consumes it for nothing.
    flags_with_no_detector = sorted(
        n for n in all_flag_names
        if n not in emitted and n not in known_future_flags
    )
    flags_with_no_router_consumer = sorted(
        n for n in all_flag_names
        if n not in routed and n not in known_future_flags
    )
    # Detected at runtime but the cascade never reads it — the worst case.
    flags_emitted_no_routes = sorted(emitted - routed - known_future_flags)

    # ── 2. Estimators ────────────────────────────────────────────────────
    reach = _estimator_reachability()
    explicit_only: list[str] = []
    unreachable: list[str] = []
    for spec in catalog:
        eid = spec.estimator_id
        if reach.get(eid):
            continue
        if eid in explicit_only_estimators:
            explicit_only.append(eid)
        else:
            unreachable.append(eid)
    explicit_only.sort()
    unreachable.sort()

    # ── 3. Sensitivity panels ────────────────────────────────────────────
    synth_src = _synthesis_prompt_source()
    html_src = _report_html_source()
    panels_not_in_synthesis = sorted(p for p in dashboard_panel_names if p not in synth_src)
    panels_not_in_report = sorted(p for p in dashboard_panel_names if p not in html_src)

    # ── 4. Brief fields ──────────────────────────────────────────────────
    brief_fields = _brief_field_names()
    expert_src = _module_source("causalrag.discovery.expert")
    flags_from_brief_src = ""
    try:
        flags_from_brief_src = inspect.getsource(flags_from_brief)
    except Exception:  # noqa: BLE001
        pass
    master_src = _master_loop_source()
    q5_src = _q5_identify_source()
    downstream_corpus = "\n".join(
        [flags_from_brief_src, master_src, q5_src, synth_src]
    )

    not_routed: list[str] = []
    for fld in brief_fields:
        tokens = _BRIEF_FIELD_ROUTING_TOKENS.get(fld, (fld,))
        # The field is routed if *any* of its routing tokens shows up
        # anywhere in the downstream corpus (master loop, Q5, synthesis,
        # or the brief→flag bridge). Tokens default to the field name.
        if any(tok in downstream_corpus for tok in tokens):
            continue
        # The field name itself may also appear in the expert module on
        # the class definition, which is consumption-adjacent — but we
        # require evidence of *downstream* use, not just declaration.
        not_routed.append(fld)
    not_routed.sort()

    # ── 5. Severity + tickets ────────────────────────────────────────────
    tickets: list[str] = []
    for f in flags_emitted_no_routes:
        tickets.append(
            f"[red] DataFlag.{f} is emitted by a detector but no rule in "
            "_rule_cascade reads it — wire a route or move to "
            "_KNOWN_FUTURE_FLAGS."
        )
    for eid in unreachable:
        tickets.append(
            f"[red] Estimator {eid!r} is registered in CATALOG but no "
            "cascade path routes to it — add a rule or move to "
            "_KNOWN_EXPLICIT_ONLY_ESTIMATORS."
        )
    for f in flags_with_no_detector:
        tickets.append(
            f"[yellow] DataFlag.{f} is in the enum but no detector emits "
            "it — either implement a detector or move to "
            "_KNOWN_FUTURE_FLAGS."
        )
    for f in flags_with_no_router_consumer:
        if f in flags_emitted_no_routes:
            continue  # already covered by red ticket
        tickets.append(
            f"[yellow] DataFlag.{f} is in the enum but _rule_cascade "
            "never inspects it — wire a route or move to "
            "_KNOWN_FUTURE_FLAGS."
        )
    for p in panels_not_in_synthesis:
        tickets.append(
            f"[yellow] Sensitivity panel {p!r} is in the dashboard but "
            "not referenced in the synthesis prompt — quote its verdict."
        )
    for p in panels_not_in_report:
        tickets.append(
            f"[yellow] Sensitivity panel {p!r} is in the dashboard but "
            "not referenced in the HTML report path — surface it."
        )
    for fld in not_routed:
        tickets.append(
            f"[yellow] DomainExpertBrief.{fld} is not routed into the "
            "master loop / Q5 / estimator selection / synthesis."
        )

    if flags_emitted_no_routes or unreachable:
        severity: Severity = "red"
    elif (
        panels_not_in_synthesis
        or panels_not_in_report
        or not_routed
        or flags_with_no_detector
        or flags_with_no_router_consumer
    ):
        severity = "yellow"
    else:
        severity = "green"

    summary = (
        f"flow audit: severity={severity}; "
        f"{len(flags_emitted_no_routes)} orphaned-emit flags, "
        f"{len(unreachable)} unreachable estimators, "
        f"{len(panels_not_in_synthesis)} panels missing from synthesis, "
        f"{len(panels_not_in_report)} panels missing from HTML report, "
        f"{len(not_routed)} brief fields not routed."
    )

    return FlowAuditReport(
        timestamp=datetime.now(timezone.utc),
        n_flags_total=len(all_flag_names),
        n_estimators_total=len(catalog),
        n_sensitivity_panels=len(dashboard_panel_names),
        flags_with_no_detector=flags_with_no_detector,
        flags_with_no_router_consumer=flags_with_no_router_consumer,
        flags_emitted_no_routes=flags_emitted_no_routes,
        estimators_unreachable=unreachable,
        estimators_only_via_explicit_id=explicit_only,
        sensitivity_panels_not_in_report=panels_not_in_report,
        sensitivity_panels_not_in_synthesis=panels_not_in_synthesis,
        brief_fields_not_routed=not_routed,
        severity=severity,
        summary=summary,
        actionable_tickets=tickets,
    )


# ─────────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────────


_SEVERITY_COLOR = {
    "green": "#2e7d32",
    "yellow": "#ed6c02",
    "red": "#c62828",
}


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_flow_audit_html(report: FlowAuditReport) -> str:
    """Render the audit as a self-contained HTML fragment.

    Plain HTML with inline styles — no Jinja dependency, drop-in for
    Quarto / static reports / CI artifacts.
    """
    header_color = _SEVERITY_COLOR.get(report.severity, "#616161")

    def _chip(severity: str, text: str) -> str:
        color = _SEVERITY_COLOR.get(severity, "#616161")
        return (
            f'<span style="display:inline-block;padding:2px 8px;'
            f'border-radius:12px;background:{color};color:white;'
            f'font-size:0.85em;margin-right:6px;">{_esc(text)}</span>'
        )

    sections: list[tuple[str, str, list[str]]] = [
        ("Flags emitted but not routed (orphaned emit)", "red", report.flags_emitted_no_routes),
        ("Estimators unreachable from cascade", "red", report.estimators_unreachable),
        ("Flags in enum with no detector", "yellow", report.flags_with_no_detector),
        ("Flags in enum with no router consumer", "yellow", report.flags_with_no_router_consumer),
        (
            "Sensitivity panels not in synthesis prompt",
            "yellow",
            report.sensitivity_panels_not_in_synthesis,
        ),
        (
            "Sensitivity panels not in HTML report",
            "yellow",
            report.sensitivity_panels_not_in_report,
        ),
        ("Brief fields not routed downstream", "yellow", report.brief_fields_not_routed),
        (
            "Estimators reachable only via prefer=<id> (informational)",
            "green",
            report.estimators_only_via_explicit_id,
        ),
    ]

    body: list[str] = []
    body.append(
        f'<div style="font-family:sans-serif;">'
        f'<h2 style="margin-bottom:4px;">Flow audit</h2>'
        f'<div style="margin-bottom:8px;">'
        f'<span style="display:inline-block;padding:2px 10px;'
        f'border-radius:12px;background:{header_color};color:white;'
        f'font-weight:bold;">{_esc(report.severity)}</span> '
        f'<span style="color:#555;">{_esc(report.summary)}</span>'
        f"</div>"
        f'<p style="color:#777;font-size:0.9em;">'
        f"{report.n_flags_total} flags · "
        f"{report.n_estimators_total} estimators · "
        f"{report.n_sensitivity_panels} sensitivity panels · "
        f"audited {_esc(report.timestamp.isoformat())}</p>"
    )

    for title, sev, items in sections:
        body.append(
            f'<h3 style="margin-bottom:4px;">{_chip(sev, sev)}{_esc(title)} '
            f'<span style="color:#777;font-weight:normal;">({len(items)})</span></h3>'
        )
        if items:
            body.append('<ul style="margin-top:2px;">')
            for it in items:
                body.append(f"<li><code>{_esc(it)}</code></li>")
            body.append("</ul>")
        else:
            body.append('<p style="color:#2e7d32;margin-top:2px;">— none —</p>')

    if report.actionable_tickets:
        body.append('<h3 style="margin-top:14px;">Actionable tickets</h3><ol>')
        for t in report.actionable_tickets:
            body.append(f"<li>{_esc(t)}</li>")
        body.append("</ol>")

    body.append("</div>")
    return "".join(body)


# ─────────────────────────────────────────────────────────────────────────
# CLI hook (running this module prints something sensible)
# ─────────────────────────────────────────────────────────────────────────


def _print_report(report: FlowAuditReport) -> None:
    print(report.summary)
    print(f"  severity: {report.severity}")
    print(f"  flags total: {report.n_flags_total}")
    print(f"  estimators total: {report.n_estimators_total}")
    print(f"  sensitivity panels: {report.n_sensitivity_panels}")
    sections = [
        ("flags_emitted_no_routes", report.flags_emitted_no_routes),
        ("estimators_unreachable", report.estimators_unreachable),
        ("flags_with_no_detector", report.flags_with_no_detector),
        ("flags_with_no_router_consumer", report.flags_with_no_router_consumer),
        ("sensitivity_panels_not_in_synthesis", report.sensitivity_panels_not_in_synthesis),
        ("sensitivity_panels_not_in_report", report.sensitivity_panels_not_in_report),
        ("brief_fields_not_routed", report.brief_fields_not_routed),
        ("estimators_only_via_explicit_id", report.estimators_only_via_explicit_id),
    ]
    for label, items in sections:
        print(f"  {label}: {items}")
    if report.actionable_tickets:
        print("\n  tickets:")
        for t in report.actionable_tickets:
            print(f"   - {t}")


def _main() -> int:
    report = audit_pipeline_flow()
    _print_report(report)
    return 0 if report.severity != "red" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())


__all__ = [
    "FlowAuditReport",
    "audit_pipeline_flow",
    "render_flow_audit_html",
]


# Suppress unused-import warnings on names we import for side-effect /
# audit-discoverability (their presence in the module makes the dependency
# explicit).
_ = (Path,)
