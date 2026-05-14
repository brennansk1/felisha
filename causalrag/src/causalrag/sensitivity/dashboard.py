"""Unified sensitivity dashboard — Sprint 2.6.

Felisha runs several sensitivity diagnostics on every Q7 estimate: an
E-value, a sensemakr partial-R² robustness value, a tipping-point
analysis, anomaly audit, multiple-testing adjustment, and (per the SEDR)
Rosenbaum bounds, Manski bounds, negative-control / OVB-Chernozhukov,
plus a per-walk refutation summary. Each lives in its own module; the
master loop has historically wired them up ad-hoc, producing slightly
different output shapes per call site.

This module is the *single* aggregator. It runs every panel that is
applicable, captures per-panel failure rather than letting one missing
backend (R unavailable, sensemakr not installed, ...) take down the
whole dashboard, and returns a Pydantic ``SensitivityDashboard`` that
report renderers, the TUI, and downstream synthesis can quote verbatim.

Contract:

- Each panel is wrapped in ``try/except``. A failure sets
  ``available=False`` and puts the reason in ``result['error']``.
- ``aggregate_verdict`` follows the strictest-wins rule: ``red`` if any
  panel is red, ``yellow`` if any panel is yellow, ``green`` only when
  every available, non-neutral panel is green. ``unknown`` otherwise.
- The HTML renderer is intentionally template-free and only produces
  plain HTML so reports can embed it without an extra Jinja dependency.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.result import EstimationResult

# Pull each component lazily-but-statically — these are first-party
# modules in the same package and must already be importable. The
# *backends* they call (R, sensemakr, etc.) are what fail at runtime;
# the module imports themselves do not.
from causalrag.sensitivity.anomaly_audit import audit_for_anomalies
from causalrag.sensitivity.evalue import evalue_for_estimator
from causalrag.sensitivity.sensemakr_py import sensemakr as run_sensemakr
from causalrag.sensitivity.verdict import aggregate as aggregate_two_panel

VerdictColor = Literal["green", "yellow", "red", "unknown", "neutral"]
AggregateColor = Literal["green", "yellow", "red", "unknown"]

# Panel name vocabulary — frozen here so reports and downstream consumers
# can rely on exact strings.
_PANEL_NAMES = (
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


class SensitivityPanel(BaseModel):
    """One sensitivity-test result for one estimate."""

    model_config = ConfigDict(extra="forbid")

    name: str
    backend: str
    result: dict[str, Any]
    verdict_contribution: VerdictColor
    rationale: str
    available: bool


class SensitivityDashboard(BaseModel):
    """Aggregated view across all sensitivity panels for one Q7 estimate."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    aggregate_verdict: AggregateColor
    aggregate_rationale: str
    panels: list[SensitivityPanel]


# ─────────── Per-panel helpers ───────────────────────────────────────────


def _outcome_dtype(protocol: StudyProtocol | None) -> str:
    """Best-effort recovery of the outcome dtype. Falls back to ``continuous``
    when the protocol's flag set is unavailable — the standardized E-value
    branch then refuses to compute on implausibly large magnitudes, which
    is the safe failure mode."""
    if protocol is None:
        return "continuous"
    try:
        from causalrag.core.flags import DataFlag  # local: avoids cycle at import

        if DataFlag.RIGHT_CENSORED_OUTCOME in protocol.flags:
            return "survival"
        if DataFlag.BINARY_OUTCOME in protocol.flags:
            return "binary"
    except Exception:  # noqa: BLE001 — flags layer is best-effort here
        pass
    return "continuous"


def _confounders(protocol: StudyProtocol | None, df_columns: set[str]) -> tuple[str, ...]:
    """Pull confounder names from the discovery report. Empty tuple is fine —
    sensemakr will then fit OLS on T alone and report the marginal RV."""
    if protocol is None or protocol.discovery is None:
        return ()
    try:
        from causalrag.core.flags import VariableRole

        return tuple(
            v.name
            for v in protocol.discovery.columns
            if v.role is VariableRole.CONFOUNDER and v.name in df_columns
        )
    except Exception:  # noqa: BLE001
        return ()


def _evalue_panel(result: EstimationResult, *, outcome_dtype: str) -> SensitivityPanel:
    ev = evalue_for_estimator(result, outcome_dtype=outcome_dtype)
    if ev.reason is not None:
        return SensitivityPanel(
            name="e_value",
            backend="python.evalue",
            result=ev.model_dump(),
            verdict_contribution="unknown",
            rationale=ev.reason,
            available=True,
        )
    if ev.e_value_ci is not None and ev.e_value_ci <= 1.25:
        color: VerdictColor = "red"
    elif ev.e_value >= 3.0:
        color = "green"
    elif ev.e_value >= 1.75:
        color = "yellow"
    else:
        color = "red"
    return SensitivityPanel(
        name="e_value",
        backend="python.evalue",
        result=ev.model_dump(),
        verdict_contribution=color,
        rationale=(
            f"E-value={ev.e_value:.2f} ({ev.scale}); "
            f"CI-bound E-value={ev.e_value_ci if ev.e_value_ci is not None else 'n/a'}."
        ),
        available=True,
    )


def _sensemakr_panel(
    df: Any,
    *,
    treatment: str,
    outcome: str,
    covariates: tuple[str, ...],
) -> SensitivityPanel:
    sm = run_sensemakr(df, treatment=treatment, outcome=outcome, covariates=covariates)
    rv = sm.robustness_value
    if rv >= 0.2:
        color: VerdictColor = "green"
    elif rv >= 0.075:
        color = "yellow"
    else:
        color = "red"
    return SensitivityPanel(
        name="sensemakr",
        backend=sm.backend,
        result=sm.model_dump(),
        verdict_contribution=color,
        rationale=f"Partial-R² robustness value RV={rv:.3f} (q=1).",
        available=True,
    )


def _tipping_panel(result: EstimationResult) -> SensitivityPanel:
    """Tipping-point via the R bridge. Routed through a try/except in the
    caller, so any R-unavailability collapses to ``available=False``."""
    from causalrag.estimators.rbridge.sensitivity_r import tipping_point

    if result.se is None:
        return SensitivityPanel(
            name="tipping_point",
            backend="rbridge.tipr",
            result={"error": "se is None — tipping-point requires a standard error"},
            verdict_contribution="unknown",
            rationale="No SE on the estimate; tipping-point analysis skipped.",
            available=False,
        )
    out = tipping_point(
        estimate=result.point_estimate,
        se=result.se,
        n_treated=max(result.n_used // 2, 1),
        n_untreated=max(result.n_used - result.n_used // 2, 1),
    )
    strength = out.get("tipping_confounder_strength") or out.get("tipping_smd")
    # Larger required confounder = more robust.
    if isinstance(strength, int | float):
        if strength >= 2.0:
            color: VerdictColor = "green"
        elif strength >= 1.0:
            color = "yellow"
        else:
            color = "red"
    else:
        color = "unknown"
    return SensitivityPanel(
        name="tipping_point",
        backend="rbridge.tipr",
        result=out,
        verdict_contribution=color,
        rationale=(
            f"Tipping-point confounder strength ≈ {strength}"
            if strength is not None
            else "Tipping-point produced no numeric strength."
        ),
        available=True,
    )


def _refutation_summary_panel(result: EstimationResult) -> SensitivityPanel:
    """Distill ``result.refutations`` into a single colored panel.

    A refutation 'passes' when the placebo / random-confounder shift is
    within a few SE of zero; large shifts indicate the estimate is not
    robust to that refutation. We're conservative: any test with
    |delta_in_se_units| > 3 forces red.
    """
    refs = result.refutations or {}
    tests: list[dict[str, Any]] = []
    if isinstance(refs.get("tests"), list):
        tests = [t for t in refs["tests"] if isinstance(t, dict)]
    else:
        tests = [v for v in refs.values() if isinstance(v, dict)]

    if not tests:
        return SensitivityPanel(
            name="refutation_summary",
            backend="python.refutation_summary",
            result={"n_tests": 0},
            verdict_contribution="neutral",
            rationale="No refutation tests on this estimate.",
            available=True,
        )

    deltas: list[float] = []
    for t in tests:
        d = t.get("delta_in_se_units")
        if isinstance(d, int | float):
            deltas.append(float(d))
    max_abs = max((abs(d) for d in deltas), default=None)
    if max_abs is None:
        color: VerdictColor = "neutral"
        rationale = f"{len(tests)} refutation tests ran but reported no SE-scaled delta."
    elif max_abs > 3.0:
        color = "red"
        rationale = (
            f"Worst refutation shift = {max_abs:.2f} SE; estimate is not robust."
        )
    elif max_abs > 1.0:
        color = "yellow"
        rationale = (
            f"Worst refutation shift = {max_abs:.2f} SE; mild sensitivity."
        )
    else:
        color = "green"
        rationale = f"All refutations within {max_abs:.2f} SE of the point estimate."

    return SensitivityPanel(
        name="refutation_summary",
        backend="python.refutation_summary",
        result={"n_tests": len(tests), "max_abs_delta_in_se_units": max_abs, "tests": tests},
        verdict_contribution=color,
        rationale=rationale,
        available=True,
    )


def _anomaly_audit_panel(
    result: EstimationResult,
    *,
    walk: RoadmapWalk,
    treatment: str,
    outcome: str,
) -> SensitivityPanel:
    audit = audit_for_anomalies(
        result=result,
        walk=walk,
        treatment=treatment,
        outcome=outcome,
        client=None,  # deterministic pre-screen only — LLM is optional
    )
    n_flags = len(audit.flags)
    if audit.recommendation == "disqualify":
        color: VerdictColor = "red"
    elif audit.recommendation == "rerun_with_different_estimator":
        color = "yellow"
    elif n_flags == 0:
        color = "green"
    else:
        color = "yellow"
    return SensitivityPanel(
        name="anomaly_audit",
        backend="python.anomaly_audit",
        result=audit.model_dump(),
        verdict_contribution=color,
        rationale=audit.overall_note or f"{n_flags} anomaly flag(s).",
        available=True,
    )


def _unavailable_panel(name: str, reason: str) -> SensitivityPanel:
    """Construct an explicitly-unavailable panel slot.

    Some sensitivity tests (Rosenbaum Γ on matched-pair sets, Manski
    bounds with monotone-IV, negative-control outcomes, OVB-Chernozhukov
    via DoWhy 0.12) are catalog items not yet wired into the pipeline.
    The dashboard surfaces them as unavailable placeholders so reports
    visibly track what is *not* yet running.
    """
    return SensitivityPanel(
        name=name,
        backend="not_implemented",
        result={"error": reason},
        verdict_contribution="unknown",
        rationale=reason,
        available=False,
    )


# ─────────── Aggregation ─────────────────────────────────────────────────


def _aggregate(panels: list[SensitivityPanel]) -> tuple[AggregateColor, str]:
    """Strictest-wins aggregation across panels.

    Only panels with ``available=True`` and a non-``neutral`` contribution
    influence the verdict. ``unknown`` is treated as a 'cannot-rule-out'
    signal; if every contributing panel is ``unknown``, the aggregate is
    ``unknown``.
    """
    contributing = [
        p for p in panels if p.available and p.verdict_contribution != "neutral"
    ]
    if not contributing:
        return "unknown", "No sensitivity panels produced a contributing verdict."

    colors = [p.verdict_contribution for p in contributing]
    if "red" in colors:
        agg: AggregateColor = "red"
    elif "yellow" in colors:
        agg = "yellow"
    elif all(c == "green" for c in colors):
        agg = "green"
    else:  # at least one unknown and no red/yellow
        agg = "unknown"

    rationale_parts = [f"{p.name}={p.verdict_contribution}" for p in contributing]
    return agg, "; ".join(rationale_parts)


# ─────────── Public API ──────────────────────────────────────────────────


def run_sensitivity_dashboard(
    *,
    result: EstimationResult,
    walk: RoadmapWalk,
    df: Any,
    candidate: Any,
    protocol: StudyProtocol | None = None,
    propose_client: Any = None,  # noqa: ARG001 — accepted for future LLM-driven panels
) -> SensitivityDashboard:
    """Build the full sensitivity dashboard for one Q7 estimate.

    Every panel call is wrapped in ``try/except``; one failure does not
    take down the dashboard. The exit verdict is the strictest-wins
    aggregate over panels whose computation succeeded.

    Parameters
    ----------
    result:
        The Q7 estimation result to audit.
    walk:
        The :class:`RoadmapWalk` carrying the hypothesis id and chain
        context. The walk's refutation history is read off the estimate
        directly, not from the walk, so a freshly-constructed walk works.
    df:
        The analysis frame. Needed for sensemakr (it refits OLS).
    candidate:
        The :class:`CandidateExperiment` describing the (T, Y) pair.
    protocol:
        Optional :class:`StudyProtocol`. Used to read outcome dtype +
        confounder columns; absent → fall back to ``continuous`` outcome
        and empty confounder list.
    propose_client:
        Reserved for future LLM-driven panel proposals (e.g., the
        anomaly-audit LLM consultation). Currently unused; declared so
        the master loop can pass its existing client without a typecheck
        failure.
    """
    treatment = getattr(candidate, "treatment", None) or ""
    outcome = getattr(candidate, "outcome", None) or ""
    outcome_dtype = _outcome_dtype(protocol)
    df_columns = set(getattr(df, "columns", []))
    covariates = _confounders(protocol, df_columns)

    panels: list[SensitivityPanel] = []

    # 1. E-value
    try:
        panels.append(_evalue_panel(result, outcome_dtype=outcome_dtype))
    except Exception as exc:  # noqa: BLE001 — failure-safe
        panels.append(
            SensitivityPanel(
                name="e_value",
                backend="python.evalue",
                result={"error": f"{type(exc).__name__}: {exc}"},
                verdict_contribution="unknown",
                rationale=f"E-value failed: {type(exc).__name__}",
                available=False,
            )
        )

    # 2. Sensemakr
    if treatment and outcome and treatment in df_columns and outcome in df_columns:
        try:
            panels.append(
                _sensemakr_panel(
                    df,
                    treatment=treatment,
                    outcome=outcome,
                    covariates=covariates,
                )
            )
        except Exception as exc:  # noqa: BLE001
            panels.append(
                SensitivityPanel(
                    name="sensemakr",
                    backend="pysensemakr",
                    result={"error": f"{type(exc).__name__}: {exc}"},
                    verdict_contribution="unknown",
                    rationale=f"Sensemakr failed: {type(exc).__name__}",
                    available=False,
                )
            )
    else:
        panels.append(
            _unavailable_panel(
                "sensemakr",
                "Sensemakr requires treatment + outcome columns to be in df.",
            )
        )

    # 3. Tipping point (R bridge)
    try:
        panels.append(_tipping_panel(result))
    except Exception as exc:  # noqa: BLE001 — typically R / tipr missing
        panels.append(
            SensitivityPanel(
                name="tipping_point",
                backend="rbridge.tipr",
                result={"error": f"{type(exc).__name__}: {exc}"},
                verdict_contribution="unknown",
                rationale="tipr unavailable (likely missing R or R package).",
                available=False,
            )
        )

    # 4. Refutation summary
    try:
        panels.append(_refutation_summary_panel(result))
    except Exception as exc:  # noqa: BLE001
        panels.append(
            SensitivityPanel(
                name="refutation_summary",
                backend="python.refutation_summary",
                result={"error": f"{type(exc).__name__}: {exc}"},
                verdict_contribution="unknown",
                rationale="Refutation summary failed.",
                available=False,
            )
        )

    # 5. Anomaly audit (deterministic pre-screen)
    try:
        panels.append(
            _anomaly_audit_panel(
                result, walk=walk, treatment=treatment, outcome=outcome
            )
        )
    except Exception as exc:  # noqa: BLE001
        panels.append(
            SensitivityPanel(
                name="anomaly_audit",
                backend="python.anomaly_audit",
                result={"error": f"{type(exc).__name__}: {exc}"},
                verdict_contribution="unknown",
                rationale="Anomaly audit failed.",
                available=False,
            )
        )

    # 6-9. Not-yet-implemented catalog slots — Sprint 6 / Sprint 9 will fill.
    panels.append(
        _unavailable_panel(
            "rosenbaum",
            "Rosenbaum Γ bounds require a matched-pair fit; not yet wired.",
        )
    )
    panels.append(
        _unavailable_panel(
            "manski",
            "Manski bounds module is on the Sprint-9 roadmap (no-assumption + MIV).",
        )
    )
    panels.append(
        _unavailable_panel(
            "negative_control",
            "Negative-control outcome panel not yet implemented.",
        )
    )
    panels.append(
        _unavailable_panel(
            "ovb_chernozhukov",
            "DoWhy 0.12 Chernozhukov-Cinelli-Newey OVB not yet wired.",
        )
    )

    aggregate_color, aggregate_rationale = _aggregate(panels)

    # Cross-check sanity: the legacy two-panel aggregator (verdict.aggregate)
    # should not contradict ours when both E-value and sensemakr ran cleanly.
    # We don't override; we just annotate the rationale if there's a
    # disagreement so reviewers can spot the discrepancy.
    try:
        ev_panel = next(p for p in panels if p.name == "e_value" and p.available)
        sm_panel = next(p for p in panels if p.name == "sensemakr" and p.available)
        if ev_panel.result.get("reason") is None and sm_panel.result.get("robustness_value") is not None:
            from causalrag.sensitivity.evalue import EValueResult
            from causalrag.sensitivity.sensemakr_py import SensemakrResult

            ev_obj = EValueResult.model_validate(ev_panel.result)
            sm_obj = SensemakrResult.model_validate(sm_panel.result)
            two = aggregate_two_panel(evalue=ev_obj, sensemakr=sm_obj, rule="min")
            if two.color != aggregate_color and aggregate_color in ("green", "yellow", "red"):
                aggregate_rationale = (
                    f"{aggregate_rationale} | legacy two-panel verdict={two.color}"
                )
    except StopIteration:
        pass
    except Exception:  # noqa: BLE001 — diagnostic only
        pass

    return SensitivityDashboard(
        hypothesis_id=walk.hypothesis_id,
        aggregate_verdict=aggregate_color,
        aggregate_rationale=aggregate_rationale,
        panels=panels,
    )


# ─────────── HTML rendering ──────────────────────────────────────────────


_COLOR_CSS = {
    "green": "#2e7d32",
    "yellow": "#ed6c02",
    "red": "#c62828",
    "unknown": "#616161",
    "neutral": "#90a4ae",
}


def _escape(s: str) -> str:
    """Minimal HTML escape — avoids a Jinja / markupsafe dep."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_sensitivity_dashboard_html(d: SensitivityDashboard) -> str:
    """Render the dashboard as a self-contained HTML fragment.

    Intentionally template-free — the report layer embeds this in a
    Quarto chunk, so we want plain HTML with inline styles rather than
    a Jinja-dependency.
    """
    header_color = _COLOR_CSS.get(d.aggregate_verdict, "#616161")
    rows: list[str] = []
    for p in d.panels:
        chip_color = _COLOR_CSS.get(p.verdict_contribution, "#616161")
        avail_marker = "" if p.available else " (unavailable)"
        rows.append(
            "<tr>"
            f'<td style="padding:4px 8px;">{_escape(p.name)}{avail_marker}</td>'
            f'<td style="padding:4px 8px;">{_escape(p.backend)}</td>'
            f'<td style="padding:4px 8px;">'
            f'<span style="display:inline-block;padding:2px 8px;border-radius:12px;'
            f'background:{chip_color};color:white;font-size:0.85em;">'
            f"{_escape(p.verdict_contribution)}</span></td>"
            f'<td style="padding:4px 8px;">{_escape(p.rationale)}</td>'
            "</tr>"
        )
    table = (
        '<table style="border-collapse:collapse;width:100%;font-family:sans-serif;'
        'font-size:0.95em;">'
        '<thead><tr style="background:#f5f5f5;border-bottom:1px solid #ccc;">'
        '<th style="text-align:left;padding:4px 8px;">Panel</th>'
        '<th style="text-align:left;padding:4px 8px;">Backend</th>'
        '<th style="text-align:left;padding:4px 8px;">Verdict</th>'
        '<th style="text-align:left;padding:4px 8px;">Rationale</th>'
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    header = (
        f'<div style="font-family:sans-serif;">'
        f'<h3 style="margin-bottom:4px;">Sensitivity dashboard — '
        f"{_escape(d.hypothesis_id)}</h3>"
        f'<div style="margin-bottom:8px;">'
        f'<span style="display:inline-block;padding:2px 10px;border-radius:12px;'
        f'background:{header_color};color:white;font-weight:bold;">'
        f"{_escape(d.aggregate_verdict)}</span> "
        f'<span style="color:#555;">{_escape(d.aggregate_rationale)}</span>'
        f"</div>"
    )
    return header + table + "</div>"


__all__ = [
    "AggregateColor",
    "SensitivityDashboard",
    "SensitivityPanel",
    "VerdictColor",
    "render_sensitivity_dashboard_html",
    "run_sensitivity_dashboard",
]
