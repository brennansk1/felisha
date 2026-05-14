"""Self-contained HTML / Markdown renderer for the StudyProtocol.

PDD §12. We deliberately avoid pulling in Jinja2 / Quarto / weasyprint for
v0.1 — a hand-rolled HTML template keeps the dependency footprint small and
the report is fully reproducible from the YAML alone. The rendered HTML
embeds inline CSS matching the TUI palette (dark navy + sky-blue accent) so
the report looks of a piece with the terminal experience.

The report includes:

- Cover card (project · tier · dataset · generated-at).
- Research question + domain summary.
- Discovery: column roles, flag chips, candidate-DAG list, Layer-4 audit
  contradictions.
- Feasibility: admissible / borderline / underpowered counts.
- Hypothesis queue (top 10).
- Per-hypothesis Roadmap walks: identification strategy, adjustment set,
  estimate card with point + CI + p-value, refutations, sensitivity.
- Analyst-decision ledger (every default accepted + every override).
- Provenance section: LLM model digests, seeds, Python / R versions,
  timestamps.
- BH-adjusted summary table when multiple hypotheses produced estimates.
- "Limitations & Failure Modes" appendix (always shipped per PDD §31).
"""

from __future__ import annotations

import html
from datetime import datetime
from typing import TYPE_CHECKING, Any

from causalrag.core.protocol import StudyProtocol

if TYPE_CHECKING:
    from causalrag.reporting.synthesis import ExecutiveSynthesis


_CSS = """
:root {
  --bg: #0c1422;
  --bg-soft: #10182a;
  --surface: #152034;
  --border: #2a3a55;
  --rule: #24314a;
  --text: #eef2f9;
  --text-soft: #cfd6e4;
  --text-muted: #9aa3b5;
  --text-faint: #4d5773;
  --accent: #5fa8ff;
  --accent-hi: #9ec2ff;
  --success: #7ed2e6;
  --warning: #a3b6da;
  --danger: #e08877;
  --serif: 'Merriweather', Georgia, serif;
  --mono: 'JetBrains Mono', ui-monospace, Menlo, monospace;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--serif);
  font-size: 15px;
  line-height: 1.65;
  letter-spacing: 0.012em;
}
.container { max-width: 980px; margin: 0 auto; padding: 32px 28px 80px; }
h1, h2, h3, h4 { font-family: var(--serif); color: var(--text); letter-spacing: 0.005em; }
h1 { font-size: 28px; margin: 0 0 6px; }
h2 { font-size: 20px; margin: 28px 0 10px; padding-top: 18px; border-top: 1px dashed var(--rule); }
h3 { font-size: 16px; margin: 18px 0 8px; color: var(--accent-hi); }
.eyebrow { font-family: var(--mono); font-size: 11px; color: var(--text-faint); letter-spacing: 0.02em; }
.sub { font-family: var(--serif); font-style: italic; font-size: 14px; color: var(--text-muted); }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px; margin: 12px 0; }
.kv { display: grid; grid-template-columns: 200px 1fr; gap: 4px 16px; font-family: var(--mono); font-size: 13px; }
.kv .k { color: var(--text-faint); }
.kv .v { color: var(--text-soft); }
table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12.5px; margin: 8px 0; }
th { text-align: left; font-weight: 400; color: var(--text-faint); font-size: 11px; padding: 4px 12px 6px; border-bottom: 1px dashed var(--rule); }
td { padding: 4px 12px; color: var(--text-soft); border-bottom: none; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.chip { display: inline-block; padding: 1px 7px; border-radius: 3px; font-family: var(--mono); font-size: 11px; color: var(--text-muted); margin-right: 4px; border: 1px solid var(--border); }
.chip.ok { color: var(--success); border-color: rgba(126, 210, 230, 0.4); }
.chip.warn { color: var(--warning); border-color: rgba(163, 182, 218, 0.4); }
.chip.err { color: var(--danger); border-color: rgba(224, 136, 119, 0.4); }
.chip.acc { color: var(--accent-hi); border-color: rgba(158, 194, 255, 0.4); }
.divider { height: 1px; background: var(--rule); margin: 16px 0; }
.headline {
  font-family: var(--mono);
  font-size: 28px;
  font-weight: 900;
  color: var(--accent-hi);
  line-height: 1;
}
.headline-ci {
  font-family: var(--mono);
  font-size: 14px;
  color: var(--text-soft);
}
details { margin: 10px 0; padding: 6px 10px; background: var(--bg-soft); border: 1px solid var(--border); border-radius: 4px; }
summary { font-family: var(--mono); font-size: 12px; color: var(--accent); cursor: pointer; }
pre, code { font-family: var(--mono); font-size: 12px; color: var(--text-soft); background: var(--bg-soft); padding: 1px 4px; border-radius: 2px; }
.serif { font-family: var(--serif); }
.dim { color: var(--text-faint); }
.muted { color: var(--text-muted); }
.acc { color: var(--accent-hi); }
.ok  { color: var(--success); }
.warn { color: var(--warning); }
.err { color: var(--danger); }
.lede { font-family: var(--serif); font-size: 16px; line-height: 1.75; color: var(--text-soft); }
"""


def _e(text: Any) -> str:
    return html.escape("" if text is None else str(text))


def _chip(text: str, kind: str = "") -> str:
    cls = f"chip {kind}".strip()
    return f'<span class="{cls}">{_e(text)}</span>'


def render_report(
    protocol: StudyProtocol,
    fmt: str = "html",
    *,
    executive_synthesis: "ExecutiveSynthesis | None" = None,
) -> str:
    if fmt == "md":
        return _render_markdown(protocol, executive_synthesis=executive_synthesis)
    return _render_html(protocol, executive_synthesis=executive_synthesis)


def _render_html(
    protocol: StudyProtocol,
    *,
    executive_synthesis: "ExecutiveSynthesis | None" = None,
) -> str:
    parts: list[str] = []
    parts.append("<!doctype html><html lang='en'><head>")
    parts.append(f"<meta charset='utf-8'><title>{_e(protocol.name)} · CausalRoadmap</title>")
    parts.append("<style>" + _CSS + "</style>")
    parts.append("</head><body><div class='container'>")

    # Cover
    parts.append(_cover(protocol))

    # Executive synthesis — domain-aware findings, rendered FIRST so a
    # reader sees the headline conclusions before the technical detail.
    if executive_synthesis is not None:
        from causalrag.reporting.synthesis import render_executive_synthesis_html

        parts.append(render_executive_synthesis_html(executive_synthesis))

    # Research question + brief
    if protocol.research_question:
        parts.append("<h2>Research question</h2>")
        parts.append(f"<p class='lede'>{_e(protocol.research_question)}</p>")
    if protocol.discovery and protocol.discovery.domain_brief:
        parts.append("<h2>Domain brief</h2>")
        parts.append(f"<p class='lede'>{_e(protocol.discovery.domain_brief)}</p>")
    # Discovery
    parts.append(_discovery_section(protocol))
    # Feasibility
    parts.append(_feasibility_section(protocol))
    # Hypothesis queue
    parts.append(_hypothesis_section(protocol))
    # Roadmap walks
    parts.append(_walks_section(protocol))
    # Decision ledger
    parts.append(_ledger_section(protocol))
    # Provenance
    parts.append(_provenance_section(protocol))
    # Limitations appendix
    parts.append(_limitations_appendix())

    parts.append("</div></body></html>")
    return "\n".join(parts)


def _cover(protocol: StudyProtocol) -> str:
    rows = [
        ("project", protocol.name),
        ("tier", protocol.tier),
        ("version", protocol.version),
        ("created", protocol.created.isoformat(timespec="seconds")),
        ("updated", protocol.updated.isoformat(timespec="seconds")),
        ("generated", datetime.utcnow().isoformat(timespec="seconds") + "Z"),
    ]
    if protocol.dataset:
        rows.append(("source", protocol.dataset.source))
        rows.append(("n_rows × n_cols", f"{protocol.dataset.n_rows} × {protocol.dataset.n_cols}"))
    kv = "".join(f"<div class='k'>{_e(k)}</div><div class='v'>{_e(v)}</div>" for k, v in rows)
    return f"""
<h1>{_e(protocol.name)}</h1>
<p class='sub'>CausalRoadmap study report — Petersen-van der Laan</p>
<div class='card'><div class='kv'>{kv}</div></div>
"""


def _discovery_section(protocol: StudyProtocol) -> str:
    if protocol.discovery is None:
        return ""
    out = ["<h2>Phase 1 · Discovery</h2>"]
    # Flag chips with hover descriptions
    try:
        from causalrag.core.flag_descriptions import describe_safe

        flag_rows: list[str] = []
        for f in sorted(protocol.flags, key=lambda x: x.value):
            d = describe_safe(f)
            tooltip = f"{d.summary} · {d.implication}"
            flag_rows.append(
                f"<span class='chip acc' title='{_e(tooltip)}'>{_e(f.value)}</span>"
            )
        if flag_rows:
            out.append("<h3>Active data flags</h3>")
            out.append(f"<p>{''.join(flag_rows)}</p>")
            # Render full semantics in a collapsible block so the
            # report stays self-explanatory for non-LLM readers.
            out.append("<details><summary>What each flag means</summary><ul>")
            for f in sorted(protocol.flags, key=lambda x: x.value):
                d = describe_safe(f)
                out.append(
                    f"<li><code>{_e(f.value)}</code> — {_e(d.summary)} "
                    f"<em>{_e(d.implication)}</em></li>"
                )
            out.append("</ul></details>")
    except Exception:
        # Fall back to the original chip rendering when
        # flag_descriptions isn't importable.
        chips = ""
        for f in sorted(protocol.flags, key=lambda x: x.value):
            chips += _chip(f.value, kind="acc")
        if chips:
            out.append(f"<p>{chips}</p>")
    # Identification warnings from the brief.
    if getattr(protocol.discovery, "identification_warnings", ()):
        out.append("<h3>Identification warnings (expert brief)</h3>")
        out.append("<ul>")
        for w in protocol.discovery.identification_warnings:
            out.append(f"<li class='warn'>{_e(w)}</li>")
        out.append("</ul>")
    # Variable role table
    if protocol.discovery.columns:
        out.append("<h3>Variable specifications</h3>")
        out.append("<table><thead><tr><th>Column</th><th>Role</th><th>Temporal</th><th>Dtype</th><th>Description</th></tr></thead><tbody>")
        for v in protocol.discovery.columns:
            desc = (v.semantic_description or "")[:120]
            out.append(
                f"<tr><td><code>{_e(v.name)}</code></td><td>{_e(v.role.value)}</td>"
                f"<td>{_e(v.measured_at or '—')}</td><td>{_e(v.dtype)}</td>"
                f"<td>{_e(desc)}</td></tr>"
            )
        out.append("</tbody></table>")
    # Candidate DAGs
    if protocol.discovery.candidate_graphs:
        out.append("<h3>Candidate DAGs</h3>")
        for g in protocol.discovery.candidate_graphs[:3]:
            edges = ", ".join(f"{_e(e.source)} → {_e(e.target)}" for e in g.edges)
            out.append(f"<details><summary>rank {g.rank} · {len(g.edges)} edges</summary><p class='dim'>{edges}</p></details>")
    return "".join(out)


def _feasibility_section(protocol: StudyProtocol) -> str:
    if protocol.feasibility is None:
        return ""
    out = ["<h2>Phase 2 · Feasibility</h2>"]
    n_admissible = len(protocol.feasibility.admissible_pairs)
    out.append(
        f"<p class='muted'>{n_admissible} admissible (treatment, outcome) pairs · "
        f"α = {protocol.feasibility.alpha} · target power = {protocol.feasibility.power_target}</p>"
    )
    if n_admissible:
        out.append("<table><thead><tr><th>Treatment</th><th>Outcome</th></tr></thead><tbody>")
        for t, y in protocol.feasibility.admissible_pairs:
            out.append(f"<tr><td><code>{_e(t)}</code></td><td><code>{_e(y)}</code></td></tr>")
        out.append("</tbody></table>")
    if protocol.feasibility.notes:
        out.append(f"<p class='dim'>{_e(protocol.feasibility.notes)}</p>")
    return "".join(out)


def _hypothesis_section(protocol: StudyProtocol) -> str:
    if not protocol.hypothesis_queue:
        return ""
    out = ["<h2>Phase 3 · Hypothesis queue</h2>"]
    out.append(
        f"<p class='muted'>{len(protocol.hypothesis_queue)} hypotheses · "
        f"counterfactual share {protocol.counterfactual_ratio:.0%}</p>"
    )
    out.append("<table><thead><tr><th>ID</th><th>Treatment</th><th>Outcome</th><th>Estimand</th><th>Impact</th><th>Rationale</th></tr></thead><tbody>")
    for h in protocol.hypothesis_queue[:10]:
        ek = h.estimand.klass.value if h.estimand else "?"
        out.append(
            f"<tr><td><code>{_e(h.id)}</code></td>"
            f"<td><code>{_e(h.treatment)}</code></td>"
            f"<td><code>{_e(h.outcome)}</code></td>"
            f"<td>{_e(ek)}</td>"
            f"<td class='num'>{_e(f'{h.impact_score:.2f}' if h.impact_score else '—')}</td>"
            f"<td>{_e((h.rationale or '')[:80])}</td></tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _walks_section(protocol: StudyProtocol) -> str:
    if not protocol.roadmap_walks:
        return ""
    out = ["<h2>Phase 4-5 · Roadmap walks</h2>"]
    # BH-adjusted summary if multiple estimates
    estimates = []
    for key, walk in protocol.roadmap_walks.items():
        if walk.q7_estimates:
            estimates.append((key, walk.q7_estimates[-1]))
    if len(estimates) > 1:
        bh = _bh_adjust([e.p_value or 1.0 for _, e in estimates])
        out.append("<h3>Multiple-hypothesis summary (BH-adjusted)</h3>")
        out.append("<table><thead><tr><th>Hypothesis</th><th>Point</th><th>95% CI</th><th>p (raw)</th><th>p (BH)</th></tr></thead><tbody>")
        for (key, est), q_adj in zip(estimates, bh):
            ci = f"[{est.ci_low:+.4f}, {est.ci_high:+.4f}]" if est.ci_low is not None else "—"
            out.append(
                f"<tr><td><code>{_e(key)}</code></td>"
                f"<td class='num'>{est.point_estimate:+.4f}</td>"
                f"<td class='num'>{ci}</td>"
                f"<td class='num'>{est.p_value:.4g if est.p_value else '—'}</td>"
                f"<td class='num'>{q_adj:.4g}</td></tr>"
            )
        out.append("</tbody></table>")
    # Per-walk details
    for key, walk in protocol.roadmap_walks.items():
        # Chain-linkage label for foundation children
        chain_label = ""
        if walk.parent_id:
            chain_label = (
                f" <span class='chip dim'>foundation of "
                f"<code>{_e(walk.parent_id)}</code> · chain="
                f"<code>{_e(walk.chain_id or '—')}</code></span>"
            )
        elif walk.chain_id:
            chain_label = (
                f" <span class='chip dim'>chain root · "
                f"<code>{_e(walk.chain_id)}</code></span>"
            )

        out.append(f"<details open><summary>{_e(key)}{chain_label}</summary>")
        ident = walk.q5_identification or {}
        out.append("<div class='kv'>")
        out.append(f"<div class='k'>identification</div><div class='v'>{_e(ident.get('strategy', '—'))}</div>")
        out.append(f"<div class='k'>identifiable</div><div class='v'>{_e(ident.get('identifiable', '—'))}</div>")
        out.append(f"<div class='k'>adjustment set</div><div class='v'>{_e(', '.join(ident.get('adjustment_set', [])))}</div>")

        # Identification narration (LLM-generated)
        narration = ident.get("narration") if isinstance(ident, dict) else None
        if isinstance(narration, dict):
            if narration.get("strategy_explanation"):
                out.append(
                    f"<div class='k'>why this works</div>"
                    f"<div class='v serif'>{_e(narration['strategy_explanation'])}</div>"
                )
            if narration.get("blocked_paths"):
                bp = " · ".join(narration["blocked_paths"])
                out.append(
                    f"<div class='k'>blocked paths</div><div class='v'>{_e(bp)}</div>"
                )
            if narration.get("unblocked_paths"):
                up = " · ".join(narration["unblocked_paths"])
                out.append(
                    f"<div class='k'>open backdoor paths</div>"
                    f"<div class='v err'>{_e(up)}</div>"
                )
            if narration.get("analyst_assertions"):
                out.append(
                    "<div class='k'>analyst trusts</div><div class='v'>"
                    + "; ".join(_e(a) for a in narration["analyst_assertions"])
                    + "</div>"
                )
            if narration.get("confidence"):
                out.append(
                    f"<div class='k'>identification confidence</div>"
                    f"<div class='v'>{_e(narration['confidence'])}</div>"
                )

        if walk.q7_estimates:
            est = walk.q7_estimates[-1]
            out.append(f"<div class='k'>estimator</div><div class='v'>{_e(est.estimator_id)}</div>")
            out.append(f"<div class='k'>point estimate</div><div class='v acc'>{est.point_estimate:+.4f}</div>")
            if est.ci_low is not None and est.ci_high is not None:
                out.append(f"<div class='k'>95% CI</div><div class='v'>[{est.ci_low:+.4f}, {est.ci_high:+.4f}]</div>")
            if est.p_value is not None:
                out.append(f"<div class='k'>p-value</div><div class='v'>{est.p_value:.4g}</div>")
            # Adjusted p-value if multiple-testing applied
            diag = est.diagnostics if isinstance(est.diagnostics, dict) else {}
            if diag.get("adjusted_p_value") is not None:
                adj_method = diag.get("adjustment_method", "")
                out.append(
                    f"<div class='k'>adjusted p ({_e(adj_method)})</div>"
                    f"<div class='v'>{diag['adjusted_p_value']:.4g}</div>"
                )
            out.append(f"<div class='k'>n used</div><div class='v'>{est.n_used}</div>")
            if est.refutations:
                n_pass = est.refutations.get("n_passed", 0)
                out.append(f"<div class='k'>refutations</div><div class='v'>{n_pass} / 3 passed</div>")

            # Anomaly audit findings
            audit = diag.get("anomaly_audit") if isinstance(diag, dict) else None
            if isinstance(audit, dict):
                flags = audit.get("flags") or []
                rec = audit.get("recommendation", "accept")
                if flags or rec != "accept":
                    color_class = (
                        "err" if rec == "disqualify"
                        else "warn" if rec == "rerun_with_different_estimator"
                        else "dim"
                    )
                    flag_str = ", ".join(flags) if flags else "(no flags)"
                    out.append(
                        f"<div class='k'>anomaly audit</div>"
                        f"<div class='v {color_class}'>{_e(rec)} — {_e(flag_str)}</div>"
                    )

            # Sensitivity interpretation (LLM-translated)
            interp = diag.get("sensitivity_interpretation") if isinstance(diag, dict) else None
            if isinstance(interp, dict) and interp.get("plain_language"):
                out.append(
                    f"<div class='k'>sensitivity narrative</div>"
                    f"<div class='v serif'>{_e(interp['plain_language'])}</div>"
                )
                if interp.get("plausibility_of_threshold_confounder"):
                    out.append(
                        f"<div class='k'>plausibility of threshold confounder</div>"
                        f"<div class='v dim'>{_e(interp['plausibility_of_threshold_confounder'])}</div>"
                    )

            # Tipping-point + negative-control auto-fired
            tip = diag.get("tipping_point") if isinstance(diag, dict) else None
            if isinstance(tip, dict) and not tip.get("error"):
                tip_smd = tip.get("tipping_smd")
                if tip_smd is not None:
                    out.append(
                        f"<div class='k'>tipping confounder SMD</div>"
                        f"<div class='v'>{tip_smd:.3f}</div>"
                    )
            negcontrol = diag.get("negative_control_scan") if isinstance(diag, dict) else None
            if isinstance(negcontrol, dict):
                interp_str = negcontrol.get("interpretation", "")
                if interp_str:
                    out.append(
                        f"<div class='k'>negative-control scan</div>"
                        f"<div class='v dim'>{_e(interp_str)}</div>"
                    )

        if walk.q8_interpretation:
            out.append(f"<div class='k'>Step 8</div><div class='v serif'>{_e(walk.q8_interpretation)}</div>")

        # Surface every sensitivity panel by name — keeps the report
        # transparent about which tests were actually run vs which were
        # skipped due to estimator-path or missing optional deps.
        diag_for_panels = diag if isinstance(diag, dict) else {}
        panel_chips: list[str] = []
        for panel_name in (
            "e_value", "sensemakr", "tipping_point", "refutation_summary",
            "anomaly_audit", "rosenbaum", "manski", "ovb_chernozhukov",
            "negative_control",
        ):
            panel_data = diag_for_panels.get(panel_name)
            if isinstance(panel_data, dict):
                available = panel_data.get("available", True)
                state = "ran" if available else "skipped"
            elif panel_data is None:
                state = "n/a"
            else:
                state = "ran"
            css = "ok" if state == "ran" else ("dim" if state == "skipped" else "muted")
            panel_chips.append(
                f"<span class='chip {css}'>{_e(panel_name)} · {state}</span>"
            )
        if panel_chips:
            out.append(
                "<div class='k'>sensitivity panels</div>"
                f"<div class='v'>{''.join(panel_chips)}</div>"
            )

        if walk.sensitivity_verdict:
            color_class = {
                "green": "ok", "yellow": "warn", "red": "err",
                "unknown": "dim", "errored": "err",
            }.get(walk.sensitivity_verdict, "dim")
            out.append(
                f"<div class='k'>sensitivity verdict</div>"
                f"<div class='v {color_class}'>● {_e(walk.sensitivity_verdict)}</div>"
            )

        if walk.failure_reason:
            out.append(
                f"<div class='k'>failure reason</div>"
                f"<div class='v err'>{_e(walk.failure_reason)}</div>"
            )

        out.append("</div></details>")
    return "".join(out)


def _ledger_section(protocol: StudyProtocol) -> str:
    if not protocol.decision_ledger and not protocol.overrides:
        return ""
    out = ["<h2>Analyst-decision ledger</h2>"]
    if protocol.decision_ledger:
        out.append("<table><thead><tr><th>Timestamp</th><th>Phase</th><th>Decision</th><th>Chose</th><th>Source</th></tr></thead><tbody>")
        for d in protocol.decision_ledger:
            out.append(
                f"<tr><td>{_e(d.timestamp.isoformat(timespec='seconds'))}</td>"
                f"<td>{_e(d.phase)}</td><td>{_e(d.decision)}</td>"
                f"<td>{_e(d.chose)}</td><td>{_e(d.source)}</td></tr>"
            )
        out.append("</tbody></table>")
    if protocol.overrides:
        out.append("<h3>Overrides</h3>")
        out.append("<table><thead><tr><th>Site</th><th>LLM proposed</th><th>Analyst chose</th><th>Reason</th></tr></thead><tbody>")
        for o in protocol.overrides:
            out.append(
                f"<tr><td><code>{_e(o.site)}</code></td>"
                f"<td>{_e(o.llm_value)}</td><td>{_e(o.analyst_value)}</td>"
                f"<td>{_e(o.reason or '—')}</td></tr>"
            )
        out.append("</tbody></table>")
    return "".join(out)


def _provenance_section(protocol: StudyProtocol) -> str:
    out = ["<h2>Provenance</h2><div class='kv'>"]
    llm = protocol.llm
    out.append(f"<div class='k'>llm backend</div><div class='v'><code>{_e(llm.backend)}</code></div>")
    out.append(f"<div class='k'>discovery model</div><div class='v'><code>{_e(llm.reasoning_model or '—')}</code></div>")
    out.append(f"<div class='k'>general model</div><div class='v'><code>{_e(llm.general_model or '—')}</code></div>")
    out.append(f"<div class='k'>model digest</div><div class='v'><code>{_e(llm.model_digest or '—')}</code></div>")
    out.append(f"<div class='k'>seed</div><div class='v'><code>{llm.seed}</code></div>")
    out.append(f"<div class='k'>temperature</div><div class='v'><code>{llm.temperature}</code></div>")
    out.append(f"<div class='k'>hardware tier</div><div class='v'><code>{llm.hardware_tier or '—'}</code></div>")
    out.append(f"<div class='k'>prompt pack</div><div class='v'><code>{_e(llm.prompt_pack_version)}</code></div>")
    out.append(f"<div class='k'>multiple testing</div><div class='v'><code>{_e(protocol.multiple_testing)}</code></div>")
    out.append("</div>")
    return "".join(out)


def _limitations_appendix() -> str:
    return """
<h2>Appendix · Limitations &amp; Failure Modes</h2>
<p class='lede'>
This study uses the Petersen-van der Laan Causal Roadmap and was generated
with an LLM-assisted discovery agent. The following failure modes are
documented here for transparency (PDD §31):
</p>
<ul>
  <li><strong>Causal Parrot</strong>: an LLM may propose a DAG it memorized
      from the literature rather than one supported by the data. Every
      LLM-proposed edge in this study was tested via partial-correlation
      audit (Layer 4); contradicted edges are surfaced in the decision
      ledger above.</li>
  <li><strong>Post-treatment bias</strong>: variables realized after the
      treatment cannot serve as confounders. The temporal-lattice check at
      Stage 1c blocked any column tagged
      <code>temporal_position=post_treatment</code> from entering the
      adjustment set.</li>
  <li><strong>Unmeasured confounding</strong>: the E-value and Sensemakr
      robustness value quantify how strong an unmeasured confounder would
      need to be to nullify the headline effect. See the per-walk
      sensitivity card.</li>
  <li><strong>Multiple-testing inflation</strong>: when more than one
      hypothesis was estimated, the headline conclusions quote
      BH-adjusted p-values.</li>
</ul>
"""


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted q-values, ordered same as input."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda kv: kv[1])
    adjusted = [0.0] * m
    prev = 1.0
    for rank_from_end, (orig_i, p) in enumerate(reversed(indexed), start=1):
        i_rank = m - rank_from_end + 1  # 1-indexed
        q = min(prev, p * m / i_rank)
        adjusted[orig_i] = q
        prev = q
    return adjusted


def _render_markdown(
    protocol: StudyProtocol,
    *,
    executive_synthesis: "ExecutiveSynthesis | None" = None,
) -> str:
    lines = [f"# {protocol.name}", "", "_CausalRoadmap study report_", ""]
    if executive_synthesis is not None:
        lines.extend([
            "## Executive synthesis",
            "",
            f"_Inferred domain: **{executive_synthesis.inferred_domain}**_",
            "",
            executive_synthesis.tldr,
            "",
        ])
        for f in executive_synthesis.findings:
            lines.extend([
                f"### Finding {f.rank} · {f.confidence.upper()} confidence",
                "",
                f"**{f.headline}**",
                "",
                f"- Effect: {f.quantified_effect}",
                f"- Implication: {f.domain_implication}",
                f"- Suggested next step: {f.suggested_next_step}",
            ])
            if f.caveats:
                lines.append(f"- Caveats: {' · '.join(f.caveats)}")
            lines.append("")
        if executive_synthesis.overall_caveats:
            lines.append("**Overall caveats:**")
            for c in executive_synthesis.overall_caveats:
                lines.append(f"- {c}")
            lines.append("")
    if protocol.research_question:
        lines.extend(["## Research question", "", protocol.research_question, ""])
    if protocol.discovery and protocol.discovery.domain_brief:
        lines.extend(["## Domain brief", "", protocol.discovery.domain_brief, ""])
    for key, walk in protocol.roadmap_walks.items():
        lines.append(f"### {key}")
        if walk.q7_estimates:
            est = walk.q7_estimates[-1]
            lines.append(f"- estimator: `{est.estimator_id}`")
            lines.append(f"- point: {est.point_estimate:+.4f}")
            if est.ci_low is not None:
                lines.append(f"- 95% CI: [{est.ci_low:+.4f}, {est.ci_high:+.4f}]")
            if est.p_value is not None:
                lines.append(f"- p-value: {est.p_value:.4g}")
        if walk.q8_interpretation:
            lines.append(f"- interpretation: {walk.q8_interpretation}")
        lines.append("")
    return "\n".join(lines)


__all__ = ["render_report"]
