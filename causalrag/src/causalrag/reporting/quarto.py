"""Quarto report renderer for CausalRoadmap (Sprint 1.4 — PDD §12).

Produces a ``.qmd`` source file that can be rendered to HTML / PDF /
DOCX with the ``quarto`` CLI. The renderer is self-contained: it
constructs the Quarto markdown body from the :class:`StudyProtocol`
(and optional :class:`ExecutiveSynthesis`) using simple string
substitution against ``templates/report.qmd.template``.

If ``run_quarto=True`` AND the ``quarto`` executable is on ``PATH``,
the function additionally invokes ``quarto render`` and returns the
path to the rendered HTML. Otherwise the ``.qmd`` source path is
returned and no subprocess is spawned. This keeps the function
testable on machines without a Quarto install.

Sensitivity verdict colour mapping:

* ``green``  → ``::: {.callout-note}``    (positive / acceptable)
* ``yellow`` → ``::: {.callout-warning}`` (caution)
* ``red``    → ``::: {.callout-important}`` (concerning)
* ``unknown`` / ``errored`` / missing → plain text (no callout)
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from causalrag.core.protocol import StudyProtocol

if TYPE_CHECKING:
    from causalrag.reporting.synthesis import ExecutiveSynthesis


_TEMPLATE_PATH = Path(__file__).parent / "templates" / "report.qmd.template"


# Mapping from sensitivity verdict colour → Quarto callout kind.
SENSITIVITY_CALLOUT: dict[str, str] = {
    "green": "note",
    "yellow": "warning",
    "red": "important",
}


def _esc(text: Any) -> str:
    """Minimal markdown escape — strip newlines that would break a list item."""
    if text is None:
        return ""
    s = str(text)
    # Collapse newlines so a single value can sit inside a markdown row.
    return s.replace("\r", " ").replace("\n", " ").strip()


def render_quarto(
    protocol: StudyProtocol,
    *,
    executive_synthesis: "ExecutiveSynthesis | None" = None,
    output_dir: Path,
    run_quarto: bool = False,
    project_dir: Path | None = None,
) -> Path:
    """Render ``protocol`` to a Quarto ``.qmd`` (and optionally HTML).

    Parameters
    ----------
    protocol:
        The study protocol to render.
    executive_synthesis:
        Optional synthesis object; when supplied it is rendered at the
        top of the report.
    output_dir:
        Directory the ``.qmd`` (and rendered HTML when applicable) will
        be written to. Created if missing.
    run_quarto:
        When ``True`` AND the ``quarto`` CLI is on ``PATH``, invoke
        ``quarto render <path> --to html`` after writing the source.
        When the CLI is missing this is a no-op.
    project_dir:
        Optional directory containing ``run.lock.json`` — used for the
        reproducibility-manifest excerpt. Defaults to ``output_dir``'s
        parent if not given.

    Returns
    -------
    Path:
        The rendered HTML path when Quarto was successfully invoked,
        otherwise the ``.qmd`` source path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if project_dir is None:
        project_dir = output_dir.parent

    body = _build_body(
        protocol,
        executive_synthesis=executive_synthesis,
        project_dir=project_dir,
    )

    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    qmd = (
        template
        .replace("{{TITLE}}", _esc(protocol.name))
        .replace("{{DATE}}", datetime.now(UTC).date().isoformat())
        .replace("{{PROJECT_NAME}}", _esc(protocol.name))
        .replace("{{TIER}}", _esc(protocol.tier))
        .replace("{{BODY}}", body)
    )

    qmd_path = output_dir / "report.qmd"
    qmd_path.write_text(qmd, encoding="utf-8")

    if run_quarto and shutil.which("quarto") is not None:
        try:
            subprocess.run(
                ["quarto", "render", str(qmd_path), "--to", "html"],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, OSError):
            # Quarto failed at runtime — fall back to the source path
            # rather than raising. The caller can inspect the .qmd.
            return qmd_path
        html_path = qmd_path.with_suffix(".html")
        if html_path.exists():
            return html_path

    return qmd_path


# ─────────────────────────── body construction ──────────────────────────


def _build_body(
    protocol: StudyProtocol,
    *,
    executive_synthesis: "ExecutiveSynthesis | None",
    project_dir: Path,
) -> str:
    parts: list[str] = []

    # Executive synthesis (TL;DR + ranked findings).
    if executive_synthesis is not None:
        parts.append(_synthesis_section(executive_synthesis))

    # Research question + domain brief.
    if protocol.research_question:
        parts.append("## Research question\n")
        parts.append(_esc(protocol.research_question) + "\n")
    if protocol.discovery and protocol.discovery.domain_brief:
        parts.append("## Domain brief\n")
        parts.append(_esc(protocol.discovery.domain_brief) + "\n")

    # Discovery: variables + flags + candidate DAGs.
    parts.append(_discovery_section(protocol))

    # Per-walk Roadmap sections (Q5 - Q8).
    if protocol.roadmap_walks:
        parts.append(_walks_section(protocol))
    elif (
        executive_synthesis is None
        and not protocol.hypothesis_queue
    ):
        # Empty-protocol notice expected by the unit tests.
        parts.append(
            "::: {.callout-note}\n"
            "No experiments run yet — the protocol has no Roadmap walks "
            "and no hypothesis queue.\n"
            ":::\n"
        )

    # Method cards (one per distinct estimator used).
    method_cards = _method_cards_section(protocol)
    if method_cards:
        parts.append(method_cards)

    # Reproducibility manifest excerpt.
    manifest = _manifest_section(project_dir)
    if manifest:
        parts.append(manifest)

    # Decision ledger.
    if protocol.decision_ledger or protocol.overrides:
        parts.append(_ledger_section(protocol))

    # Honest caveats appendix.
    parts.append(_caveats_appendix())

    return "\n".join(p for p in parts if p)


def _confidence_chip(confidence: str) -> str:
    """Return a Quarto-compatible coloured chip for a confidence label."""
    c = confidence.lower()
    colour = {"high": "#7ed2e6", "medium": "#a3b6da", "low": "#e08877"}.get(
        c, "#9aa3b5"
    )
    return (
        f"[**{c.upper()}**]{{style='color:{colour}; "
        "font-family:monospace; font-size:0.85em;'}"
    )


def _synthesis_section(synth: "ExecutiveSynthesis") -> str:
    lines: list[str] = []
    lines.append("## Executive synthesis\n")
    lines.append(f"_Inferred domain: **{_esc(synth.inferred_domain)}**_\n")
    lines.append("> " + _esc(synth.tldr) + "\n")
    for f in synth.findings:
        chip = _confidence_chip(f.confidence)
        lines.append(
            f"### Finding {f.rank} · {chip}\n"
        )
        lines.append(f"**{_esc(f.headline)}**\n")
        lines.append(f"- Effect: {_esc(f.quantified_effect)}")
        lines.append(f"- Implication: {_esc(f.domain_implication)}")
        lines.append(f"- Suggested next step: {_esc(f.suggested_next_step)}")
        lines.append(f"- Source hypothesis: `{_esc(f.hypothesis_id)}`")
        lines.append(f"- Estimator: `{_esc(f.estimator_used)}`")
        if f.caveats:
            lines.append("- Caveats: " + " · ".join(_esc(c) for c in f.caveats))
        lines.append("")
    if synth.overall_caveats:
        lines.append("**Overall caveats**\n")
        for c in synth.overall_caveats:
            lines.append(f"- {_esc(c)}")
        lines.append("")
    return "\n".join(lines)


def _discovery_section(protocol: StudyProtocol) -> str:
    if protocol.discovery is None and not protocol.flags:
        return ""
    lines: list[str] = []
    lines.append("## Discovery\n")

    # Flag chips.
    if protocol.flags:
        chips = " ".join(
            f"`{_esc(f.value)}`" for f in sorted(protocol.flags, key=lambda x: x.value)
        )
        lines.append(f"**Flags:** {chips}\n")

    discovery = protocol.discovery
    if discovery is None:
        return "\n".join(lines)

    # Variables table.
    if discovery.columns:
        lines.append("### Variables\n")
        lines.append("| Column | Role | Temporal | Dtype | Description |")
        lines.append("|---|---|---|---|---|")
        for v in discovery.columns:
            desc = (v.semantic_description or "")[:120]
            lines.append(
                f"| `{_esc(v.name)}` | {_esc(v.role.value)} | "
                f"{_esc(v.measured_at or '—')} | {_esc(v.dtype)} | {_esc(desc)} |"
            )
        lines.append("")

    # Candidate DAGs as Mermaid graphs.
    if discovery.candidate_graphs:
        lines.append("### Candidate DAGs\n")
        for g in discovery.candidate_graphs[:3]:
            lines.append(f"**Rank {g.rank}** — {len(g.edges)} edges")
            lines.append("")
            lines.append("```{mermaid}")
            lines.append("graph LR")
            for e in g.edges:
                src = _mermaid_id(e.source)
                tgt = _mermaid_id(e.target)
                lines.append(f"  {src} --> {tgt}")
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def _mermaid_id(name: str) -> str:
    """Sanitise a column name into a Mermaid-safe node id."""
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(name))
    if not safe:
        safe = "node"
    if safe[0].isdigit():
        safe = "n_" + safe
    return safe


def _walks_section(protocol: StudyProtocol) -> str:
    lines: list[str] = ["## Roadmap walks\n"]
    for key, walk in protocol.roadmap_walks.items():
        lines.append(f"### {_esc(key)}\n")

        # Q5 — identification narration.
        ident = walk.q5_identification or {}
        if ident:
            lines.append("#### Q5 · Identification\n")
            lines.append(
                f"- Strategy: {_esc(ident.get('strategy', '—'))}"
            )
            lines.append(
                f"- Identifiable: {_esc(ident.get('identifiable', '—'))}"
            )
            adj = ident.get("adjustment_set") or []
            if adj:
                lines.append(
                    "- Adjustment set: "
                    + ", ".join(f"`{_esc(c)}`" for c in adj)
                )
            narration = ident.get("narration")
            if isinstance(narration, dict):
                if narration.get("strategy_explanation"):
                    lines.append(
                        f"- Why this works: {_esc(narration['strategy_explanation'])}"
                    )
                if narration.get("blocked_paths"):
                    lines.append(
                        "- Blocked paths: "
                        + " · ".join(_esc(p) for p in narration["blocked_paths"])
                    )
                if narration.get("unblocked_paths"):
                    lines.append(
                        "- Open backdoor paths: "
                        + " · ".join(_esc(p) for p in narration["unblocked_paths"])
                    )
            lines.append("")

        # Q6 — statistical estimand.
        if walk.q6_statistical_estimand is not None:
            lines.append("#### Q6 · Statistical estimand\n")
            try:
                estimand_repr = walk.q6_statistical_estimand.model_dump_json(indent=2)
            except Exception:
                estimand_repr = str(walk.q6_statistical_estimand)
            lines.append("```json")
            lines.append(estimand_repr)
            lines.append("```\n")

        # Q7 — estimate + diagnostics + anomaly audit.
        if walk.q7_estimates:
            est = walk.q7_estimates[-1]
            lines.append("#### Q7 · Estimate\n")
            lines.append(f"- Estimator: `{_esc(est.estimator_id)}`")
            lines.append(f"- Point estimate: **{est.point_estimate:+.4f}**")
            if est.ci_low is not None and est.ci_high is not None:
                lines.append(
                    f"- 95% CI: [{est.ci_low:+.4f}, {est.ci_high:+.4f}]"
                )
            if est.p_value is not None:
                lines.append(f"- p-value: {est.p_value:.4g}")
            lines.append(f"- n used: {est.n_used}")

            diag = est.diagnostics if isinstance(est.diagnostics, dict) else {}
            audit = diag.get("anomaly_audit") if isinstance(diag, dict) else None
            if isinstance(audit, dict):
                rec = audit.get("recommendation", "accept")
                flags = audit.get("flags") or []
                flag_str = ", ".join(flags) if flags else "(no flags)"
                lines.append(f"- Anomaly audit: **{_esc(rec)}** — {_esc(flag_str)}")

            if est.refutations:
                n_pass = est.refutations.get("n_passed", 0)
                lines.append(f"- Refutations: {n_pass} / 3 passed")
            lines.append("")

        # Q8 — sensitivity verdict + interpretation in colour-coded callout.
        verdict = walk.sensitivity_verdict
        callout = SENSITIVITY_CALLOUT.get(verdict) if verdict else None
        if callout is not None or walk.q8_interpretation:
            lines.append("#### Q8 · Sensitivity & interpretation\n")
            if callout is not None:
                lines.append(f"::: {{.callout-{callout}}}")
                lines.append(f"**Sensitivity verdict: {_esc(verdict)}**\n")
                if walk.q8_interpretation:
                    lines.append(_esc(walk.q8_interpretation))
                lines.append(":::\n")
            else:
                if verdict:
                    lines.append(f"- Sensitivity verdict: `{_esc(verdict)}`")
                if walk.q8_interpretation:
                    lines.append("")
                    lines.append(_esc(walk.q8_interpretation))
                lines.append("")

        if walk.failure_reason:
            lines.append("::: {.callout-important}")
            lines.append(f"**Failure:** {_esc(walk.failure_reason)}")
            lines.append(":::\n")

    return "\n".join(lines)


def _method_cards_section(protocol: StudyProtocol) -> str:
    """One method card per distinct estimator used across walks."""
    cards: dict[str, dict[str, Any]] = {}
    for walk in protocol.roadmap_walks.values():
        if not walk.q7_estimates:
            continue
        for est in walk.q7_estimates:
            if est.estimator_id in cards:
                continue
            cards[est.estimator_id] = {
                "backend_version": est.backend_version,
                "diagnostics_keys": sorted(
                    (est.diagnostics or {}).keys()
                ) if isinstance(est.diagnostics, dict) else [],
            }
    if not cards:
        return ""
    lines = ["## Method cards\n"]
    for eid, info in cards.items():
        lines.append(f"### `{_esc(eid)}`\n")
        lines.append(_method_citation(eid))
        lines.append("")
        lines.append("**Assumptions**\n")
        for a in _method_assumptions(eid):
            lines.append(f"- {a}")
        lines.append("")
        if info["backend_version"]:
            lines.append(f"- Backend version: `{_esc(info['backend_version'])}`")
        if info["diagnostics_keys"]:
            lines.append(
                "- Diagnostics emitted: "
                + ", ".join(f"`{_esc(k)}`" for k in info["diagnostics_keys"])
            )
        lines.append("")
    return "\n".join(lines)


def _method_citation(estimator_id: str) -> str:
    eid = estimator_id.lower()
    if "tmle" in eid:
        return (
            "Targeted Maximum Likelihood Estimation (van der Laan & Rose, "
            "2011). Doubly robust, asymptotically efficient."
        )
    if "dml" in eid:
        return (
            "Double / Debiased Machine Learning (Chernozhukov et al., 2018). "
            "Cross-fitted, orthogonalised."
        )
    if "did" in eid:
        return (
            "Difference-in-Differences — see Callaway-Sant'Anna (2021), "
            "Roth-Sant'Anna (2025) Practitioner's Guide."
        )
    if "rd" in eid:
        return "Regression Discontinuity (Calonico-Cattaneo-Titiunik, rdrobust)."
    if "ipw" in eid or "weighting" in eid:
        return "Inverse-Probability Weighting (Hernán & Robins, 2020)."
    return "See estimator registry for citation."


def _method_assumptions(estimator_id: str) -> list[str]:
    eid = estimator_id.lower()
    base = ["No unmeasured confounders (conditional exchangeability)", "Positivity / overlap"]
    if "did" in eid:
        return ["Parallel trends", "No anticipation"] + base
    if "rd" in eid:
        return ["Continuity of potential outcomes at cutoff", "No manipulation"] + base
    return base


def _manifest_section(project_dir: Path) -> str:
    """Excerpt key reproducibility fields from ``run.lock.json``."""
    if not project_dir:
        return ""
    manifest_path = project_dir / "run.lock.json"
    if not manifest_path.exists():
        return ""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    lines = ["## Reproducibility manifest (excerpt)\n"]
    for k in ("data_sha256", "dag_hash", "estimand_hash", "code_sha", "seed"):
        if k in data:
            lines.append(f"- **{k}**: `{_esc(data[k])}`")
    lines.append("")
    return "\n".join(lines)


def _ledger_section(protocol: StudyProtocol) -> str:
    lines = ["## Decision ledger\n"]
    if protocol.decision_ledger:
        lines.append("| Timestamp | Phase | Decision | Chose | Source |")
        lines.append("|---|---|---|---|---|")
        for d in protocol.decision_ledger:
            lines.append(
                f"| {d.timestamp.isoformat(timespec='seconds')} | "
                f"{_esc(d.phase)} | {_esc(d.decision)} | "
                f"{_esc(d.chose)} | {_esc(d.source)} |"
            )
        lines.append("")
    if protocol.overrides:
        lines.append("### Overrides\n")
        lines.append("| Site | LLM proposed | Analyst chose | Reason |")
        lines.append("|---|---|---|---|")
        for o in protocol.overrides:
            lines.append(
                f"| `{_esc(o.site)}` | {_esc(o.llm_value)} | "
                f"{_esc(o.analyst_value)} | {_esc(o.reason or '—')} |"
            )
        lines.append("")
    return "\n".join(lines)


def _caveats_appendix() -> str:
    return (
        "## Appendix · Honest caveats\n\n"
        "This study uses the Petersen-van der Laan Causal Roadmap and was "
        "generated with an LLM-assisted discovery agent. The following "
        "failure modes are documented for transparency (PDD §31):\n\n"
        "- **Causal Parrot**: an LLM may propose a DAG memorised from the "
        "literature rather than one supported by the data. LLM-proposed "
        "edges were audited via partial-correlation tests; contradictions "
        "appear in the decision ledger.\n"
        "- **Post-treatment bias**: variables realised after the treatment "
        "cannot serve as confounders. The temporal-lattice check blocked "
        "any column tagged `temporal_position=post_treatment` from the "
        "adjustment set.\n"
        "- **Unmeasured confounding**: the E-value and Sensemakr robustness "
        "value quantify how strong an unmeasured confounder would need to be "
        "to nullify the headline effect — see the per-walk sensitivity card.\n"
        "- **Multiple-testing inflation**: when more than one hypothesis was "
        "estimated, the headline conclusions quote BH-adjusted p-values.\n"
    )


__all__ = ["render_quarto", "SENSITIVITY_CALLOUT"]
