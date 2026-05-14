"""LLM-guided auto pipeline.

Single entry point that walks the full Causal Roadmap end to end on a dataset
the analyst hands over with minimal prior knowledge. Each phase is run with
sensible defaults; treatment / outcome are inferred from the discovery
report when not supplied; decisions are logged to the analyst-decision
ledger as ``source=auto`` so the report shows exactly what the auto path
chose.

The function is used by both the CLI ``causalrag run`` and the TUI
``/run`` so the behavior is consistent across surfaces.

It is intentionally written as an iterator that yields per-phase
events — the CLI prints them as Rich tables, the TUI streams them into
the LogView. This keeps the pipeline orchestration logic out of both
front-ends.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalGraph
from causalrag.core.ledger import record_decision
from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.discovery import run_discovery
from causalrag.feasibility import default_thresholds, run_feasibility
from causalrag.hypothesize import (
    proposals_to_hypotheses,
    rank_by_impact,
    run_automated,
)
from causalrag.llm.ollama_client import OllamaClient
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q6_statistical_estimand import derive_statistical_estimand
from causalrag.roadmap.q7_estimate import estimate as run_step7
from causalrag.sensitivity.evalue import evalue as run_evalue
from causalrag.sensitivity.sensemakr_py import sensemakr as run_sensemakr
from causalrag.sensitivity.verdict import aggregate as aggregate_sensitivity


@dataclass
class AutoEvent:
    """One step of the auto pipeline — yielded by :func:`run_auto`.

    Front-ends pattern-match on ``kind`` to render the right widget.
    """

    kind: str  # "phase_start" | "phase_end" | "log" | "card" | "error"
    phase: str
    message: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


def _infer_treatment_outcome(
    protocol: StudyProtocol,
    treatment_hint: str | None,
    outcome_hint: str | None,
) -> tuple[str | None, str | None]:
    """Resolve (treatment, outcome). Explicit hints win; otherwise read off
    the investigator-assigned roles on the discovery report."""
    t = treatment_hint
    y = outcome_hint
    if (t is None or y is None) and protocol.discovery is not None:
        for v in protocol.discovery.columns:
            if t is None and v.role is VariableRole.TREATMENT:
                t = v.name
            if y is None and v.role is VariableRole.OUTCOME:
                y = v.name
    return t, y


def _build_default_graph(
    protocol: StudyProtocol,
    treatment: str,
    outcome: str,
) -> CausalGraph:
    """Pick the rank-1 candidate DAG if discovery surfaced one; otherwise
    construct a 'naive backdoor' DAG using every other variable as a
    confounder."""
    if protocol.candidate_graphs:
        idx = min(protocol.selected_graph_index, len(protocol.candidate_graphs) - 1)
        return protocol.candidate_graphs[idx]
    if protocol.discovery and protocol.discovery.candidate_graphs:
        return protocol.discovery.candidate_graphs[0]
    confounders: list[str] = []
    if protocol.discovery is not None:
        confounders = [
            v.name
            for v in protocol.discovery.columns
            if v.role is VariableRole.CONFOUNDER
        ]
    edges = [(c, treatment) for c in confounders] + [(c, outcome) for c in confounders]
    edges.append((treatment, outcome))
    roles = {c: VariableRole.CONFOUNDER for c in confounders}
    roles[treatment] = VariableRole.TREATMENT
    roles[outcome] = VariableRole.OUTCOME
    return CausalGraph.from_edge_list(edges, roles=roles)


def run_auto(
    *,
    protocol: StudyProtocol,
    project_dir: Path,
    dataset_path: Path,
    treatment_hint: str | None = None,
    outcome_hint: str | None = None,
    research_question: str | None = None,
    discovery_client: OllamaClient | None = None,
    expert_client: OllamaClient | None = None,
    counterfactual_ratio: float = 0.30,
    max_hypotheses: int = 3,
    report_format: str = "html",
) -> Iterator[AutoEvent]:
    """Walk the full pipeline. Yields :class:`AutoEvent` per step.

    ``protocol`` is mutated in place and persisted back to YAML at the end
    so the caller can read off the final state via
    ``StudyProtocol.read_yaml(project_dir / 'study.causalrag.yaml')``.
    """
    protocol_path = project_dir / "study.causalrag.yaml"

    # --- Phase 1 -- discover ------------------------------------------------
    yield AutoEvent(kind="phase_start", phase="discover", message="Phase 1 · discover")
    discovery = run_discovery(
        source=dataset_path,
        client=discovery_client,
        expert_client=expert_client,
        research_question=research_question,
        treatment=treatment_hint,
        outcome=outcome_hint,
    )
    protocol.discovery = discovery.to_report()
    protocol.flags |= discovery.flags
    if discovery.candidate_graphs and not protocol.candidate_graphs:
        protocol.candidate_graphs = discovery.candidate_graphs
    if not protocol.dataset:
        from causalrag.core.protocol import DatasetSpec

        protocol.dataset = DatasetSpec(
            source=f"csv://{dataset_path}",
            n_rows=discovery.profile.n_rows,
            n_cols=discovery.profile.n_cols,
            columns=discovery.columns,
        )
    if research_question and not protocol.research_question:
        protocol.research_question = research_question
    record_decision(
        protocol,
        phase="discover",
        decision="rank-1 DAG accepted",
        chose=f"K={len(discovery.candidate_graphs)} candidates, rank=1 selected",
        source="auto",
    )
    n_contradicted = sum(1 for a in discovery.dag_audit if a.verdict == "contradicted")
    yield AutoEvent(
        kind="phase_end",
        phase="discover",
        message=f"flags={','.join(sorted(f.value for f in discovery.flags))} · K={len(discovery.candidate_graphs)} DAGs · {n_contradicted} edges contradicted",
        payload={"flags": sorted(f.value for f in discovery.flags)},
    )

    # --- Resolve (T, Y) ----------------------------------------------------
    # In master mode (LLM client + no hint) we deliberately DEFER picking
    # (T, Y) — the master generator will propose multiple (T, Y) pairs of
    # its own. We only require an (T, Y) here when the analyst passed
    # explicit hints OR there's no LLM available to delegate to.
    treatment, outcome = _infer_treatment_outcome(protocol, treatment_hint, outcome_hint)
    has_llm = discovery_client is not None or expert_client is not None
    explicit_hints = bool(treatment_hint and outcome_hint)
    if (treatment is None or outcome is None) and not has_llm and not explicit_hints:
        yield AutoEvent(
            kind="error",
            phase="discover",
            message="Treatment / outcome could not be inferred and no LLM client provided. Pass --treatment and --outcome, or run with LLM access for master mode.",
        )
        return
    if treatment and outcome:
        yield AutoEvent(
            kind="log",
            phase="auto",
            message=f"Anchor hypothesis: treatment={treatment}, outcome={outcome}",
            payload={"treatment": treatment, "outcome": outcome},
        )
    else:
        yield AutoEvent(
            kind="log",
            phase="auto",
            message="No explicit treatment/outcome — master mode will propose them.",
        )

    # --- Load data ---------------------------------------------------------
    df = pd.read_csv(dataset_path)

    # --- Phase 2 -- feasibility -------------------------------------------
    yield AutoEvent(kind="phase_start", phase="feasibility", message="Phase 2 · feasibility")
    flags = set(protocol.flags)
    thresholds = default_thresholds(flags)
    feasibility_report = run_feasibility(df, protocol, thresholds=thresholds)
    protocol.feasibility = feasibility_report.to_protocol()
    yield AutoEvent(
        kind="phase_end",
        phase="feasibility",
        message=f"{len(feasibility_report.admissible)} admissible · {len(feasibility_report.borderline)} borderline · {len(feasibility_report.underpowered)} underpowered",
    )

    # --- Phase 3 -- hypothesize (deterministic, single-pass) -------------
    # Master autonomous mode lives in causalrag.master_loop, invoked only
    # via the TUI's `auto run` command. The generic `/run` path stays
    # deterministic + predictable.
    yield AutoEvent(kind="phase_start", phase="hypothesize", message="Phase 3 · hypothesize")
    proposals = run_automated(
        protocol=protocol,
        brief=None,
        client=None,
        counterfactual_ratio=counterfactual_ratio,
    )
    hypotheses = rank_by_impact(proposals_to_hypotheses(proposals))[:max_hypotheses]
    protocol.hypothesis_queue = tuple(hypotheses)
    protocol.counterfactual_ratio = counterfactual_ratio
    record_decision(
        protocol,
        phase="hypothesize",
        decision="deterministic queue",
        chose=f"{len(hypotheses)} hypotheses (top {max_hypotheses})",
        source="auto",
    )
    yield AutoEvent(
        kind="phase_end",
        phase="hypothesize",
        message=f"{len(hypotheses)} hypothesis/-es queued",
        payload={"ids": [h.id for h in hypotheses]},
    )

    # --- Phase 4 + 5 -- estimate + sensitivity per hypothesis -------------
    graph = _build_default_graph(protocol, treatment, outcome)

    for h in hypotheses:
        if h.estimand is None:
            continue
        est = h.estimand
        yield AutoEvent(
            kind="phase_start",
            phase="estimate",
            message=f"Phase 4 · estimate {h.id}: {est.treatment} → {est.outcome}",
        )
        ident = identify_effect(est, graph, df=df)
        if not ident.identifiable:
            yield AutoEvent(
                kind="log",
                phase="estimate",
                message=f"{h.id}: non-identifiable ({ident.notes}); skipped",
            )
            continue
        try:
            result = run_step7(
                df=df,
                estimand=est,
                identification=ident,
                protocol=protocol,
                flags=set(protocol.flags),
            )
        except Exception as e:
            yield AutoEvent(
                kind="error",
                phase="estimate",
                message=f"{h.id} estimate failed: {type(e).__name__}: {e}",
            )
            continue

        walk = protocol.roadmap_walks.get(h.id) or RoadmapWalk(hypothesis_id=h.id)
        walk.q3_estimand = est
        walk.q5_identification = {
            "identifiable": ident.identifiable,
            "strategy": ident.strategy,
            "adjustment_set": list(ident.adjustment_set),
            "estimand_expression": ident.estimand_expression,
        }
        walk.q6_statistical_estimand = derive_statistical_estimand(est, ident)
        walk.q7_estimates = tuple(list(walk.q7_estimates) + [result])
        p_str = f"{result.p_value:.4g}" if result.p_value is not None else "NA"
        ci_str = (
            f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]"
            if result.ci_low is not None and result.ci_high is not None
            else "—"
        )
        record_decision(
            protocol,
            phase="estimate",
            decision=f"{h.id} · selected_estimator",
            chose=result.estimator_id,
            source="auto",
            note=f"strategy={ident.strategy} · p={p_str}",
        )
        yield AutoEvent(
            kind="card",
            phase="estimate",
            message=f"{h.id}: {result.point_estimate:+.4f} · 95% CI {ci_str} · p={p_str}",
            payload={"hypothesis": h.id, "result": result.model_dump(mode="json")},
        )

        # Sensitivity for this hypothesis
        yield AutoEvent(
            kind="phase_start",
            phase="sensitivity",
            message=f"Phase 5 · sensitivity {h.id}",
        )
        auto_scale = "odds_ratio" if DataFlag.BINARY_OUTCOME in protocol.flags else (
            "hazard_ratio" if DataFlag.RIGHT_CENSORED_OUTCOME in protocol.flags else "standardized"
        )
        point = result.point_estimate
        ci_low_e = result.ci_low
        ci_high_e = result.ci_high
        if auto_scale == "standardized" and est.outcome in df.columns:
            y_sd = float(df[est.outcome].std(ddof=1)) or 1.0
            point = point / y_sd
            if ci_low_e is not None:
                ci_low_e = ci_low_e / y_sd
            if ci_high_e is not None:
                ci_high_e = ci_high_e / y_sd
        ev = run_evalue(point, scale=auto_scale, ci_low=ci_low_e, ci_high=ci_high_e)  # type: ignore[arg-type]

        confounders_for_sm = tuple(
            v.name
            for v in (protocol.discovery.columns if protocol.discovery else ())
            if v.role is VariableRole.CONFOUNDER and v.name in df.columns
        )
        sm = run_sensemakr(
            df,
            treatment=est.treatment,
            outcome=est.outcome,
            covariates=confounders_for_sm,
        )
        verdict = aggregate_sensitivity(evalue=ev, sensemakr=sm, rule="min")
        walk.q8_interpretation = (
            f"Sensitivity {verdict.color}. {verdict.rationale}. "
            f"E-value={ev.e_value:.2f} ({ev.scale}). RV={sm.robustness_value:.3f}."
        )
        record_decision(
            protocol,
            phase="sensitivity",
            decision=f"{h.id} · verdict_rule=min",
            chose=verdict.color,
            source="auto",
        )
        new_walks = dict(protocol.roadmap_walks)
        new_walks[h.id] = walk
        protocol.roadmap_walks = new_walks
        yield AutoEvent(
            kind="phase_end",
            phase="sensitivity",
            message=f"{h.id}: {verdict.color} ({verdict.rationale})",
            payload={"hypothesis": h.id, "verdict": verdict.color},
        )

    # --- Phase 6 -- report -------------------------------------------------
    yield AutoEvent(kind="phase_start", phase="report", message="Phase 6 · report")
    from datetime import datetime

    from causalrag.reporting.render_html import render_report

    reports_dir = project_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"{protocol.name}_{ts}.{report_format}"

    # Load executive synthesis if the master loop produced one.
    synthesis_obj = None
    synth_path = project_dir / "executive_synthesis.json"
    if synth_path.exists():
        try:
            from causalrag.reporting.synthesis import ExecutiveSynthesis

            synthesis_obj = ExecutiveSynthesis.model_validate_json(
                synth_path.read_text()
            )
        except Exception:
            synthesis_obj = None

    content = render_report(
        protocol, fmt=report_format, executive_synthesis=synthesis_obj
    )
    path.write_text(content, encoding="utf-8")
    protocol.write_yaml(protocol_path)
    yield AutoEvent(
        kind="phase_end",
        phase="report",
        message=f"{path} · {len(content):,} bytes",
        payload={"report_path": str(path)},
    )


__all__ = ["AutoEvent", "run_auto"]
