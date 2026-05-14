"""Slash-command runners for the TUI.

Each runner is an ``async`` function that takes the App and any string args,
streams log lines + cards into the LogView, mutates the StudyProtocol on disk,
and updates the status bar. Runners are intentionally NOT pure Typer
re-invocations — the TUI controls the streaming cadence and we want to feed
the user updates as work progresses, not all-at-once at the end.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.text import Text

from causalrag.cli.doctor import recommend, report_dict, run_doctor
from causalrag.cli.main import PROTOCOL_FILENAME, _scaffold_project
from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.discovery import run_discovery
from causalrag.llm.ollama_client import OllamaClient
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q7_estimate import estimate as run_estimate
from causalrag.sensitivity.evalue import evalue as run_evalue
from causalrag.sensitivity.sensemakr_py import sensemakr as run_sensemakr
from causalrag.sensitivity.verdict import aggregate as aggregate_sensitivity
from causalrag.tui.errors import hint_for
from causalrag.tui.widgets.cards import column_table, kv_table
from causalrag.tui.widgets.composer import COMMANDS

if TYPE_CHECKING:
    from causalrag.tui.app import CausalRoadmapTUI


# --- Helpers ---------------------------------------------------------------


def _split_args(rest: str) -> list[str]:
    try:
        return shlex.split(rest) if rest.strip() else []
    except ValueError:
        return rest.split()


async def _sleep(ms: int) -> None:
    await asyncio.sleep(ms / 1000)


def _spinner_glyph(step: int) -> str:
    glyphs = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    return glyphs[step % len(glyphs)]


# --- Runners ---------------------------------------------------------------


async def run_help(app: "CausalRoadmapTUI", _args: list[str]) -> None:
    rows = []
    for c in COMMANDS:
        rows.append((c.name, c.description))
    table = kv_table(rows)
    app.log_view.card(title="Commands", body=table)
    app.log_view.line(
        "Use Tab to autocomplete · ↑/↓ to walk history · /quit to exit",
        kind="dim",
        gutter="·",
    )


async def run_clear(app: "CausalRoadmapTUI", _args: list[str]) -> None:
    app.log_view.clear_log()


async def run_init(app: "CausalRoadmapTUI", args: list[str]) -> None:
    if not args:
        app.log_view.line("Usage: /init <project-name> [--tier=<tier>]", kind="err", gutter="✗")
        return
    name = args[0]
    tier = "academic"
    for a in args[1:]:
        if a.startswith("--tier="):
            tier = a.split("=", 1)[1]
    project_dir = (app.project_dir / name).resolve()
    if project_dir.exists() and any(project_dir.iterdir()):
        app.log_view.line(
            f"Directory {project_dir} is not empty.", kind="err", gutter="✗"
        )
        return
    await _sleep(100)
    path = _scaffold_project(project_dir, name=name, tier=tier)
    app.set_project_dir(project_dir)
    app.log_view.line(
        f"Scaffold created at {project_dir}",
        kind="ok",
        gutter="✓",
    )
    app.log_view.line(
        f"Wrote {path.name} · .causalrag/cassettes/ · .causalrag/history.jsonl",
        kind="dim",
        gutter="·",
    )
    app.log_view.line(f"tier · {tier}", kind="ok", gutter="✓")
    app.log_view.line("Next: /doctor", kind="acc", gutter="→")
    app.set_phase(0)


async def run_doctor(app: "CausalRoadmapTUI", _args: list[str]) -> None:
    app.log_view.line("Probing hardware…", kind="dim", gutter=_spinner_glyph(0))
    await _sleep(100)
    profile = await asyncio.to_thread(run_doctor.__wrapped__) if hasattr(run_doctor, "__wrapped__") else None
    # Direct call (run_doctor imported from cli.doctor is the function we want)
    from causalrag.cli.doctor import run_doctor as _probe

    profile = await asyncio.to_thread(_probe)
    slots, missing = recommend(profile)
    payload = report_dict(profile)

    py_ok = profile.python_version.split(".")[:2] >= ["3", "11"]
    app.log_view.line(
        f"Python {profile.python_version} {'OK' if py_ok else 'TOO OLD'} · "
        f"{profile.platform}",
        kind="ok" if py_ok else "err",
        gutter="✓" if py_ok else "✗",
    )
    app.log_view.line(
        f"{profile.cpu_logical} logical cores · {profile.total_ram_gb:.1f} GB RAM "
        f"({profile.available_ram_gb:.1f} free)",
        kind="dim",
        gutter="·",
    )
    if profile.gpus:
        for g in profile.gpus:
            app.log_view.line(
                f"{g.name} · {g.vram_total_gb:.1f} GB · {g.backend}",
                kind="ok",
                gutter="✓",
            )
    if profile.ollama.reachable:
        app.log_view.line(
            f"Ollama {profile.ollama.version or '?'} reachable · "
            f"{len(profile.ollama.models)} models installed",
            kind="ok",
            gutter="✓",
        )
    else:
        app.log_view.line(
            f"Ollama not reachable at {profile.ollama.base_url}",
            kind="warn",
            gutter="⚠",
        )

    rows = [
        ("discovery (general)", Text(slots.discovery, style="#9ec2ff")),
        ("hypothesize (reasoning)", Text(slots.hypothesize, style="#9ec2ff")),
        ("utility", Text(slots.utility, style="#9ec2ff")),
        ("effective VRAM", Text(f"{profile.effective_vram_gb:.1f} GB")),
        ("tier", Text(profile.tier_label, style="#7ed2e6")),
    ]
    app.log_view.card(
        title="Resolved tier",
        step=f"T{profile.tier}",
        meta="hardware-aware default",
        body=kv_table(rows),
    )
    if missing:
        app.log_view.line(
            "Missing models: " + ", ".join(missing) + " · using fallbacks",
            kind="warn",
            gutter="⚠",
        )
    for w in profile.warnings:
        app.log_view.line(w, kind="warn", gutter="⚠")
    app.log_view.line("Next: /discover <dataset>", kind="acc", gutter="→")

    # Persist to .causalrag/hardware.json when a project is loaded
    cr = app.project_dir / ".causalrag"
    if cr.exists():
        (cr / "hardware.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Update chrome
    app.title_bar.tier = app.protocol.tier if app.protocol else "academic"
    app.title_bar.model = slots.discovery
    app.cached_slots = slots
    app.cached_profile = profile


async def run_discover(app: "CausalRoadmapTUI", args: list[str]) -> None:
    if not args:
        app.log_view.line(
            "Usage: /discover <path/to/data.csv> [--treatment T] [--outcome Y] [--question \"...\"] [--no-llm]",
            kind="err",
            gutter="✗",
        )
        return
    source = args[0]
    treatment: str | None = None
    outcome: str | None = None
    question: str | None = None
    no_llm = False
    i = 1
    while i < len(args):
        a = args[i]
        if a == "--treatment" and i + 1 < len(args):
            treatment = args[i + 1]
            i += 2
        elif a == "--outcome" and i + 1 < len(args):
            outcome = args[i + 1]
            i += 2
        elif a == "--question" and i + 1 < len(args):
            question = args[i + 1]
            i += 2
        elif a == "--no-llm":
            no_llm = True
            i += 1
        else:
            i += 1

    project_path = app.project_dir / PROTOCOL_FILENAME
    if not project_path.exists():
        app.log_view.line(
            f"No StudyProtocol at {project_path} · run /init first",
            kind="err",
            gutter="✗",
        )
        return
    protocol = StudyProtocol.read_yaml(project_path)

    data_path = Path(source)
    if not data_path.is_absolute():
        data_path = (app.project_dir / data_path).resolve()
    if not data_path.exists():
        app.log_view.line(f"Dataset not found at {data_path}", kind="err", gutter="✗")
        return

    app.set_phase(1)
    app.log_view.line("Stage 1a · connector · arrow read…", kind="dim", gutter=_spinner_glyph(0))
    await _sleep(80)

    client: OllamaClient | None = None
    expert_client: OllamaClient | None = None
    if not no_llm:
        slots = getattr(app, "cached_slots", None)
        if slots is None:
            from causalrag.cli.doctor import run_doctor as _probe

            profile = await asyncio.to_thread(_probe)
            slots, _ = recommend(profile)
        discovery_model = slots.discovery
        expert_model = slots.hypothesize
        cassette_dir = app.project_dir / ".causalrag" / "cassettes"
        cassette_dir.mkdir(parents=True, exist_ok=True)
        client = OllamaClient(
            model=discovery_model,
            cassette_dir=cassette_dir,
            allow_live=True,
        )
        if expert_model != discovery_model:
            expert_client = OllamaClient(
                model=expert_model,
                cassette_dir=cassette_dir,
                allow_live=True,
            )
            app.log_view.line(
                f"Stage 1c · {discovery_model} (general)    Stage 1e · {expert_model} (reasoning)",
                kind="dim",
                gutter="·",
            )
        else:
            app.log_view.line(f"LLM · {discovery_model}", kind="dim", gutter="·")

    try:
        result = await asyncio.to_thread(
            run_discovery,
            source=data_path,
            client=client,
            expert_client=expert_client,
            research_question=question,
            treatment=treatment,
            outcome=outcome,
        )
    except Exception as e:
        app.log_view.line(f"discover failed · {type(e).__name__}: {e}", kind="err", gutter="✗")
        hint = hint_for(e)
        if hint:
            app.log_view.line(hint, kind="acc", gutter="→")
        return

    app.log_view.line(
        f"Loaded · {result.profile.n_rows:,} × {result.profile.n_cols}",
        kind="ok",
        gutter="✓",
    )

    # Show per-column quick profile
    rows = []
    for col in result.profile.columns:
        rows.append((
            col.name,
            col.logical_dtype,
            f"{col.missing_rate:.0%}",
            str(col.cardinality),
        ))
    app.log_view.card(
        title="Profile · all columns",
        step="1b",
        body=column_table(
            [("Column", "left"), ("Logical", "left"), ("Missing", "right"), ("Card.", "right")],
            rows,
        ),
    )

    if result.flags:
        flag_text = Text("Flags emitted: ")
        for i, f in enumerate(sorted(result.flags, key=lambda x: x.value)):
            if i:
                flag_text.append("  ", style="")
            flag_text.append(f.value, style="#9ec2ff")
        app.log_view.line(flag_text, kind="ok", gutter="✓")

    # LLM-stage output
    if result.investigator is not None:
        app.log_view.line(
            f"Stage 1c · investigator · domain_tag = {result.investigator.domain_tag}",
            kind="ok",
            gutter="✓",
        )
    if result.expert is not None:
        body = Text(result.expert.domain_summary, style="#cfd6e4")
        app.log_view.card(
            title="Domain Expert Brief",
            step="1e",
            meta=f"tag · {result.investigator.domain_tag if result.investigator else '?'}",
            body=body,
        )
        if result.expert.identification_warnings:
            for w in result.expert.identification_warnings:
                app.log_view.line(w, kind="warn", gutter="⚠")
        if result.expert.unmeasured_confounders:
            for u in result.expert.unmeasured_confounders[:5]:
                proxies = ", ".join(u.observed_proxies) if u.observed_proxies else "—"
                app.log_view.line(
                    f"unmeasured · {u.name} · {u.reason} (proxies: {proxies})",
                    kind="warn",
                    gutter="·",
                )
        if result.candidate_graphs:
            app.log_view.line(
                f"K={len(result.candidate_graphs)} candidate DAGs · rank-1 selected",
                kind="acc",
                gutter="·",
            )
        if result.dag_audit:
            contradicted = [a for a in result.dag_audit if a.verdict == "contradicted"]
            supported = [a for a in result.dag_audit if a.verdict == "supported"]
            app.log_view.line(
                f"Layer-4 audit · {len(supported)} supported · {len(contradicted)} contradicted",
                kind="warn" if contradicted else "ok",
                gutter="⚠" if contradicted else "✓",
            )
            for a in contradicted[:5]:
                app.log_view.line(
                    f"{a.source} → {a.target} · |r|={abs(a.partial_correlation):.3f} · p={a.p_value:.3f}",
                    kind="warn",
                    gutter="·",
                )

    # Persist back to YAML
    protocol.discovery = result.to_report()
    protocol.flags |= result.flags
    if result.candidate_graphs and not protocol.candidate_graphs:
        protocol.candidate_graphs = result.candidate_graphs
    if not protocol.dataset:
        from causalrag.core.protocol import DatasetSpec

        protocol.dataset = DatasetSpec(
            source=f"csv://{data_path}",
            n_rows=result.profile.n_rows,
            n_cols=result.profile.n_cols,
            columns=result.columns,
        )
    if question and not protocol.research_question:
        protocol.research_question = question
    protocol.write_yaml(project_path)
    app.protocol = protocol

    sidecar = app.project_dir / ".causalrag" / "discovery.json"
    sidecar.write_text(
        json.dumps(
            {
                "source": result.source_describe,
                "research_question": result.research_question,
                "flags": sorted(f.value for f in result.flags),
                "investigator": result.investigator.model_dump() if result.investigator else None,
                "variables": [v.model_dump() for v in result.columns],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    app.log_view.line(
        f"discovery.json written · {sidecar}",
        kind="dim",
        gutter="·",
    )
    app.log_view.line("Next: /estimate --treatment T --outcome Y", kind="acc", gutter="→")


async def run_estimate_cmd(app: "CausalRoadmapTUI", args: list[str]) -> None:
    treatment = None
    outcome = None
    estimand_class = "ATE"
    prefer = None
    allow_nonid = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--treatment" and i + 1 < len(args):
            treatment = args[i + 1]
            i += 2
        elif a == "--outcome" and i + 1 < len(args):
            outcome = args[i + 1]
            i += 2
        elif a == "--estimand" and i + 1 < len(args):
            estimand_class = args[i + 1]
            i += 2
        elif a == "--prefer" and i + 1 < len(args):
            prefer = args[i + 1]
            i += 2
        elif a == "--allow-nonidentifiable":
            allow_nonid = True
            i += 1
        else:
            i += 1

    project_path = app.project_dir / PROTOCOL_FILENAME
    if not project_path.exists():
        app.log_view.line("No StudyProtocol · run /init first", kind="err", gutter="✗")
        return
    protocol = StudyProtocol.read_yaml(project_path)

    # Infer treatment/outcome from discovery if not provided
    if (treatment is None or outcome is None) and protocol.discovery is not None:
        for v in protocol.discovery.columns:
            if treatment is None and v.role is VariableRole.TREATMENT:
                treatment = v.name
            if outcome is None and v.role is VariableRole.OUTCOME:
                outcome = v.name
    if treatment is None or outcome is None:
        app.log_view.line(
            "Treatment / outcome could not be inferred · pass --treatment and --outcome",
            kind="err",
            gutter="✗",
        )
        return

    app.set_phase(4)
    klass = EstimandClass(estimand_class.upper())
    est = CausalEstimand.model_validate(
        {
            "class": klass,
            "treatment": treatment,
            "outcome": outcome,
            "formal_expression": "E[Y(1) - Y(0)]",
        }
    )

    graph = _resolve_graph(protocol)
    if graph is None:
        graph = CausalGraph.from_edge_list(
            [(treatment, outcome)],
            roles={treatment: VariableRole.TREATMENT, outcome: VariableRole.OUTCOME},
        )

    # Load data
    from causalrag.cli.main import _load_dataframe, _infer_confounders

    try:
        df = _load_dataframe(protocol, app.project_dir)
    except Exception as e:
        app.log_view.line(f"Could not load dataset · {e}", kind="err", gutter="✗")
        return

    app.log_view.line(f"Q5 · identify_effect …", kind="dim", gutter=_spinner_glyph(0))
    ident = await asyncio.to_thread(identify_effect, est, graph, df)
    app.log_view.line(
        f"Q5 · {ident.strategy} · {'identifiable' if ident.identifiable else 'NOT identifiable'}",
        kind="ok" if ident.identifiable else "warn",
        gutter="✓" if ident.identifiable else "⚠",
    )

    confounders = _infer_confounders(graph, treatment, outcome, protocol, df)
    app.log_view.line(
        f"Adjustment candidates · {len(confounders)} variables",
        kind="dim",
        gutter="·",
    )

    app.log_view.line("auto_preprocess · select_variables · overlap_summary …", kind="dim", gutter=_spinner_glyph(1))

    try:
        result = await asyncio.to_thread(
            run_estimate,
            df=df,
            estimand=est,
            identification=ident,
            protocol=protocol,
            confounders=tuple(confounders),
            flags=set(protocol.flags),
            prefer=prefer,
            allow_nonidentifiable=allow_nonid,
        )
    except Exception as e:
        app.log_view.line(f"estimate failed · {type(e).__name__}: {e}", kind="err", gutter="✗")
        hint = hint_for(e)
        if hint:
            app.log_view.line(hint, kind="acc", gutter="→")
        return

    # Headline card
    body = kv_table(
        [
            ("estimator", Text(result.estimator_id, style="#9ec2ff")),
            ("estimand", Text(result.estimand_class, style="#cfd6e4")),
            ("strategy", Text(ident.strategy, style="#cfd6e4")),
            (
                "adjustment used",
                Text(
                    ", ".join(result.diagnostics.get("adjustment_set_used", []))[:80]
                    or "∅",
                    style="#cfd6e4",
                ),
            ),
            (
                "point estimate",
                Text(f"{result.point_estimate:+.4f}", style="#9ec2ff bold"),
            ),
            (
                "95% CI",
                Text(
                    f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]"
                    if result.ci_low is not None and result.ci_high is not None
                    else "—",
                    style="#cfd6e4",
                ),
            ),
            (
                "p-value",
                Text(
                    f"{result.p_value:.4g}" if result.p_value is not None else "—",
                    style="#cfd6e4",
                ),
            ),
            ("n used", Text(str(result.n_used))),
            ("fit seconds", Text(f"{result.fit_seconds:.2f}" if result.fit_seconds else "—")),
        ]
    )
    app.log_view.card(title=f"ATE on {outcome}", step="Q7", body=body)

    # Refutations table
    refs = result.refutations or {}
    table_rows = []
    for name in ("placebo_treatment", "random_common_cause", "subset_bootstrap"):
        entry = refs.get(name, {})
        if "error" in entry:
            table_rows.append((name, "error", entry["error"][:40], ""))
        elif "refuted" in entry:
            passed = entry.get("passed", False)
            table_rows.append(
                (
                    name,
                    "✓ pass" if passed else "✗ fail",
                    f"{entry['refuted']:+.4f}",
                    f"orig {entry['original']:+.4f}",
                )
            )
    if table_rows:
        app.log_view.card(
            title="Refutations",
            step="Q7+",
            body=column_table(
                [("Check", "left"), ("Verdict", "left"), ("Refuted", "right"), ("Original", "right")],
                table_rows,
            ),
        )

    overlap = result.diagnostics.get("overlap") or {}
    if overlap:
        pos = overlap.get("positivity", {})
        app.log_view.line(
            f"overlap · {pos.get('verdict', '?')} · π ∈ "
            f"[{pos.get('propensity_min', 0):.3f}, {pos.get('propensity_max', 0):.3f}]",
            kind={"green": "ok", "yellow": "warn", "red": "err"}.get(pos.get("verdict", "?"), "dim"),
            gutter="·",
        )

    sel = result.diagnostics.get("variable_selection") or {}
    if sel:
        app.log_view.line(
            f"selection · {sel.get('method', '?')} · "
            f"selected {len(sel.get('selected', []))} · dropped {len(sel.get('dropped', []))}",
            kind="dim",
            gutter="·",
        )

    # Persist into RoadmapWalk
    key = f"{treatment}->{outcome}"
    walk = protocol.roadmap_walks.get(key) or RoadmapWalk(hypothesis_id=key)
    walk.q3_estimand = est
    walk.q5_identification = {
        "identifiable": ident.identifiable,
        "strategy": ident.strategy,
        "adjustment_set": list(ident.adjustment_set),
        "estimand_expression": ident.estimand_expression,
    }
    walk.q7_estimates = tuple(list(walk.q7_estimates) + [result])
    new_walks = dict(protocol.roadmap_walks)
    new_walks[key] = walk
    protocol.roadmap_walks = new_walks
    protocol.write_yaml(project_path)
    app.protocol = protocol
    app.log_view.line(f"Next: /sensitivity --treatment {treatment} --outcome {outcome}", kind="acc", gutter="→")


async def run_sensitivity_cmd(app: "CausalRoadmapTUI", args: list[str]) -> None:
    treatment = None
    outcome = None
    rule = "min"
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--treatment" and i + 1 < len(args):
            treatment = args[i + 1]
            i += 2
        elif a == "--outcome" and i + 1 < len(args):
            outcome = args[i + 1]
            i += 2
        elif a == "--rule" and i + 1 < len(args):
            rule = args[i + 1]
            i += 2
        else:
            i += 1

    project_path = app.project_dir / PROTOCOL_FILENAME
    if not project_path.exists():
        app.log_view.line("No StudyProtocol · run /init first", kind="err", gutter="✗")
        return
    protocol = StudyProtocol.read_yaml(project_path)
    if (treatment is None or outcome is None) and protocol.discovery is not None:
        for v in protocol.discovery.columns:
            if treatment is None and v.role is VariableRole.TREATMENT:
                treatment = v.name
            if outcome is None and v.role is VariableRole.OUTCOME:
                outcome = v.name
    if treatment is None or outcome is None:
        app.log_view.line("Pass --treatment and --outcome", kind="err", gutter="✗")
        return

    key = f"{treatment}->{outcome}"
    walk = protocol.roadmap_walks.get(key)
    if walk is None or not walk.q7_estimates:
        app.log_view.line(
            f"No estimate found for {key} · run /estimate first",
            kind="err",
            gutter="✗",
        )
        return
    last = walk.q7_estimates[-1]
    app.set_phase(5)

    from causalrag.cli.main import _load_dataframe, _infer_confounders

    df = _load_dataframe(protocol, app.project_dir)

    auto_scale = "standardized"
    if DataFlag.BINARY_OUTCOME in protocol.flags:
        auto_scale = "odds_ratio"
    elif DataFlag.RIGHT_CENSORED_OUTCOME in protocol.flags:
        auto_scale = "hazard_ratio"

    point = last.point_estimate
    ci_low_e = last.ci_low
    ci_high_e = last.ci_high
    if auto_scale == "standardized" and outcome in df.columns:
        y_sd = float(df[outcome].std(ddof=1)) or 1.0
        point = point / y_sd
        if ci_low_e is not None:
            ci_low_e = ci_low_e / y_sd
        if ci_high_e is not None:
            ci_high_e = ci_high_e / y_sd

    app.log_view.line("E-value · sensemakr …", kind="dim", gutter=_spinner_glyph(0))
    e = await asyncio.to_thread(
        run_evalue, point, scale=auto_scale, ci_low=ci_low_e, ci_high=ci_high_e
    )
    confounders = _infer_confounders(
        _resolve_graph(protocol)
        or CausalGraph.from_edge_list([(treatment, outcome)]),
        treatment,
        outcome,
        protocol,
        df,
    )
    s = await asyncio.to_thread(
        run_sensemakr, df, treatment=treatment, outcome=outcome, covariates=tuple(confounders)
    )
    verdict = aggregate_sensitivity(evalue=e, sensemakr=s, rule=rule)

    color_map = {"green": "ok", "yellow": "warn", "red": "err"}
    app.log_view.card(
        title=f"sensitivity · {treatment} → {outcome}",
        body=kv_table(
            [
                ("E-value (point)", Text(f"{e.e_value:.2f}", style="#9ec2ff")),
                (
                    "E-value (CI bound)",
                    Text(f"{e.e_value_ci:.2f}" if e.e_value_ci else "—"),
                ),
                ("scale", Text(e.scale)),
                ("sensemakr estimate", Text(f"{s.estimate:+.4f}")),
                ("robustness value", Text(f"{s.robustness_value:.4f}")),
                ("backend", Text(s.backend)),
                (
                    "verdict",
                    Text(verdict.color, style={"green": "#7ed2e6", "yellow": "#a3b6da", "red": "#e08877"}[verdict.color] + " bold"),
                ),
                ("components", Text(verdict.rationale)),
            ]
        ),
    )
    walk.q8_interpretation = (
        f"Sensitivity · {verdict.color}. {verdict.rationale}. "
        f"E-value={e.e_value:.2f} ({e.scale}). RV={s.robustness_value:.3f}."
    )
    new_walks = dict(protocol.roadmap_walks)
    new_walks[key] = walk
    protocol.roadmap_walks = new_walks
    protocol.write_yaml(project_path)
    app.log_view.line(
        verdict.color.upper(),
        kind=color_map.get(verdict.color, "dim"),
        gutter="●",
    )


async def run_feasibility(app: "CausalRoadmapTUI", args: list[str]) -> None:
    alpha = 0.05
    power = 0.80
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--alpha" and i + 1 < len(args):
            alpha = float(args[i + 1])
            i += 2
        elif a == "--power" and i + 1 < len(args):
            power = float(args[i + 1])
            i += 2
        else:
            i += 1

    project_path = app.project_dir / PROTOCOL_FILENAME
    if not project_path.exists():
        app.log_view.line("No StudyProtocol · run /init first", kind="err", gutter="✗")
        return
    protocol = StudyProtocol.read_yaml(project_path)
    from causalrag.cli.main import _load_dataframe
    from causalrag.feasibility import default_thresholds, run_feasibility as do_feas

    df = await asyncio.to_thread(_load_dataframe, protocol, app.project_dir)
    flags = set(protocol.flags)
    thresholds = default_thresholds(flags)
    thresholds.alpha = alpha
    thresholds.target_power = power

    app.set_phase(2)
    app.log_view.line(
        f"Mode {thresholds.mode} · n={len(df):,} · α={alpha} · target_power={power}",
        kind="dim",
        gutter="·",
    )
    report = await asyncio.to_thread(do_feas, df, protocol, thresholds=thresholds)
    protocol.feasibility = report.to_protocol()
    protocol.write_yaml(project_path)
    app.protocol = protocol

    rows = []
    for r in report.results:
        verdict_chip = {"admissible": "● admissible", "borderline": "◐ borderline", "underpowered": "○ underpowered"}.get(
            r.verdict, r.verdict
        )
        rows.append(
            (
                r.treatment,
                r.outcome,
                r.family,
                f"{r.mde:.3f}",
                f"{r.achieved_power_at_band:.2f}" if r.achieved_power_at_band is not None else "—",
                verdict_chip,
            )
        )
    app.log_view.card(
        title="Power × MDE",
        step="Phase 2",
        meta=f"{len(report.admissible)} admissible / {len(report.results)} pairs",
        body=column_table(
            [
                ("Treatment", "left"),
                ("Outcome", "left"),
                ("Family", "left"),
                ("MDE", "right"),
                ("Power", "right"),
                ("Verdict", "left"),
            ],
            rows,
        ),
    )
    if report.thresholds.rationale:
        app.log_view.line(report.thresholds.rationale, kind="dim", gutter="·")
    app.log_view.line("Next: /hypothesize", kind="acc", gutter="→")


async def run_hypothesize(app: "CausalRoadmapTUI", args: list[str]) -> None:
    mode = "automated"
    ratio = 0.30
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif a == "--counterfactual-ratio" and i + 1 < len(args):
            ratio = float(args[i + 1])
            i += 2
        else:
            i += 1
    project_path = app.project_dir / PROTOCOL_FILENAME
    if not project_path.exists():
        app.log_view.line("No StudyProtocol · run /init first", kind="err", gutter="✗")
        return
    protocol = StudyProtocol.read_yaml(project_path)
    app.set_phase(3)
    from causalrag.hypothesize import (
        from_pairs as hyp_from_pairs,
        proposals_to_hypotheses,
        rank_by_impact,
        run_automated,
    )

    if mode == "manual":
        if protocol.feasibility is None or not protocol.feasibility.admissible_pairs:
            app.log_view.line(
                "No admissible pairs · run /feasibility first or use --mode automated",
                kind="err",
                gutter="✗",
            )
            return
        hypotheses = hyp_from_pairs(list(protocol.feasibility.admissible_pairs))
    else:
        proposals = run_automated(
            protocol=protocol,
            brief=None,
            client=None,
            counterfactual_ratio=ratio,
        )
        hypotheses = proposals_to_hypotheses(proposals)
    hypotheses = rank_by_impact(hypotheses)
    protocol.hypothesis_queue = tuple(hypotheses)
    protocol.counterfactual_ratio = ratio
    protocol.write_yaml(project_path)
    app.protocol = protocol

    rows = []
    for h in hypotheses[:10]:
        rows.append(
            (
                h.id,
                h.treatment,
                h.outcome,
                h.estimand.klass.value if h.estimand else "?",
                f"{h.impact_score:.2f}" if h.impact_score is not None else "—",
                (h.rationale or "")[:60],
            )
        )
    app.log_view.card(
        title="Hypothesis queue",
        step="Phase 3",
        meta=f"{len(hypotheses)} queued · CF share {ratio:.0%}",
        body=column_table(
            [
                ("ID", "left"),
                ("Treatment", "left"),
                ("Outcome", "left"),
                ("Estimand", "left"),
                ("Impact", "right"),
                ("Rationale", "left"),
            ],
            rows,
        ),
    )
    app.log_view.line("Next: /estimate", kind="acc", gutter="→")


async def run_report(app: "CausalRoadmapTUI", args: list[str]) -> None:
    fmt = "html"
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--format" and i + 1 < len(args):
            fmt = args[i + 1]
            i += 2
        else:
            i += 1
    project_path = app.project_dir / PROTOCOL_FILENAME
    if not project_path.exists():
        app.log_view.line("No StudyProtocol · run /init first", kind="err", gutter="✗")
        return
    protocol = StudyProtocol.read_yaml(project_path)
    app.set_phase(6)
    from datetime import datetime as _dt

    from causalrag.reporting.render_html import render_report

    reports_dir = app.project_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = _dt.utcnow().strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"{protocol.name}_{ts}.{fmt}"
    app.log_view.line(f"Rendering {fmt.upper()} report…", kind="dim", gutter=_spinner_glyph(0))

    synthesis_obj = None
    synth_path = app.project_dir / "executive_synthesis.json"
    if synth_path.exists():
        try:
            from causalrag.reporting.synthesis import ExecutiveSynthesis

            synthesis_obj = ExecutiveSynthesis.model_validate_json(
                synth_path.read_text()
            )
        except Exception:
            synthesis_obj = None

    content = await asyncio.to_thread(
        render_report, protocol, fmt, executive_synthesis=synthesis_obj
    )
    path.write_text(content, encoding="utf-8")
    app.log_view.line(f"{path} · {len(content):,} bytes", kind="ok", gutter="✓")
    app.log_view.line(f"open {path}", kind="acc", gutter="→")


async def run_run(app: "CausalRoadmapTUI", args: list[str]) -> None:
    """Auto pilot — discover → feasibility → hypothesize → estimate → sensitivity → report."""
    if not args:
        app.log_view.line(
            "Pass a dataset path to /run: /run data/cohort.csv [--treatment T] [--outcome Y]",
            kind="dim",
            gutter="·",
        )
        return
    source = args[0]
    treatment = None
    outcome = None
    question = None
    no_llm = False
    i = 1
    while i < len(args):
        a = args[i]
        if a == "--treatment" and i + 1 < len(args):
            treatment = args[i + 1]
            i += 2
        elif a == "--outcome" and i + 1 < len(args):
            outcome = args[i + 1]
            i += 2
        elif a == "--question" and i + 1 < len(args):
            question = args[i + 1]
            i += 2
        elif a == "--no-llm":
            no_llm = True
            i += 1
        else:
            i += 1
    project_path = app.project_dir / PROTOCOL_FILENAME
    if not project_path.exists():
        app.log_view.line("No StudyProtocol · run /init first", kind="err", gutter="✗")
        return
    protocol = StudyProtocol.read_yaml(project_path)
    data_path = Path(source)
    if not data_path.is_absolute():
        data_path = (app.project_dir / data_path).resolve()
    if not data_path.exists():
        app.log_view.line(f"Dataset not found: {data_path}", kind="err", gutter="✗")
        return

    discovery_client: OllamaClient | None = None
    expert_client: OllamaClient | None = None
    if not no_llm:
        slots = getattr(app, "cached_slots", None)
        if slots is None:
            from causalrag.cli.doctor import run_doctor as _probe

            slots, _ = recommend(await asyncio.to_thread(_probe))
        cassette_dir = app.project_dir / ".causalrag" / "cassettes"
        cassette_dir.mkdir(parents=True, exist_ok=True)
        discovery_client = OllamaClient(
            model=slots.discovery, cassette_dir=cassette_dir, allow_live=True
        )
        if slots.hypothesize != slots.discovery:
            expert_client = OllamaClient(
                model=slots.hypothesize, cassette_dir=cassette_dir, allow_live=True
            )

    from causalrag.auto import run_auto

    def _iter() -> list:
        return list(
            run_auto(
                protocol=protocol,
                project_dir=app.project_dir,
                dataset_path=data_path,
                treatment_hint=treatment,
                outcome_hint=outcome,
                research_question=question,
                discovery_client=discovery_client,
                expert_client=expert_client,
            )
        )

    events = await asyncio.to_thread(_iter)
    phase_map = {
        "discover": 1,
        "feasibility": 2,
        "hypothesize": 3,
        "estimate": 4,
        "sensitivity": 5,
        "report": 6,
    }
    for ev in events:
        if ev.kind == "phase_start":
            app.set_phase(phase_map.get(ev.phase, 0))
            app.log_view.line(ev.message or "", kind="acc", gutter="▸")
        elif ev.kind == "phase_end":
            app.log_view.line(ev.message or "", kind="ok", gutter="✓")
        elif ev.kind == "card":
            app.log_view.line(ev.message or "", kind="acc", gutter="·")
        elif ev.kind == "error":
            app.log_view.line(ev.message or "", kind="err", gutter="✗")
        else:
            app.log_view.line(ev.message or "", kind="dim", gutter="·")
    app.log_view.line("Auto-pilot complete.", kind="ok", gutter="✓")


async def run_quit(app: "CausalRoadmapTUI", _args: list[str]) -> None:
    app.exit()


async def run_layout(app: "CausalRoadmapTUI", args: list[str]) -> None:
    """Toggle visibility of the /auto-mode side panels.

    Usage:
        /layout                  show/hide both queue + chain panels
        /layout queue            toggle just the candidate queue
        /layout chains           toggle just the chain forest
        /layout show             force both visible (if auto_mode)
        /layout hide             hide both
    """
    queue = getattr(app, "queue_panel", None)
    chains = getattr(app, "chain_forest", None)
    if queue is None and chains is None:
        app.log_view.line(
            "/layout has no effect — start the TUI with --auto to mount panels.",
            kind="dim",
            gutter="·",
        )
        return
    target = (args[0].lower() if args else "toggle")
    if target == "show":
        if queue is not None:
            queue.display = True
        if chains is not None:
            chains.display = True
    elif target == "hide":
        if queue is not None:
            queue.display = False
        if chains is not None:
            chains.display = False
    elif target == "queue":
        if queue is not None:
            queue.display = not queue.display
    elif target == "chains":
        if chains is not None:
            chains.display = not chains.display
    else:  # default toggle
        if queue is not None:
            queue.display = not queue.display
        if chains is not None:
            chains.display = not chains.display
    states = []
    if queue is not None:
        states.append(f"queue · {'on' if queue.display else 'off'}")
    if chains is not None:
        states.append(f"chains · {'on' if chains.display else 'off'}")
    app.log_view.line(" · ".join(states), kind="acc", gutter="◧")


# --- Dispatcher ------------------------------------------------------------


async def run_auto(app: "CausalRoadmapTUI", args: list[str]) -> None:
    """Autonomous master mode — LLM-driven, multi-experiment loop.

    Usage:
        /auto run <data.csv> --experiments K [--foundation]
                              [--max-foundation-iterations N]
                              [--max-foundation-depth D]

    Distinct from /run (which is a deterministic single-pass pipeline). In
    /auto:
      - The reasoning LLM proposes experiments one at a time.
      - Each experiment walks the full Causal Roadmap (Steps 1-8).
      - --foundation enables the LLM to propose follow-up experiments that
        build on prior results (e.g., significant ATE → CATE on a modifier).
      - The loop terminates when the experiment budget is met OR the LLM
        votes 'stop' OR the safety circuit-breakers trip.
    """
    # Expect "run <data.csv> ..." or just "<data.csv> ..."
    if not args:
        app.log_view.line(
            "Usage: /auto run <data.csv> --experiments 5 [--foundation]",
            kind="dim",
            gutter="·",
        )
        return
    if args[0].lower() == "run":
        args = args[1:]
    if not args:
        app.log_view.line(
            "Usage: /auto run <data.csv> --experiments 5 [--foundation]",
            kind="err",
            gutter="✗",
        )
        return
    source = args[0]
    n_experiments = 5
    foundation = False
    max_foundation_iterations = 8
    max_foundation_depth = 4
    question = None
    i = 1
    while i < len(args):
        a = args[i]
        if a == "--experiments" and i + 1 < len(args):
            n_experiments = int(args[i + 1])
            i += 2
        elif a == "--foundation":
            foundation = True
            i += 1
        elif a == "--max-foundation-iterations" and i + 1 < len(args):
            max_foundation_iterations = int(args[i + 1])
            i += 2
        elif a == "--max-foundation-depth" and i + 1 < len(args):
            max_foundation_depth = int(args[i + 1])
            i += 2
        elif a == "--question" and i + 1 < len(args):
            question = args[i + 1]
            i += 2
        else:
            i += 1

    project_path = app.project_dir / PROTOCOL_FILENAME
    if not project_path.exists():
        app.log_view.line("No StudyProtocol · run /init first", kind="err", gutter="✗")
        return
    protocol = StudyProtocol.read_yaml(project_path)
    if question and not protocol.research_question:
        protocol.research_question = question
    data_path = Path(source)
    if not data_path.is_absolute():
        data_path = (app.project_dir / data_path).resolve()
    if not data_path.exists():
        app.log_view.line(f"Dataset not found: {data_path}", kind="err", gutter="✗")
        return

    # Need an LLM client for master mode
    slots = getattr(app, "cached_slots", None)
    if slots is None:
        from causalrag.cli.doctor import run_doctor as _probe

        slots, _ = recommend(await asyncio.to_thread(_probe))
        app.cached_slots = slots
    cassette_dir = app.project_dir / ".causalrag" / "cassettes"
    cassette_dir.mkdir(parents=True, exist_ok=True)
    discovery_client = OllamaClient(
        model=slots.discovery, cassette_dir=cassette_dir, allow_live=True
    )
    expert_client = None
    if slots.hypothesize != slots.discovery:
        expert_client = OllamaClient(
            model=slots.hypothesize, cassette_dir=cassette_dir, allow_live=True
        )

    app.log_view.line(
        f"AUTO master mode · {n_experiments} experiments target · "
        f"foundation={'on' if foundation else 'off'} "
        f"(iterations≤{max_foundation_iterations}, depth≤{max_foundation_depth})",
        kind="acc",
        gutter="▸",
    )
    app.log_view.line(
        f"Reasoning model: {slots.hypothesize}", kind="dim", gutter="·"
    )

    from causalrag.master_loop import LoopConfig, run_master_loop

    config = LoopConfig(
        n_experiments=n_experiments,
        foundation_allowed=foundation,
        max_foundation_iterations=max_foundation_iterations,
        max_foundation_depth=max_foundation_depth,
    )

    def _drain() -> list:
        return list(
            run_master_loop(
                protocol=protocol,
                project_dir=app.project_dir,
                dataset_path=data_path,
                discovery_client=discovery_client,
                expert_client=expert_client,
                config=config,
            )
        )

    events = await asyncio.to_thread(_drain)
    phase_map = {
        "discover": 1, "feasibility": 2, "hypothesize": 3,
        "estimate": 4, "sensitivity": 5, "report": 6,
    }
    # Auto-mode sub-phase labels — surfaced on the status bar so the user
    # can tell whether the loop is still planning, critiquing, or walking.
    sub_phase_label: dict[str, str] = {
        "plan": "3 · plan",
        "planner": "3 · plan",
        "critic": "3 · critic",
        "auto": "4 · auto",
        "synthesis": "6 · synthesis",
        "synthesize": "6 · synthesis",
    }
    queue_panel = getattr(app, "queue_panel", None)
    chain_forest = getattr(app, "chain_forest", None)
    for ev in events:
        phase_idx = phase_map.get(ev.phase, 0)
        # Push planner + result events to the live panels (skip cleanly
        # if --auto mode wasn't enabled).
        if ev.kind == "plan" and queue_panel is not None:
            queue_panel.update_panel(ev.payload)
        elif ev.kind == "card":
            if queue_panel is not None:
                queue_panel.update_panel(ev.payload)
            if chain_forest is not None:
                chain_forest.update_panel(ev.payload)

        if ev.kind == "phase_start":
            label = sub_phase_label.get(ev.phase)
            if label is not None:
                app.set_phase(phase_idx, label=label)
            else:
                app.set_phase(phase_idx)
            app.log_view.line(ev.message or "", kind="acc", gutter="▸")
        elif ev.kind == "phase_end":
            app.log_view.line(ev.message or "", kind="ok", gutter="✓")
        elif ev.kind == "card":
            _render_auto_card(app, ev)
        elif ev.kind == "plan":
            top = ev.payload.get("top") if isinstance(ev.payload, dict) else None
            n = len(top) if isinstance(top, list) else 0
            app.log_view.line(
                f"plan · top-{n} candidates ready",
                kind="acc",
                gutter="◆",
            )
        elif ev.kind == "done":
            app.log_view.line(ev.message or "", kind="ok", gutter="✓")
        elif ev.kind == "error":
            app.log_view.line(ev.message or "", kind="err", gutter="✗")
            hint = hint_for(ev.message or "")
            if hint:
                app.log_view.line(hint, kind="acc", gutter="→")
        else:
            app.log_view.line(ev.message or "", kind="dim", gutter="·")
    app.log_view.line(
        "Master loop complete. Run /report to render the HTML.",
        kind="acc",
        gutter="→",
    )


def _render_auto_card(app: "CausalRoadmapTUI", ev: Any) -> None:
    """Render a `kind=\"card\"` event in /auto mode with a readable layout.

    Falls back to the plain message when the payload is a non-dict.
    Verdict / magnitude / CI get their own indented lines with colored
    chips so the user can scan the result at a glance.
    """
    payload = ev.payload if isinstance(ev.payload, dict) else {}
    if not payload:
        app.log_view.line(ev.message or "", kind="acc", gutter="·")
        return
    hid = payload.get("id") or payload.get("hypothesis_id") or "?"
    treat = payload.get("treatment", "?")
    out = payload.get("outcome", "?")
    point = payload.get("point") or payload.get("point_estimate")
    ci_low = payload.get("ci_low")
    ci_high = payload.get("ci_high")
    verdict = (payload.get("sensitivity_verdict") or "").lower()
    method = payload.get("method") or payload.get("estimator_id")
    chip_color = {"green": "ok", "yellow": "warn", "red": "err"}.get(verdict, "dim")
    chip_glyph = {"green": "●", "yellow": "◐", "red": "○"}.get(verdict, "·")

    header = Text()
    header.append(f"[{hid}] ", style="#9ec2ff bold")
    header.append(f"{treat} → {out}", style="#cfd6e4")
    if method:
        header.append(f"  · {method}", style="#9aa3b5")
    app.log_view.line(header, kind="acc", gutter="◆")

    if point is not None:
        mag = Text()
        mag.append("magnitude  ", style="#4d5773")
        try:
            mag.append(f"{float(point):+.4f}", style="#9ec2ff bold")
        except (TypeError, ValueError):
            mag.append(str(point), style="#cfd6e4")
        if ci_low is not None and ci_high is not None:
            try:
                mag.append(
                    f"   95% CI [{float(ci_low):+.4f}, {float(ci_high):+.4f}]",
                    style="#cfd6e4",
                )
            except (TypeError, ValueError):
                pass
        app.log_view.line(mag, kind="", gutter=" ")
    if verdict:
        chip = Text()
        chip.append("sensitivity ", style="#4d5773")
        chip.append(f"{chip_glyph} {verdict.upper()}", style="bold")
        app.log_view.line(chip, kind=chip_color, gutter=" ")
    elif ev.message:
        app.log_view.line(ev.message, kind="dim", gutter="·")


DISPATCH: dict[str, Any] = {
    "/help": run_help,
    "/?": run_help,
    "/clear": run_clear,
    "/init": run_init,
    "/doctor": run_doctor,
    "/discover": run_discover,
    "/estimate": run_estimate_cmd,
    "/sensitivity": run_sensitivity_cmd,
    "/feasibility": run_feasibility,
    "/hypothesize": run_hypothesize,
    "/report": run_report,
    "/run": run_run,
    "/auto": run_auto,
    "/layout": run_layout,
    "/quit": run_quit,
    "/exit": run_quit,
}


def _resolve_graph(protocol: StudyProtocol) -> CausalGraph | None:
    if protocol.candidate_graphs:
        idx = min(protocol.selected_graph_index, len(protocol.candidate_graphs) - 1)
        return protocol.candidate_graphs[idx]
    if protocol.discovery and protocol.discovery.candidate_graphs:
        return protocol.discovery.candidate_graphs[0]
    return None


async def dispatch(app: "CausalRoadmapTUI", line: str) -> None:
    line = line.strip()
    if not line:
        return
    head, _, rest = line.partition(" ")
    cmd = head.lower()
    args = _split_args(rest)
    handler = DISPATCH.get(cmd)
    if handler is None:
        app.log_view.line(
            f"Unknown command: {head} · try /help",
            kind="err",
            gutter="✗",
        )
        return
    try:
        await handler(app, args)
    except Exception as e:
        app.log_view.line(
            f"{head} crashed · {type(e).__name__}: {e}",
            kind="err",
            gutter="✗",
        )
        hint = hint_for(e)
        if hint:
            app.log_view.line(hint, kind="acc", gutter="→")
