"""Typer entry point — ``causalrag`` CLI.

Week 1 commands: ``init``, ``doctor``, ``validate``. No LLM calls yet.
The full command set (PDD §6, §29.1) is added in later weeks.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from causalrag import __version__
from causalrag.cli.doctor import recommend, report_dict, run_doctor
from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.graph import CausalGraph
from causalrag.core.ledger import record_decision
from causalrag.core.protocol import RoadmapWalk, StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.discovery import run_discovery
from causalrag.feasibility import run_feasibility as do_feasibility
from causalrag.hypothesize import (
    from_pairs as hypotheses_from_pairs,
    proposals_to_hypotheses,
    rank_by_impact,
    run_automated as run_automated_hypotheses,
)
from causalrag.llm.cassette import CassetteMiss
from causalrag.llm.ollama_client import OllamaClient, SchemaValidationFailed
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q7_estimate import estimate as run_estimate
from causalrag.sensitivity.evalue import evalue as run_evalue
from causalrag.sensitivity.sensemakr_py import sensemakr as run_sensemakr
from causalrag.sensitivity.verdict import aggregate as aggregate_sensitivity

app = typer.Typer(
    name="causalrag",
    no_args_is_help=True,
    add_completion=False,
    help="LLM-assisted causal inference exploratory & estimation pipeline.",
)

console = Console()

PROTOCOL_FILENAME = "study.causalrag.yaml"


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"causalrag {__version__}")
        raise typer.Exit()


@app.callback()
def root(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version and exit."),
    ] = None,
) -> None:
    """CausalRoadmap — causal inference with hardware-aware local LLMs."""


# --- init ---------------------------------------------------------------------


def _scaffold_project(
    root: Path,
    name: str,
    tier: str,
) -> Path:
    """Create the §6.1 project skeleton and return the protocol path."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(exist_ok=True)
    (root / "reports").mkdir(exist_ok=True)
    cr = root / ".causalrag"
    cr.mkdir(exist_ok=True)
    (cr / "cassettes").mkdir(exist_ok=True)
    (cr / "cassettes" / ".gitkeep").touch()
    (cr / "history.jsonl").touch()

    readme = root / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {name}\n\nScaffolded by `causalrag init`. See `{PROTOCOL_FILENAME}` "
            "for the study protocol.\n",
            encoding="utf-8",
        )

    protocol_path = root / PROTOCOL_FILENAME
    if not protocol_path.exists():
        protocol = StudyProtocol(name=name, tier=tier)  # type: ignore[arg-type]
        protocol.write_yaml(protocol_path)

    return protocol_path


@app.command()
def init(
    name: Annotated[
        str | None,
        typer.Argument(
            help="Project name. If a directory is given, the basename is used."
        ),
    ] = None,
    tier: Annotated[
        str,
        typer.Option(
            "--tier",
            help="Default tier: data-scientist | academic | domain-expert | auto",
        ),
    ] = "academic",
    path: Annotated[
        Path | None,
        typer.Option("--path", help="Target directory (defaults to ./<name>)."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Allow init in a non-empty directory."),
    ] = False,
) -> None:
    """Scaffold a new CausalRoadmap project — see PDD §6.1."""
    if tier == "auto":
        tier = "academic"
    if tier not in {"data-scientist", "academic", "domain-expert"}:
        console.print(f"[red]Unknown tier:[/red] {tier}")
        raise typer.Exit(2)

    if name is None and path is None:
        console.print("[red]Provide a project name or --path.[/red]")
        raise typer.Exit(2)

    target: Path
    project_name: str
    if path is not None:
        target = path.expanduser().resolve()
        project_name = name or target.name
    else:
        assert name is not None
        target = Path.cwd() / name
        project_name = name

    if target.exists() and any(target.iterdir()) and not force:
        console.print(
            f"[red]Directory {target} is not empty.[/red] Use --force to scaffold anyway."
        )
        raise typer.Exit(1)

    protocol_path = _scaffold_project(target, name=project_name, tier=tier)
    console.print(
        Panel.fit(
            f"Initialized [bold]{project_name}[/bold] at {target}\n"
            f"Protocol: {protocol_path.relative_to(target.parent) if target.parent != target else protocol_path}\n"
            f"Tier: [cyan]{tier}[/cyan]",
            title="causalrag init",
            border_style="green",
        )
    )


# --- doctor -------------------------------------------------------------------


def _render_doctor(profile) -> None:
    table = Table(title="causalrag doctor", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Value")
    py_ok = profile.python_version.split(".")[:2] >= ["3", "11"]
    table.add_row("Python", f"{profile.python_version} {'✓' if py_ok else '✗'}")
    table.add_row("Platform", profile.platform)
    table.add_row(
        "CPU", f"{profile.cpu_logical} logical / {profile.cpu_physical or '?'} physical cores"
    )
    table.add_row(
        "RAM", f"{profile.available_ram_gb:.1f} GB free / {profile.total_ram_gb:.1f} GB total"
    )
    if profile.disk_free_gb is not None:
        table.add_row("Disk", f"{profile.disk_free_gb:.1f} GB free")
    if profile.gpus:
        devs = ", ".join(
            f"{g.name} ({g.vram_total_gb:.1f} GB, {g.backend})" for g in profile.gpus
        )
        table.add_row("GPU", devs)
    else:
        table.add_row("GPU", "[yellow]none detected[/yellow]")
    if profile.ollama.reachable:
        v = profile.ollama.version or "?"
        n_models = len(profile.ollama.models)
        table.add_row(
            "Ollama", f"reachable @ {profile.ollama.base_url} (v{v}, {n_models} models installed)"
        )
    else:
        table.add_row(
            "Ollama", f"[yellow]not reachable @ {profile.ollama.base_url}[/yellow]"
        )
    table.add_row("rpy2", "ok" if profile.rpy2_importable else "not installed")
    table.add_row("R binary", profile.r_binary or "[dim]not found[/dim]")
    table.add_row("Effective VRAM", f"{profile.effective_vram_gb:.1f} GB")
    table.add_row("Tier", profile.tier_label)
    slots, missing = recommend(profile)
    table.add_row("Discovery model", slots.discovery)
    table.add_row("Hypothesize model", slots.hypothesize)
    table.add_row("Utility model", slots.utility)
    if missing:
        table.add_row("Missing", "[yellow]" + ", ".join(missing) + "[/yellow]")
    console.print(table)
    for w in profile.warnings:
        console.print(f"[yellow]![/yellow] {w}")


@app.command()
def doctor(
    base_url: Annotated[
        str, typer.Option("--base-url", help="Ollama base URL.")
    ] = "http://127.0.0.1:11434",
    out_json: Annotated[
        bool, typer.Option("--json", help="Emit JSON to stdout instead of the table.")
    ] = False,
    save: Annotated[
        bool,
        typer.Option(
            "--save",
            help="Persist the report to .causalrag/hardware.json (project-cwd-relative).",
        ),
    ] = False,
) -> None:
    """Run an environment audit — see PDD §6.2."""
    profile = run_doctor(base_url=base_url)
    payload = report_dict(profile)
    if save:
        cr = Path.cwd() / ".causalrag"
        if not cr.exists():
            console.print(
                "[yellow]No .causalrag directory found — run `causalrag init` first or omit --save.[/yellow]"
            )
        else:
            (cr / "hardware.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if out_json:
        typer.echo(json.dumps(payload, indent=2))
    else:
        _render_doctor(profile)


# --- validate -----------------------------------------------------------------


@app.command()
def validate(
    protocol: Annotated[
        Path | None,
        typer.Argument(
            help="Path to study.causalrag.yaml. Defaults to ./study.causalrag.yaml.",
        ),
    ] = None,
) -> None:
    """Validate a StudyProtocol YAML file and round-trip it through Pydantic."""
    path = (protocol or Path.cwd() / PROTOCOL_FILENAME).resolve()
    if not path.exists():
        console.print(f"[red]Not found:[/red] {path}")
        raise typer.Exit(2)
    try:
        loaded = StudyProtocol.read_yaml(path)
    except Exception as e:
        console.print(Panel(f"[red]Validation failed[/red]\n\n{e}", title=str(path)))
        raise typer.Exit(1)

    # Round-trip — confirm the dump is self-consistent.
    redump = loaded.to_yaml()
    StudyProtocol.from_yaml(redump)

    table = Table(title=str(path), show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("name", loaded.name)
    table.add_row("version", loaded.version)
    table.add_row("tier", loaded.tier)
    table.add_row("research_question", loaded.research_question or "[dim]—[/dim]")
    table.add_row("flags", ", ".join(f.value for f in sorted(loaded.flags)) or "[dim]—[/dim]")
    table.add_row("hypotheses", str(len(loaded.hypothesis_queue)))
    table.add_row("candidate graphs", str(len(loaded.candidate_graphs)))
    console.print(table)
    console.print("[green]OK[/green] — protocol is valid and round-trips cleanly.")


# --- discover -----------------------------------------------------------------


@app.command()
def discover(
    source: Annotated[
        Path,
        typer.Argument(help="Path to the dataset (CSV / Parquet)."),
    ],
    project: Annotated[
        Path | None,
        typer.Option(
            "--project",
            help="Project directory (default: cwd). study.causalrag.yaml is updated in place.",
        ),
    ] = None,
    treatment: Annotated[
        str | None, typer.Option("--treatment", help="Treatment column hint.")
    ] = None,
    outcome: Annotated[
        str | None, typer.Option("--outcome", help="Outcome column hint.")
    ] = None,
    research_question: Annotated[
        str | None,
        typer.Option("--question", "-q", help="One-sentence research question."),
    ] = None,
    no_llm: Annotated[
        bool,
        typer.Option(
            "--no-llm",
            help="Skip Stage 1c — emit deterministic profile + flags only.",
        ),
    ] = False,
    model: Annotated[
        str | None, typer.Option("--model", help="Override the discovery-slot model.")
    ] = None,
    base_url: Annotated[
        str, typer.Option("--base-url", help="Ollama base URL.")
    ] = "http://127.0.0.1:11434",
    seed: Annotated[
        int, typer.Option("--seed", help="Deterministic seed forwarded to Ollama.")
    ] = 0,
) -> None:
    """Run Phase 1 discovery (PDD §7) on a dataset and update the StudyProtocol."""
    project_dir = (project or Path.cwd()).resolve()
    protocol_path = project_dir / PROTOCOL_FILENAME
    if not protocol_path.exists():
        console.print(f"[red]No StudyProtocol at[/red] {protocol_path}. Run `causalrag init` first.")
        raise typer.Exit(2)

    protocol = StudyProtocol.read_yaml(protocol_path)

    client: OllamaClient | None = None
    expert_client: OllamaClient | None = None
    if not no_llm:
        profile = run_doctor(base_url=base_url)
        slots, _ = recommend(profile)
        # Stage 1c — column-by-column extraction → general/instruction-tuned model.
        chosen_model = model or slots.discovery
        # Stage 1e — synthesis + DAG proposals → reasoning ("thinking") model.
        expert_model = slots.hypothesize
        cassette_dir = project_dir / ".causalrag" / "cassettes"
        cassette_dir.mkdir(parents=True, exist_ok=True)
        client = OllamaClient(
            model=chosen_model,
            base_url=base_url,
            seed=seed,
            cassette_dir=cassette_dir,
            allow_live=True,  # CLI invocation = user explicitly asked for a live call
        )
        if expert_model and expert_model != chosen_model:
            expert_client = OllamaClient(
                model=expert_model,
                base_url=base_url,
                seed=seed,
                cassette_dir=cassette_dir,
                allow_live=True,
            )
            console.print(
                f"[dim]Stage 1c: {chosen_model} (general)   Stage 1e: {expert_model} (reasoning)[/dim]"
            )
        else:
            console.print(f"[dim]LLM model: {chosen_model}[/dim]")

    try:
        result = run_discovery(
            source=source,
            client=client,
            expert_client=expert_client,
            research_question=research_question,
            treatment=treatment,
            outcome=outcome,
        )
    except CassetteMiss as e:
        console.print(
            f"[red]Cassette miss[/red] for the discovery prompt ({e.key[:8]}…). "
            "Set CAUSALRAG_REFRESH_LLM=1 to record a fresh cassette, or pass --no-llm."
        )
        raise typer.Exit(3) from e
    except SchemaValidationFailed as e:
        console.print(
            f"[red]LLM schema validation failed after retries.[/red]\nLast raw response:\n{e.last_response}"
        )
        raise typer.Exit(4) from e

    # Persist the discovery report into the protocol
    protocol.discovery = result.to_report()
    protocol.flags |= result.flags
    if result.candidate_graphs and not protocol.candidate_graphs:
        protocol.candidate_graphs = result.candidate_graphs
    if not protocol.dataset:
        from causalrag.core.protocol import DatasetSpec

        protocol.dataset = DatasetSpec(
            source=result.source_describe.get("source", str(source)),
            n_rows=result.profile.n_rows,
            n_cols=result.profile.n_cols,
            columns=result.columns,
        )
    if research_question and not protocol.research_question:
        protocol.research_question = research_question
    protocol.write_yaml(protocol_path)

    # JSON sidecar so downstream tools (and humans) can grep the raw profile
    sidecar = project_dir / ".causalrag" / "discovery.json"
    sidecar.write_text(
        json.dumps(
            {
                "source": result.source_describe,
                "research_question": result.research_question,
                "flags": sorted(f.value for f in result.flags),
                "profile": result.profile.model_dump(),
                "investigator": result.investigator.model_dump()
                if result.investigator
                else None,
                "variables": [v.model_dump() for v in result.columns],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Render summary
    table = Table(title="causalrag discover", show_lines=False)
    table.add_column("Column", style="bold")
    table.add_column("Logical dtype")
    table.add_column("Missing", justify="right")
    table.add_column("Role")
    table.add_column("Temporal")
    for col in result.profile.columns:
        var = next((v for v in result.columns if v.name == col.name), None)
        info = result.investigator.column(col.name) if result.investigator else None
        table.add_row(
            col.name,
            col.logical_dtype,
            f"{col.missing_rate:.0%}",
            (var.role.value if var else "-"),
            (info.temporal_position if info else "-"),
        )
    console.print(table)
    if result.flags:
        console.print(
            "Flags: " + ", ".join(f"[cyan]{f.value}[/cyan]" for f in sorted(result.flags))
        )

    if result.expert is not None:
        console.print()
        console.print(
            Panel.fit(
                result.expert.domain_summary,
                title=f"[bold]Domain Expert Brief[/bold] (tag: {result.investigator.domain_tag if result.investigator else '?'})",
                border_style="cyan",
            )
        )
        if result.expert.identification_warnings:
            console.print("[yellow]Identification warnings:[/yellow]")
            for w in result.expert.identification_warnings:
                console.print(f"  • {w}")
        if result.expert.unmeasured_confounders:
            console.print("[yellow]Suspected unmeasured confounders:[/yellow]")
            for u in result.expert.unmeasured_confounders[:5]:
                proxies = ", ".join(u.observed_proxies) if u.observed_proxies else "—"
                console.print(f"  • {u.name}: {u.reason} (proxies: {proxies})")
        if result.candidate_graphs:
            console.print(f"[cyan]Candidate DAGs:[/cyan] {len(result.candidate_graphs)} (rank-1 selected)")
        if result.dag_audit:
            contradicted = [a for a in result.dag_audit if a.verdict == "contradicted"]
            if contradicted:
                console.print(
                    f"[red]⚠ {len(contradicted)} LLM-proposed edge(s) contradicted by Layer-4 audit[/red]"
                )
                for a in contradicted[:5]:
                    console.print(
                        f"  • {a.source} → {a.target}: |r|={abs(a.partial_correlation):.3f}, p={a.p_value:.3f}"
                    )

    console.print(f"[green]✓[/green] discovery.json written to {sidecar}")
    console.print(f"[green]✓[/green] {protocol_path} updated")


# --- estimate -----------------------------------------------------------------


def _load_dataframe(protocol: StudyProtocol, project_dir: Path) -> "pd.DataFrame":
    import pandas as pd

    if protocol.dataset is None:
        raise typer.Exit(code=2)
    source = protocol.dataset.source
    raw_path = source[len("csv://") :] if source.startswith("csv://") else source
    p = Path(raw_path)
    if not p.is_absolute():
        p = (project_dir / p).resolve()
    if not p.exists():
        console.print(f"[red]Dataset file not found at {p}[/red]")
        raise typer.Exit(code=2)
    if p.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _resolve_graph(protocol: StudyProtocol) -> CausalGraph | None:
    if protocol.candidate_graphs:
        idx = min(protocol.selected_graph_index, len(protocol.candidate_graphs) - 1)
        return protocol.candidate_graphs[idx]
    if protocol.discovery and protocol.discovery.candidate_graphs:
        return protocol.discovery.candidate_graphs[0]
    return None


def _infer_treatment_outcome(
    protocol: StudyProtocol,
    treatment: str | None,
    outcome: str | None,
) -> tuple[str, str]:
    t = treatment
    y = outcome
    if (t is None or y is None) and protocol.discovery is not None:
        for v in protocol.discovery.columns:
            if t is None and v.role is VariableRole.TREATMENT:
                t = v.name
            if y is None and v.role is VariableRole.OUTCOME:
                y = v.name
    if t is None or y is None:
        raise typer.BadParameter(
            "Treatment/outcome could not be inferred. Pass --treatment and --outcome."
        )
    return t, y


@app.command()
def estimate(
    project: Annotated[
        Path | None, typer.Option("--project", help="Project directory (default: cwd).")
    ] = None,
    treatment: Annotated[str | None, typer.Option("--treatment")] = None,
    outcome: Annotated[str | None, typer.Option("--outcome")] = None,
    estimand_class: Annotated[
        str,
        typer.Option(
            "--estimand", help="Estimand class: ATE, ATT, CATE, LATE, NDE, NIE."
        ),
    ] = "ATE",
    prefer: Annotated[
        str | None,
        typer.Option(
            "--prefer",
            help="Force an estimator id or family (dml, sparse, forest, meta, bart).",
        ),
    ] = None,
    allow_nonidentifiable: Annotated[
        bool,
        typer.Option(
            "--allow-nonidentifiable",
            help="Proceed even if Step 5 says the estimand is not identifiable.",
        ),
    ] = False,
) -> None:
    """Run Steps 5-7 for one hypothesis and persist the result into the
    StudyProtocol (PDD §10.5-10.7)."""
    project_dir = (project or Path.cwd()).resolve()
    protocol_path = project_dir / PROTOCOL_FILENAME
    if not protocol_path.exists():
        console.print(f"[red]No StudyProtocol at[/red] {protocol_path}")
        raise typer.Exit(2)
    protocol = StudyProtocol.read_yaml(protocol_path)

    t, y = _infer_treatment_outcome(protocol, treatment, outcome)
    klass = EstimandClass(estimand_class.upper())
    est = CausalEstimand(  # type: ignore[call-arg]
        **{
            "class": klass,
            "treatment": t,
            "outcome": y,
            "formal_expression": _default_expression(klass),
        }
    )

    graph = _resolve_graph(protocol)
    if graph is None:
        # Fall back to a no-confounder DAG; identification check will note empty adjustment
        graph = CausalGraph.from_edge_list([(t, y)], roles={t: VariableRole.TREATMENT, y: VariableRole.OUTCOME})

    df = _load_dataframe(protocol, project_dir)
    ident = identify_effect(est, graph, df=df)

    flag_set = set(protocol.flags)
    confounders = _infer_confounders(graph, t, y, protocol, df)
    result = run_estimate(
        df=df,
        estimand=est,
        identification=ident,
        protocol=protocol,
        confounders=tuple(confounders),
        flags=flag_set,
        prefer=prefer,
        allow_nonidentifiable=allow_nonidentifiable,
    )

    walk = protocol.roadmap_walks.get(f"{t}->{y}") or RoadmapWalk(hypothesis_id=f"{t}->{y}")
    walk.q3_estimand = est
    walk.q5_identification = {
        "identifiable": ident.identifiable,
        "strategy": ident.strategy,
        "adjustment_set": list(ident.adjustment_set),
        "estimand_expression": ident.estimand_expression,
        "notes": ident.notes,
    }
    walk.q7_estimates = tuple(list(walk.q7_estimates) + [result])
    # Step 6 — derive the canonical statistical functional
    from causalrag.roadmap.q6_statistical_estimand import derive_statistical_estimand

    walk.q6_statistical_estimand = derive_statistical_estimand(est, ident)
    new_walks = dict(protocol.roadmap_walks)
    new_walks[f"{t}->{y}"] = walk
    protocol.roadmap_walks = new_walks
    p_str = f"{result.p_value:.4g}" if result.p_value is not None else "NA"
    adj_n = len(result.diagnostics.get("adjustment_set_used", []))
    record_decision(
        protocol,
        phase="estimate",
        decision=f"selected_estimator (prefer={prefer or 'auto'})",
        chose=result.estimator_id,
        source="analyst" if prefer else "default",
        note=f"strategy={ident.strategy} · adj={adj_n} · p={p_str}",
    )
    protocol.write_yaml(protocol_path)

    table = Table(title=f"causalrag estimate — {t} → {y}", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("Estimator", result.estimator_id)
    table.add_row("Estimand", result.estimand_class)
    table.add_row("Strategy", ident.strategy)
    table.add_row("Adjustment", ", ".join(ident.adjustment_set) or "[dim]∅[/dim]")
    table.add_row("Point estimate", f"{result.point_estimate:.4f}")
    if result.ci_low is not None and result.ci_high is not None:
        table.add_row("95% CI", f"[{result.ci_low:.4f}, {result.ci_high:.4f}]")
    if result.p_value is not None:
        table.add_row("p-value", f"{result.p_value:.4g}")
    table.add_row("n used", str(result.n_used))
    if result.fit_seconds is not None:
        table.add_row("Fit seconds", f"{result.fit_seconds:.2f}")
    console.print(table)
    if not ident.identifiable:
        console.print(
            "[yellow]⚠ Step 5 flagged the estimand as non-identifiable; estimate retained because --allow-nonidentifiable was passed.[/yellow]"
        )


def _default_expression(klass: EstimandClass) -> str:
    mapping = {
        EstimandClass.ATE: "E[Y(1) - Y(0)]",
        EstimandClass.ATT: "E[Y(1) - Y(0) | T=1]",
        EstimandClass.ATC: "E[Y(1) - Y(0) | T=0]",
        EstimandClass.CATE: "E[Y(1) - Y(0) | X=x]",
        EstimandClass.LATE: "Local ATE among compliers",
        EstimandClass.NDE: "Natural direct effect",
        EstimandClass.NIE: "Natural indirect effect",
    }
    return mapping.get(klass, "")


def _build_q8_interpretation(walk, estimate, verdict, e_value, sensemakr) -> str:
    """Compose the Step 8 narrative from the estimate + sensitivity outputs.

    Uses the deterministic template path in ``roadmap.q8_interpret`` so the
    narrative is always defensible even without a live LLM. Falls back to a
    one-line summary if the walk doesn't have an estimand attached."""
    if walk.q3_estimand is None:
        return (
            f"Sensitivity verdict: {verdict.color}. {verdict.rationale}. "
            f"E-value={e_value.e_value:.2f} (scale={e_value.scale}). "
            f"RV={sensemakr.robustness_value:.3f}."
        )
    from causalrag.roadmap.q5_identify import IdentificationResult
    from causalrag.roadmap.q8_interpret import interpret as q8_interpret

    ident_dict = walk.q5_identification or {}
    ident = IdentificationResult(
        identifiable=bool(ident_dict.get("identifiable", True)),
        strategy=ident_dict.get("strategy", "backdoor"),
        adjustment_set=tuple(ident_dict.get("adjustment_set", [])),
        estimand_expression=ident_dict.get("estimand_expression"),
    )
    interp = q8_interpret(
        estimand=walk.q3_estimand,
        identification=ident,
        estimate=estimate,
        verdict=verdict,
    )
    return f"{interp.headline}  {interp.magnitude}  {interp.robustness}"


def _infer_confounders(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    protocol: StudyProtocol,
    df: "pd.DataFrame",
) -> list[str]:
    declared = [n for n, r in graph.roles.items() if r is VariableRole.CONFOUNDER]
    if declared:
        return [c for c in declared if c in df.columns]
    if protocol.discovery is not None:
        from_specs = [
            v.name
            for v in protocol.discovery.columns
            if v.role is VariableRole.CONFOUNDER
        ]
        kept = [c for c in from_specs if c in df.columns]
        if kept:
            return kept
    # Last resort: every column except treatment/outcome
    return [c for c in df.columns if c not in {treatment, outcome}]


# --- sensitivity --------------------------------------------------------------


@app.command()
def sensitivity(
    project: Annotated[Path | None, typer.Option("--project")] = None,
    treatment: Annotated[str | None, typer.Option("--treatment")] = None,
    outcome: Annotated[str | None, typer.Option("--outcome")] = None,
    scale: Annotated[
        str,
        typer.Option(
            "--scale",
            help="E-value scale: auto, risk_ratio, odds_ratio, hazard_ratio, standardized.",
        ),
    ] = "auto",
    rule: Annotated[
        str,
        typer.Option(
            "--rule", help="Verdict aggregation rule: min, average, strict."
        ),
    ] = "min",
) -> None:
    """Run sensitivity analyses on the latest estimate and persist the verdict."""
    project_dir = (project or Path.cwd()).resolve()
    protocol_path = project_dir / PROTOCOL_FILENAME
    if not protocol_path.exists():
        console.print(f"[red]No StudyProtocol at[/red] {protocol_path}")
        raise typer.Exit(2)
    protocol = StudyProtocol.read_yaml(protocol_path)

    t, y = _infer_treatment_outcome(protocol, treatment, outcome)
    key = f"{t}->{y}"
    walk = protocol.roadmap_walks.get(key)
    if walk is None or not walk.q7_estimates:
        console.print(
            f"[red]No estimate found for {key}. Run `causalrag estimate` first.[/red]"
        )
        raise typer.Exit(2)
    last = walk.q7_estimates[-1]

    df = _load_dataframe(protocol, project_dir)

    auto_scale = scale
    if auto_scale == "auto":
        from causalrag.core.flags import DataFlag

        if DataFlag.BINARY_OUTCOME in protocol.flags:
            auto_scale = "odds_ratio"
        elif DataFlag.RIGHT_CENSORED_OUTCOME in protocol.flags:
            auto_scale = "hazard_ratio"
        else:
            auto_scale = "standardized"

    point = last.point_estimate
    ci_low_e = last.ci_low
    ci_high_e = last.ci_high
    if auto_scale == "standardized":
        # The standardized E-value expects a Cohen's d input. Our DML point
        # estimate is in raw outcome units, so we convert via the outcome's SD.
        if y in df.columns:
            y_sd = float(df[y].std(ddof=1)) or 1.0
            point = point / y_sd
            if ci_low_e is not None:
                ci_low_e = ci_low_e / y_sd
            if ci_high_e is not None:
                ci_high_e = ci_high_e / y_sd

    e = run_evalue(
        point,
        scale=auto_scale,  # type: ignore[arg-type]
        ci_low=ci_low_e,
        ci_high=ci_high_e,
    )
    confounders = _infer_confounders(
        _resolve_graph(protocol) or CausalGraph.from_edge_list([(t, y)]),
        t,
        y,
        protocol,
        df,
    )
    s = run_sensemakr(df, treatment=t, outcome=y, covariates=tuple(confounders))

    verdict = aggregate_sensitivity(evalue=e, sensemakr=s, rule=rule)  # type: ignore[arg-type]

    table = Table(title=f"causalrag sensitivity — {t} → {y}", show_lines=False)
    table.add_column("Method", style="bold")
    table.add_column("Statistic")
    table.add_column("Value")
    table.add_row("E-value", "point", f"{e.e_value:.2f}")
    if e.e_value_ci is not None:
        table.add_row("", "CI bound", f"{e.e_value_ci:.2f}")
    table.add_row("", "scale", e.scale)
    table.add_row("Sensemakr", "estimate", f"{s.estimate:.4f}")
    table.add_row("", "robustness value", f"{s.robustness_value:.4f}")
    table.add_row("", "backend", s.backend)
    table.add_row("Verdict", "color", f"[bold]{verdict.color}[/bold]")
    table.add_row("", "components", verdict.rationale)
    console.print(table)

    # Step 8 — produce a structured interpretation (template-mode by default)
    walk.q8_interpretation = _build_q8_interpretation(walk, last, verdict, e, s)
    new_walks = dict(protocol.roadmap_walks)
    new_walks[key] = walk
    protocol.roadmap_walks = new_walks
    record_decision(
        protocol,
        phase="sensitivity",
        decision=f"verdict_rule={rule}",
        chose=verdict.color,
        source="default",
        note=f"e-value={e.e_value:.2f} · rv={s.robustness_value:.3f}",
    )
    protocol.write_yaml(protocol_path)


# --- tui ----------------------------------------------------------------------


@app.command()
def feasibility(
    project: Annotated[Path | None, typer.Option("--project")] = None,
    alpha: Annotated[float, typer.Option("--alpha")] = 0.05,
    power: Annotated[float, typer.Option("--power")] = 0.80,
) -> None:
    """Run Phase 2 — power × MDE feasibility filter (PDD §8)."""
    project_dir = (project or Path.cwd()).resolve()
    protocol_path = project_dir / PROTOCOL_FILENAME
    if not protocol_path.exists():
        console.print(f"[red]No StudyProtocol at[/red] {protocol_path}")
        raise typer.Exit(2)
    protocol = StudyProtocol.read_yaml(protocol_path)
    df = _load_dataframe(protocol, project_dir)

    from causalrag.feasibility import default_thresholds

    flags = set(protocol.flags)
    thresholds = default_thresholds(flags)
    thresholds.alpha = alpha
    thresholds.target_power = power

    report = do_feasibility(df, protocol, thresholds=thresholds)
    protocol.feasibility = report.to_protocol()
    protocol.write_yaml(protocol_path)

    table = Table(title="causalrag feasibility", show_lines=False)
    table.add_column("Treatment", style="bold")
    table.add_column("Outcome")
    table.add_column("Family")
    table.add_column("MDE", justify="right")
    table.add_column("Achieved", justify="right")
    table.add_column("Verdict")
    for r in report.results:
        verdict_color = {
            "admissible": "green",
            "borderline": "yellow",
            "underpowered": "red",
            "unsupported": "dim",
        }.get(r.verdict, "white")
        table.add_row(
            r.treatment,
            r.outcome,
            r.family,
            f"{r.mde:.3f}",
            f"{r.achieved_power_at_band:.2f}" if r.achieved_power_at_band is not None else "—",
            f"[{verdict_color}]{r.verdict}[/{verdict_color}]",
        )
    console.print(table)
    console.print(
        f"{len(report.admissible)} admissible · {len(report.borderline)} borderline · "
        f"{len(report.underpowered)} underpowered"
    )


@app.command()
def hypothesize(
    project: Annotated[Path | None, typer.Option("--project")] = None,
    mode: Annotated[
        str,
        typer.Option("--mode", help="manual | automated | hybrid"),
    ] = "automated",
    counterfactual_ratio: Annotated[
        float, typer.Option("--counterfactual-ratio")
    ] = 0.30,
) -> None:
    """Run Phase 3 — generate the ranked hypothesis queue (PDD §9)."""
    project_dir = (project or Path.cwd()).resolve()
    protocol_path = project_dir / PROTOCOL_FILENAME
    if not protocol_path.exists():
        console.print(f"[red]No StudyProtocol at[/red] {protocol_path}")
        raise typer.Exit(2)
    protocol = StudyProtocol.read_yaml(protocol_path)

    if mode == "manual":
        if protocol.feasibility is None or not protocol.feasibility.admissible_pairs:
            console.print(
                "[yellow]No admissible pairs in feasibility — run /feasibility first or pass --mode automated.[/yellow]"
            )
            raise typer.Exit(1)
        hypotheses = hypotheses_from_pairs(list(protocol.feasibility.admissible_pairs))
    else:
        # Automated path (no live LLM here — uses the cached expert brief).
        brief = None
        try:
            from causalrag.discovery.expert import DomainExpertBrief

            if protocol.discovery and protocol.discovery.domain_brief:
                # Brief was persisted as plain text on the protocol; we still
                # need the structured object for the automated generator. Use
                # deterministic fallback if not available.
                brief = None
        except Exception:
            brief = None
        proposals = run_automated_hypotheses(
            protocol=protocol,
            brief=brief,
            client=None,
            counterfactual_ratio=counterfactual_ratio,
        )
        hypotheses = proposals_to_hypotheses(proposals)

    hypotheses = rank_by_impact(hypotheses)
    protocol.hypothesis_queue = tuple(hypotheses)
    protocol.counterfactual_ratio = counterfactual_ratio
    protocol.write_yaml(protocol_path)

    table = Table(title="causalrag hypothesize", show_lines=False)
    table.add_column("ID", style="bold")
    table.add_column("Treatment")
    table.add_column("Outcome")
    table.add_column("Estimand")
    table.add_column("Impact", justify="right")
    table.add_column("Rationale")
    for h in hypotheses:
        table.add_row(
            h.id,
            h.treatment,
            h.outcome,
            h.estimand.klass.value if h.estimand else "?",
            f"{h.impact_score:.2f}" if h.impact_score is not None else "—",
            (h.rationale or "")[:80],
        )
    console.print(table)
    console.print(f"{len(hypotheses)} hypotheses queued · counterfactual ratio {counterfactual_ratio:.0%}")


@app.command()
def report(
    project: Annotated[Path | None, typer.Option("--project")] = None,
    fmt: Annotated[
        str, typer.Option("--format", help="html | md")
    ] = "html",
) -> None:
    """Render Phase 6 report (PDD §12)."""
    project_dir = (project or Path.cwd()).resolve()
    protocol_path = project_dir / PROTOCOL_FILENAME
    if not protocol_path.exists():
        console.print(f"[red]No StudyProtocol at[/red] {protocol_path}")
        raise typer.Exit(2)
    protocol = StudyProtocol.read_yaml(protocol_path)

    from causalrag.reporting.render_html import render_report

    reports_dir = project_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    path = reports_dir / f"{protocol.name}_{timestamp}.{fmt}"
    content = render_report(protocol, fmt=fmt)
    path.write_text(content, encoding="utf-8")
    console.print(f"[green]✓[/green] {path} · {len(content):,} bytes")


@app.command()
def run(
    source: Annotated[Path, typer.Argument(help="Path/URI to the dataset.")],
    project: Annotated[Path | None, typer.Option("--project")] = None,
    treatment: Annotated[str | None, typer.Option("--treatment")] = None,
    outcome: Annotated[str | None, typer.Option("--outcome")] = None,
    question: Annotated[
        str | None,
        typer.Option("--question", "-q", help="One-sentence research question."),
    ] = None,
    no_llm: Annotated[bool, typer.Option("--no-llm")] = False,
    max_hypotheses: Annotated[int, typer.Option("--max-hypotheses")] = 3,
    counterfactual_ratio: Annotated[float, typer.Option("--counterfactual-ratio")] = 0.30,
    report_format: Annotated[str, typer.Option("--report-format")] = "html",
    base_url: Annotated[str, typer.Option("--base-url")] = "http://127.0.0.1:11434",
) -> None:
    """Auto-pilot — run the full pipeline end to end (PDD §29.1 ``run``).

    Sequence: discover → feasibility → hypothesize → estimate → sensitivity →
    report. Treatment / outcome are inferred from the LLM investigator when
    not supplied. Every decision is logged to ``protocol.decision_ledger``
    with ``source=auto`` so the analyst can see exactly what was chosen.
    """
    from causalrag.auto import run_auto

    project_dir = (project or Path.cwd()).resolve()
    protocol_path = project_dir / PROTOCOL_FILENAME
    if not protocol_path.exists():
        # Auto-scaffold so /run can be the only command the user types.
        from causalrag.cli.main import _scaffold_project

        project_dir = (project_dir / source.stem).resolve()
        _scaffold_project(project_dir, name=source.stem, tier="academic")
        protocol_path = project_dir / PROTOCOL_FILENAME
        console.print(f"[dim]Auto-scaffolded project at {project_dir}[/dim]")

    protocol = StudyProtocol.read_yaml(protocol_path)
    data_path = source if source.is_absolute() else (project_dir / source).resolve()
    if not data_path.exists():
        # Try cwd fallback
        if source.exists():
            data_path = source.resolve()
        else:
            console.print(f"[red]Dataset not found: {source}[/red]")
            raise typer.Exit(2)

    # Build LLM clients (unless --no-llm)
    discovery_client = None
    expert_client = None
    if not no_llm:
        profile = run_doctor(base_url=base_url)
        slots, _ = recommend(profile)
        cassette_dir = project_dir / ".causalrag" / "cassettes"
        cassette_dir.mkdir(parents=True, exist_ok=True)
        discovery_client = OllamaClient(
            model=slots.discovery,
            base_url=base_url,
            cassette_dir=cassette_dir,
            allow_live=True,
        )
        if slots.hypothesize != slots.discovery:
            expert_client = OllamaClient(
                model=slots.hypothesize,
                base_url=base_url,
                cassette_dir=cassette_dir,
                allow_live=True,
            )
        console.print(
            f"[dim]LLM · discovery={slots.discovery} · expert={slots.hypothesize}[/dim]"
        )

    for event in run_auto(
        protocol=protocol,
        project_dir=project_dir,
        dataset_path=data_path,
        treatment_hint=treatment,
        outcome_hint=outcome,
        research_question=question,
        discovery_client=discovery_client,
        expert_client=expert_client,
        counterfactual_ratio=counterfactual_ratio,
        max_hypotheses=max_hypotheses,
        report_format=report_format,
    ):
        if event.kind == "phase_start":
            console.print(f"\n[bold cyan]▸ {event.message}[/bold cyan]")
        elif event.kind == "phase_end":
            console.print(f"[green]✓[/green] {event.message}")
        elif event.kind == "card":
            console.print(f"  {event.message}")
        elif event.kind == "error":
            console.print(f"[red]✗ {event.message}[/red]")
        else:
            console.print(f"  · {event.message}")

    console.print("\n[bold green]Pipeline complete.[/bold green]")


@app.command()
def tui(
    project: Annotated[
        Path | None,
        typer.Option("--project", help="Project directory (default: cwd)."),
    ] = None,
) -> None:
    """Launch the CausalRoadmap TUI (Textual-based terminal UI)."""
    from causalrag.tui import run as run_tui

    run_tui(project_dir=project)


if __name__ == "__main__":
    app()
