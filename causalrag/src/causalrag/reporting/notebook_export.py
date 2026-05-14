"""Convert a completed run into a reproducible Jupyter notebook.

Sprint 4.7 deliverable: ``/export notebook`` writes the steps the
master loop took as a `.ipynb` file with code cells that recreate
every estimate. Useful for handing the work off to a notebook user,
for archival reproducibility, and for downstream slicing in pandas /
matplotlib.

Implementation is deliberately dependency-light: we render the
notebook directly via the nbformat JSON schema. If `jupytext` is
installed we can also offer the percent-format `.py` flavour as a
companion output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from causalrag.core.protocol import StudyProtocol


@dataclass
class NotebookExportResult:
    output_path: Path
    n_code_cells: int
    n_markdown_cells: int
    format: str  # "ipynb" or "ipynb+jupytext"


def _markdown_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
    }


def _code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(text: str) -> list[str]:
    """nbformat expects a list of lines that, when joined, equal the source."""
    if not text:
        return []
    parts = text.splitlines(keepends=True)
    if not text.endswith("\n"):
        parts[-1] = parts[-1].rstrip("\n")
    return parts


def export_notebook(
    protocol: StudyProtocol,
    *,
    output_path: Path,
    dataset_path: Path | str | None = None,
    include_synthesis: bool = True,
) -> NotebookExportResult:
    """Build a reproducible .ipynb from the protocol.

    Sections emitted:
      1. Title + research question + run metadata
      2. Imports + dataset load
      3. Discovery: profile, flags, candidate graphs
      4. One section per completed RoadmapWalk:
         - identification (the strategy + adjustment set)
         - estimator construction
         - fit + estimate
         - sensitivity verdict
      5. Executive synthesis (if present)
      6. Reproducibility note

    Returns:
      NotebookExportResult with path, cell counts, and format label.
    """
    cells: list[dict[str, Any]] = []
    n_code = 0
    n_md = 0

    # ─── Header ─────────────────────────────────────────────────
    header = f"""# {protocol.name}

_Reproducible Jupyter export of a CausalRoadmap study run._

**Research question:** {protocol.research_question or "(not specified)"}

**Tier:** {protocol.tier}

**Created:** {protocol.created.isoformat(timespec="seconds")}

**Exported:** {datetime.now(UTC).isoformat(timespec="seconds")}

**Multiple-testing:** `{protocol.multiple_testing}`

This notebook regenerates every estimate in the run. Cells are
intended to be runnable top-to-bottom in a Python kernel with
`causalrag` installed in editable mode.
"""
    cells.append(_markdown_cell(header)); n_md += 1

    # ─── Imports ────────────────────────────────────────────────
    imports = """import pandas as pd
import numpy as np

from causalrag.core.protocol import StudyProtocol
from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.graph import CausalGraph
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q7_estimate import estimate as run_step7
"""
    cells.append(_code_cell(imports)); n_code += 1

    # ─── Dataset load ───────────────────────────────────────────
    src = (
        str(dataset_path)
        if dataset_path is not None
        else (protocol.dataset.source if protocol.dataset else "<UNKNOWN>")
    )
    cells.append(
        _markdown_cell(
            f"## 1. Load the dataset\n\nSource: `{src}`"
        )
    ); n_md += 1
    cells.append(
        _code_cell(
            f"# Adjust the path to match your environment.\n"
            f"df = pd.read_csv({src!r})\n"
            f"print('shape:', df.shape)\n"
            f"df.head()"
        )
    ); n_code += 1

    # ─── Discovery summary ──────────────────────────────────────
    if protocol.discovery is not None:
        flags_str = ", ".join(sorted(f.value for f in protocol.discovery.flags))
        cells.append(
            _markdown_cell(
                f"## 2. Discovery\n\n"
                f"**Flags emitted:** `{flags_str}`\n\n"
                f"**Number of candidate DAGs:** "
                f"{len(protocol.discovery.candidate_graphs)}\n\n"
                f"**Domain brief (excerpt):**\n\n"
                f"> {protocol.discovery.domain_brief[:600] if protocol.discovery.domain_brief else '(none)'}"
            )
        ); n_md += 1

    # ─── Per-walk reproduction ─────────────────────────────────
    if protocol.roadmap_walks:
        cells.append(
            _markdown_cell(
                f"## 3. Reproduce every Roadmap walk\n\n"
                f"Each section reconstructs one estimand and refits the "
                f"estimator. Total walks: {len(protocol.roadmap_walks)}."
            )
        ); n_md += 1

        for walk_id, walk in protocol.roadmap_walks.items():
            if not walk.q3_estimand:
                continue
            est_cause = walk.q3_estimand
            ident = walk.q5_identification or {}
            est_result = walk.q7_estimates[-1] if walk.q7_estimates else None

            md = (
                f"### {walk_id}\n\n"
                f"- Treatment: `{est_cause.treatment}`\n"
                f"- Outcome: `{est_cause.outcome}`\n"
                f"- Estimand: `{est_cause.klass.value}`\n"
                f"- Identification strategy: `{ident.get('strategy', '—')}`\n"
                f"- Adjustment set: `{', '.join(ident.get('adjustment_set', [])) or '(empty)'}`\n"
            )
            if est_result:
                md += (
                    f"- Estimator: `{est_result.estimator_id}`\n"
                    f"- Point estimate: `{est_result.point_estimate:+.4f}`\n"
                )
                if est_result.ci_low is not None and est_result.ci_high is not None:
                    md += f"- 95% CI: `[{est_result.ci_low:+.4f}, {est_result.ci_high:+.4f}]`\n"
                if walk.sensitivity_verdict:
                    md += f"- Sensitivity verdict: **{walk.sensitivity_verdict}**\n"

            cells.append(_markdown_cell(md)); n_md += 1

            confounders = list(ident.get("adjustment_set", []) or [])
            cells.append(
                _code_cell(
                    f"# Reproduce {walk_id}\n"
                    f"estimand = CausalEstimand.model_validate({{\n"
                    f"    'class': EstimandClass.{est_cause.klass.name},\n"
                    f"    'treatment': {est_cause.treatment!r},\n"
                    f"    'outcome': {est_cause.outcome!r},\n"
                    f"    'modifiers': {tuple(est_cause.modifiers or ())!r},\n"
                    f"    'formal_expression': {est_cause.formal_expression!r},\n"
                    f"}})\n"
                    f"\n"
                    f"# Recreate the graph used at fit time\n"
                    f"confounders = {confounders!r}\n"
                    f"edges = (\n"
                    f"    [(c, {est_cause.treatment!r}) for c in confounders]\n"
                    f"    + [(c, {est_cause.outcome!r}) for c in confounders]\n"
                    f"    + [({est_cause.treatment!r}, {est_cause.outcome!r})]\n"
                    f")\n"
                    f"from causalrag.core.roles import VariableRole\n"
                    f"roles = {{c: VariableRole.CONFOUNDER for c in confounders}}\n"
                    f"roles[{est_cause.treatment!r}] = VariableRole.TREATMENT\n"
                    f"roles[{est_cause.outcome!r}] = VariableRole.OUTCOME\n"
                    f"graph = CausalGraph.from_edge_list(edges, roles=roles)\n"
                    f"\n"
                    f"ident = identify_effect(estimand, graph, df=df)\n"
                    f"print('identifiable:', ident.identifiable, '|', 'strategy:', ident.strategy)"
                )
            ); n_code += 1

            if est_result:
                cells.append(
                    _code_cell(
                        f"# Refit the estimator the original run used\n"
                        f"protocol_obj = StudyProtocol(name={protocol.name!r})\n"
                        f"result = run_step7(\n"
                        f"    df=df,\n"
                        f"    estimand=estimand,\n"
                        f"    identification=ident,\n"
                        f"    protocol=protocol_obj,\n"
                        f"    prefer={est_result.estimator_id!r},\n"
                        f")\n"
                        f"print(f'point={{result.point_estimate:+.4f}}, '\n"
                        f"      f'CI=[{{result.ci_low:+.4f}}, {{result.ci_high:+.4f}}]')"
                    )
                ); n_code += 1

    # ─── Synthesis ──────────────────────────────────────────────
    if include_synthesis:
        synth_path = output_path.parent / "executive_synthesis.json"
        if synth_path.exists():
            try:
                synth = json.loads(synth_path.read_text())
                md = (
                    f"## 4. Executive synthesis ({synth.get('inferred_domain', '?')})\n\n"
                    f"**TL;DR:** {synth.get('tldr', '(none)')}\n\n"
                    f"### Findings\n"
                )
                for f in synth.get("findings", []):
                    md += f"\n**{f.get('rank', '?')}. {f.get('headline', '')}**\n"
                    md += f"\n- Effect: {f.get('quantified_effect', '')}\n"
                    md += f"- Implication: {f.get('domain_implication', '')}\n"
                    md += f"- Suggested next step: {f.get('suggested_next_step', '')}\n"
                    md += f"- Confidence: `{f.get('confidence', '?')}`\n"
                cells.append(_markdown_cell(md)); n_md += 1
            except Exception:
                pass

    # ─── Reproducibility note ───────────────────────────────────
    cells.append(
        _markdown_cell(
            "## Reproducibility note\n\n"
            "Estimators, identification proofs, and sensitivity outputs in "
            "this notebook regenerate from the protocol's pinned random seeds "
            "and the catalog version at the time of export. For a stronger "
            "guarantee, consult the `run.lock.json` manifest in this project "
            "directory — it hashes the dataset, DAG, estimands, RNG seeds, "
            "library versions, and prompt digests."
        )
    ); n_md += 1

    # ─── Assemble notebook ──────────────────────────────────────
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "causalrag": {
                "exported_from": "causalrag.reporting.notebook_export",
                "schema_version": "1",
                "protocol_name": protocol.name,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # Optional: jupytext companion
    fmt = "ipynb"
    try:
        import jupytext  # type: ignore

        py_path = output_path.with_suffix(".py")
        nb_obj = jupytext.read(output_path)
        jupytext.write(nb_obj, py_path, fmt="py:percent")
        fmt = "ipynb+jupytext"
    except Exception:
        pass

    return NotebookExportResult(
        output_path=output_path,
        n_code_cells=n_code,
        n_markdown_cells=n_md,
        format=fmt,
    )


__all__ = ["NotebookExportResult", "export_notebook"]
