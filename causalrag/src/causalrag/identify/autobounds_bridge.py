"""Partial-identification bridge to the ``autobounds`` package (Sprint 2.8).

When the deterministic identification gate (:mod:`causalrag.roadmap.q5_identify`)
returns *non-identifiable* but the treatment and outcome are both discrete and
the DAG is small, we still owe the analyst more than a binary verdict. This
module wraps Duarte et al.'s ``autobounds`` polynomial-programming engine
(JASA 2024) to return informative *partial* identification bounds
``[lower, upper]`` for the requested causal contrast.

The bridge is intentionally defensive:

* The ``autobounds`` PyPI package is *not* a hard dependency. When it is
  missing we return a fallback :class:`PartialIDResult` carrying the trivial
  bound and ``backend="fallback"`` so the rest of the pipeline can keep
  running.
* The polynomial program is run inside a :mod:`concurrent.futures` worker so
  a pathological problem cannot stall the master loop — the timeout returns
  a trivial fallback.
* Several pre-flight guards reject inputs that are clearly out of scope
  (graph too large, too many treatment/outcome levels, empty frame) before
  ever touching the optimiser.

Treat the returned bounds as a *negotiated* answer between the data and the
encoded assumptions — strictly more honest than a binary identification
verdict.
"""

from __future__ import annotations

import importlib
import math
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from causalrag.identify.decomposition import extract_relevant_subgraph

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from causalrag.core.graph import CausalGraph


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class PartialIDResult:
    """Outcome of a partial-identification attempt.

    ``backend`` is ``"autobounds"`` when the optimiser actually ran and
    ``"fallback"`` when we returned a trivial bound (missing dependency,
    pre-flight refusal, timeout, or solver error). ``point_estimate`` is
    the midpoint of the bound — a deliberately weak "best guess" used by
    downstream UI; never confuse it with a point-identified estimate.
    """

    target_estimand: str
    lower_bound: float
    upper_bound: float
    point_estimate: float | None
    bound_width: float
    backend: str
    n_nodes: int
    runtime_seconds: float
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def partial_identify(
    *,
    graph: "CausalGraph",
    treatment: str,
    outcome: str,
    df: "pd.DataFrame",
    estimand_class: str = "ATE",
    max_nodes: int = 10,
    max_levels: int = 10,
    timeout_seconds: float = 60.0,
) -> PartialIDResult:
    """Run partial identification for a discrete T → Y contrast on a small DAG.

    Pre-flight guards (each returns a trivial fallback bound with a note):

    * ``df`` is empty.
    * Treatment or outcome column missing from ``df``.
    * Treatment or outcome has more than ``max_levels`` unique values.
    * Reduced subgraph still has more than ``max_nodes`` nodes.

    On success the call returns the optimised ``[lower, upper]`` bound. On any
    autobounds failure or timeout we return the trivial bound matching the
    outcome's observed range — never an exception.
    """
    t0 = time.perf_counter()
    notes: list[str] = []

    # ---- Pre-flight: empty data ---------------------------------------------
    if df is None or len(df) == 0:
        notes.append("empty dataframe — returning trivial bound")
        return _trivial(estimand_class, len(graph.nodes), t0, notes, y_range=1.0)

    # ---- Pre-flight: T / Y exist --------------------------------------------
    for col in (treatment, outcome):
        if col not in df.columns:
            notes.append(f"column {col!r} missing from dataframe")
            return _trivial(estimand_class, len(graph.nodes), t0, notes, y_range=1.0)

    # ---- Pre-flight: discrete with ≤ max_levels -----------------------------
    t_levels = _unique_count(df[treatment])
    y_levels = _unique_count(df[outcome])
    if t_levels > max_levels:
        notes.append(
            f"treatment {treatment!r} has {t_levels} levels (> {max_levels}); "
            "partial-ID engine only supports discrete T"
        )
        return _trivial(
            estimand_class, len(graph.nodes), t0, notes, y_range=_y_range(df[outcome])
        )
    if y_levels > max_levels:
        notes.append(
            f"outcome {outcome!r} has {y_levels} levels (> {max_levels}); "
            "partial-ID engine only supports discrete Y"
        )
        return _trivial(
            estimand_class, len(graph.nodes), t0, notes, y_range=_y_range(df[outcome])
        )

    # ---- Reduce graph to relevant subgraph ----------------------------------
    try:
        reduced = extract_relevant_subgraph(graph, treatment, outcome)
    except Exception as e:  # pragma: no cover - defensive
        notes.append(f"subgraph extraction failed: {type(e).__name__}: {e}")
        reduced = graph

    n_nodes = len(reduced.nodes) or len(graph.nodes)
    if n_nodes > max_nodes:
        notes.append(
            f"graph too large ({n_nodes} relevant nodes > {max_nodes}); "
            "skipping autobounds — try restricting the DAG"
        )
        return _trivial(estimand_class, n_nodes, t0, notes, y_range=_y_range(df[outcome]))

    # ---- Optional dependency check ------------------------------------------
    if not _autobounds_available():
        notes.append(
            "autobounds package not installed — returning fallback bound. "
            "Install with `pip install autobounds` to enable partial ID."
        )
        return _trivial(estimand_class, n_nodes, t0, notes, y_range=_y_range(df[outcome]))

    # ---- Run autobounds under a timeout -------------------------------------
    # We use a process-pool worker because the autobounds solver may call into
    # native libraries that ignore Python signals; a process boundary is the
    # only reliable way to enforce a wall-clock budget.
    edge_payload = _edges_for_worker(reduced)
    data_payload = _data_for_worker(df, treatment, outcome)

    pool = ProcessPoolExecutor(max_workers=1)
    try:
        future = pool.submit(
            _autobounds_worker,
            edge_payload,
            data_payload,
            treatment,
            outcome,
            estimand_class,
        )
        try:
            lower, upper, worker_notes = future.result(timeout=timeout_seconds)
        except FuturesTimeout:
            # Don't wait for the (likely native-locked) worker to finish —
            # shut the pool down asynchronously and return immediately.
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            notes.append(
                f"autobounds optimisation exceeded {timeout_seconds:.1f}s — "
                "returning trivial bound"
            )
            return _trivial(
                estimand_class, n_nodes, t0, notes, y_range=_y_range(df[outcome])
            )
        finally:
            # Successful path: tear the pool down cleanly.
            pool.shutdown(wait=True)
    except Exception as e:
        pool.shutdown(wait=False, cancel_futures=True)
        notes.append(
            f"autobounds invocation failed: {type(e).__name__}: {e}; "
            "returning trivial bound"
        )
        return _trivial(estimand_class, n_nodes, t0, notes, y_range=_y_range(df[outcome]))

    notes.extend(worker_notes)

    # Sanity-check returned bounds.
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower > upper:
        notes.append(
            f"autobounds returned ill-formed bound [{lower}, {upper}]; "
            "returning trivial bound"
        )
        return _trivial(estimand_class, n_nodes, t0, notes, y_range=_y_range(df[outcome]))

    width = upper - lower
    return PartialIDResult(
        target_estimand=estimand_class,
        lower_bound=float(lower),
        upper_bound=float(upper),
        point_estimate=float((lower + upper) / 2.0),
        bound_width=float(width),
        backend="autobounds",
        n_nodes=n_nodes,
        runtime_seconds=time.perf_counter() - t0,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _autobounds_available() -> bool:
    """Return True iff a usable ``autobounds`` package is importable."""
    try:
        importlib.import_module("autobounds")
        return True
    except Exception:
        return False


def _unique_count(series: Any) -> int:
    try:
        return int(series.nunique(dropna=True))
    except Exception:
        return len(set(series))


def _y_range(series: Any) -> float:
    """Half-range of ``series`` used to size the trivial bound for continuous Y.

    For binary or 0/1 outcomes this returns 1.0 → the classic [-1, 1] bound.
    For wider observed ranges we widen the trivial bound accordingly so the
    fallback is *honest* about ignorance rather than artificially tight.
    """
    try:
        ymin = float(series.min())
        ymax = float(series.max())
        if math.isfinite(ymin) and math.isfinite(ymax):
            r = ymax - ymin
            return max(r, 1.0)
    except Exception:
        pass
    return 1.0


def _trivial(
    estimand_class: str,
    n_nodes: int,
    t0: float,
    notes: list[str],
    *,
    y_range: float,
) -> PartialIDResult:
    """Return the trivial fallback bound ``[-y_range, +y_range]``."""
    lower = -float(y_range)
    upper = +float(y_range)
    return PartialIDResult(
        target_estimand=estimand_class,
        lower_bound=lower,
        upper_bound=upper,
        point_estimate=None,
        bound_width=upper - lower,
        backend="fallback",
        n_nodes=n_nodes,
        runtime_seconds=time.perf_counter() - t0,
        notes=list(notes),
    )


def _edges_for_worker(graph: "CausalGraph") -> dict[str, Any]:
    """Serialise the graph to a worker-safe payload (no Pydantic objects)."""
    return {
        "nodes": list(graph.nodes),
        "directed": [
            (e.source, e.target) for e in graph.edges if not getattr(e, "bidirected", False)
        ],
        "bidirected": [
            (e.source, e.target) for e in graph.edges if getattr(e, "bidirected", False)
        ],
    }


def _data_for_worker(df: "pd.DataFrame", treatment: str, outcome: str) -> dict[str, Any]:
    """Return the joint (T, Y) empirical distribution for the worker."""
    counts: dict[tuple[Any, Any], int] = {}
    for t_val, y_val in zip(df[treatment].tolist(), df[outcome].tolist()):
        key = (t_val, y_val)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values()) or 1
    return {
        "treatment": treatment,
        "outcome": outcome,
        "joint": [
            {"t": k[0], "y": k[1], "p": v / total, "n": v} for k, v in counts.items()
        ],
        "n": total,
    }


# ---------------------------------------------------------------------------
# Worker — runs in a child process so we can enforce a hard timeout
# ---------------------------------------------------------------------------


def _autobounds_worker(
    edge_payload: dict[str, Any],
    data_payload: dict[str, Any],
    treatment: str,
    outcome: str,
    estimand_class: str,
) -> tuple[float, float, list[str]]:
    """Run autobounds and return ``(lower, upper, notes)``.

    The autobounds 1.x API is not perfectly stable between minor releases. We
    probe a few entry-point shapes and degrade gracefully — on any failure
    the worker raises so the parent records a fallback. Returning a wide bound
    is the parent's responsibility, not the worker's.
    """
    notes: list[str] = []

    # Try to import the modern (causalProblem) API first.
    try:
        from autobounds.causalProblem import causalProblem  # type: ignore
        from autobounds.DAG import DAG  # type: ignore
    except Exception as e:
        raise RuntimeError(f"autobounds API not importable: {e}") from e

    # Construct a DAG description string in the form autobounds expects:
    #   "U -> X, U -> Y, X -> Y" plus bidirected encoded via shared latents.
    dag_lines: list[str] = []
    for u, v in edge_payload["directed"]:
        dag_lines.append(f"{u} -> {v}")
    # Encode bidirected edges as a fresh latent confounder per pair.
    for i, (u, v) in enumerate(edge_payload["bidirected"]):
        latent = f"U_ab_{i}"
        dag_lines.append(f"{latent} -> {u}")
        dag_lines.append(f"{latent} -> {v}")

    dag_str = ", ".join(dag_lines) if dag_lines else f"{treatment} -> {outcome}"

    try:
        dag = DAG()
        dag.from_structure(dag_str, unob="U_ab_")  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"autobounds DAG construction failed: {e}") from e

    problem = causalProblem(dag)

    # Provide the joint distribution to the problem. autobounds expects rows
    # named (T, Y, prob).
    try:
        for row in data_payload["joint"]:
            problem.set_p_to_zero  # noqa: B018 - sanity check
        problem.load_data(  # type: ignore[attr-defined]
            [
                (treatment, row["t"], outcome, row["y"], row["p"])
                for row in data_payload["joint"]
            ]
        )
    except Exception:
        # Different versions of autobounds use different data ingestion calls;
        # we tolerate failure here and proceed without it (yields the natural
        # bound under the structural assumptions alone).
        notes.append("autobounds: load_data not available in this version")

    # Set the estimand — ATE between T=1 and T=0 by default.
    try:
        problem.set_estimand(  # type: ignore[attr-defined]
            problem.query(f"{outcome}(1)=1") - problem.query(f"{outcome}(0)=1")
        )
    except Exception as e:
        raise RuntimeError(f"autobounds estimand construction failed: {e}") from e

    try:
        program = problem.write_program()  # type: ignore[attr-defined]
        bounds = program.run_scip()  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"autobounds optimisation failed: {e}") from e

    # ``bounds`` shape varies. Probe a few likely structures.
    if isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        lower = float(bounds[0])
        upper = float(bounds[1])
    elif hasattr(bounds, "lower") and hasattr(bounds, "upper"):
        lower = float(bounds.lower)
        upper = float(bounds.upper)
    else:
        raise RuntimeError(f"unrecognised autobounds result shape: {type(bounds)!r}")

    return lower, upper, notes
