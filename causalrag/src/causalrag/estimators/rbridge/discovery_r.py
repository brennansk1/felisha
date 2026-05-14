"""``bnlearn`` wrapper — Bayesian-network structure learning for the
discovery phase.

Complements the LLM-driven DAG proposal with a data-driven alternative:
PC algorithm, Grow-Shrink, Max-Min Parents-and-Children, hill-climbing
score-based search, etc. Useful as a Layer-4 anchor — does the
LLM-proposed DAG match what a constraint-based discovery algorithm
finds in the data?

Returns a CausalGraph with rank=0 (data-derived; ranked beside the
LLM-proposed graphs in the candidate-DAG carousel).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.estimators.rbridge._r import converter, r_session, require


def discover_dag(
    df: pd.DataFrame,
    *,
    treatment: str | None = None,
    outcome: str | None = None,
    algorithm: Literal["pc", "gs", "hc", "tabu", "iamb", "mmhc"] = "pc",
    blacklist: list[tuple[str, str]] | None = None,
    whitelist: list[tuple[str, str]] | None = None,
) -> CausalGraph:
    """Run a bnlearn structure-learning algorithm and return a CausalGraph.

    Algorithms:
    - ``pc``: PC algorithm (constraint-based, Spirtes-Glymour-Scheines)
    - ``gs``: Grow-Shrink (constraint-based)
    - ``hc``: Hill-Climbing (score-based)
    - ``tabu``: Tabu Search (score-based)
    - ``iamb``: Incremental Association MB
    - ``mmhc``: Max-Min Hill-Climbing (hybrid)
    """
    require("bnlearn")
    ro = r_session()
    work = df.dropna()
    with converter():
        ro.globalenv["df_"] = ro.conversion.py2rpy(work)
    # bnlearn needs factor types for discrete; we'll cast everything numeric to a
    # standardized scale and feed via continuous algos when available.
    is_continuous = all(
        pd.api.types.is_numeric_dtype(work[c]) and work[c].nunique() > 5 for c in work.columns
    )
    if is_continuous and algorithm in ("hc", "tabu"):
        algo_fn = {"hc": "hc", "tabu": "tabu"}[algorithm]
        ro.r(f"net_ <- bnlearn::{algo_fn}(df_)")
    else:
        # Constraint-based — discretize if needed
        ro.r(
            "df_disc <- as.data.frame(lapply(df_, function(x) "
            "if (is.numeric(x)) cut(x, breaks=quantile(x, probs=seq(0,1,0.25), na.rm=TRUE), include.lowest=TRUE) else as.factor(x)))"
        )
        ro.r(f"net_ <- bnlearn::{algorithm}(df_disc)")

    edges_r = ro.r("bnlearn::arcs(net_)")
    # Convert R matrix → Python list of tuples
    n_edges = int(list(ro.r("nrow(bnlearn::arcs(net_))"))[0])
    edges: list[tuple[str, str]] = []
    if n_edges > 0:
        from_col = list(ro.r("bnlearn::arcs(net_)[,1]"))
        to_col = list(ro.r("bnlearn::arcs(net_)[,2]"))
        edges = list(zip(from_col, to_col))

    roles: dict[str, VariableRole] = {}
    if treatment:
        roles[treatment] = VariableRole.TREATMENT
    if outcome:
        roles[outcome] = VariableRole.OUTCOME
    for c in work.columns:
        if c not in roles:
            roles[c] = VariableRole.CONFOUNDER

    return CausalGraph(
        nodes=tuple(work.columns),
        edges=tuple(
            CausalEdge(source=s, target=t, llm_proposed=False, note=f"bnlearn::{algorithm}")
            for s, t in edges
        ),
        roles=roles,
        rank=0,  # data-derived; rank=0 distinguishes from LLM-proposed (rank≥1)
    )


def discover_markov_boundary(
    df: pd.DataFrame,
    *,
    target: str,
    method: Literal["iamb", "fast.iamb", "inter.iamb", "iamb.fdr", "hpc", "mmpc", "si.hiton.pc"] = "iamb",
    alpha: float = 0.05,
) -> dict[str, object]:
    """Discover the Markov boundary of ``target`` via bnlearn.

    The Markov boundary MB(T) is the minimal subset S of V \\ {T} such
    that T ⊥ V \\ (S ∪ {T}) | S. Under faithfulness it equals the
    Markov blanket (parents, children, spouses) in the underlying DAG
    — i.e., the minimal sufficient adjustment + descendant set for any
    predictive query on T.

    This wrapper calls bnlearn's MB algorithms directly (faster +
    simpler than learning the whole DAG when you only care about one
    target). Output is consumed by the discovery layer as a cross-
    check against the LLM investigator's CONFOUNDER labels.

    Methods (all from bnlearn):
    - ``iamb`` — Incremental Association MB (Tsamardinos 2003)
    - ``fast.iamb`` — fast variant; less aggressive shrinking
    - ``inter.iamb`` — interleaved; better in low-power regimes
    - ``iamb.fdr`` — FDR-controlled (Pena 2008) — recommended for
      high-dim/low-sample where multiple-testing matters
    - ``hpc`` — Hybrid Parents-and-Children (returns MB via union)
    - ``mmpc`` — Max-Min PC (parents-and-children only; NOT the full MB)
    - ``si.hiton.pc`` — semi-interleaved HITON-PC

    Returns
    -------
    dict with keys:
    - ``target``: the column queried
    - ``mb``: list[str] — discovered Markov-boundary columns
    - ``method``: the algorithm used
    - ``alpha``: significance level
    - ``n``: rows used (after dropna)
    - ``test``: name of the conditional-independence test used
    """
    require("bnlearn")
    ro = r_session()
    work = df.dropna()
    if target not in work.columns:
        raise ValueError(f"target {target!r} not in df columns")

    with converter():
        ro.globalenv["df_"] = ro.conversion.py2rpy(work)

    # Detect continuous vs discrete for bnlearn's test selection.
    is_continuous = all(
        pd.api.types.is_numeric_dtype(work[c]) and work[c].nunique() > 5
        for c in work.columns
    )
    # bnlearn's MB-learning functions return a full `bn` object — we
    # learn the network on the dataset, then extract the MB of the
    # target node via `learned$nodes[[target]]$mb`.
    if is_continuous:
        ro.r(
            f'net_ <- bnlearn::{method}(df_, test = "cor", alpha = {alpha})'
        )
        test_name = "cor"
    else:
        ro.r(
            "df_disc <- as.data.frame(lapply(df_, function(x) "
            "if (is.numeric(x)) cut(x, breaks=quantile(x, probs=seq(0,1,0.25), na.rm=TRUE), "
            "include.lowest=TRUE) else as.factor(x)))"
        )
        ro.r(
            f'net_ <- bnlearn::{method}(df_disc, test = "mi", alpha = {alpha})'
        )
        test_name = "mi"

    # Extract MB of `target`. bnlearn::mb(bn, target) returns a character
    # vector. Wrap in `as.character` to guarantee plain strings.
    mb_cols = list(ro.r(f'as.character(bnlearn::mb(net_, "{target}"))'))
    return {
        "target": target,
        "mb": [str(c) for c in mb_cols],
        "method": method,
        "alpha": alpha,
        "n": int(len(work)),
        "test": test_name,
    }


__all__ = ["discover_dag", "discover_markov_boundary"]
