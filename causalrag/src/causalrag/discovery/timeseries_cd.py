"""Tigramite back-end for time-series causal discovery (Sprint 7.1).

The standard constraint-based PC/FCI back-end (``discovery.ci_backend``,
``estimators.rbridge.discovery_r``) silently ignores autocorrelation:
it treats each row as an i.i.d. observation, conflates contemporaneous
and lagged dependencies, and produces wildly inflated false-positive
edge sets on panel / longitudinal data. Once the discovery layer raises
:class:`DataFlag.PANEL_STRUCTURE` or :class:`DataFlag.LONGITUDINAL`, the
master loop should route DAG discovery here instead.

This module wraps three algorithms from Jakob Runge's *tigramite*
package, each appropriate for a different regime:

- ``pcmci_plus`` (Runge 2020, UAI) — single multivariate time series,
  no latent confounders. Combines a Markov-set-style PC pre-selection
  on lagged parents with an MCI-style contemporaneous orientation step;
  controls FDR at the cost of conservatism. Default.
- ``lpcmci`` (Gerhardus & Runge 2020, NeurIPS) — Latent-PCMCI; the
  FCI-style sibling that admits unobserved common drivers. Returns a
  PAG with bidirected edges (``<->``) and "o" mark uncertainties; we
  fold the bidirected ones onto ``CausalEdge.bidirected``.
- ``j_pcmci_plus`` (Günther, Ninad & Runge 2023, UAI) — joint PCMCI+
  for panels: multiple short series sharing a single underlying
  process. Requires a ``unit_column`` to split rows into per-unit
  trajectories.

Tigramite is treated as an optional dependency. The module top-level is
import-safe without it; the actual import happens inside
:func:`discover_timeseries_dag`. Callers without tigramite installed
get an :class:`ImportError` only at call time, mirroring the
``rbridge.*`` modules' "require on use" pattern.

Output is a :class:`CausalGraph` rooted at ``rank=0`` (data-derived,
same provenance tier as ``discover_dag`` from the bnlearn bridge). Each
edge carries a ``note`` of the form ``"<alg> lag-<k>"`` (e.g.
``"pcmci_plus lag-2"`` or ``"pcmci_plus contemporaneous"``) so the DAG
carousel and downstream identification audits can see *when* each
mechanism fires — temporal precedence is the strongest single piece of
evidence we can attach to a discovered edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole

if TYPE_CHECKING:  # pragma: no cover - typing-only import
    pass


TigramiteAlgorithm = Literal["pcmci_plus", "lpcmci", "j_pcmci_plus"]
CITest = Literal["parcorr", "gpdc", "cmiknn"]


def _make_ci_test(ci_test: CITest) -> Any:
    """Instantiate a tigramite ``CondIndTest`` lazily.

    Each test pulls in its own optional dependency stack
    (``gpdc`` needs ``dcor``, ``cmiknn`` needs ``sklearn`` neighbours
    plus shuffle tests), so we defer the import until the caller has
    actually selected one.
    """
    if ci_test == "parcorr":
        from tigramite.independence_tests.parcorr import ParCorr

        return ParCorr(significance="analytic")
    if ci_test == "gpdc":
        from tigramite.independence_tests.gpdc import GPDC

        return GPDC(significance="analytic")
    if ci_test == "cmiknn":
        from tigramite.independence_tests.cmiknn import CMIknn

        return CMIknn(significance="shuffle_test")
    raise ValueError(f"unknown ci_test {ci_test!r}")


def _split_panel(
    df: pd.DataFrame, *, time_column: str, unit_column: str | None
) -> tuple[list[np.ndarray], list[str]]:
    """Return per-unit ``(T_i, N)`` arrays plus the variable-name list.

    Variables are every column except ``time_column`` / ``unit_column``;
    rows are sorted by time within each unit so tigramite sees a proper
    monotonically-ordered trajectory. Non-numeric variables are cast to
    ``float`` via ``pd.to_numeric`` (``coerce``) — tigramite is numeric-
    only, and any column that fails to coerce becomes NaN, which the
    PCMCI ``missing_flag`` handling will skip.
    """
    if time_column not in df.columns:
        raise ValueError(f"time_column {time_column!r} not in df columns")
    if unit_column is not None and unit_column not in df.columns:
        raise ValueError(f"unit_column {unit_column!r} not in df columns")

    var_cols = [
        c for c in df.columns if c != time_column and c != unit_column
    ]
    if not var_cols:
        raise ValueError(
            "no variable columns left after removing time / unit columns"
        )

    if unit_column is None:
        ordered = df.sort_values(by=time_column)
        arr = ordered[var_cols].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )
        return [arr], var_cols

    arrays: list[np.ndarray] = []
    for _, sub in df.groupby(unit_column, sort=True):
        ordered = sub.sort_values(by=time_column)
        arr = ordered[var_cols].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )
        arrays.append(arr)
    if not arrays:
        raise ValueError("unit_column produced zero groups")
    return arrays, var_cols


def _build_dataframe(
    arrays: list[np.ndarray], var_names: list[str], *, multiple: bool
):
    """Wrap per-unit arrays in a ``tigramite.data_processing.DataFrame``.

    Single-series callers (``unit_column=None`` and PCMCI+/LPCMCI) get
    ``analysis_mode='single'`` over the lone trajectory; panel callers
    (J-PCMCI+ or anything multi-unit) get ``analysis_mode='multiple'``
    with a dict-keyed dataset so tigramite can vary ``T_i`` per unit.
    """
    from tigramite import data_processing as pp

    # tigramite uses ``np.nan`` as the implicit missing flag when
    # ``missing_flag`` is set — pass that through so coerced non-numerics
    # don't crash the CI test.
    if not multiple:
        return pp.DataFrame(
            arrays[0],
            var_names=var_names,
            missing_flag=np.nan,
            analysis_mode="single",
        )
    data_dict = {i: arr for i, arr in enumerate(arrays)}
    return pp.DataFrame(
        data_dict,
        var_names=var_names,
        missing_flag=np.nan,
        analysis_mode="multiple",
    )


def _graph_array_to_edges(
    graph: np.ndarray,
    var_names: list[str],
    *,
    algorithm: str,
) -> list[CausalEdge]:
    """Decode tigramite's ``(N, N, tau_max+1)`` link-mark array.

    Each cell ``graph[i, j, tau]`` is a string mark:

    - ``""`` — no link at this lag
    - ``"-->"`` / ``"<--"`` — directed link i→j or i←j (tau=0 is
      contemporaneous; tau>0 means i at time t-tau → j at time t)
    - ``"<->"`` — bidirected (latent common cause, FCI/LPCMCI only)
    - ``"o-o"`` / ``"o->"`` / ``"<-o"`` — circle endpoints, "we don't
      know the orientation". We emit them as directed best-guess from
      the non-circle end where one exists, else drop.

    PCMCI+ writes each lagged link once at ``(i, j, tau)``; for
    ``tau>0`` it does *not* mirror at ``(j, i, tau)``, so we walk every
    cell once. Contemporaneous links *are* mirrored, so we dedupe with a
    ``seen`` set to avoid emitting both directions of a single ``-->``.
    """
    edges: list[CausalEdge] = []
    seen: set[tuple[str, str, int, bool]] = set()
    n = len(var_names)
    if graph.ndim != 3 or graph.shape[0] != n or graph.shape[1] != n:
        return edges
    tau_max = graph.shape[2] - 1

    def _note(tau: int) -> str:
        if tau == 0:
            return f"{algorithm} contemporaneous"
        return f"{algorithm} lag-{tau}"

    def _add(src: str, dst: str, tau: int, bidirected: bool) -> None:
        # Canonicalise bidirected so we don't store both directions.
        if bidirected and src > dst:
            src, dst = dst, src
        key = (src, dst, tau, bidirected)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            CausalEdge(
                source=src,
                target=dst,
                bidirected=bidirected,
                llm_proposed=False,
                note=_note(tau),
            )
        )

    for i in range(n):
        for j in range(n):
            for tau in range(tau_max + 1):
                mark = graph[i, j, tau]
                if not mark:
                    continue
                src, dst = var_names[i], var_names[j]
                if mark == "-->":
                    _add(src, dst, tau, False)
                elif mark == "<--":
                    _add(dst, src, tau, False)
                elif mark == "<->":
                    _add(src, dst, tau, True)
                elif mark in ("o->", "x->"):
                    # circle/cross at i, arrowhead at j → i causes j
                    # under the partial-knowledge reading.
                    _add(src, dst, tau, False)
                elif mark in ("<-o", "<-x"):
                    _add(dst, src, tau, False)
                # ``o-o``, ``x-x``, ``o-x``, ``x-o`` are fully
                # ambiguous; skip rather than guessing a direction.
    return edges


def discover_timeseries_dag(
    df: pd.DataFrame,
    *,
    time_column: str,
    unit_column: str | None = None,
    algorithm: TigramiteAlgorithm = "pcmci_plus",
    tau_max: int = 3,
    alpha: float = 0.05,
    ci_test: CITest = "parcorr",
) -> CausalGraph:
    """Discover a (possibly cyclic-looking-but-temporally-resolved) DAG
    from time-series data via tigramite.

    Parameters
    ----------
    df:
        Long-format frame with one row per (unit, timestep). Variable
        columns may be numeric or coercible-to-numeric; anything else
        becomes ``NaN`` and is skipped by the CI test.
    time_column:
        Column used to sort rows within each unit. Values need only be
        sortable; the absolute scale is irrelevant — tigramite indexes
        by row position once sorted.
    unit_column:
        Optional grouping column for panel data. When ``None``, ``df``
        is treated as a single trajectory. Required (in spirit) for
        ``j_pcmci_plus`` — passing ``None`` collapses J-PCMCI+ to a
        single-unit run, which is well-defined but wasteful.
    algorithm:
        One of ``"pcmci_plus"`` (default), ``"lpcmci"``, or
        ``"j_pcmci_plus"``. See the module docstring for selection
        guidance.
    tau_max:
        Maximum lag (in row-positions) the algorithm is allowed to
        consider. Default 3 mirrors the tigramite tutorial; ``tau_max=0``
        collapses to contemporaneous-only and is rarely what you want.
    alpha:
        Significance level for the conditional-independence test
        (``pc_alpha`` in tigramite parlance).
    ci_test:
        ``"parcorr"`` (linear/Gaussian, fast), ``"gpdc"`` (Gaussian-
        process distance-correlation, nonlinear, needs ``dcor``), or
        ``"cmiknn"`` (kNN conditional mutual information, fully
        nonparametric, slow). Each is lazy-imported.

    Returns
    -------
    CausalGraph with ``rank=0`` (data-derived). Edge ``note`` carries
    the lag annotation, e.g. ``"pcmci_plus lag-2"``.
    """
    # Lazy import — keeps module-level import safe when tigramite is
    # not installed (mirrors the ``rbridge`` modules' lazy-R pattern).
    from tigramite.pcmci import PCMCI

    arrays, var_names = _split_panel(
        df, time_column=time_column, unit_column=unit_column
    )

    cond_ind_test = _make_ci_test(ci_test)

    if algorithm == "j_pcmci_plus":
        from tigramite.jpcmciplus import JPCMCIplus

        dataframe = _build_dataframe(arrays, var_names, multiple=True)
        # All variables in ``df`` (minus time/unit) are system
        # variables; the unit_column itself is the implicit
        # space_context but tigramite handles that via analysis_mode +
        # the multi-dataset dict. We classify every variable as
        # "system" since the unit_column was stripped out by
        # ``_split_panel``.
        node_classification = {i: "system" for i in range(len(var_names))}
        runner = JPCMCIplus(
            node_classification=node_classification,
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=0,
        )
        results = runner.run_jpcmciplus(tau_max=tau_max, pc_alpha=alpha)
        graph = results["graph"]
        edges = _graph_array_to_edges(graph, var_names, algorithm=algorithm)
    elif algorithm == "lpcmci":
        from tigramite.lpcmci import LPCMCI

        # LPCMCI is single-series only; if a panel was passed we
        # concatenate by stacking units along time. This is the
        # conventional fallback when an analyst wants latent-confounder
        # awareness but doesn't have J-LPCMCI available.
        multiple = len(arrays) > 1
        if multiple:
            dataframe = _build_dataframe(arrays, var_names, multiple=True)
        else:
            dataframe = _build_dataframe(arrays, var_names, multiple=False)
        runner = LPCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=0,
        )
        results = runner.run_lpcmci(tau_max=tau_max, pc_alpha=alpha)
        graph = results["graph"]
        edges = _graph_array_to_edges(graph, var_names, algorithm=algorithm)
    elif algorithm == "pcmci_plus":
        multiple = len(arrays) > 1
        dataframe = _build_dataframe(arrays, var_names, multiple=multiple)
        runner = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=0,
        )
        results = runner.run_pcmciplus(tau_max=tau_max, pc_alpha=alpha)
        graph = results["graph"]
        edges = _graph_array_to_edges(graph, var_names, algorithm=algorithm)
    else:
        raise ValueError(f"unknown algorithm {algorithm!r}")

    roles: dict[str, VariableRole] = {
        c: VariableRole.CONFOUNDER for c in var_names
    }

    return CausalGraph(
        nodes=tuple(var_names),
        edges=tuple(edges),
        roles=roles,
        rank=0,
    )


__all__ = ["discover_timeseries_dag"]
