"""Time-varying / panel DAG layering (PDD §33 — Sprint 6.5.3).

A :class:`TimeVaryingDAG` wraps a base :class:`CausalGraph` and unrolls it
across ``T`` discrete time periods, generating time-indexed nodes
(``X_t0``, ``X_t1``, ...) and the appropriate temporal edges. This is the
data structure that longitudinal estimators (LTMLE, parametric g-formula,
``lmtp`` longitudinal) consume when the master loop flags
``TIME_VARYING_TREATMENT`` or ``LONGITUDINAL``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole


@dataclass(frozen=True)
class TimeIndexedNode:
    """A node materialised at a specific time index."""

    name: str  # base variable name, e.g. 'A' for treatment
    time_index: int  # 0, 1, 2, ...
    qualified_name: str  # e.g. 'A_t0'


def _qualify(name: str, t: int) -> str:
    return f"{name}_t{t}"


@dataclass
class TimeVaryingDAG:
    """Time-layered DAG built from a base CausalGraph + T time periods.

    Rules:
    - For every base node, materialise T copies (one per time period).
    - For every base directed edge ``(u, v)``, add a contemporaneous copy
      at every time index (when ``contemporaneous=True``).
    - When ``u`` is a treatment and ``v`` is an outcome, add lagged copies
      ``u_t -> v_{t+k}`` for each ``k`` in ``treatment_outcome_lags``.
    - Confounders evolve with lag-1 autoregression by default
      (``X_t0 -> X_t1 -> X_t2 -> ...``) when ``confounder_persistence`` is
      True.
    - Bidirected (latent confounder) edges propagate at every time index.

    Used by LTMLE / parametric g-formula / lmtp longitudinal estimators
    that need a time-indexed adjustment-set graph.
    """

    base_graph: CausalGraph
    n_periods: int
    contemporaneous: bool = True  # T_t -> Y_t edges
    lag_1_autoregression: bool = True
    treatment_outcome_lags: tuple[int, ...] = (0, 1)
    confounder_persistence: bool = True  # X_t -> X_{t+1}
    _materialised: CausalGraph | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_periods < 1:
            raise ValueError("n_periods must be >= 1")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def time_indexed_nodes(self, base_name: str) -> tuple[TimeIndexedNode, ...]:
        """All time copies of a base node."""
        if base_name not in self.base_graph.nodes:
            raise KeyError(f"{base_name!r} is not a node in the base graph")
        return tuple(
            TimeIndexedNode(
                name=base_name,
                time_index=t,
                qualified_name=_qualify(base_name, t),
            )
            for t in range(self.n_periods)
        )

    def materialise(self) -> CausalGraph:
        """Build and return the full time-layered CausalGraph."""
        if self._materialised is not None:
            return self._materialised

        treatments = set(self.base_graph.variables_with_role(VariableRole.TREATMENT))
        outcomes = set(self.base_graph.variables_with_role(VariableRole.OUTCOME))
        confounders = set(self.base_graph.variables_with_role(VariableRole.CONFOUNDER))

        # ---- Nodes & roles ------------------------------------------------
        nodes: list[str] = []
        roles: dict[str, VariableRole] = {}
        for base in self.base_graph.nodes:
            base_role = self.base_graph.roles.get(base, VariableRole.AUXILIARY)
            for t in range(self.n_periods):
                q = _qualify(base, t)
                nodes.append(q)
                roles[q] = base_role

        # ---- Edges --------------------------------------------------------
        edges: list[CausalEdge] = []
        # Track (source, target, bidirected) to avoid duplicates.
        seen: set[tuple[str, str, bool]] = set()

        def _add_edge(src: str, tgt: str, *, bidirected: bool = False,
                      template: CausalEdge | None = None) -> None:
            key = (src, tgt, bidirected)
            if key in seen:
                return
            seen.add(key)
            if template is not None:
                edges.append(
                    CausalEdge(
                        source=src,
                        target=tgt,
                        bidirected=bidirected,
                        llm_proposed=template.llm_proposed,
                        ci_test_passed=template.ci_test_passed,
                        note=template.note,
                    )
                )
            else:
                edges.append(CausalEdge(source=src, target=tgt, bidirected=bidirected))

        for e in self.base_graph.edges:
            u, v = e.source, e.target
            if e.bidirected:
                # Bidirected edges propagate at every time index.
                for t in range(self.n_periods):
                    _add_edge(_qualify(u, t), _qualify(v, t),
                              bidirected=True, template=e)
                continue

            is_treat_out = (u in treatments and v in outcomes)

            if is_treat_out:
                # Treatment -> outcome: lagged copies per
                # treatment_outcome_lags. Skip the default contemporaneous
                # duplicate when 0 is not in the lag list.
                for k in self.treatment_outcome_lags:
                    if k < 0:
                        continue
                    for t in range(self.n_periods - k):
                        _add_edge(_qualify(u, t), _qualify(v, t + k),
                                  template=e)
            else:
                # General contemporaneous edge.
                if self.contemporaneous:
                    for t in range(self.n_periods):
                        _add_edge(_qualify(u, t), _qualify(v, t),
                                  template=e)

        # ---- Autoregressive / persistence edges ---------------------------
        # Confounder persistence has dedicated semantics; lag_1_autoregression
        # is the catch-all flag enabling X_t -> X_{t+1} for every variable.
        for base in self.base_graph.nodes:
            base_role = self.base_graph.roles.get(base, VariableRole.AUXILIARY)
            enable = False
            if base in confounders:
                enable = self.confounder_persistence
            else:
                enable = self.lag_1_autoregression
            # Confounder gets persistence when confounder_persistence is set,
            # regardless of lag_1_autoregression. (Tests rely on disabling
            # confounder_persistence to remove X_t -> X_{t+1}.)
            if base in confounders and not self.confounder_persistence:
                enable = False
            if not enable:
                continue
            for t in range(self.n_periods - 1):
                _add_edge(_qualify(base, t), _qualify(base, t + 1))
            # Touch role to avoid unused-variable; the role inheritance is
            # already wired in via the roles dict above.
            _ = base_role

        graph = CausalGraph(
            nodes=tuple(nodes),
            edges=tuple(edges),
            roles=roles,
            rank=self.base_graph.rank,
        )
        self._materialised = graph
        return graph

    def adjustment_set_at_time(self, t: int) -> frozenset[str]:
        """Nodes to condition on to estimate treatment-at-time-t's effect on
        outcome-at-time-t.

        Returns the union of:
        - All confounders at time ``<= t`` (history of measured confounders).
        - All prior treatments (``A_t'`` for ``t' < t``) — needed for
          sequential ignorability in g-methods.
        - All prior outcomes (``Y_t'`` for ``t' < t``) when present.
        """
        if t < 0 or t >= self.n_periods:
            raise ValueError(
                f"t={t} is out of range for n_periods={self.n_periods}"
            )
        confounders = self.base_graph.variables_with_role(VariableRole.CONFOUNDER)
        treatments = self.base_graph.variables_with_role(VariableRole.TREATMENT)
        outcomes = self.base_graph.variables_with_role(VariableRole.OUTCOME)

        adj: set[str] = set()
        # Confounders up to and including t.
        for x in confounders:
            for k in range(t + 1):
                adj.add(_qualify(x, k))
        # Prior treatments (strictly before t).
        for a in treatments:
            for k in range(t):
                adj.add(_qualify(a, k))
        # Prior outcomes (strictly before t) — past Y is part of the
        # time-varying history that confounds future treatment.
        for y in outcomes:
            for k in range(t):
                adj.add(_qualify(y, k))
        return frozenset(adj)
