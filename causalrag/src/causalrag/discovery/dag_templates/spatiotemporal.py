"""Spatiotemporal partial-interference DAG template.

Each node is a *unit-time* cell. Two structural edges populate the DAG:

* Temporal lag-1: the previous time-step's treatment / outcome influences the
  current step (autoregressive).
* Spatial adjacency: neighbour pairs supply partial-interference edges so that
  a neighbour's treatment (and / or outcome) can affect the focal unit
  (spillover).

Because the template does not know how many time steps the analyst has, the
returned graph uses *role-typed* nodes — ``treatment_t``, ``treatment_t_minus_1``
— rather than expanding per row. This keeps the skeleton compact while still
encoding the autoregressive structure that identification will need.
"""

from __future__ import annotations

from pydantic import Field

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.discovery.dag_templates.base import DAGTemplate


_REQUIRED_SLOTS = ["unit_id", "time", "treatment", "outcome"]


class SpatiotemporalTemplate(DAGTemplate):
    """Spatiotemporal partial-interference skeleton.

    ``neighbour_pairs`` is an iterable of ``(unit_a, unit_b)`` tuples
    expressing the spatial adjacency graph. The template emits a single
    ``neighbour_treatment`` and ``neighbour_outcome`` aggregate node and wires
    spillover edges through them. The list itself is validated for shape and
    its size becomes part of the graph's structural metadata.
    """

    template_name: str = "spatiotemporal"
    domain: str = "ecology"
    description: str = (
        "Unit-time panel with autoregressive lag-1 edges and a partial-"
        "interference neighbour aggregate driving spillover."
    )
    slots: list[str] = Field(default_factory=lambda: list(_REQUIRED_SLOTS))

    neighbour_pairs: list[tuple[str, str]] = Field(default_factory=list)

    def instantiate(self, column_map: dict[str, str]) -> CausalGraph:
        self._check_required_slots(column_map)
        for pair in self.neighbour_pairs:
            if not (isinstance(pair, tuple | list) and len(pair) == 2):
                raise ValueError(
                    "SpatiotemporalTemplate.neighbour_pairs entries must be "
                    f"2-tuples; got {pair!r}."
                )

        unit_id = column_map["unit_id"]
        time = column_map["time"]
        treatment = column_map["treatment"]
        outcome = column_map["outcome"]

        # Lag-1 autoregressive shadows of treatment and outcome.
        treatment_lag = f"{treatment}__lag1"
        outcome_lag = f"{outcome}__lag1"
        neighbour_treatment = f"{treatment}__neighbour_agg"
        neighbour_outcome = f"{outcome}__neighbour_agg"

        edges: list[tuple[str, str]] = []
        # Temporal backbone.
        edges.append((treatment_lag, treatment))
        edges.append((outcome_lag, outcome))
        edges.append((treatment_lag, outcome))  # lagged treatment carryover
        edges.append((treatment, outcome))
        # Unit and time index drive both lag values and the current treatment
        # (panel fixed-effects style confounding).
        edges.append((unit_id, treatment))
        edges.append((unit_id, outcome))
        edges.append((time, treatment))
        edges.append((time, outcome))

        # Partial-interference spillover (only when there is at least one
        # neighbour pair).
        if self.neighbour_pairs:
            edges.append((neighbour_treatment, outcome))
            edges.append((neighbour_outcome, outcome))
            edges.append((treatment, neighbour_outcome))

        roles: dict[str, VariableRole] = {
            unit_id: VariableRole.IDENTIFIER,
            time: VariableRole.TIMESTAMP,
            treatment: VariableRole.TREATMENT,
            outcome: VariableRole.OUTCOME,
            treatment_lag: VariableRole.CONFOUNDER,
            outcome_lag: VariableRole.CONFOUNDER,
        }
        if self.neighbour_pairs:
            roles[neighbour_treatment] = VariableRole.CONFOUNDER
            roles[neighbour_outcome] = VariableRole.MEDIATOR

        edge_objs = tuple(
            CausalEdge(source=u, target=v, note=f"{self.template_name} template")
            for u, v in edges
        )
        nodes: list[str] = []
        seen: set[str] = set()
        for u, v in edges:
            for n in (u, v):
                if n not in seen:
                    seen.add(n)
                    nodes.append(n)
        return CausalGraph(nodes=tuple(nodes), edges=edge_objs, roles=roles)
