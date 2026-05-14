"""Touchpoint-sequence attribution DAG template.

Sequential mediation: each touchpoint mediates the effect of the previous one
on the eventual conversion. For ordered touchpoints ``t1, t2, ..., tK`` and a
``conversion`` outcome:

    t1 -> t2 -> t3 -> ... -> tK -> conversion

Each ``t_i`` (i < K) also has a direct edge to ``conversion`` to allow the
estimator to distinguish direct from indirect (through later touchpoints)
effects.
"""

from __future__ import annotations

from pydantic import Field

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.discovery.dag_templates.base import DAGTemplate


_REQUIRED_SLOTS = ["conversion"]


class AttributionTemplate(DAGTemplate):
    """Sequential-touchpoint attribution skeleton.

    ``touchpoint_columns`` is an ordered list of column names (resolved or
    slot-style); the template treats ``touchpoint_columns[0]`` as the root
    treatment and every later touchpoint as a mediator that also has a direct
    arrow to ``conversion``.
    """

    template_name: str = "attribution"
    domain: str = "marketing"
    description: str = (
        "Ordered touchpoint sequence with each touchpoint mediating the "
        "previous and also pointing directly at conversion."
    )
    slots: list[str] = Field(default_factory=lambda: list(_REQUIRED_SLOTS))

    touchpoint_columns: list[str] = Field(default_factory=list)

    def instantiate(self, column_map: dict[str, str]) -> CausalGraph:
        if len(self.touchpoint_columns) < 1:
            raise ValueError(
                "AttributionTemplate requires at least one touchpoint in "
                "touchpoint_columns."
            )
        self._check_required_slots(column_map)

        conversion = column_map["conversion"]
        # Allow either pre-resolved names or slot lookups.
        touchpoints = [column_map.get(t, t) for t in self.touchpoint_columns]
        if len(set(touchpoints)) != len(touchpoints):
            raise ValueError(
                "AttributionTemplate touchpoints must be unique after "
                f"column_map resolution; got {touchpoints!r}."
            )

        edges: list[tuple[str, str]] = []
        # Sequential chain.
        for prev, curr in zip(touchpoints, touchpoints[1:]):
            edges.append((prev, curr))
        # Every touchpoint has a direct arrow to conversion.
        for tp in touchpoints:
            edges.append((tp, conversion))

        roles: dict[str, VariableRole] = {conversion: VariableRole.OUTCOME}
        # First touchpoint is the treatment of interest; the rest are mediators.
        roles[touchpoints[0]] = VariableRole.TREATMENT
        for tp in touchpoints[1:]:
            roles[tp] = VariableRole.MEDIATOR

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
