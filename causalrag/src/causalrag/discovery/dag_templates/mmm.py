"""Media-mix-modelling (MMM) DAG template.

For each channel ``c``:

    spend_c -> reach_c -> conversion -> revenue

Seasonality and (optional) competition_index are baseline confounders that
fork into both spend (budget pacing reacts to season) and conversion / revenue
(demand varies with season).
"""

from __future__ import annotations

from pydantic import Field

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.discovery.dag_templates.base import DAGTemplate


_REQUIRED_SLOTS = ["revenue", "conversion", "seasonality"]


class MMMTemplate(DAGTemplate):
    """Media-mix modelling skeleton.

    ``channels`` is the list of channel base names; for each channel ``c`` the
    template expects ``column_map[f"spend_{c}"]`` and ``column_map[f"reach_{c}"]``.
    """

    template_name: str = "mmm"
    domain: str = "marketing"
    description: str = (
        "Per-channel spend drives reach, which drives conversion and then "
        "revenue. Seasonality (and optionally a competition index) is a "
        "baseline confounder of spend and the downstream outcomes."
    )
    slots: list[str] = Field(default_factory=lambda: list(_REQUIRED_SLOTS))

    channels: list[str] = Field(default_factory=list)
    include_competition_index: bool = False

    def instantiate(self, column_map: dict[str, str]) -> CausalGraph:
        # Build the dynamic slot list so the user gets one clear error message.
        dynamic_slots = list(_REQUIRED_SLOTS)
        for c in self.channels:
            dynamic_slots.append(f"spend_{c}")
            dynamic_slots.append(f"reach_{c}")
        if self.include_competition_index:
            dynamic_slots.append("competition_index")
        self.slots = dynamic_slots
        self._check_required_slots(column_map)

        revenue = column_map["revenue"]
        conversion = column_map["conversion"]
        seasonality = column_map["seasonality"]
        competition = (
            column_map["competition_index"] if self.include_competition_index else None
        )

        edges: list[tuple[str, str]] = []
        roles: dict[str, VariableRole] = {
            revenue: VariableRole.OUTCOME,
            conversion: VariableRole.MEDIATOR,
            seasonality: VariableRole.CONFOUNDER,
        }
        if competition is not None:
            roles[competition] = VariableRole.CONFOUNDER

        for c in self.channels:
            spend = column_map[f"spend_{c}"]
            reach = column_map[f"reach_{c}"]
            edges.append((spend, reach))
            edges.append((reach, conversion))
            # Seasonality forks into spend and into conversion / revenue.
            edges.append((seasonality, spend))
            if competition is not None:
                edges.append((competition, spend))
            roles[spend] = VariableRole.TREATMENT
            roles[reach] = VariableRole.MEDIATOR

        edges.append((conversion, revenue))
        edges.append((seasonality, conversion))
        edges.append((seasonality, revenue))
        if competition is not None:
            edges.append((competition, conversion))
            edges.append((competition, revenue))

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
