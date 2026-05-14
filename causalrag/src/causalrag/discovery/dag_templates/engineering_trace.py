"""Engineering / OpenTelemetry-trace DAG template.

For an ordered service call chain ``[s1, s2, ..., sN]`` and an SLO metric:

    s1 -> s2 -> ... -> sN -> slo_metric

Each service is a treatment (its latency / error rate is the lever the
operator can twiddle); intermediate services are mediators with respect to the
SLO. An optional ``tenant`` column attaches as a confounder forking into the
first service and the SLO (multi-tenant heterogeneity).
"""

from __future__ import annotations

from pydantic import Field

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.discovery.dag_templates.base import DAGTemplate


_REQUIRED_SLOTS = ["slo_metric"]


class EngineeringTraceTemplate(DAGTemplate):
    """Engineering-trace skeleton: ordered service chain -> SLO metric."""

    template_name: str = "engineering_trace"
    domain: str = "engineering"
    description: str = (
        "Ordered OpenTelemetry-style service call DAG terminating at an SLO "
        "metric; an optional tenant column acts as a confounder."
    )
    slots: list[str] = Field(default_factory=lambda: list(_REQUIRED_SLOTS))

    services: list[str] = Field(default_factory=list)
    tenant: str | None = None

    def instantiate(self, column_map: dict[str, str]) -> CausalGraph:
        if len(self.services) < 1:
            raise ValueError(
                "EngineeringTraceTemplate requires at least one service in "
                "services."
            )
        self._check_required_slots(column_map)

        slo = column_map["slo_metric"]
        services = [column_map.get(s, s) for s in self.services]
        if len(set(services)) != len(services):
            raise ValueError(
                "EngineeringTraceTemplate services must resolve to unique "
                f"column names; got {services!r}."
            )
        tenant = column_map.get(self.tenant, self.tenant) if self.tenant else None

        edges: list[tuple[str, str]] = []
        # Chain edges.
        for prev, curr in zip(services, services[1:]):
            edges.append((prev, curr))
        # Last service -> SLO.
        edges.append((services[-1], slo))
        # Tenant confounder.
        if tenant is not None:
            edges.append((tenant, services[0]))
            edges.append((tenant, slo))

        roles: dict[str, VariableRole] = {slo: VariableRole.OUTCOME}
        roles[services[0]] = VariableRole.TREATMENT
        for s in services[1:]:
            roles[s] = VariableRole.MEDIATOR
        if tenant is not None:
            roles[tenant] = VariableRole.CONFOUNDER

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
