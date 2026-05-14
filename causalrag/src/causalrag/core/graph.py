"""CausalGraph — thin wrapper around NetworkX with GML I/O (PDD §13 core/graph.py)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.roles import VariableRole


class CausalEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source: str
    target: str
    llm_proposed: bool = False
    ci_test_passed: bool | None = None
    note: str | None = None


class CausalGraph(BaseModel):
    """Directed acyclic graph over named variables.

    Wraps NetworkX for traversal; serializes to a plain edge list for YAML
    round-trip in the StudyProtocol. The graph is *not* enforced acyclic at
    construction time — Step 2 emits acyclicity diagnostics rather than raising,
    so that LLM-proposed cycles can be surfaced to the analyst.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    nodes: tuple[str, ...] = ()
    edges: tuple[CausalEdge, ...] = ()
    roles: dict[str, VariableRole] = Field(default_factory=dict)
    rank: int = Field(default=1, description="Discovery rank — 1 is the top candidate DAG")

    def to_networkx(self) -> nx.DiGraph:
        g: nx.DiGraph = nx.DiGraph()
        for node in self.nodes:
            g.add_node(node, role=self.roles.get(node, VariableRole.AUXILIARY).value)
        for edge in self.edges:
            g.add_edge(
                edge.source,
                edge.target,
                llm_proposed=edge.llm_proposed,
                ci_test_passed=edge.ci_test_passed,
                note=edge.note,
            )
        return g

    @classmethod
    def from_networkx(cls, g: nx.DiGraph, rank: int = 1) -> CausalGraph:
        nodes = tuple(g.nodes())
        roles = {
            n: VariableRole(g.nodes[n]["role"])
            for n in g.nodes()
            if "role" in g.nodes[n]
        }
        edges = tuple(
            CausalEdge(
                source=u,
                target=v,
                llm_proposed=bool(d.get("llm_proposed", False)),
                ci_test_passed=d.get("ci_test_passed"),
                note=d.get("note"),
            )
            for u, v, d in g.edges(data=True)
        )
        return cls(nodes=nodes, edges=edges, roles=roles, rank=rank)

    def is_acyclic(self) -> bool:
        return nx.is_directed_acyclic_graph(self.to_networkx())

    def parents(self, node: str) -> tuple[str, ...]:
        return tuple(self.to_networkx().predecessors(node))

    def descendants(self, node: str) -> frozenset[str]:
        return frozenset(nx.descendants(self.to_networkx(), node))

    def variables_with_role(self, role: VariableRole) -> tuple[str, ...]:
        return tuple(n for n, r in self.roles.items() if r is role)

    def is_collider_on_path(self, path: list[str], node: str) -> bool:
        """Return True iff ``node`` is an interior collider on ``path``.

        A node C at position i (0 < i < len(path)-1) is a collider when both the
        edge from path[i-1] to C and the edge from path[i+1] to C exist in the
        directed graph — i.e. both arrows point *into* C.
        """
        if node not in path:
            return False
        i = path.index(node)
        if i == 0 or i == len(path) - 1:
            return False
        g = self.to_networkx()
        prev_in = g.has_edge(path[i - 1], node)
        next_in = g.has_edge(path[i + 1], node)
        return prev_in and next_in

    def colliders_between(self, source: str, target: str) -> frozenset[str]:
        """Return nodes that are colliders on at least one undirected path
        connecting ``source`` and ``target``.

        A node C is reported when there exists a simple path (ignoring edge
        direction) from ``source`` to ``target`` such that on that path both the
        incoming and outgoing edges at C point *into* C in the directed graph
        (i.e. ``X -> C <- Y`` structure for some neighbors X, Y on the path).
        """
        g = self.to_networkx()
        if source not in g or target not in g:
            return frozenset()
        ug = g.to_undirected()
        try:
            paths = nx.all_simple_paths(ug, source, target)
        except nx.NodeNotFound:
            return frozenset()
        out: set[str] = set()
        for path in paths:
            for i in range(1, len(path) - 1):
                node = path[i]
                if g.has_edge(path[i - 1], node) and g.has_edge(path[i + 1], node):
                    out.add(node)
        return frozenset(out)

    @classmethod
    def empty(cls) -> CausalGraph:
        return cls()

    @classmethod
    def from_edge_list(
        cls,
        edges: Iterable[tuple[str, str]],
        roles: dict[str, VariableRole] | None = None,
        rank: int = 1,
    ) -> CausalGraph:
        edge_objs = tuple(CausalEdge(source=u, target=v) for u, v in edges)
        nodes: list[str] = []
        seen: set[str] = set()
        for e in edge_objs:
            for n in (e.source, e.target):
                if n not in seen:
                    seen.add(n)
                    nodes.append(n)
        return cls(nodes=tuple(nodes), edges=edge_objs, roles=roles or {}, rank=rank)

    def model_dump_yaml_safe(self) -> dict[str, Any]:
        """Plain-Python representation for YAML serialization (no Pydantic types)."""
        return {
            "rank": self.rank,
            "nodes": list(self.nodes),
            "roles": {k: v.value for k, v in self.roles.items()},
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "llm_proposed": e.llm_proposed,
                    "ci_test_passed": e.ci_test_passed,
                    "note": e.note,
                }
                for e in self.edges
            ],
        }
