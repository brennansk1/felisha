"""Large-DAG scope reduction utilities (Sprint 6.5.6).

Three pure-graph algorithms that shrink identification work on big DAGs:

1. :func:`c_components` — partition nodes by the bidirected-edge equivalence
   relation (Tian's c-components / Pearl's confounded components).
2. :func:`extract_relevant_subgraph` — drop nodes that cannot influence T or Y
   along any directed or bidirected path.
3. :func:`d_separation_prune` — remove redundant adjusters from a candidate
   adjustment set, using a fast moralised-undirected-graph approximation of
   d-separation.

All functions are pure: they read a :class:`CausalGraph` and return new
objects. They never mutate the input. They tolerate the current ``CausalGraph``
schema which has no explicit bidirected-edge type — bidirected edges are
detected by looking for a truthy ``bidirected`` attribute on an edge object,
falling back to False when absent.

Wiring into :mod:`causalrag.roadmap.q5_identify` is a separate ticket.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from causalrag.core.graph import CausalEdge, CausalGraph


def _is_bidirected(edge: CausalEdge | Any) -> bool:
    """Return True if ``edge`` represents a bidirected (latent-confounder) link.

    The current :class:`CausalEdge` schema does not include a ``bidirected``
    field. To stay forward-compatible, we look up the attribute defensively;
    absent ⇒ treated as a directed edge.
    """
    return bool(getattr(edge, "bidirected", False))


def _bidirected_edges(graph: CausalGraph) -> list[tuple[str, str]]:
    return [(e.source, e.target) for e in graph.edges if _is_bidirected(e)]


def _directed_edges(graph: CausalGraph) -> list[tuple[str, str]]:
    return [(e.source, e.target) for e in graph.edges if not _is_bidirected(e)]


def c_components(graph: CausalGraph) -> list[frozenset[str]]:
    """Partition ``graph.nodes`` by the bidirected-edge equivalence relation.

    A *c-component* (Tian, 2002) is a maximal set of nodes connected via
    bidirected edges, which encode the presence of a latent common cause. In a
    purely directed acyclic graph with no bidirected edges, every node is its
    own singleton c-component.

    Returns
    -------
    list[frozenset[str]]
        A list of disjoint frozensets whose union is ``set(graph.nodes)``.
        Order is stable: components are sorted by their lexicographically
        smallest member.
    """
    if not graph.nodes:
        return []

    ug: nx.Graph = nx.Graph()
    ug.add_nodes_from(graph.nodes)
    for u, v in _bidirected_edges(graph):
        # Tolerate bidirected edges that reference unknown nodes by adding them.
        ug.add_edge(u, v)

    components = [frozenset(c) for c in nx.connected_components(ug)]
    components.sort(key=lambda s: min(s))
    return components


def extract_relevant_subgraph(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    adjustment_set: frozenset[str] | set[str] | tuple[str, ...] | None = None,
) -> CausalGraph:
    """Return the minimal subgraph relevant for identifying T → Y.

    A node is kept if it is on some directed path T ⇝ Y, an ancestor of T, Y,
    or any adjustment-set member, or shares a c-component with any kept node.
    Everything else cannot influence the identification verdict for the T → Y
    estimand and is dropped.

    Parameters
    ----------
    graph
        Source DAG (possibly with bidirected edges).
    treatment, outcome
        Names of the treatment and outcome nodes.
    adjustment_set
        Optional iterable of variable names whose ancestors should also be
        retained (their measurement constrains identification).

    Returns
    -------
    CausalGraph
        A new :class:`CausalGraph` over the relevant node subset, preserving
        edges, roles, and rank.
    """
    adjustment_set = frozenset(adjustment_set or ())

    if treatment not in graph.nodes or outcome not in graph.nodes:
        # Nothing sensible to extract; return an empty graph of the same rank.
        return CausalGraph(nodes=(), edges=(), roles={}, rank=graph.rank)

    # Build the directed view to compute ancestors / descendants / forward paths.
    dg: nx.DiGraph = nx.DiGraph()
    dg.add_nodes_from(graph.nodes)
    for u, v in _directed_edges(graph):
        dg.add_edge(u, v)

    relevant: set[str] = {treatment, outcome}

    # Forward cone from T intersected with backward cone from Y → nodes on
    # some directed T ⇝ Y path.
    descendants_t = nx.descendants(dg, treatment) | {treatment}
    ancestors_y = nx.ancestors(dg, outcome) | {outcome}
    relevant |= descendants_t & ancestors_y

    # Ancestors of T, Y, and every adjustment-set member.
    relevant |= nx.ancestors(dg, treatment)
    relevant |= nx.ancestors(dg, outcome)
    for z in adjustment_set:
        if z in dg:
            relevant.add(z)
            relevant |= nx.ancestors(dg, z)

    # Close under c-components: latent confounders pull additional nodes in.
    comps = c_components(graph)
    node_to_comp = {n: comp for comp in comps for n in comp}
    expanded: set[str] = set(relevant)
    for n in list(relevant):
        comp = node_to_comp.get(n)
        if comp is not None:
            expanded |= set(comp)
    # Only keep nodes that actually appear in the graph (c_components may have
    # auto-added dangling bidirected endpoints).
    relevant = expanded & set(graph.nodes)

    # Subset edges and roles.
    new_edges = tuple(
        e for e in graph.edges if e.source in relevant and e.target in relevant
    )
    # Preserve original node ordering for determinism.
    new_nodes = tuple(n for n in graph.nodes if n in relevant)
    new_roles = {n: r for n, r in graph.roles.items() if n in relevant}

    return CausalGraph(
        nodes=new_nodes, edges=new_edges, roles=new_roles, rank=graph.rank
    )


def _moralised_undirected(graph: CausalGraph) -> nx.Graph:
    """Return the moral graph: skeleton + edges between co-parents.

    This is a standard fast over-approximation of d-separation reachability:
    if T and Y are disconnected in the moralised graph after removing Z, then
    Z d-separates them in the original DAG.
    """
    dg: nx.DiGraph = nx.DiGraph()
    dg.add_nodes_from(graph.nodes)
    for u, v in _directed_edges(graph):
        dg.add_edge(u, v)

    mg: nx.Graph = nx.Graph()
    mg.add_nodes_from(graph.nodes)
    # Skeleton.
    for u, v in dg.edges():
        mg.add_edge(u, v)
    # Marry parents.
    for node in dg.nodes():
        parents = list(dg.predecessors(node))
        for i, p1 in enumerate(parents):
            for p2 in parents[i + 1 :]:
                mg.add_edge(p1, p2)
    # Bidirected edges connect their endpoints in the moralised view.
    for u, v in _bidirected_edges(graph):
        if u in mg and v in mg:
            mg.add_edge(u, v)
    return mg


def _separates(mg: nx.Graph, t: str, y: str, z: set[str]) -> bool:
    """True iff removing ``z`` disconnects T from Y in the moralised graph."""
    if t == y:
        return False
    if t not in mg or y not in mg:
        return True
    if t in z or y in z:
        # Conditioning on T or Y is nonsensical for backdoor blocking; treat
        # as non-separating to keep the variable (caller controls inclusion).
        return False
    h = mg.copy()
    h.remove_nodes_from(z)
    if t not in h or y not in h:
        return True
    return not nx.has_path(h, t, y)


def d_separation_prune(
    graph: CausalGraph,
    treatment: str,
    outcome: str,
    candidate_adjustment_set: frozenset[str] | set[str] | tuple[str, ...],
) -> frozenset[str]:
    """Drop redundant adjusters from ``candidate_adjustment_set``.

    A node ``z`` is redundant when ``Z \\ {z}`` still d-separates T and Y in
    the moralised undirected graph — a fast bounded approximation of exact
    d-separation. Removal is greedy: variables are tried in a deterministic
    order (sorted) and dropped one at a time as long as the remaining set
    still separates T and Y.

    If the *full* candidate set fails to separate T and Y (or T == Y, or
    either endpoint is missing), the input set is returned unchanged — we do
    not invent adjusters here.
    """
    z = set(candidate_adjustment_set)
    mg = _moralised_undirected(graph)

    if treatment not in mg or outcome not in mg or treatment == outcome:
        return frozenset(z)

    # Only prune if the full set already separates; otherwise we can't claim
    # any subset does.
    if not _separates(mg, treatment, outcome, z):
        return frozenset(z)

    for candidate in sorted(z):
        trial = z - {candidate}
        if _separates(mg, treatment, outcome, trial):
            z = trial

    return frozenset(z)


def summarise_dag(graph: CausalGraph) -> dict[str, Any]:
    """Return a compact size/shape summary of ``graph``.

    Keys
    ----
    n_nodes, n_edges, max_in_degree, max_out_degree,
    n_c_components, has_bidirected_edges, n_strongly_connected_components
    """
    dg: nx.DiGraph = nx.DiGraph()
    dg.add_nodes_from(graph.nodes)
    for u, v in _directed_edges(graph):
        dg.add_edge(u, v)

    if dg.number_of_nodes() == 0:
        max_in = max_out = 0
    else:
        max_in = max((d for _, d in dg.in_degree()), default=0)
        max_out = max((d for _, d in dg.out_degree()), default=0)

    comps = c_components(graph)
    has_bidir = any(_is_bidirected(e) for e in graph.edges)
    n_scc = nx.number_strongly_connected_components(dg) if dg.number_of_nodes() else 0

    return {
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "max_in_degree": max_in,
        "max_out_degree": max_out,
        "n_c_components": len(comps),
        "has_bidirected_edges": has_bidir,
        "n_strongly_connected_components": n_scc,
    }
