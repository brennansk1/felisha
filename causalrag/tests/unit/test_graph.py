from __future__ import annotations

from causalrag.core.graph import CausalGraph
from causalrag.core.roles import VariableRole


def test_from_edge_list_collects_nodes_in_first_seen_order() -> None:
    g = CausalGraph.from_edge_list([("A", "B"), ("C", "B"), ("B", "D")])
    assert g.nodes == ("A", "B", "C", "D")
    assert {(e.source, e.target) for e in g.edges} == {("A", "B"), ("C", "B"), ("B", "D")}


def test_acyclicity_detects_cycles() -> None:
    dag = CausalGraph.from_edge_list([("A", "B"), ("B", "C")])
    cycle = CausalGraph.from_edge_list([("A", "B"), ("B", "A")])
    assert dag.is_acyclic()
    assert not cycle.is_acyclic()


def test_parents_and_descendants() -> None:
    g = CausalGraph.from_edge_list([("A", "B"), ("B", "C"), ("D", "C")])
    assert set(g.parents("C")) == {"B", "D"}
    assert g.descendants("A") == frozenset({"B", "C"})


def test_networkx_roundtrip_preserves_roles_and_metadata() -> None:
    g = CausalGraph.from_edge_list(
        [("T", "Y"), ("X", "T"), ("X", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "X": VariableRole.CONFOUNDER,
        },
    )
    g2 = CausalGraph.from_networkx(g.to_networkx())
    assert set(g2.nodes) == set(g.nodes)
    assert {(e.source, e.target) for e in g2.edges} == {(e.source, e.target) for e in g.edges}
    assert g2.roles["T"] is VariableRole.TREATMENT
    assert g2.roles["Y"] is VariableRole.OUTCOME
    assert g2.roles["X"] is VariableRole.CONFOUNDER
