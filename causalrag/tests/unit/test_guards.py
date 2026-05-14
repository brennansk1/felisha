from __future__ import annotations

import numpy as np
import pandas as pd

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.llm.guards import (
    audit_dag_edges,
    check_columns_exist,
    check_iv_relevance,
    check_temporal_consistency,
)


def test_check_columns_exist_reports_unknowns() -> None:
    missing = check_columns_exist(["a", "b", "c"], known={"a", "c"}, context="dag")
    assert missing == ["b"]


def test_check_temporal_consistency_flags_post_to_pre() -> None:
    edges = [("post_outcome", "baseline_var"), ("baseline_var", "post_outcome")]
    positions = {"post_outcome": "outcome", "baseline_var": "baseline"}
    violations = check_temporal_consistency(edges, positions)
    assert len(violations) == 1
    assert violations[0][0] == "post_outcome"


def test_check_iv_relevance_passes_strong_iv() -> None:
    rng = np.random.default_rng(0)
    z = rng.integers(0, 2, size=500).astype(float)
    t = (z + rng.normal(0, 0.1, size=500) > 0.5).astype(float)
    df = pd.DataFrame({"Z": z, "T": t})
    out = check_iv_relevance(df, "Z", "T")
    assert out.passes_relevance


def test_check_iv_relevance_downgrades_weak_iv() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"Z": rng.integers(0, 2, size=500), "T": rng.integers(0, 2, size=500)}
    )
    out = check_iv_relevance(df, "Z", "T")
    assert not out.passes_relevance


def test_audit_supports_a_true_edge() -> None:
    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n)
    df = pd.DataFrame({"X": x, "Y": y})
    g = CausalGraph(
        nodes=("X", "Y"),
        edges=(CausalEdge(source="X", target="Y", llm_proposed=True),),
        roles={"X": VariableRole.CONFOUNDER, "Y": VariableRole.OUTCOME},
    )
    audits = audit_dag_edges(g, df)
    assert audits[0].verdict == "supported"
    assert abs(audits[0].partial_correlation) > 0.5


def test_audit_contradicts_a_spurious_edge() -> None:
    rng = np.random.default_rng(2)
    n = 800
    df = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.normal(size=n)})
    g = CausalGraph(
        nodes=("X", "Y"),
        edges=(CausalEdge(source="X", target="Y", llm_proposed=True),),
        roles={"X": VariableRole.CONFOUNDER, "Y": VariableRole.OUTCOME},
    )
    audits = audit_dag_edges(g, df)
    assert audits[0].verdict == "contradicted"


def test_audit_handles_missing_columns_gracefully() -> None:
    df = pd.DataFrame({"X": [1, 2, 3]})
    g = CausalGraph(
        nodes=("X", "Y"),
        edges=(CausalEdge(source="X", target="Y", llm_proposed=True),),
        roles={"X": VariableRole.CONFOUNDER, "Y": VariableRole.OUTCOME},
    )
    audits = audit_dag_edges(g, df)
    assert audits[0].verdict == "inconclusive"
    assert audits[0].note and "column missing" in audits[0].note
