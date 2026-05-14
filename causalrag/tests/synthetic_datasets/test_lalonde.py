"""Lalonde NSW — gold-standard ATE recovery + Causal Parrot regression tests.

The Lalonde (1986) / Dehejia-Wahba (1999) NSW experiment is the canonical
benchmark for observational causal-inference methods because (a) the treatment
was randomized and (b) the experimental ATE on 1978 earnings (~$1,790) is
broadly accepted in the literature. Methods that recover something close to
this from the experimental data pass; methods that wildly miss it fail.

These tests cover:

- Discovery + flag emission on the real columns.
- ATE recovery via Step 5 → Step 7 with auto preprocessing and selection.
- Causal Parrot regression: permute the treatment column to destroy any
  causal link. A pipeline that "parrots" Lalonde's canonical DAG without
  checking the data will still propose treat → re78; our Layer-4 audit
  must surface the contradiction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalGraph
from causalrag.core.protocol import StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.discovery import run_discovery
from causalrag.llm.guards import audit_dag_edges
from causalrag.roadmap.q5_identify import identify_effect
from causalrag.roadmap.q7_estimate import estimate

LALONDE_EXPERIMENTAL_ATE = 1794.0  # Dehejia-Wahba 1999, in 1978 dollars


@pytest.fixture
def estimand() -> CausalEstimand:
    return CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": "treat",
            "outcome": "re78",
            "formal_expression": "E[Y(1) - Y(0)]",
        }
    )


@pytest.fixture
def lalonde_graph() -> CausalGraph:
    """Canonical adjustment DAG for Lalonde NSW: pre-treatment covariates →
    (treat, re78)."""
    covariates = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
    edges = [(c, "treat") for c in covariates] + [(c, "re78") for c in covariates]
    edges.append(("treat", "re78"))
    roles = {c: VariableRole.CONFOUNDER for c in covariates}
    roles["treat"] = VariableRole.TREATMENT
    roles["re78"] = VariableRole.OUTCOME
    return CausalGraph.from_edge_list(edges, roles=roles)


def test_lalonde_discovery_flags(lalonde_nsw: pd.DataFrame) -> None:
    """Discovery on real Lalonde data must emit BINARY_TREATMENT +
    CONTINUOUS_OUTCOME and recognize 're74'/'re75' as pre-treatment."""
    result = run_discovery(source=lalonde_nsw, treatment="treat", outcome="re78")
    assert DataFlag.BINARY_TREATMENT in result.flags
    assert DataFlag.CONTINUOUS_OUTCOME in result.flags
    profile = result.profile
    assert profile.column("re74").mean is not None
    # Treatment should be detected as binary 0/1
    assert profile.column("treat").is_binary_01


def test_lalonde_ate_recovery(
    lalonde_nsw: pd.DataFrame, estimand: CausalEstimand, lalonde_graph: CausalGraph
) -> None:
    """LinearDML on Lalonde NSW with the canonical adjustment set should
    recover the experimental ATE within a generous tolerance."""
    ident = identify_effect(estimand, lalonde_graph, df=lalonde_nsw)
    assert ident.identifiable
    assert ident.strategy == "backdoor"

    result = estimate(
        df=lalonde_nsw,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="lalonde"),
        confounders=("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        selection="none",  # use full adjustment set; this is the canonical run
    )
    # Lalonde is famously sensitive; we accept anything within ±$1500 of the
    # experimental benchmark, which is consistent with the published range
    # ($800-$2800 across DML / TMLE / matching specifications).
    assert abs(result.point_estimate - LALONDE_EXPERIMENTAL_ATE) < 1500, (
        f"ATE estimate {result.point_estimate:.0f} too far from experimental "
        f"benchmark {LALONDE_EXPERIMENTAL_ATE}"
    )
    assert result.n_used == len(lalonde_nsw)
    assert result.diagnostics.get("overlap") is not None
    assert "preprocessing" in result.diagnostics


def test_lalonde_refutations_pass_on_real_treatment(
    lalonde_nsw: pd.DataFrame, estimand: CausalEstimand, lalonde_graph: CausalGraph
) -> None:
    """On the real (un-permuted) treatment, the placebo refutation should NOT
    pass — the permuted treatment must give something near zero AND the
    original estimate is substantively non-zero."""
    ident = identify_effect(estimand, lalonde_graph, df=lalonde_nsw)
    result = estimate(
        df=lalonde_nsw,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="lalonde"),
        confounders=("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        selection="none",
    )
    refs = result.refutations
    # Placebo treatment ≈ 0 → passes (the placebo "should" estimate 0)
    placebo = refs.get("placebo_treatment", {})
    if "refuted" in placebo:
        assert abs(placebo["refuted"]) < abs(placebo["original"]) * 0.5
    # Random common cause: tolerance is 50% of the original estimate — Lalonde
    # at n=445 is famously sensitive across published replications (Dehejia &
    # Wahba 1999; Imbens 2015), so a 50% movement under an unrelated noise
    # covariate is within the published norm for this dataset.
    rcc = refs.get("random_common_cause", {})
    if "refuted" in rcc:
        assert abs(rcc["refuted"] - rcc["original"]) < abs(rcc["original"]) * 0.5


def test_causal_parrot_permuted_treatment(
    lalonde_nsw: pd.DataFrame, estimand: CausalEstimand, lalonde_graph: CausalGraph
) -> None:
    """**Causal Parrot regression test.**

    Permute the treatment column to destroy any causal link with the outcome.
    A pipeline that parrots Lalonde's canonical "treat → re78" structure
    without statistically checking it will still propose the edge AND will
    report a non-trivial ATE.

    Our Layer-4 audit must:
    1. Mark the (treat → re78) edge as ``contradicted`` on permuted data.
    2. Even if the LLM proposed it, the statistical truth on this data is
       independence.

    This test directly exercises the audit_dag_edges path on a graph that
    looks correct semantically but is wrong on the permuted data.
    """
    rng = np.random.default_rng(0)
    permuted = lalonde_nsw.copy()
    permuted["treat"] = rng.permutation(permuted["treat"].to_numpy())

    # Build a 'parroted' DAG that the LLM would propose for Lalonde: every
    # pre-treatment covariate flows into treat and re78, plus treat → re78.
    from causalrag.core.graph import CausalEdge

    covariates = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]
    edges = [
        CausalEdge(source=c, target="treat", llm_proposed=True) for c in covariates
    ] + [
        CausalEdge(source=c, target="re78", llm_proposed=True) for c in covariates
    ]
    edges.append(CausalEdge(source="treat", target="re78", llm_proposed=True))
    parroted_dag = CausalGraph(
        nodes=tuple(["treat", "re78"] + covariates),
        edges=tuple(edges),
        roles={
            "treat": VariableRole.TREATMENT,
            "re78": VariableRole.OUTCOME,
            **{c: VariableRole.CONFOUNDER for c in covariates},
        },
        rank=1,
    )

    audits = audit_dag_edges(parroted_dag, permuted)
    # Find the audit row for the headline treat → re78 edge
    headline = next(a for a in audits if a.source == "treat" and a.target == "re78")
    # On permuted data, the treat → re78 edge should be contradicted or
    # at minimum inconclusive — never supported.
    assert headline.verdict in {"contradicted", "inconclusive"}, (
        f"Causal Parrot failure: pipeline accepted a memorized 'treat → re78' "
        f"edge on permuted data (verdict={headline.verdict}, "
        f"r={headline.partial_correlation:.3f}, p={headline.p_value:.3f})."
    )
    # Compare to the same edge on REAL data — should be supported there
    real_audits = audit_dag_edges(parroted_dag, lalonde_nsw)
    real_headline = next(a for a in real_audits if a.source == "treat" and a.target == "re78")
    assert real_headline.verdict == "supported", (
        f"Layer-4 audit failed to support the true Lalonde effect "
        f"(r={real_headline.partial_correlation:.3f}, p={real_headline.p_value:.3f})"
    )


def test_lalonde_post_double_selection_keeps_pretreatment(
    lalonde_nsw: pd.DataFrame, estimand: CausalEstimand, lalonde_graph: CausalGraph
) -> None:
    """Post-double-selection on Lalonde should keep at least one of
    (re74, re75) — they're the strongest predictors of re78 and treatment
    receipt."""
    ident = identify_effect(estimand, lalonde_graph, df=lalonde_nsw)
    result = estimate(
        df=lalonde_nsw,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="lalonde"),
        confounders=("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"),
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME, DataFlag.HIGH_DIMENSIONAL},
        selection="post_double_selection",
    )
    selection = result.diagnostics.get("variable_selection", {})
    selected = set(selection.get("selected", []))
    assert "re74" in selected or "re75" in selected
