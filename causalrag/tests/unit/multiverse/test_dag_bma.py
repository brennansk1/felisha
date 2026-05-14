"""Unit tests for Sprint 6.3 — multiverse-of-DAGs Bayesian model averaging.

The dag_bma orchestrator is exercised with a fake ``run_single`` injection
so we don't pull in DoWhy + EconML for unit tests; the math (weight
renormalisation, BMA point and SE, consensus classification) is what we
care about here.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.graph import CausalGraph
from causalrag.multiverse.dag_bma import (
    DAGBMAFinding,
    DAGBMAReport,
    dag_bma,
)


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def estimand() -> CausalEstimand:
    return CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": "t",
            "outcome": "y",
            "formal_expression": "E[Y(1) - Y(0)]",
        }
    )


@pytest.fixture
def df() -> pd.DataFrame:
    # The runner is faked in these tests, so the data isn't actually
    # consumed — but dag_bma takes a DataFrame in its signature.
    return pd.DataFrame({"t": [0, 1, 0, 1], "y": [1.0, 2.0, 1.5, 2.5]})


def _g(rank: int) -> CausalGraph:
    """Tiny T→Y graph at a given discovery rank."""
    return CausalGraph.from_edge_list([("t", "y")], rank=rank)


def _make_runner(scripted: dict[int, tuple[bool, float | None, float | None]]):
    """Build a fake ``run_single`` that returns ``scripted[idx]`` keyed
    by the order the DAG appears in ``candidate_graphs``."""

    counter = {"i": 0}

    def runner(graph, df, estimand, estimator_id):
        idx = counter["i"]
        counter["i"] += 1
        identifiable, point, se = scripted[idx]
        notes = [] if identifiable else ["scripted: non-identifiable"]
        return identifiable, point, se, notes

    return runner


# --- Empty / degenerate inputs ----------------------------------------------


def test_no_candidate_graphs_returns_non_identifiable(df, estimand):
    report = dag_bma(
        candidate_graphs=[],
        df=df,
        estimand=estimand,
        run_single=_make_runner({}),
    )
    assert report.n_dags == 0
    assert report.n_identifiable == 0
    assert report.consensus_verdict == "non_identifiable"
    assert report.bma_point == 0.0


# --- Majority non-identifiable case (spec test 1) ---------------------------


def test_majority_non_identifiable_yields_non_identifiable_verdict(df, estimand):
    """3 candidate DAGs, only the first is identifiable.

    The remaining two fail Step 5 so the multiverse cannot reach a
    confident verdict even though *some* answer is available.
    """
    runner = _make_runner({
        0: (True, 2.0, 0.10),
        1: (False, None, None),
        2: (False, None, None),
    })
    graphs = [_g(1), _g(2), _g(3)]
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        run_single=runner,
    )

    assert report.n_dags == 3
    assert report.n_identifiable == 1
    assert report.consensus_verdict == "non_identifiable"
    # Only the identifiable finding gets weight.
    weights = [f.posterior_weight for f in report.findings]
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == 0.0
    assert weights[2] == 0.0
    # BMA collapses to the single identifiable DAG.
    assert report.bma_point == pytest.approx(2.0)


# --- Three identifiable DAGs near same answer (spec test 2) -----------------


def test_three_identifiable_near_same_answer_yields_consensus(df, estimand):
    runner = _make_runner({
        0: (True, 2.00, 0.10),
        1: (True, 2.05, 0.11),
        2: (True, 1.95, 0.09),
    })
    graphs = [_g(1), _g(2), _g(3)]
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        run_single=runner,
    )

    assert report.n_identifiable == 3
    assert report.consensus_verdict == "consensus"
    # Uniform weights — BMA point is the simple mean.
    assert report.bma_point == pytest.approx((2.00 + 2.05 + 1.95) / 3.0)
    # SE must include both within-DAG and between-DAG variance.
    assert report.bma_se > 0.0
    # CI is symmetric around the BMA point.
    assert report.bma_ci_low is not None and report.bma_ci_high is not None
    assert report.bma_ci_low < report.bma_point < report.bma_ci_high


# --- Differing answers → split (spec test 3) --------------------------------


def test_three_identifiable_disagreeing_signs_yields_split(df, estimand):
    runner = _make_runner({
        0: (True, 2.0, 0.10),
        1: (True, -1.5, 0.12),
        2: (True, 0.5, 0.20),
    })
    graphs = [_g(1), _g(2), _g(3)]
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        run_single=runner,
    )

    assert report.n_identifiable == 3
    assert report.consensus_verdict == "split"
    # BMA point is the uniform-weighted average.
    assert report.bma_point == pytest.approx((2.0 - 1.5 + 0.5) / 3.0)


def test_same_sign_but_large_magnitude_spread_is_split(df, estimand):
    """Two DAGs agree on sign but disagree wildly in magnitude — the
    weighted spread relative to the BMA point exceeds the 50% threshold
    and we flag this as ``split`` rather than papering over it with a
    BMA point that lives in neither candidate's posterior."""
    runner = _make_runner({
        0: (True, 0.1, 0.01),
        1: (True, 10.0, 0.01),
    })
    graphs = [_g(1), _g(2)]
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        run_single=runner,
    )
    assert report.consensus_verdict == "split"


# --- BMA arithmetic --------------------------------------------------------


def test_bma_point_is_posterior_weighted_mean(df, estimand):
    """With a non-uniform posterior, the BMA point should equal the
    posterior-weighted mean (renormalised over identifiable DAGs)."""
    runner = _make_runner({
        0: (True, 1.0, 0.1),
        1: (True, 3.0, 0.1),
        2: (True, 2.0, 0.1),
    })
    graphs = [_g(1), _g(2), _g(3)]
    posterior = {0: 0.5, 1: 0.3, 2: 0.2}
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        bootstrapped_cd_posterior=posterior,
        run_single=runner,
    )
    # All three are identifiable so the prior posterior is preserved.
    expected = 0.5 * 1.0 + 0.3 * 3.0 + 0.2 * 2.0
    assert report.bma_point == pytest.approx(expected)
    assert sum(f.posterior_weight for f in report.findings) == pytest.approx(1.0)


def test_bma_se_combines_within_and_between_variance(df, estimand):
    """BMA SE = sqrt(Σ w_i (SE_i^2 + (point_i - BMA_point)^2))."""
    runner = _make_runner({
        0: (True, 1.0, 0.2),
        1: (True, 3.0, 0.4),
    })
    graphs = [_g(1), _g(2)]
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        run_single=runner,
    )
    # Uniform w = 0.5 each. BMA point = 2.0.
    assert report.bma_point == pytest.approx(2.0)
    expected_var = 0.5 * (0.2 ** 2 + (1.0 - 2.0) ** 2) + 0.5 * (0.4 ** 2 + (3.0 - 2.0) ** 2)
    assert report.bma_se == pytest.approx(math.sqrt(expected_var))


def test_posterior_renormalises_over_identifiable_dags(df, estimand):
    """Mass on a non-identifiable DAG is rolled into the identifiable
    ones — otherwise the BMA point would shrink toward 0 simply because
    some DAGs cannot be evaluated."""
    runner = _make_runner({
        0: (True, 2.0, 0.1),
        1: (False, None, None),
    })
    graphs = [_g(1), _g(2)]
    posterior = {0: 0.4, 1: 0.6}
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        bootstrapped_cd_posterior=posterior,
        run_single=runner,
    )
    # The identifiable DAG now carries the full 1.0 mass for BMA.
    assert report.findings[0].posterior_weight == pytest.approx(1.0)
    assert report.findings[1].posterior_weight == 0.0
    assert report.bma_point == pytest.approx(2.0)


def test_uniform_weights_default_when_posterior_missing(df, estimand):
    runner = _make_runner({
        0: (True, 1.0, 0.1),
        1: (True, 2.0, 0.1),
        2: (True, 3.0, 0.1),
    })
    graphs = [_g(1), _g(2), _g(3)]
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        run_single=runner,
    )
    for f in report.findings:
        assert f.posterior_weight == pytest.approx(1.0 / 3.0)


def test_findings_carry_dag_index_and_rank(df, estimand):
    runner = _make_runner({
        0: (True, 1.0, 0.1),
        1: (False, None, None),
    })
    # Note the rank is *not* sequential with index — discovery may
    # supply DAGs in arbitrary rank order.
    graphs = [_g(rank=5), _g(rank=2)]
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        run_single=runner,
    )
    assert report.findings[0].dag_index == 0
    assert report.findings[0].dag_rank == 5
    assert report.findings[1].dag_index == 1
    assert report.findings[1].dag_rank == 2
    # Non-identifiable finding has notes.
    assert report.findings[1].notes


def test_all_zero_posterior_falls_back_to_uniform(df, estimand):
    runner = _make_runner({
        0: (True, 1.0, 0.1),
        1: (True, 2.0, 0.1),
    })
    graphs = [_g(1), _g(2)]
    # Posterior keyed on indices that don't match — caller bug.
    posterior = {99: 0.5, 100: 0.5}
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        bootstrapped_cd_posterior=posterior,
        run_single=runner,
    )
    # Falls back to uniform 1/k.
    for f in report.findings:
        assert f.posterior_weight == pytest.approx(0.5)
    assert report.bma_point == pytest.approx(1.5)


def test_missing_se_yields_no_ci(df, estimand):
    """When no identifiable DAG provides an SE, the BMA CI is undefined
    but the point + between-DAG variance are still reported."""
    runner = _make_runner({
        0: (True, 1.0, None),
        1: (True, 3.0, None),
    })
    graphs = [_g(1), _g(2)]
    report = dag_bma(
        candidate_graphs=graphs,
        df=df,
        estimand=estimand,
        run_single=runner,
    )
    assert report.bma_point == pytest.approx(2.0)
    # Between-DAG variance is still present.
    assert report.bma_se > 0.0
    # But no within-DAG SE means no calibrated CI.
    assert report.bma_ci_low is None
    assert report.bma_ci_high is None
