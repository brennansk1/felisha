"""Tests for the domain-specific DAG templates (Sprint 6.5.7)."""

from __future__ import annotations

import networkx as nx
import pytest

from causalrag.core.graph import CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.discovery.dag_templates import (
    AttributionTemplate,
    ClinicalTTETemplate,
    EngineeringTraceTemplate,
    MMMTemplate,
    SpatiotemporalTemplate,
)


# ----------------------------------------------------------------------
# Shared structural-assertion helpers
# ----------------------------------------------------------------------


def _assert_structural_invariants(graph: CausalGraph, treatment: str, outcome: str) -> None:
    """Every template-built graph must satisfy these basic invariants."""
    g = graph.to_networkx()
    assert isinstance(graph, CausalGraph)
    assert graph.is_acyclic(), "template-built graph must be a DAG"
    # No self-loops anywhere.
    for u, v in g.edges():
        assert u != v, f"self-loop on {u!r} is not allowed"
    # Treatment must have at least one directed path to outcome.
    assert nx.has_path(g, treatment, outcome), (
        f"no path from treatment={treatment!r} to outcome={outcome!r}"
    )
    # Round-trip: yaml-safe dump is non-empty.
    dump = graph.model_dump_yaml_safe()
    assert dump["nodes"], "expected non-empty nodes"
    assert dump["edges"], "expected non-empty edges"


# ----------------------------------------------------------------------
# Clinical TTE
# ----------------------------------------------------------------------


def test_clinical_tte_instantiates_with_full_column_map() -> None:
    tpl = ClinicalTTETemplate(
        baseline_confounders=["age", "sex"],
        time_varying_confounders=["sbp"],
    )
    column_map = {
        "eligibility_indicator": "eligible",
        "washout_period_complete": "washed",
        "treatment_strategy": "rx_strategy",
        "treatment_received": "rx_received",
        "adherence": "adher",
        "loss_to_followup_censoring": "censored",
        "outcome_observed": "event",
        "age": "age_yrs",
        "sex": "sex_m",
        "sbp": "sbp_mmhg",
    }
    g = tpl.instantiate(column_map)
    assert g.roles["rx_strategy"] is VariableRole.TREATMENT
    assert g.roles["event"] is VariableRole.OUTCOME
    assert g.roles["age_yrs"] is VariableRole.CONFOUNDER
    assert g.roles["sbp_mmhg"] is VariableRole.MEDIATOR
    _assert_structural_invariants(g, treatment="rx_strategy", outcome="event")


def test_clinical_tte_missing_slot_raises() -> None:
    tpl = ClinicalTTETemplate()
    with pytest.raises(ValueError, match="missing required slot"):
        tpl.instantiate({"eligibility_indicator": "e"})


# ----------------------------------------------------------------------
# MMM
# ----------------------------------------------------------------------


def test_mmm_template_with_two_channels_and_competition() -> None:
    tpl = MMMTemplate(channels=["search", "social"], include_competition_index=True)
    column_map = {
        "revenue": "rev",
        "conversion": "conv",
        "seasonality": "season",
        "competition_index": "comp",
        "spend_search": "spend_s",
        "reach_search": "reach_s",
        "spend_social": "spend_so",
        "reach_social": "reach_so",
    }
    g = tpl.instantiate(column_map)
    assert g.roles["rev"] is VariableRole.OUTCOME
    assert g.roles["spend_s"] is VariableRole.TREATMENT
    assert g.roles["reach_s"] is VariableRole.MEDIATOR
    assert g.roles["season"] is VariableRole.CONFOUNDER
    assert g.roles["comp"] is VariableRole.CONFOUNDER
    _assert_structural_invariants(g, treatment="spend_s", outcome="rev")
    _assert_structural_invariants(g, treatment="spend_so", outcome="rev")


def test_mmm_missing_channel_slot_raises() -> None:
    tpl = MMMTemplate(channels=["search"])
    with pytest.raises(ValueError, match="missing required slot"):
        tpl.instantiate(
            {
                "revenue": "rev",
                "conversion": "conv",
                "seasonality": "season",
                # spend_search and reach_search missing
            }
        )


# ----------------------------------------------------------------------
# Attribution
# ----------------------------------------------------------------------


def test_attribution_sequence_three_touchpoints() -> None:
    tpl = AttributionTemplate(touchpoint_columns=["t1", "t2", "t3"])
    column_map = {
        "conversion": "conv",
        "t1": "tp_email",
        "t2": "tp_ad",
        "t3": "tp_retarget",
    }
    g = tpl.instantiate(column_map)
    assert g.roles["tp_email"] is VariableRole.TREATMENT
    assert g.roles["tp_ad"] is VariableRole.MEDIATOR
    assert g.roles["tp_retarget"] is VariableRole.MEDIATOR
    assert g.roles["conv"] is VariableRole.OUTCOME
    # Sequential chain present.
    edges = {(e.source, e.target) for e in g.edges}
    assert ("tp_email", "tp_ad") in edges
    assert ("tp_ad", "tp_retarget") in edges
    assert ("tp_email", "conv") in edges
    _assert_structural_invariants(g, treatment="tp_email", outcome="conv")


def test_attribution_requires_at_least_one_touchpoint() -> None:
    tpl = AttributionTemplate(touchpoint_columns=[])
    with pytest.raises(ValueError, match="at least one touchpoint"):
        tpl.instantiate({"conversion": "c"})


def test_attribution_missing_conversion_slot_raises() -> None:
    tpl = AttributionTemplate(touchpoint_columns=["t1"])
    with pytest.raises(ValueError, match="missing required slot"):
        tpl.instantiate({"t1": "tp"})


# ----------------------------------------------------------------------
# Spatiotemporal
# ----------------------------------------------------------------------


def test_spatiotemporal_template_with_neighbours() -> None:
    tpl = SpatiotemporalTemplate(neighbour_pairs=[("u1", "u2"), ("u2", "u3")])
    column_map = {
        "unit_id": "uid",
        "time": "t",
        "treatment": "trt",
        "outcome": "y",
    }
    g = tpl.instantiate(column_map)
    assert g.roles["trt"] is VariableRole.TREATMENT
    assert g.roles["y"] is VariableRole.OUTCOME
    assert g.roles["uid"] is VariableRole.IDENTIFIER
    assert g.roles["t"] is VariableRole.TIMESTAMP
    assert "trt__lag1" in g.nodes
    assert "trt__neighbour_agg" in g.nodes
    _assert_structural_invariants(g, treatment="trt", outcome="y")


def test_spatiotemporal_without_neighbours_still_dag() -> None:
    tpl = SpatiotemporalTemplate(neighbour_pairs=[])
    column_map = {
        "unit_id": "uid",
        "time": "t",
        "treatment": "trt",
        "outcome": "y",
    }
    g = tpl.instantiate(column_map)
    _assert_structural_invariants(g, treatment="trt", outcome="y")


def test_spatiotemporal_missing_slot_raises() -> None:
    tpl = SpatiotemporalTemplate()
    with pytest.raises(ValueError, match="missing required slot"):
        tpl.instantiate({"unit_id": "u", "time": "t"})


def test_spatiotemporal_bad_neighbour_pair_raises() -> None:
    tpl = SpatiotemporalTemplate(neighbour_pairs=[("only_one",)])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        tpl.instantiate(
            {
                "unit_id": "uid",
                "time": "t",
                "treatment": "trt",
                "outcome": "y",
            }
        )


# ----------------------------------------------------------------------
# Engineering trace
# ----------------------------------------------------------------------


def test_engineering_trace_with_tenant_confounder() -> None:
    tpl = EngineeringTraceTemplate(
        services=["api_gateway", "auth_svc", "billing_svc"],
        tenant="tenant",
    )
    column_map = {
        "slo_metric": "p99_latency",
        "api_gateway": "svc_api",
        "auth_svc": "svc_auth",
        "billing_svc": "svc_bill",
        "tenant": "tenant_id",
    }
    g = tpl.instantiate(column_map)
    assert g.roles["svc_api"] is VariableRole.TREATMENT
    assert g.roles["svc_auth"] is VariableRole.MEDIATOR
    assert g.roles["svc_bill"] is VariableRole.MEDIATOR
    assert g.roles["p99_latency"] is VariableRole.OUTCOME
    assert g.roles["tenant_id"] is VariableRole.CONFOUNDER
    edges = {(e.source, e.target) for e in g.edges}
    assert ("svc_api", "svc_auth") in edges
    assert ("svc_auth", "svc_bill") in edges
    assert ("svc_bill", "p99_latency") in edges
    _assert_structural_invariants(g, treatment="svc_api", outcome="p99_latency")


def test_engineering_trace_single_service_no_tenant() -> None:
    tpl = EngineeringTraceTemplate(services=["only_svc"])
    column_map = {"slo_metric": "slo", "only_svc": "the_svc"}
    g = tpl.instantiate(column_map)
    assert g.roles["the_svc"] is VariableRole.TREATMENT
    _assert_structural_invariants(g, treatment="the_svc", outcome="slo")


def test_engineering_trace_requires_at_least_one_service() -> None:
    tpl = EngineeringTraceTemplate(services=[])
    with pytest.raises(ValueError, match="at least one service"):
        tpl.instantiate({"slo_metric": "slo"})


def test_engineering_trace_missing_slo_slot_raises() -> None:
    tpl = EngineeringTraceTemplate(services=["svc1"])
    with pytest.raises(ValueError, match="missing required slot"):
        tpl.instantiate({"svc1": "the_svc"})
