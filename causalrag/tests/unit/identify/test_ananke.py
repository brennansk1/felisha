"""Tests for :mod:`causalrag.identify.ananke_bridge` (Sprint 6.1).

Two cohorts of tests:

* Ananke-gated tests (``pytest.importorskip("ananke")``) — full
  Tian-Shpitser identification on backdoor / front-door / hedge DAGs and
  c-component sanity checks. These also exercise :func:`reconcile`.
* Pure-Python tests — never import ananke; they pin down the fallback path
  and verify ``backend == "fallback"``.

The DoWhy side is mocked with a tiny stand-in object that exposes the
two attributes ``reconcile`` reads (``identifiable``, ``adjustment_set``,
``strategy``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.identify.ananke_bridge import (
    AnankeIDResult,
    ananke_identify,
    reconcile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockDoWhyResult:
    identifiable: bool
    adjustment_set: tuple[str, ...] = ()
    strategy: str | None = None


def _backdoor_dag_with_observed_confounder() -> CausalGraph:
    """U → T → Y, U → Y (classic confounder DAG)."""
    edges = (
        CausalEdge(source="U", target="T"),
        CausalEdge(source="U", target="Y"),
        CausalEdge(source="T", target="Y"),
    )
    return CausalGraph(nodes=("U", "T", "Y"), edges=edges)


def _frontdoor_dag() -> CausalGraph:
    """T → M → Y plus bidirected T <-> Y (unobserved confounder)."""
    edges = (
        CausalEdge(source="T", target="M"),
        CausalEdge(source="M", target="Y"),
        CausalEdge(source="T", target="Y", bidirected=True),
    )
    return CausalGraph(nodes=("T", "M", "Y"), edges=edges)


def _five_node_hedge_admg() -> CausalGraph:
    """5-node ADMG with a T<->Y bidirected edge — Tian-ID says non-identifiable."""
    edges = (
        CausalEdge(source="W", target="T"),
        CausalEdge(source="W", target="Y"),
        CausalEdge(source="T", target="Z"),
        CausalEdge(source="Z", target="Y"),
        CausalEdge(source="T", target="Y"),
        CausalEdge(source="T", target="Y", bidirected=True),
    )
    return CausalGraph(nodes=("W", "T", "Z", "Y", "X"), edges=edges)


def _four_node_admg_one_bidirected() -> CausalGraph:
    """4-node ADMG with a bidirected U1 <-> U2 inside; one non-singleton component."""
    edges = (
        CausalEdge(source="U1", target="T"),
        CausalEdge(source="U2", target="Y"),
        CausalEdge(source="T", target="Y"),
        CausalEdge(source="U1", target="U2", bidirected=True),
    )
    return CausalGraph(nodes=("U1", "U2", "T", "Y"), edges=edges)


# ---------------------------------------------------------------------------
# Ananke-gated tests
# ---------------------------------------------------------------------------


def test_ananke_backdoor_agrees_with_dowhy() -> None:
    pytest.importorskip("ananke")
    graph = _backdoor_dag_with_observed_confounder()
    result = ananke_identify(graph=graph, treatment="T", outcome="Y")
    assert result.identified is True
    assert result.backend == "ananke"
    assert set(result.adjustment_set) == {"U"}

    dowhy_mock = _MockDoWhyResult(
        identifiable=True, adjustment_set=("U",), strategy="backdoor"
    )
    verdict = reconcile(dowhy_mock, result)
    assert verdict["agree"] is True
    assert verdict["disagreement_note"] is None


def test_ananke_frontdoor_identifies_does_not_crash_reconcile() -> None:
    pytest.importorskip("ananke")
    graph = _frontdoor_dag()
    result = ananke_identify(graph=graph, treatment="T", outcome="Y")
    # Front-door is identifiable; ananke should agree.
    assert result.identified is True
    assert result.backend == "ananke"
    assert result.method in {"frontdoor", "tian-id"}

    # DoWhy may or may not identify front-door; reconcile() must not crash
    # in either case.
    dowhy_unidentified = _MockDoWhyResult(identifiable=False, strategy="non-identifiable")
    verdict = reconcile(dowhy_unidentified, result)
    assert verdict["agree"] is False
    assert verdict["primary"] == "ananke"
    assert verdict["disagreement_note"]


def test_ananke_five_node_hedge_non_identifiable() -> None:
    pytest.importorskip("ananke")
    graph = _five_node_hedge_admg()
    result = ananke_identify(graph=graph, treatment="T", outcome="Y")
    assert result.identified is False
    assert result.method != "backdoor"
    # backend should still be ananke (ananke ran and said no).
    assert result.backend == "ananke"


def test_ananke_four_node_admg_c_component() -> None:
    pytest.importorskip("ananke")
    graph = _four_node_admg_one_bidirected()
    result = ananke_identify(graph=graph, treatment="T", outcome="Y")
    # Exactly one c-component should be non-singleton ({U1, U2}).
    non_singleton = [c for c in result.c_component_decomposition if len(c) > 1]
    assert len(non_singleton) == 1
    assert non_singleton[0] == frozenset({"U1", "U2"})


# ---------------------------------------------------------------------------
# Pure-Python fallback tests (no ananke needed)
# ---------------------------------------------------------------------------


def test_fallback_backdoor_when_ananke_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the fallback path even if ananke happens to be installed locally.
    monkeypatch.setattr(
        "causalrag.identify.ananke_bridge._try_import_ananke", lambda: None
    )
    graph = _backdoor_dag_with_observed_confounder()
    result = ananke_identify(graph=graph, treatment="T", outcome="Y")
    assert isinstance(result, AnankeIDResult)
    assert result.backend == "fallback"
    assert result.identified is True
    assert result.method == "backdoor"
    assert set(result.adjustment_set) == {"U"}
    # Fallback always emits a warning explaining itself.
    assert any("fallback" in w for w in result.warnings)


def test_fallback_hedge_non_identifiable_and_reconcile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "causalrag.identify.ananke_bridge._try_import_ananke", lambda: None
    )
    graph = _five_node_hedge_admg()
    result = ananke_identify(graph=graph, treatment="T", outcome="Y")
    assert result.backend == "fallback"
    assert result.identified is False
    assert result.method == "none"
    # Reconcile with a DoWhy result that *did* claim backdoor (forced
    # disagreement) — must not crash and must report disagreement.
    dowhy_wrong = _MockDoWhyResult(
        identifiable=True, adjustment_set=("W",), strategy="backdoor"
    )
    verdict = reconcile(dowhy_wrong, result)
    assert verdict["agree"] is False
    assert verdict["primary"] == "dowhy"
    assert "mismatch" in (verdict["disagreement_note"] or "")


def test_fallback_c_component_decomposition_4_node_admg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "causalrag.identify.ananke_bridge._try_import_ananke", lambda: None
    )
    graph = _four_node_admg_one_bidirected()
    result = ananke_identify(graph=graph, treatment="T", outcome="Y")
    non_singleton = [c for c in result.c_component_decomposition if len(c) > 1]
    assert len(non_singleton) == 1
    assert non_singleton[0] == frozenset({"U1", "U2"})


def test_fallback_missing_treatment_returns_unidentifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "causalrag.identify.ananke_bridge._try_import_ananke", lambda: None
    )
    graph = _backdoor_dag_with_observed_confounder()
    result = ananke_identify(graph=graph, treatment="MISSING", outcome="Y")
    assert result.identified is False
    assert result.backend == "fallback"
    assert any("MISSING" in w for w in result.warnings)


def test_reconcile_both_unidentifiable_agree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "causalrag.identify.ananke_bridge._try_import_ananke", lambda: None
    )
    graph = _five_node_hedge_admg()
    ananke_res = ananke_identify(graph=graph, treatment="T", outcome="Y")
    dowhy_mock = _MockDoWhyResult(identifiable=False, strategy="non-identifiable")
    verdict = reconcile(dowhy_mock, ananke_res)
    assert verdict["agree"] is True
    assert verdict["disagreement_note"] is None


def test_reconcile_adjustment_set_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "causalrag.identify.ananke_bridge._try_import_ananke", lambda: None
    )
    graph = _backdoor_dag_with_observed_confounder()
    ananke_res = ananke_identify(graph=graph, treatment="T", outcome="Y")
    assert ananke_res.method == "backdoor"
    # DoWhy claims a different adjustment set — should disagree.
    dowhy_mock = _MockDoWhyResult(
        identifiable=True, adjustment_set=("OTHER",), strategy="backdoor"
    )
    verdict = reconcile(dowhy_mock, ananke_res)
    assert verdict["agree"] is False
    assert verdict["disagreement_note"] is not None
    assert "adjustment" in verdict["disagreement_note"]
