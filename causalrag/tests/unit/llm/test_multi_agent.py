"""Unit tests for ``causalrag.llm.multi_agent`` (Sprint 8.4).

We stub the OllamaClient at the ``.parse(...)`` boundary; this module
should never have to know about cassettes / retries. What we verify:

  * the three-client setup (planner / skeptic / statistician) is
    honored — each agent's call routes to its own stubbed client when a
    dict is supplied;
  * the consensus rule (``keep iff >= 2 of 3 agents accept/revise``)
    holds across all 8 verdict triples (accept/revise/reject ^ 3) for
    the two challengers, with the planner's vote inferred per the
    documented rule;
  * a transport failure on either agent aborts to
    ``DebateConsensus(keep=True, rationale='debate aborted: ...')``;
  * ``revised_method`` is only populated when the *statistician*
    requested an actual catalog id, and only when at least one
    challenger asked for a revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import pytest
from pydantic import BaseModel

from causalrag.llm.multi_agent import (
    DebateConsensus,
    SkepticChallenge,
    StatisticianChallenge,
    Verdict,
    run_debate,
)


# ─────────── Helpers ──────────────────────────────────────────────────────


@dataclass
class _FakeCandidate:
    candidate_id: str = "c-abc123"
    research_question: str = "Does T cause Y?"
    treatment: str = "T_drug"
    outcome: str = "Y_recovery"
    estimand_class: str = "ATE"
    recommended_method: str | None = "dowhy.linear_regression"
    mediator: str | None = None
    instrument: str | None = None


@dataclass
class _StubResp:
    parsed: BaseModel


class _StubClient:
    """Records calls and returns scripted parsed objects in order.

    Distinguishes itself from peers via the ``name`` attribute so tests
    can assert each agent was routed to the correct client."""

    def __init__(self, name: str, responses: list[BaseModel | Exception]) -> None:
        self.name = name
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def parse(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        system: str = "",
        json_schema: dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> _StubResp:
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "system": system,
                "json_schema": json_schema,
            }
        )
        if not self._responses:
            raise RuntimeError(f"{self.name}: no scripted response left")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, schema), (
            f"{self.name}: scripted response {type(item).__name__} does not "
            f"match requested schema {schema.__name__}"
        )
        return _StubResp(parsed=item)


def _mk_skeptic(verdict: Verdict, cid: str = "c-abc123") -> SkepticChallenge:
    return SkepticChallenge(
        candidate_id=cid,
        identification_concerns=["unmeasured U: income drives both T and Y"]
        if verdict != "accept"
        else [],
        rob_concerns=[],
        sutva_concerns=[],
        overall_verdict=verdict,
    )


def _mk_stat(
    verdict: Verdict,
    cid: str = "c-abc123",
    recommended_changes: list[str] | None = None,
) -> StatisticianChallenge:
    return StatisticianChallenge(
        candidate_id=cid,
        estimator_concerns=[]
        if verdict == "accept"
        else ["linear_regression cannot recover NDE under heterogeneity"],
        power_concerns=[],
        cross_fit_concerns=[],
        recommended_changes=recommended_changes or [],
        overall_verdict=verdict,
    )


_CATALOG_MD = (
    "| id | family |\n"
    "|---|---|\n"
    "| rbridge.weightit | weighting |\n"
    "| econml.linear_dml | dml |\n"
)


# ─────────── Three-client routing ─────────────────────────────────────────


def test_three_distinct_clients_route_to_correct_agents() -> None:
    cand = _FakeCandidate()
    skeptic_resp = _mk_skeptic("accept")
    stat_resp = _mk_stat("accept")

    planner = _StubClient("planner", [])  # never called in current design
    skeptic = _StubClient("skeptic", [skeptic_resp])
    statistician = _StubClient("stat", [stat_resp])

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief="A small RCT in primary care.",
        client={
            "planner": planner,
            "skeptic": skeptic,
            "statistician": statistician,
        },
        max_rounds=1,
    )

    assert out.keep is True
    assert len(planner.calls) == 0  # planner is not invoked in this version
    assert len(skeptic.calls) == 1
    assert len(statistician.calls) == 1
    assert skeptic.calls[0]["schema"] is SkepticChallenge
    assert statistician.calls[0]["schema"] is StatisticianChallenge
    # The catalog block should ONLY be substituted into the statistician's
    # system prompt, not the skeptic's.
    assert "rbridge.weightit" in statistician.calls[0]["system"]
    assert "rbridge.weightit" not in skeptic.calls[0]["system"]
    assert "{CATALOG_MARKDOWN}" not in statistician.calls[0]["system"]
    # Honesty preamble should be wired in via with_honesty().
    assert "Honesty rules" in skeptic.calls[0]["system"]
    assert "Honesty rules" in statistician.calls[0]["system"]


def test_single_client_is_reused_for_all_agents() -> None:
    """When the user passes a single client (not a tuple/dict), it gets
    used for all three agent calls."""
    cand = _FakeCandidate()
    client = _StubClient(
        "shared",
        [_mk_skeptic("accept"), _mk_stat("accept")],
    )

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=1,
    )

    assert out.keep is True
    assert len(client.calls) == 2


# ─────────── Consensus across all 8 verdict triples ───────────────────────


# Expected planner-self verdict per (skeptic, statistician):
#   both accept -> accept; both reject -> reject; else revise.
def _expected_planner_verdict(s: Verdict, t: Verdict) -> Verdict:
    if s == "reject" and t == "reject":
        return "reject"
    if s == "accept" and t == "accept":
        return "accept"
    return "revise"


def _expected_keep(s: Verdict, t: Verdict) -> bool:
    p = _expected_planner_verdict(s, t)
    return sum(1 for v in (s, t, p) if v != "reject") >= 2


VERDICTS: tuple[Verdict, Verdict, Verdict] = ("accept", "revise", "reject")


@pytest.mark.parametrize("s,t", list(product(VERDICTS, repeat=2)))
def test_consensus_keep_across_all_verdict_pairs(s: Verdict, t: Verdict) -> None:
    """Skeptic x Statistician = 9 verdict pairs (3x3). With the planner's
    inferred vote that's 9 distinct (s, t, planner) triples; the spec
    asks for coverage of the eight non-degenerate cases. We assert the
    deterministic keep rule for every (s, t) pair."""
    cand = _FakeCandidate()
    client = _StubClient("shared", [_mk_skeptic(s), _mk_stat(t)])

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=1,
    )

    assert out.keep is _expected_keep(s, t), (
        f"keep mismatch for (skeptic={s}, stat={t}): "
        f"got {out.keep}, expected {_expected_keep(s, t)}"
    )
    # The rationale should always name all three verdicts so downstream
    # logs are auditable.
    expected_planner = _expected_planner_verdict(s, t)
    assert f"skeptic={s}" in out.rationale
    assert f"statistician={t}" in out.rationale
    assert f"planner={expected_planner}" in out.rationale


def test_eight_planner_verdict_triples_cover_keep_logic() -> None:
    """Explicit enumeration of the 8 documented cases the spec asks for:
    every triple where keep is decided by majority-not-reject."""
    # Each row is (skeptic, statistician, expected_keep). The planner's
    # vote is derived per _expected_planner_verdict.
    cases: list[tuple[Verdict, Verdict, bool]] = [
        # 1. both accept -> planner accept -> 3 non-reject -> KEEP.
        ("accept", "accept", True),
        # 2. one accept, one revise -> planner revise -> 3 non-reject -> KEEP.
        ("accept", "revise", True),
        ("revise", "accept", True),
        # 3. one accept, one reject -> planner revise -> 2 non-reject -> KEEP.
        ("accept", "reject", True),
        ("reject", "accept", True),
        # 4. both revise -> planner revise -> 3 non-reject -> KEEP.
        ("revise", "revise", True),
        # 5. one revise, one reject -> planner revise -> 2 non-reject -> KEEP.
        ("revise", "reject", True),
        ("reject", "revise", True),
        # 6. both reject -> planner reject -> 0 non-reject -> DROP.
        ("reject", "reject", False),
    ]
    for s, t, expected in cases:
        cand = _FakeCandidate()
        client = _StubClient("shared", [_mk_skeptic(s), _mk_stat(t)])
        out = run_debate(
            candidate=cand,
            completed_history=[],
            catalog_markdown=_CATALOG_MD,
            domain_brief=None,
            client=client,
            max_rounds=1,
        )
        assert out.keep is expected, f"({s},{t}) expected keep={expected}, got {out.keep}"


# ─────────── Transport failure abort path ─────────────────────────────────


def test_skeptic_transport_failure_aborts_safely() -> None:
    cand = _FakeCandidate()
    client = _StubClient("shared", [RuntimeError("ollama down")])

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=2,
    )

    assert out.keep is True
    assert "debate aborted" in out.rationale
    assert "skeptic" in out.rationale
    assert "RuntimeError" in out.rationale
    assert "ollama down" in out.rationale
    assert out.revised_method is None
    assert out.revised_estimand is None
    assert out.candidate_id == cand.candidate_id


def test_statistician_transport_failure_aborts_safely() -> None:
    cand = _FakeCandidate()
    # Skeptic succeeds, then statistician blows up.
    client = _StubClient(
        "shared",
        [_mk_skeptic("revise"), ValueError("bad JSON after retries")],
    )

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=2,
    )

    assert out.keep is True
    assert "debate aborted" in out.rationale
    assert "statistician" in out.rationale
    assert "ValueError" in out.rationale


def test_abort_on_distinct_three_client_setup() -> None:
    """Even with a 3-client dict, a single agent's failure aborts cleanly."""
    cand = _FakeCandidate()
    planner = _StubClient("planner", [])
    skeptic = _StubClient("skeptic", [_mk_skeptic("revise")])
    stat = _StubClient("stat", [ConnectionError("model unloaded")])

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client={"planner": planner, "skeptic": skeptic, "statistician": stat},
        max_rounds=1,
    )

    assert out.keep is True
    assert "ConnectionError" in out.rationale
    assert len(skeptic.calls) == 1
    assert len(stat.calls) == 1


# ─────────── Revised-method only when requested ───────────────────────────


def test_revised_method_set_when_statistician_requested_catalog_id() -> None:
    """Statistician says 'revise' and names a known catalog id in
    recommended_changes -> revised_method is populated."""
    cand = _FakeCandidate(recommended_method="dowhy.linear_regression")
    skeptic_resp = _mk_skeptic("accept")
    stat_resp = _mk_stat(
        "revise",
        recommended_changes=[
            "swap to rbridge.weightit for IPW estimation",
        ],
    )
    client = _StubClient("shared", [skeptic_resp, stat_resp])

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=1,
    )

    assert out.keep is True
    assert out.revised_method == "rbridge.weightit"


def test_revised_method_not_set_when_no_one_requested_change() -> None:
    """All accept -> no revised_method even if the catalog has options."""
    cand = _FakeCandidate()
    client = _StubClient(
        "shared", [_mk_skeptic("accept"), _mk_stat("accept")]
    )

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=1,
    )

    assert out.revised_method is None


def test_revised_method_not_set_when_only_skeptic_asks_for_revision() -> None:
    """Per spec: revised_method requires the *statistician* to have
    requested a catalog id. The skeptic's verdict alone is not enough."""
    cand = _FakeCandidate(recommended_method="dowhy.linear_regression")
    # Skeptic asks for revise, but statistician accepts and doesn't
    # recommend any swap -> revised_method should stay None.
    client = _StubClient(
        "shared", [_mk_skeptic("revise"), _mk_stat("accept")]
    )

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=1,
    )

    assert out.keep is True
    assert out.revised_method is None


def test_revised_method_not_set_when_statistician_changes_lack_catalog_id() -> None:
    """Statistician says revise but only suggests vague text — no
    catalog id appears, so we don't fabricate one."""
    cand = _FakeCandidate(recommended_method="dowhy.linear_regression")
    stat_resp = _mk_stat(
        "revise",
        recommended_changes=["use a doubly-robust estimator", "improve power"],
    )
    client = _StubClient("shared", [_mk_skeptic("accept"), stat_resp])

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=1,
    )

    assert out.revised_method is None


def test_revised_method_skipped_when_change_repeats_current_method() -> None:
    """If the statistician's 'change' is just the candidate's existing
    method, we don't pretend it's a revision."""
    cand = _FakeCandidate(recommended_method="rbridge.weightit")
    stat_resp = _mk_stat(
        "revise",
        recommended_changes=["keep rbridge.weightit but tune lambda"],
    )
    client = _StubClient("shared", [_mk_skeptic("accept"), stat_resp])

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=1,
    )

    assert out.revised_method is None


# ─────────── Round behaviour ──────────────────────────────────────────────


def test_both_accept_short_circuits_to_one_round() -> None:
    """Per the cost target (~3x tokens), if round 1 ends in consensus
    accept, we should not waste tokens on a 2nd round."""
    cand = _FakeCandidate()
    client = _StubClient(
        "shared", [_mk_skeptic("accept"), _mk_stat("accept")]
    )

    run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=2,
    )

    # Only 2 LLM calls — not 4.
    assert len(client.calls) == 2


def test_two_rounds_when_challengers_disagree() -> None:
    """If round 1 has any non-accept verdict, we run a second round so
    each challenger can react to the other's critique. That's 2 + 2 = 4
    calls."""
    cand = _FakeCandidate()
    client = _StubClient(
        "shared",
        [
            _mk_skeptic("revise"),
            _mk_stat("accept"),
            _mk_skeptic("accept"),
            _mk_stat("accept"),
        ],
    )

    run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=2,
    )

    assert len(client.calls) == 4


def test_candidate_id_round_trips_into_consensus() -> None:
    cand = _FakeCandidate(candidate_id="c-zzz999")
    client = _StubClient(
        "shared",
        [
            _mk_skeptic("accept", cid="c-zzz999"),
            _mk_stat("accept", cid="c-zzz999"),
        ],
    )

    out = run_debate(
        candidate=cand,
        completed_history=[],
        catalog_markdown=_CATALOG_MD,
        domain_brief=None,
        client=client,
        max_rounds=1,
    )

    assert out.candidate_id == "c-zzz999"
    assert isinstance(out, DebateConsensus)
