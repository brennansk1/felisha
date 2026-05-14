"""Unit tests for ``causalrag.roadmap.identification_narration``.

Covers the four behaviours the module promises:

1. Happy path — a valid LLM JSON parses into an :class:`IdentificationNarration`.
2. Safe-to-fail — when the underlying client raises, the function returns a
   default narration (never re-raises).
3. Layer-3 node-name filter — paths that reference variables not in the DAG
   are dropped from the parsed narration.
4. The system prompt explicitly forbids claiming identification when
   ``result.identifiable=False`` — verified by a string-grep on the prompt.
"""

from __future__ import annotations

from typing import Any

import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.graph import CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.llm.ollama_client import FakeOllamaTransport, OllamaClient
from causalrag.roadmap.identification_narration import (
    IdentificationNarration,
    _SYSTEM_PROMPT,
    _build_prompt,
    narrate_identification,
)
from causalrag.roadmap.q5_identify import IdentificationResult


# ---- Fixtures ---------------------------------------------------------------


def _estimand(
    treatment: str = "T",
    outcome: str = "Y",
    klass: EstimandClass = EstimandClass.ATE,
) -> CausalEstimand:
    return CausalEstimand.model_validate(
        {
            "class": klass,
            "treatment": treatment,
            "outcome": outcome,
            "modifiers": (),
            "mediator": None,
            "instrument": None,
            "formal_expression": "E[Y(1)-Y(0)]",
        }
    )


def _graph() -> CausalGraph:
    return CausalGraph.from_edge_list(
        [("Z", "T"), ("Z", "Y"), ("T", "Y")],
        roles={
            "T": VariableRole.TREATMENT,
            "Y": VariableRole.OUTCOME,
            "Z": VariableRole.CONFOUNDER,
        },
    )


def _result(identifiable: bool = True) -> IdentificationResult:
    if identifiable:
        return IdentificationResult(
            identifiable=True,
            strategy="backdoor",
            adjustment_set=("Z",),
            estimand_expression="E[Y|do(T)]",
        )
    return IdentificationResult(
        identifiable=False,
        strategy="non-identifiable",
        notes=["No admissible backdoor set found."],
    )


class _RaisingClient:
    """Drop-in OllamaClient stub that always raises on ``parse``."""

    def parse(self, **_: Any) -> Any:
        raise RuntimeError("transport blew up")


def _make_client(response: dict | str) -> OllamaClient:
    """Build an :class:`OllamaClient` wired to a fake transport.

    The fake transport returns the same response for any prompt (empty-string
    key match), which is exactly the shape we want for these tests.
    """
    transport = FakeOllamaTransport(responses={"": response})
    return OllamaClient(
        model="fake-model",
        transport=transport,
        allow_live=True,
        cassette_dir=None,
    )


# ---- 1. Happy path ----------------------------------------------------------


def test_happy_path_returns_parsed_narration() -> None:
    payload = {
        "strategy_explanation": (
            "The single backdoor path T <- Z -> Y is blocked by conditioning "
            "on Z, so the do(T)-distribution of Y collapses to a "
            "covariate-adjusted average."
        ),
        "blocked_paths": ["T <- Z -> Y blocked by adjusting on Z"],
        "unblocked_paths": [],
        "analyst_assertions": [
            "no unmeasured confounders beyond Z",
            "positivity over Z",
            "consistency",
        ],
        "confidence": "high",
        "rationale": (
            "Backdoor adjustment on Z identifies the ATE under the standard "
            "no-unmeasured-confounding assumption for this two-cause DAG."
        ),
    }
    client = _make_client(payload)
    narration = narrate_identification(
        estimand=_estimand(),
        graph=_graph(),
        result=_result(identifiable=True),
        domain_brief=None,
        client=client,
    )
    assert isinstance(narration, IdentificationNarration)
    assert narration.confidence == "high"
    assert narration.blocked_paths == ["T <- Z -> Y blocked by adjusting on Z"]
    assert narration.unblocked_paths == []
    assert "Z" in narration.strategy_explanation


# ---- 2. Safe-to-fail --------------------------------------------------------


def test_returns_default_when_client_raises() -> None:
    narration = narrate_identification(
        estimand=_estimand(),
        graph=_graph(),
        result=_result(identifiable=True),
        domain_brief=None,
        client=_RaisingClient(),  # type: ignore[arg-type]
    )
    assert isinstance(narration, IdentificationNarration)
    assert narration.confidence == "low"
    # The fallback summarises the strategy + adjustment set.
    assert "backdoor" in narration.strategy_explanation
    assert "adjusted on Z" in narration.blocked_paths


def test_returns_default_when_client_raises_for_non_identifiable() -> None:
    narration = narrate_identification(
        estimand=_estimand(),
        graph=_graph(),
        result=_result(identifiable=False),
        domain_brief=None,
        client=_RaisingClient(),  # type: ignore[arg-type]
    )
    assert narration.confidence == "low"
    assert "NOT identifiable" in narration.strategy_explanation


# ---- 3. Layer-3 node-name filter --------------------------------------------


def test_filters_paths_referencing_unknown_variables() -> None:
    # The LLM mentions "Q" — a variable NOT in the DAG (DAG has T, Y, Z).
    # The blocked-path entry that names Q must be dropped; the entry that
    # only references Z must survive.
    payload = {
        "strategy_explanation": "Adjusting on Z blocks the T <- Z -> Y backdoor.",
        "blocked_paths": [
            "T <- Z -> Y blocked by adjusting on Z",
            "T <- Q -> Y blocked by adjusting on Q",  # Q is hallucinated
        ],
        "unblocked_paths": [
            "T <- Foo -> Y remains open",  # Foo is hallucinated
        ],
        "analyst_assertions": ["no unmeasured confounders"],
        "confidence": "medium",
        "rationale": "Backdoor on Z identifies the ATE.",
    }
    client = _make_client(payload)
    narration = narrate_identification(
        estimand=_estimand(),
        graph=_graph(),
        result=_result(identifiable=True),
        domain_brief=None,
        client=client,
    )
    # Z-only entry survives; Q and Foo entries are dropped.
    assert narration.blocked_paths == ["T <- Z -> Y blocked by adjusting on Z"]
    assert narration.unblocked_paths == []


def test_high_confidence_downgraded_when_not_identifiable() -> None:
    # LLM ignored the gate and claimed high confidence on a non-identifiable
    # result. The function should downgrade to 'low' without rewriting prose.
    payload = {
        "strategy_explanation": "Mistakenly claimed identification.",
        "blocked_paths": [],
        "unblocked_paths": ["T <- Z -> Y still open (no valid set)"],
        "analyst_assertions": [],
        "confidence": "high",
        "rationale": "(LLM error: claimed high confidence on a non-identifiable result)",
    }
    client = _make_client(payload)
    narration = narrate_identification(
        estimand=_estimand(),
        graph=_graph(),
        result=_result(identifiable=False),
        domain_brief=None,
        client=client,
    )
    assert narration.confidence == "low"


# ---- 4. Prompt rules --------------------------------------------------------


def test_system_prompt_contains_dont_claim_identification_rule() -> None:
    # When ``result.identifiable=False``, the system prompt must explicitly
    # forbid the LLM from claiming the effect is identified.
    assert "identifiable=False" in _SYSTEM_PROMPT
    assert (
        "MUST NOT claim the effect is identified" in _SYSTEM_PROMPT
        or "MUST NOT claim" in _SYSTEM_PROMPT
    )


def test_user_prompt_reminds_when_not_identifiable() -> None:
    prompt = _build_prompt(
        estimand=_estimand(),
        graph=_graph(),
        result=_result(identifiable=False),
        domain_brief=None,
    )
    assert "identifiable=False" in prompt
    assert "do NOT claim" in prompt


def test_user_prompt_lists_dag_nodes_and_strategy() -> None:
    prompt = _build_prompt(
        estimand=_estimand(),
        graph=_graph(),
        result=_result(identifiable=True),
        domain_brief=None,
    )
    # The DAG node list must be in the prompt verbatim so the LLM knows the
    # allowed vocabulary.
    assert "'T'" in prompt or "T" in prompt
    assert "'Z'" in prompt or "Z" in prompt
    assert "backdoor" in prompt
    assert "ATE" in prompt


def test_user_prompt_includes_diagnostics_when_present() -> None:
    result = IdentificationResult(
        identifiable=True,
        strategy="backdoor",
        adjustment_set=("Z",),
        warnings=["Dropped colliders on T-Y paths from adjustment set: ['C']"],
        diagnostics={
            "dropped_colliders": ["C"],
            "dropped_mediators": [],
            "dropped_descendants": [],
            "original_adjustment_set": ["C", "Z"],
        },
    )
    prompt = _build_prompt(
        estimand=_estimand(),
        graph=_graph(),
        result=result,
        domain_brief=None,
    )
    assert "dropped_colliders" in prompt
    assert "Dropped colliders" in prompt


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-q"])
