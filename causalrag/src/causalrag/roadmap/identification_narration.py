"""Identification narration — plain-language WHY for Step 5 outputs.

After :func:`causalrag.roadmap.q5_identify.identify_effect` returns an
:class:`IdentificationResult`, the result is technically correct but opaque to
an analyst who is not a causal-inference textbook author. This module wraps
the result in a small LLM call that explains, in 2-4 sentences plus structured
fields, why the chosen adjustment set blocks (or fails to block) the relevant
backdoor / front-door / IV criterion for the *specific* DAG in play.

Design principles (PDD §16):

- **Safe to fail**: the LLM is an enhancement layer, not a gate. If the call
  raises for any reason (cassette miss, schema validation failure, transport
  error), the function returns a deterministic default narration populated
  from the result's strategy label and adjustment set with
  ``confidence="low"``. Identification itself is unaffected.
- **Layer-3 column check** (PDD §16.6): the prompt asks the LLM to only
  reference variable names present in the DAG, and this is *enforced
  post-parse* by dropping any blocked / unblocked path mention that refers
  to an unknown node.
- **No false identifiability claims**: when ``result.identifiable=False``,
  the system prompt explicitly forbids the LLM from asserting that
  identification holds. The narration in that case explains the missing
  piece (collider in the adjustment set, weak instrument, no admissible
  backdoor set, etc.).
- **Refusal channel**: ``confidence="low"`` plus a flagging ``rationale`` is
  the explicit "I'm not confident" exit when the LLM judges identifiability
  to be fundamentally weak (single weak proxy confounder, weak instrument,
  etc.) even if DoWhy formally returned identifiable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.estimand import CausalEstimand
from causalrag.core.graph import CausalGraph
from causalrag.llm.ollama_client import OllamaClient
from causalrag.roadmap.q5_identify import IdentificationResult


class IdentificationNarration(BaseModel):
    """Plain-language narration of a Step 5 :class:`IdentificationResult`."""

    model_config = ConfigDict(extra="forbid")

    strategy_explanation: str = Field(
        ...,
        description=(
            "2-4 sentences explaining WHY the chosen strategy works (or doesn't) "
            "for this DAG."
        ),
    )
    blocked_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Informal descriptions of non-causal paths the adjustment set blocks, "
            "e.g. 'T <- age -> Y blocked by adjusting on age'."
        ),
    )
    unblocked_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Informal descriptions of any remaining open backdoor path the "
            "adjustment set fails to block."
        ),
    )
    analyst_assertions: list[str] = Field(
        default_factory=list,
        description=(
            "Assumptions the analyst is trusting (no unmeasured confounders, "
            "positivity, consistency, exclusion restriction for IVs, etc.)."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description=(
            "How confident the LLM is that identification holds. 'low' is the "
            "refusal channel: even if DoWhy declared identifiable, the LLM may "
            "downgrade to low when the evidence base is weak."
        ),
    )
    rationale: str = Field(
        ...,
        description="Short paragraph the synthesis layer can quote verbatim.",
    )


_SYSTEM_PROMPT = (
    "You are a causal-inference textbook author explaining to a careful reader "
    "WHY a particular adjustment set satisfies (or fails to satisfy) the "
    "backdoor / front-door / instrumental-variable criterion for a specific "
    "directed acyclic graph (DAG). Your audience is a working analyst who "
    "understands the vocabulary of confounders, mediators, and colliders but "
    "wants a clear, faithful explanation for THIS graph — not a generic "
    "textbook recital.\n\n"
    "RULES (these are gates, not suggestions):\n"
    "1. You MUST ONLY reference variable names that appear verbatim in the "
    "   provided DAG node list. Any variable you invent will be silently "
    "   dropped by a downstream validator, which makes your narration weaker.\n"
    "2. If the provided IdentificationResult has identifiable=False, you "
    "MUST NOT claim the effect is identified. Explain the MISSING PIECE instead: "
    "   an unblocked backdoor path, a collider that was incorrectly in the "
    "   adjustment set, a missing instrument, a weak instrument that fails "
    "   relevance/exclusion, or the absence of any valid identification "
    "   strategy under this DAG.\n"
    "3. Be specific about WHICH path each adjustment-set variable blocks. "
    "   Write paths in the form 'T <- X -> Y' (confounder) or 'T -> M -> Y' "
    "   (mediator, which you should NEVER claim to block as it's the causal "
    "   path). Use only DAG node names.\n"
    "4. State the analyst's assumptions plainly: no unmeasured confounders, "
    "   positivity (overlap), consistency (SUTVA), and — for IV — relevance "
    "   and the exclusion restriction.\n"
    "5. If the strategy is technically valid but rests on shaky evidence "
    "   (single weak proxy confounder, instrument with questionable "
    "   exclusion, structural collider risk flagged by the diagnostics), "
    "   set confidence='low' and use the rationale field to flag the concern. "
    "   This is your explicit refusal channel — use it.\n"
    "6. confidence='high' requires a clean adjustment set with multiple "
    "   sensible confounders, no diagnostic warnings, and a strategy that "
    "   fits the estimand class (e.g. IV implies LATE, not ATE).\n\n"
    "Return ONLY a JSON object conforming to the IdentificationNarration "
    "schema. No markdown, no prose preamble."
)


_DONT_CLAIM_IDENTIFICATION_RULE = (
    "the provided IdentificationResult has identifiable=False, you MUST "
    "NOT claim the effect is identified"
)


def _build_prompt(
    *,
    estimand: CausalEstimand,
    graph: CausalGraph,
    result: IdentificationResult,
    domain_brief: str | None,
) -> str:
    """Build the user-side prompt that accompanies the system prompt."""
    parts: list[str] = []
    parts.append("## Causal estimand")
    parts.append(
        f"- class: {estimand.klass.value}\n"
        f"- treatment (T): {estimand.treatment}\n"
        f"- outcome (Y): {estimand.outcome}\n"
        f"- modifiers: {list(estimand.modifiers) or '[]'}\n"
        f"- mediator: {estimand.mediator or 'None'}\n"
        f"- instrument: {estimand.instrument or 'None'}\n"
        f"- formal expression: {estimand.formal_expression}"
    )

    parts.append("\n## DAG")
    parts.append(f"- nodes (the ONLY variable names you may reference): {list(graph.nodes)}")
    if graph.edges:
        edges_str = ", ".join(f"{e.source} -> {e.target}" for e in graph.edges)
    else:
        edges_str = "(no edges)"
    parts.append(f"- edges: {edges_str}")
    if graph.roles:
        roles_str = ", ".join(
            f"{n}={r.value}" for n, r in graph.roles.items()
        )
        parts.append(f"- roles: {roles_str}")

    parts.append("\n## IdentificationResult (Step 5 output)")
    parts.append(
        f"- identifiable: {result.identifiable}\n"
        f"- strategy: {result.strategy}\n"
        f"- adjustment_set: {list(result.adjustment_set)}\n"
        f"- instrument: {result.instrument or 'None'}\n"
        f"- mediator: {result.mediator or 'None'}\n"
        f"- estimand_expression: {result.estimand_expression or 'None'}\n"
        f"- weak: {result.weak}"
    )
    if result.notes:
        parts.append("- notes:")
        for n in result.notes:
            parts.append(f"  - {n}")
    if result.warnings:
        parts.append("- warnings (from Step 5 collider/descendant/mediator guard):")
        for w in result.warnings:
            parts.append(f"  - {w}")

    diag = result.diagnostics or {}
    if any(
        diag.get(k) for k in ("dropped_descendants", "dropped_mediators", "dropped_colliders")
    ):
        parts.append("- diagnostics (variables Step 5 filtered out of the adjustment set):")
        if diag.get("dropped_descendants"):
            parts.append(f"  - dropped_descendants: {list(diag['dropped_descendants'])}")
        if diag.get("dropped_mediators"):
            parts.append(f"  - dropped_mediators: {list(diag['dropped_mediators'])}")
        if diag.get("dropped_colliders"):
            parts.append(f"  - dropped_colliders: {list(diag['dropped_colliders'])}")
        if "original_adjustment_set" in diag:
            parts.append(
                f"  - original_adjustment_set (pre-filter): "
                f"{list(diag['original_adjustment_set'])}"
            )

    if not result.identifiable:
        parts.append(
            "\n## REMINDER\nidentifiable=False — do NOT claim this effect is "
            "identified. Explain the missing piece (collider in set, unblocked "
            "backdoor path, weak/missing instrument, or unsupported estimand "
            "class)."
        )

    if domain_brief:
        parts.append("\n## Domain context (for tone — do not invent variables)")
        parts.append(domain_brief.strip())

    parts.append(
        "\n## Task\nReturn a JSON IdentificationNarration that explains WHY "
        f"the strategy '{result.strategy}' "
        f"{'identifies' if result.identifiable else 'fails to identify'} "
        f"the {estimand.klass.value} of {estimand.treatment} on "
        f"{estimand.outcome} for this DAG. List blocked and any unblocked "
        "paths using ONLY the DAG node names above."
    )
    return "\n".join(parts)


def _default_narration(
    *, result: IdentificationResult, reason: str | None = None
) -> IdentificationNarration:
    """Deterministic fallback used when the LLM call cannot be made or fails.

    The default is honest about its low confidence and surfaces whatever the
    Step 5 machinery already knows: the strategy label, the adjustment set,
    and any warnings from the collider/descendant guard.
    """
    if result.identifiable:
        explanation = (
            f"Step 5 reported the effect is identified via the "
            f"'{result.strategy}' strategy with adjustment set "
            f"{list(result.adjustment_set) or '[]'}. No LLM-generated "
            "narration is available; treat this summary as a minimal "
            "placeholder."
        )
    else:
        explanation = (
            f"Step 5 reported the effect is NOT identifiable under the "
            f"'{result.strategy}' strategy. No LLM-generated narration is "
            "available to explain the failure mode."
        )
    rationale_parts = [explanation]
    if reason:
        rationale_parts.append(f"(narration fallback reason: {reason})")
    if result.warnings:
        rationale_parts.append("Warnings from Step 5: " + "; ".join(result.warnings))

    blocked: list[str] = []
    for v in result.adjustment_set:
        blocked.append(f"adjusted on {v}")

    return IdentificationNarration(
        strategy_explanation=explanation,
        blocked_paths=blocked,
        unblocked_paths=[],
        analyst_assertions=[
            "no unmeasured confounders",
            "positivity (overlap)",
            "consistency (SUTVA)",
        ],
        confidence="low",
        rationale=" ".join(rationale_parts),
    )


def _filter_unknown_nodes(
    narration: IdentificationNarration, graph: CausalGraph
) -> IdentificationNarration:
    """Layer-3 hygiene: drop any blocked/unblocked path that mentions a node
    not present in the DAG.

    The filter is conservative: it drops the *whole* path entry if any
    whitespace-separated token in the entry looks like an unknown variable
    name. Tokens that are common path-notation glyphs (``->``, ``<-``, ``|``,
    punctuation) are ignored.
    """
    known = set(graph.nodes)
    glyphs = {
        "->",
        "<-",
        "<->",
        "→",
        "←",
        "|",
        "blocked",
        "by",
        "adjusting",
        "on",
        "via",
        "through",
        "and",
        "or",
        "the",
        "a",
        "an",
        "path",
        "set",
        "is",
        "not",
        "open",
        "closed",
        "remains",
        "(confounder)",
        "(mediator)",
        "(collider)",
    }

    def _ok(entry: str) -> bool:
        # Strip punctuation that's purely decorative so we can isolate name-like tokens.
        cleaned = (
            entry.replace(",", " ")
            .replace(".", " ")
            .replace(";", " ")
            .replace(":", " ")
            .replace("'", " ")
            .replace('"', " ")
            .replace("(", " ")
            .replace(")", " ")
        )
        for tok in cleaned.split():
            low = tok.lower()
            if low in glyphs:
                continue
            if tok in glyphs:
                continue
            # Tokens that don't look like variable identifiers (contain only
            # symbols or are purely numeric) are fine.
            if not any(ch.isalpha() for ch in tok):
                continue
            # Looks like an identifier — must be in DAG, OR be a known
            # narrative word. Accept tokens that are clearly English words
            # (lowercase, no underscores/digits) but treat any token that
            # also could be a variable name strictly: require an exact match
            # against ``known`` for tokens that resemble variable names.
            if tok in known:
                continue
            # A token resembling a variable name: starts with an uppercase
            # letter, OR contains an underscore or digit, OR is a single
            # capital letter. Reject the entry in that case.
            looks_like_varname = (
                tok[0].isupper()
                or "_" in tok
                or any(ch.isdigit() for ch in tok)
                or (len(tok) <= 3 and tok.isupper())
            )
            if looks_like_varname:
                return False
            # Otherwise it's a generic English word; allow it.
        return True

    narration.blocked_paths = [p for p in narration.blocked_paths if _ok(p)]
    narration.unblocked_paths = [p for p in narration.unblocked_paths if _ok(p)]
    return narration


def narrate_identification(
    *,
    estimand: CausalEstimand,
    graph: CausalGraph,
    result: IdentificationResult,
    domain_brief: str | None,
    client: OllamaClient,
) -> IdentificationNarration:
    """Run the identification-narration LLM call. Never raises.

    Builds a context block from ``(T, Y, estimand_class)``, the DAG edges and
    roles, the adjustment set and strategy label, plus any
    ``warnings`` / ``dropped_*`` diagnostics from Step 5. Calls ``client.parse``
    with the :class:`IdentificationNarration` schema, then applies the layer-3
    node-name filter. If anything goes wrong, returns a deterministic
    fallback narration with ``confidence="low"``.
    """
    system = _SYSTEM_PROMPT
    prompt = _build_prompt(
        estimand=estimand,
        graph=graph,
        result=result,
        domain_brief=domain_brief,
    )

    try:
        response = client.parse(
            prompt=prompt,
            schema=IdentificationNarration,
            system=system,
            json_schema=IdentificationNarration.model_json_schema(),
        )
        narration = response.parsed
        assert isinstance(narration, IdentificationNarration)
    except Exception as e:  # noqa: BLE001 — safe-to-fail by design
        return _default_narration(result=result, reason=f"{type(e).__name__}: {e}")

    # Layer-3 enforcement: drop any path that references a node not in the DAG.
    try:
        narration = _filter_unknown_nodes(narration, graph)
    except Exception:  # noqa: BLE001 — never raise out of narration
        pass

    # Safety: if the result is not identifiable but the LLM claimed high
    # confidence, downgrade. We do NOT rewrite the prose — the analyst should
    # see what the LLM said — but we do not let the confidence label lie.
    if not result.identifiable and narration.confidence == "high":
        narration.confidence = "low"

    return narration


__all__ = [
    "IdentificationNarration",
    "narrate_identification",
]
