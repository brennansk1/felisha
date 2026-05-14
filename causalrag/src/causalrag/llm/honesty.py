"""Shared honesty preamble + refusal-channel definitions used by every
LLM call in the pipeline. Goal: make HOW the model should behave
(when it should refuse, what it must not invent, output format) live
in one place rather than be re-stated inconsistently in every
prompt."""

HONESTY_PREAMBLE = (
    "## Honesty rules — non-negotiable\n"
    "  1. Return ONLY a JSON object that conforms to the schema. No "
    "     prose, no preface, no trailing remarks.\n"
    "  2. NEVER fabricate column names, estimator ids, or numerical "
    "     values that were not provided to you. Reference only what "
    "     you can quote from the context above.\n"
    "  3. If you don't know, say so via the refusal hook (the schema "
    "     provides a way — use a low confidence field, an explicit "
    "     'unknown' enum, or an empty list with a rationale).\n"
    "  4. Quantify with the units of the underlying data. Don't "
    "     invent extrapolations, annualizations, or population scaling "
    "     that wasn't supplied.\n"
    "  5. Don't restate these rules in your output.\n"
)

REFUSAL_HOOK = (
    "If the data does not support the requested judgement, prefer the "
    "schema's refusal/uncertainty option (e.g., 'unknown', empty list, "
    "or low-confidence rationale) over fabricating an answer. A "
    "well-justified refusal is better than a confident wrong answer."
)


def with_honesty(system_prompt: str) -> str:
    """Prepend the shared honesty preamble and refusal hook to a per-call system prompt."""
    return HONESTY_PREAMBLE + "\n" + REFUSAL_HOOK + "\n\n" + system_prompt


__all__ = ["HONESTY_PREAMBLE", "REFUSAL_HOOK", "with_honesty"]
