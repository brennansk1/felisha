"""Map raw exceptions / failure strings to one-line, actionable hints.

The TUI surfaces these next to ``X failed: <exception>`` so the operator
sees "what to try" instead of an opaque traceback.

Hints are pure heuristics: substring matches over the exception type +
stringified message. Nothing here imports the underlying library — we
just sniff the text. Add new patterns as we encounter them in the wild.
"""

from __future__ import annotations

from typing import Iterable


# Ordered list of (any-of needles, hint). First match wins, so put the
# most specific patterns near the top. ``needles`` are OR-joined: a hint
# fires when ANY of its needles is a substring of the lowercased message.
_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("ollama",),
        "Ollama not reachable -> run `ollama serve` (and `ollama pull` the model).",
    ),
    (
        ("connection refused", "connectionerror"),
        "Local service unreachable -> run `ollama serve` or check the service is up.",
    ),
    (
        ("timed out", "timeout"),
        "Request timed out -> the model may be cold-loading; retry in 30s.",
    ),
    (
        ("model not found", "model: not found"),
        "Model not installed -> run `ollama pull <model>` then retry.",
    ),
    (
        ("no studyprotocol", "studyprotocol",),
        "No protocol found -> run `/init <name>` first to scaffold a project.",
    ),
    (
        ("no such file", "filenotfound"),
        "Path not found -> check the file exists and is reachable from the project dir.",
    ),
    (
        ("permission denied",),
        "Permission denied -> check filesystem permissions on the project dir.",
    ),
    (
        ("empty dataframe", "no rows", "zero rows"),
        "Dataset has no rows -> verify the CSV path and any row filters.",
    ),
    (
        (
            "invalid json",
            "json decode",
            "expecting value",
            "jsondecodeerror",
            "expecting property",
        ),
        "LLM returned invalid JSON -> retry with --no-cache or try a larger reasoning model.",
    ),
    (
        ("validationerror", "field required", "missing field"),
        "Schema mismatch -> the LLM output didn't match the expected schema; retry.",
    ),
    (
        ("could not be inferred",),
        "Treatment/outcome unknown -> pass --treatment <T> --outcome <Y>.",
    ),
    (
        ("not identifiable",),
        "Effect is not identifiable -> retry with --allow-nonidentifiable or revise the DAG.",
    ),
)


def hint_for(exc: BaseException | str) -> str | None:
    """Return a one-line actionable hint, or ``None`` when no rule matches."""
    if isinstance(exc, BaseException):
        text = f"{type(exc).__name__}: {exc}".lower()
    else:
        text = str(exc).lower()
    for needles, hint in _HINTS:
        if any(n in text for n in needles):
            return hint
    return None


def hint_for_any(needles: Iterable[str]) -> str | None:
    """Lookup hint by an explicit list of needles (used by tests)."""
    text = " ".join(needles).lower()
    return hint_for(text)


__all__ = ["hint_for", "hint_for_any"]
