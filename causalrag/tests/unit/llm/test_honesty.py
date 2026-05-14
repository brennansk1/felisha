"""Tests for the shared honesty preamble + refusal-channel module."""

from __future__ import annotations

from pathlib import Path

from causalrag.llm.honesty import HONESTY_PREAMBLE, REFUSAL_HOOK, with_honesty


def test_with_honesty_contains_preamble_and_ends_with_input() -> None:
    out = with_honesty("hello")
    assert "Honesty rules" in out
    assert out.endswith("hello")


def test_with_honesty_includes_refusal_hook() -> None:
    out = with_honesty("system body")
    assert HONESTY_PREAMBLE in out
    assert REFUSAL_HOOK in out


def test_with_honesty_is_prepended_not_replaced() -> None:
    original = "Original prompt body — keep me intact."
    out = with_honesty(original)
    assert original in out
    # the preamble comes before the original
    assert out.index("Honesty rules") < out.index(original)


PROMPT_MODULES = [
    "src/causalrag/discovery/expert.py",
    "src/causalrag/discovery/investigator.py",
    "src/causalrag/hypothesize/master.py",
    "src/causalrag/hypothesize/automated.py",
    "src/causalrag/reporting/synthesis.py",
    "src/causalrag/roadmap/q8_interpret.py",
]


def _repo_root() -> Path:
    # tests/unit/llm/test_honesty.py -> repo root is three parents up
    return Path(__file__).resolve().parents[3]


def test_all_prompt_modules_use_with_honesty() -> None:
    """Grep-style assertion that every prompt-emitting module wraps its
    system prompt with the shared honesty helper."""
    root = _repo_root()
    for rel in PROMPT_MODULES:
        path = root / rel
        assert path.exists(), f"Missing prompt module: {path}"
        text = path.read_text(encoding="utf-8")
        assert "with_honesty" in text, f"{rel} does not invoke with_honesty()"
