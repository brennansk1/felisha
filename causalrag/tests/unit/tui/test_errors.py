"""Unit tests for the TUI error-hint mapper."""

from __future__ import annotations

import pytest

from causalrag.tui.errors import hint_for


def test_no_hint_for_unknown_exception() -> None:
    assert hint_for(RuntimeError("something completely unrelated")) is None


def test_hint_for_ollama_connection_refused() -> None:
    exc = ConnectionError("HTTPConnectionPool: Connection refused - ollama")
    hint = hint_for(exc)
    assert hint is not None
    assert "ollama serve" in hint.lower()


def test_hint_for_filenotfound() -> None:
    exc = FileNotFoundError("[Errno 2] No such file or directory: 'data/cohort.csv'")
    hint = hint_for(exc)
    assert hint is not None
    assert "path" in hint.lower()


def test_hint_for_invalid_json() -> None:
    hint = hint_for("JSONDecodeError: Expecting value: line 1 column 1")
    assert hint is not None
    assert "--no-cache" in hint or "json" in hint.lower()


def test_hint_for_timeout() -> None:
    hint = hint_for(TimeoutError("Request timed out after 60s"))
    assert hint is not None
    assert "retry" in hint.lower() or "cold" in hint.lower()


def test_hint_for_empty_dataset() -> None:
    hint = hint_for(ValueError("Empty dataframe — zero rows after filter"))
    assert hint is not None
    assert "rows" in hint.lower() or "csv" in hint.lower()


def test_hint_for_missing_protocol() -> None:
    hint = hint_for("No StudyProtocol at /project/study.causalrag.yaml")
    assert hint is not None
    assert "init" in hint.lower()


def test_hint_accepts_plain_string() -> None:
    """`hint_for` is the same function the dispatcher feeds with raw event
    messages; verify it handles strings."""
    assert hint_for("Connection refused to Ollama") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
