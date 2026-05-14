"""Unit tests for composer path-argument completion."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from causalrag.tui.completion import (
    apply_completion,
    common_prefix,
    complete_path,
    needs_path_completion,
)


def test_needs_completion_for_discover_path_arg() -> None:
    assert needs_path_completion("/discover data") is True
    assert needs_path_completion("/discover ") is True
    # After the path token has whitespace following it AND a flag exists,
    # we're past the path slot.
    assert needs_path_completion("/discover data/cohort.csv --treatment") is False


def test_does_not_need_completion_for_non_path_commands() -> None:
    assert needs_path_completion("/help") is False
    assert needs_path_completion("/estimate --treatment T") is False
    assert needs_path_completion("/doctor") is False


def test_needs_completion_for_auto_run_path() -> None:
    # /auto with no args is the verb slot, not a path
    assert needs_path_completion("/auto run data") is True
    assert needs_path_completion("/auto run ") is True


def test_complete_path_returns_directory_with_slash(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data2").mkdir()
    (tmp_path / "README.md").write_text("hello")
    out = complete_path("/discover dat", cwd=tmp_path)
    assert "data/" in out
    assert "data2/" in out
    # README does NOT start with 'dat'
    assert all("README" not in o for o in out)


def test_complete_path_descends_into_subdir(tmp_path: Path) -> None:
    sub = tmp_path / "data"
    sub.mkdir()
    (sub / "cohort.csv").write_text("a,b\n1,2")
    (sub / "lab.parquet").write_bytes(b"PAR1")
    out = complete_path("/discover data/coh", cwd=tmp_path)
    assert "data/cohort.csv" in out


def test_complete_path_hides_hidden_files_unless_dot_typed(tmp_path: Path) -> None:
    (tmp_path / ".secret").write_text("x")
    (tmp_path / "public.csv").write_text("x")
    plain = complete_path("/discover ", cwd=tmp_path)
    assert all(not o.startswith(".") for o in plain)
    dotted = complete_path("/discover .", cwd=tmp_path)
    assert any(o.startswith(".secret") for o in dotted)


def test_apply_completion_replaces_trailing_token() -> None:
    assert apply_completion("/discover dat", "data/") == "/discover data/"
    assert apply_completion("/discover ", "data/") == "/discover data/"
    assert apply_completion("/discover data/coh", "data/cohort.csv") == "/discover data/cohort.csv"


def test_common_prefix_basic() -> None:
    assert common_prefix(["data/", "data2/", "database/"]) == "data"
    assert common_prefix(["foo", "bar"]) == ""
    assert common_prefix(["only-one"]) == "only-one"
    assert common_prefix([]) == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
