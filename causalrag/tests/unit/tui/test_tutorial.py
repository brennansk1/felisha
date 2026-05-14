"""Unit tests for the TUI tutorial mode."""

from __future__ import annotations

import importlib
import sys

import pandas as pd
import pytest

from causalrag.tui.tutorial import (
    IHDP_TUTORIAL,
    LALONDE_TUTORIAL,
    Tutorial,
    TutorialStep,
    get_tutorial,
    list_tutorials,
    load_ihdp,
    load_lalonde,
    render_tutorial_step,
)


# --- Registry --------------------------------------------------------------


def test_list_tutorials_returns_known_names() -> None:
    names = list_tutorials()
    assert "lalonde" in names
    assert "ihdp" in names
    assert names == sorted(names), "list_tutorials must return stable, sorted order"


def test_get_tutorial_returns_correct_tutorial() -> None:
    assert get_tutorial("lalonde") is LALONDE_TUTORIAL
    assert get_tutorial("ihdp") is IHDP_TUTORIAL


def test_get_tutorial_raises_for_unknown() -> None:
    with pytest.raises(KeyError, match="Unknown tutorial"):
        get_tutorial("not-a-real-tutorial")


# --- Shape: each tutorial has ≥ 5 ordered steps -----------------------------


@pytest.mark.parametrize("tutorial", [LALONDE_TUTORIAL, IHDP_TUTORIAL], ids=lambda t: t.name)
def test_tutorial_has_at_least_five_steps(tutorial: Tutorial) -> None:
    assert len(tutorial.steps) >= 5, (
        f"{tutorial.name} must have ≥ 5 steps; has {len(tutorial.steps)}"
    )


@pytest.mark.parametrize("tutorial", [LALONDE_TUTORIAL, IHDP_TUTORIAL], ids=lambda t: t.name)
def test_steps_are_ordered_by_phase(tutorial: Tutorial) -> None:
    phases = [s.phase for s in tutorial.steps]
    assert phases == sorted(phases), (
        f"{tutorial.name} phases out of order: {phases}"
    )


@pytest.mark.parametrize("tutorial", [LALONDE_TUTORIAL, IHDP_TUTORIAL], ids=lambda t: t.name)
def test_steps_cover_full_roadmap(tutorial: Tutorial) -> None:
    """The PDD spec demands init → discover → hypothesize → estimate →
    sensitivity → report. Every tutorial must touch each one."""
    names = {s.name for s in tutorial.steps}
    required = {"init", "discover", "hypothesize", "estimate", "sensitivity", "report"}
    missing = required - names
    assert not missing, f"{tutorial.name} is missing required steps: {missing}"


@pytest.mark.parametrize("tutorial", [LALONDE_TUTORIAL, IHDP_TUTORIAL], ids=lambda t: t.name)
def test_every_step_has_non_empty_fields(tutorial: Tutorial) -> None:
    for step in tutorial.steps:
        assert step.name
        assert step.prompt.strip()
        assert step.expected_command.startswith("/"), (
            f"{step.name}: expected_command must be a slash command, got "
            f"{step.expected_command!r}"
        )
        assert step.hint.strip()


# --- Dataset loaders --------------------------------------------------------


def test_load_lalonde_returns_dataframe_and_info() -> None:
    df, info = load_lalonde()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert info["name"] == "lalonde"
    assert info["treatment"] == "treat"
    assert info["outcome"] == "re78"
    assert info["source"] in {"causaldata", "synthetic"}
    # Required columns survive both branches
    for col in ("treat", "re78"):
        assert col in df.columns


def test_load_lalonde_falls_back_when_causaldata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a clean install with no ``causaldata`` extra."""
    # Block both already-imported and to-be-imported paths.
    monkeypatch.setitem(sys.modules, "causaldata", None)

    # Force a fresh import so the loader re-evaluates the try/except.
    import causalrag.tui.tutorial as tutorial_mod

    tutorial_mod = importlib.reload(tutorial_mod)
    df, info = tutorial_mod.load_lalonde()
    assert info["source"] == "synthetic"
    assert info["true_ate"] == pytest.approx(1700.0)
    assert "treat" in df.columns and "re78" in df.columns
    assert len(df) > 100  # synthetic frame is non-trivial


def test_load_ihdp_returns_synthetic_frame() -> None:
    df, info = load_ihdp()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 747
    assert info["treatment"] == "treat"
    assert info["outcome"] == "y"
    assert info["source"] == "synthetic"
    assert isinstance(info["true_ate"], float)
    # Heterogeneity baked in via x_cont_0 and x_cont_1
    assert "x_cont_0" in df.columns
    assert "x_cont_1" in df.columns


def test_load_ihdp_is_deterministic() -> None:
    df1, info1 = load_ihdp()
    df2, info2 = load_ihdp()
    pd.testing.assert_frame_equal(df1, df2)
    assert info1["true_ate"] == info2["true_ate"]


@pytest.mark.parametrize(
    "tutorial",
    [LALONDE_TUTORIAL, IHDP_TUTORIAL],
    ids=lambda t: t.name,
)
def test_dataset_loader_is_callable_and_returns_pair(tutorial: Tutorial) -> None:
    df, info = tutorial.dataset_loader()
    assert isinstance(df, pd.DataFrame)
    assert isinstance(info, dict)
    assert "treatment" in info and "outcome" in info
    assert info["treatment"] in df.columns
    assert info["outcome"] in df.columns


# --- Rendering --------------------------------------------------------------


def test_render_tutorial_step_produces_non_empty_markdown() -> None:
    step = LALONDE_TUTORIAL.steps[0]
    rendered = render_tutorial_step(step)
    assert isinstance(rendered, str)
    assert rendered.strip(), "rendered step must not be empty"
    # Markdown features
    assert "###" in rendered  # heading
    assert step.expected_command in rendered
    assert step.hint in rendered


def test_render_tutorial_step_marks_automated_steps() -> None:
    auto_step = TutorialStep(
        name="init",
        phase=0,
        prompt="## go",
        expected_command="/init x",
        hint="why",
        automated=True,
    )
    manual_step = TutorialStep(
        name="discover",
        phase=1,
        prompt="## go",
        expected_command="/discover x",
        hint="why",
        automated=False,
    )
    assert "automated" in render_tutorial_step(auto_step).lower()
    assert "automated" not in render_tutorial_step(manual_step).lower()


@pytest.mark.parametrize("tutorial", [LALONDE_TUTORIAL, IHDP_TUTORIAL], ids=lambda t: t.name)
def test_every_step_renders(tutorial: Tutorial) -> None:
    for step in tutorial.steps:
        out = render_tutorial_step(step)
        assert out.strip()
        assert step.expected_command in out
