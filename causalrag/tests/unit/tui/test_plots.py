"""Unit tests for the PlotPanel widget (Sprint 4.2).

The FALLBACK path (no plotext / textual-plotext installed) must always
exercise — those tests monkey-patch the module flags to ``False`` so
they pass regardless of whether the optional deps are installed.

A small block of tests at the bottom uses ``pytest.importorskip`` to
exercise the *real* plotext-backed rendering when the dep is present.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrag.tui.widgets import plots as plots_mod
from causalrag.tui.widgets.plots import PlotPanel


# ---------------------------------------------------------------------------
# Fallback path — force the no-plotext branch.
# ---------------------------------------------------------------------------


@pytest.fixture
def no_plotext(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force PlotPanel to use its ASCII fallback path."""
    monkeypatch.setattr(plots_mod, "_HAS_PLOTEXT", False)
    monkeypatch.setattr(plots_mod, "_HAS_TEXTUAL_PLOTEXT", False)


def _nonempty(text: str) -> bool:
    return isinstance(text, str) and bool(text.strip())


# ----- constructor ---------------------------------------------------------


def test_plot_panel_constructs_with_title(no_plotext: None) -> None:
    p = PlotPanel(title="diagnostics")
    assert p.title == "diagnostics"
    assert p.last_kind is None
    assert p.last_payload is None
    # The header should already have been emitted.
    assert "diagnostics" in p.last_text


def test_plot_panel_constructs_without_title(no_plotext: None) -> None:
    p = PlotPanel()
    assert p.title == ""
    assert p.last_text == ""


# ----- power curve ---------------------------------------------------------


def test_render_power_curve_fallback(no_plotext: None) -> None:
    p = PlotPanel(title="power")
    p.render_power_curve([50, 100, 200, 400], [0.10, 0.45, 0.80, 0.95])
    assert p.last_kind == "power_curve"
    assert p.last_payload == {
        "ns": [50, 100, 200, 400],
        "powers": [0.10, 0.45, 0.80, 0.95],
    }
    assert _nonempty(p.last_text)
    assert "power" in p.last_text.lower()


def test_render_power_curve_empty_data(no_plotext: None) -> None:
    p = PlotPanel()
    p.render_power_curve([], [])
    assert p.last_kind == "power_curve"
    assert "no data" in p.last_text.lower()


# ----- love plot -----------------------------------------------------------


def test_render_love_plot_fallback(no_plotext: None) -> None:
    p = PlotPanel(title="balance")
    before = {"age": 0.40, "bmi": 0.25, "smoker": 0.55}
    after = {"age": 0.05, "bmi": 0.08, "smoker": 0.04}
    p.render_love_plot(before, after)

    assert p.last_kind == "love_plot"
    assert p.last_payload == {"smds_before": before, "smds_after": after}
    txt = p.last_text
    assert _nonempty(txt)
    # Every covariate label should appear.
    for k in before:
        assert k in txt
    # And the down-arrow glyph should appear since each SMD shrank.
    assert "↓" in txt


def test_render_love_plot_empty(no_plotext: None) -> None:
    p = PlotPanel()
    p.render_love_plot({}, {})
    assert p.last_kind == "love_plot"
    assert "no covariates" in p.last_text.lower()


def test_render_love_plot_handles_missing_keys(no_plotext: None) -> None:
    p = PlotPanel()
    p.render_love_plot({"age": 0.3}, {"bmi": 0.1})  # disjoint
    # Should still render without raising.
    assert "age" in p.last_text
    assert "bmi" in p.last_text


# ----- propensity overlap --------------------------------------------------


def test_render_propensity_overlap_fallback(no_plotext: None) -> None:
    rng = np.random.default_rng(0)
    treated = rng.beta(2, 5, size=200)
    control = rng.beta(5, 2, size=200)

    p = PlotPanel(title="overlap")
    p.render_propensity_overlap(treated, control)

    assert p.last_kind == "propensity_overlap"
    assert isinstance(p.last_payload, dict)
    assert p.last_payload["scores_treated"].shape == (200,)
    assert p.last_payload["scores_control"].shape == (200,)
    txt = p.last_text
    assert "treated" in txt and "control" in txt
    assert "n=200" in txt


def test_render_propensity_overlap_empty(no_plotext: None) -> None:
    p = PlotPanel()
    p.render_propensity_overlap(np.array([]), np.array([]))
    assert p.last_kind == "propensity_overlap"
    assert "no scores" in p.last_text.lower()


# ----- CATE PDP ------------------------------------------------------------


def test_render_cate_pdp_fallback(no_plotext: None) -> None:
    x = np.linspace(0, 1, 50)
    cate = 0.5 + 0.3 * np.sin(2 * np.pi * x)
    p = PlotPanel(title="cate")
    p.render_cate_pdp(x, cate, feature_name="age_z")

    assert p.last_kind == "cate_pdp"
    assert p.last_payload is not None
    assert p.last_payload["feature_name"] == "age_z"
    assert p.last_payload["x"].shape == (50,)
    txt = p.last_text
    assert "age_z" in txt
    assert "CATE" in txt


def test_render_cate_pdp_empty(no_plotext: None) -> None:
    p = PlotPanel()
    p.render_cate_pdp(np.array([]), np.array([]), feature_name="z")
    assert "no data" in p.last_text.lower()


# ----- sensemakr contour ---------------------------------------------------


def test_render_sensemakr_contour_fallback(no_plotext: None) -> None:
    r2dz = np.linspace(0, 0.5, 11)
    r2yz = np.linspace(0, 0.5, 11)
    # A toy adjusted-estimate grid.
    z = (1.0 - np.outer(r2yz, r2dz)) * 1.5
    grid = {"r2dz_x": r2dz, "r2yz_dx": r2yz, "z": z}

    p = PlotPanel(title="sense")
    p.render_sensemakr_contour(grid)

    assert p.last_kind == "sensemakr_contour"
    assert _nonempty(p.last_text)
    assert "sensemakr" in p.last_text.lower()


def test_render_sensemakr_contour_empty(no_plotext: None) -> None:
    p = PlotPanel()
    p.render_sensemakr_contour({})
    assert p.last_kind == "sensemakr_contour"
    assert "no grid" in p.last_text.lower()


# ----- payload retention ---------------------------------------------------


def test_consecutive_renders_overwrite_last_payload(no_plotext: None) -> None:
    p = PlotPanel()
    p.render_power_curve([10, 20], [0.1, 0.2])
    assert p.last_kind == "power_curve"
    p.render_love_plot({"a": 0.2}, {"a": 0.01})
    assert p.last_kind == "love_plot"
    assert "smds_before" in (p.last_payload or {})


# ---------------------------------------------------------------------------
# Real-plotext path (skipped if dep absent).
# ---------------------------------------------------------------------------


def test_render_power_curve_with_plotext() -> None:
    pytest.importorskip("plotext")
    p = PlotPanel(title="power")
    p.render_power_curve([50, 100, 200, 400], [0.1, 0.5, 0.8, 0.95])
    assert p.last_kind == "power_curve"
    assert _nonempty(p.last_text)


def test_render_love_plot_with_plotext() -> None:
    pytest.importorskip("plotext")
    p = PlotPanel()
    p.render_love_plot({"a": 0.4, "b": 0.3}, {"a": 0.05, "b": 0.04})
    assert _nonempty(p.last_text)


def test_render_propensity_overlap_with_plotext() -> None:
    pytest.importorskip("plotext")
    rng = np.random.default_rng(1)
    p = PlotPanel()
    p.render_propensity_overlap(
        rng.beta(2, 5, size=100), rng.beta(5, 2, size=100)
    )
    assert _nonempty(p.last_text)


def test_render_cate_pdp_with_plotext() -> None:
    pytest.importorskip("plotext")
    p = PlotPanel()
    x = np.linspace(0, 1, 30)
    p.render_cate_pdp(x, np.sin(x), feature_name="x")
    assert _nonempty(p.last_text)


def test_render_sensemakr_contour_with_plotext() -> None:
    pytest.importorskip("plotext")
    p = PlotPanel()
    r2 = np.linspace(0, 0.4, 6)
    grid = {"r2dz_x": r2, "r2yz_dx": r2, "z": np.outer(r2, r2)}
    p.render_sensemakr_contour(grid)
    assert _nonempty(p.last_text)
