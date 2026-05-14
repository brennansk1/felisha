from __future__ import annotations

import numpy as np
import pandas as pd

from causalrag.data.selection import resolve_method, select_variables


def test_resolve_method_high_dim_picks_post_double() -> None:
    assert resolve_method("auto", n_candidates=50, high_dimensional=True) == "post_double_selection"


def test_resolve_method_few_candidates_skips_selection() -> None:
    assert resolve_method("auto", n_candidates=3, high_dimensional=False) == "none"


def test_resolve_method_moderate_uses_correlation_pruning() -> None:
    assert resolve_method("auto", n_candidates=10, high_dimensional=False) == "correlation_pruning"


def test_correlation_pruning_drops_collinear() -> None:
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x1_twin": x1 + rng.normal(scale=0.01, size=n),  # essentially x1
            "x2": rng.normal(size=n),
            "treat": rng.binomial(1, 0.5, size=n),
            "y": rng.normal(size=n),
        }
    )
    result = select_variables(
        df,
        "treat",
        "y",
        ("x1", "x1_twin", "x2"),
        method="correlation_pruning",
    )
    assert "x1_twin" in result.dropped or "x1" in result.dropped
    assert "x2" in result.selected


def test_post_double_selection_keeps_relevant() -> None:
    """Union mode (default): keep x_relevant; some noise drops accepted."""
    rng = np.random.default_rng(1)
    n = 600
    x_relevant = rng.normal(size=n)
    noise_features = {f"noise_{i}": rng.normal(size=n) for i in range(20)}
    treat = (0.7 * x_relevant + rng.normal(scale=0.5, size=n) > 0).astype(int)
    y = 2.0 * treat + 1.5 * x_relevant + rng.normal(size=n)
    df = pd.DataFrame({"x_relevant": x_relevant, **noise_features, "treat": treat, "y": y})
    result = select_variables(
        df,
        "treat",
        "y",
        tuple(["x_relevant"] + list(noise_features.keys())),
        method="post_double_selection",
    )
    assert "x_relevant" in result.selected
    # Some noise should drop (the BCH spirit) — exact count varies by CV path.
    assert len(result.dropped) >= 3


def test_lasso_intersection_aggressive_dropping() -> None:
    """Intersection mode: only variables that BOTH Lassos pick. Should be strict."""
    rng = np.random.default_rng(1)
    n = 600
    x_relevant = rng.normal(size=n)
    noise_features = {f"noise_{i}": rng.normal(size=n) for i in range(20)}
    treat = (0.7 * x_relevant + rng.normal(scale=0.5, size=n) > 0).astype(int)
    y = 2.0 * treat + 1.5 * x_relevant + rng.normal(size=n)
    df = pd.DataFrame({"x_relevant": x_relevant, **noise_features, "treat": treat, "y": y})
    result = select_variables(
        df,
        "treat",
        "y",
        tuple(["x_relevant"] + list(noise_features.keys())),
        method="lasso_intersection",
    )
    assert "x_relevant" in result.selected
    # Intersection is strict: most noise should drop.
    assert len(result.dropped) >= 15


def test_pinned_variables_always_kept() -> None:
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "x": x,
            "x_twin": x + rng.normal(scale=0.01, size=n),
            "treat": rng.binomial(1, 0.5, size=n),
            "y": rng.normal(size=n),
        }
    )
    result = select_variables(
        df,
        "treat",
        "y",
        ("x", "x_twin"),
        method="correlation_pruning",
        pinned=("x_twin",),
    )
    assert "x_twin" in result.selected
