"""Tests for continuous_positivity_check (PDD §13)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.data.checks import continuous_positivity_check


def _df(t: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"T": t, "X": np.zeros_like(t, dtype=float)})


def test_uniform_treatment_is_green():
    rng = np.random.default_rng(0)
    t = rng.uniform(0.0, 10.0, size=2000)
    out = continuous_positivity_check(_df(t), "T", ("X",))
    assert out["verdict"] == "green"
    assert out["fraction_outside_support"] < 0.02


def test_bimodal_with_hole_flags_unsupported_range():
    rng = np.random.default_rng(1)
    left = rng.normal(loc=2.0, scale=0.4, size=1000)
    right = rng.normal(loc=8.0, scale=0.4, size=1000)
    t = np.concatenate([left, right])
    out = continuous_positivity_check(_df(t), "T", ("X",))
    assert out["verdict"] in {"yellow", "red"}
    rng_band = out["unsupported_treatment_range"]
    assert rng_band is not None
    lo, hi = rng_band
    # The reported unsupported range should overlap the [4, 6] hole.
    assert lo < 6.0 and hi > 4.0


def test_zero_variance_treatment_returns_unknown():
    t = np.full(200, 3.14)
    out = continuous_positivity_check(_df(t), "T", ("X",))
    assert out["verdict"] == "unknown"
    assert "unique" in out["interpretation"].lower() or "degenerate" in out["interpretation"].lower()


def test_only_five_distinct_values_returns_unknown():
    rng = np.random.default_rng(2)
    base = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    t = rng.choice(base, size=200, replace=True)
    out = continuous_positivity_check(_df(t), "T", ("X",))
    assert out["verdict"] == "unknown"


def test_too_few_rows_returns_unknown():
    rng = np.random.default_rng(3)
    t = rng.uniform(0, 1, size=10)
    out = continuous_positivity_check(_df(t), "T", ("X",))
    assert out["verdict"] == "unknown"


def test_return_shape_contract():
    rng = np.random.default_rng(4)
    t = rng.uniform(0.0, 1.0, size=500)
    out = continuous_positivity_check(_df(t), "T", ("X",))
    assert set(out.keys()) == {
        "verdict",
        "unsupported_treatment_range",
        "fraction_outside_support",
        "interpretation",
    }
    assert isinstance(out["fraction_outside_support"], float)
    assert isinstance(out["interpretation"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-x", "-q"])
