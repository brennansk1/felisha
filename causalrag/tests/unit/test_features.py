from __future__ import annotations

import numpy as np
import pandas as pd

from causalrag.data.features import auto_preprocess
from causalrag.data.profiler import profile_dataframe


def test_drops_constant_and_identifier():
    df = pd.DataFrame(
        {
            "patient_id": np.arange(100),
            "constant_col": [1] * 100,
            "x": np.random.normal(size=100),
            "y": np.random.normal(size=100),
        }
    )
    p = profile_dataframe(df)
    out, manifest = auto_preprocess(df, p, treatment=None, outcome="y")
    assert "constant_col" not in out.columns
    assert "patient_id" not in out.columns
    kinds = {t.kind for t in manifest.transforms}
    assert "drop_constant" in kinds
    assert "drop_identifier" in kinds


def test_onehot_encodes_low_card_categorical() -> None:
    df = pd.DataFrame(
        {
            "site": np.random.choice(["A", "B", "C"], size=200),
            "x": np.random.normal(size=200),
            "y": np.random.normal(size=200),
        }
    )
    p = profile_dataframe(df)
    out, manifest = auto_preprocess(df, p, treatment=None, outcome="y")
    site_dummies = [c for c in out.columns if c.startswith("site_")]
    assert len(site_dummies) == 2  # drop_first=True on 3-level categorical
    assert "site" not in out.columns
    assert "site" in manifest.new_columns_from


def test_standardizes_continuous_but_preserves_outcome() -> None:
    df = pd.DataFrame(
        {
            "x": np.random.normal(loc=50, scale=10, size=400),
            "treat": np.random.binomial(1, 0.5, size=400),
            "y": np.random.normal(loc=5, scale=2, size=400),
        }
    )
    p = profile_dataframe(df)
    out, _ = auto_preprocess(df, p, treatment="treat", outcome="y")
    # x is standardized
    assert abs(out["x"].mean()) < 0.1
    assert abs(out["x"].std() - 1.0) < 0.1
    # y preserved
    assert abs(out["y"].mean() - 5) < 0.5
    # treat preserved as binary
    assert set(out["treat"].unique()) <= {0, 1}


def test_log_transforms_skewed_continuous() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "skewed": rng.gamma(shape=0.5, scale=2.0, size=400) + 1,
            "treat": rng.binomial(1, 0.5, size=400),
            "y": rng.normal(loc=5, size=400),
        }
    )
    p = profile_dataframe(df)
    _, manifest = auto_preprocess(df, p, treatment="treat", outcome="y")
    log_records = [t for t in manifest.transforms if t.kind == "log_transform"]
    # At least one log transform should have fired
    assert any(t.details.get("applied") for t in log_records)
