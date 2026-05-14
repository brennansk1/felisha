from __future__ import annotations

import numpy as np
import pandas as pd

from causalrag.data.flags import emit_from_profile
from causalrag.data.profiler import profile_dataframe


def _synth(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "patient_id": np.arange(n),
            "age": rng.integers(20, 80, size=n),
            "smoker": rng.integers(0, 2, size=n),
            "treatment": rng.integers(0, 2, size=n),
            "overall_survival_days": rng.integers(30, 3000, size=n),
            "overall_survival_event": rng.integers(0, 2, size=n),
            "site": rng.choice(["A", "B", "C"], size=n),
        }
    )


def test_profile_basics() -> None:
    df = _synth()
    p = profile_dataframe(df)
    assert p.n_rows == 300
    assert p.n_cols == 7
    names = [c.name for c in p.columns]
    assert "patient_id" in names

    pid = p.column("patient_id")
    assert pid.suspected_identifier
    # patient_id is high-uniqueness integer; logical dtype should be identifier
    assert pid.logical_dtype in {"identifier", "count"}


def test_binary_detection() -> None:
    df = _synth()
    p = profile_dataframe(df)
    assert p.column("smoker").is_binary_01
    assert p.column("treatment").is_binary_01
    assert p.column("treatment").logical_dtype == "binary"


def test_continuous_stats_populated() -> None:
    df = _synth()
    p = profile_dataframe(df)
    age = p.column("age")
    assert age.mean is not None and 19 <= age.mean <= 80
    assert age.p50 is not None
    assert age.n_tukey_outliers is not None
    assert age.n_tukey_outliers >= 0


def test_categorical_top_values_populated() -> None:
    df = _synth()
    p = profile_dataframe(df)
    site = p.column("site")
    assert site.logical_dtype == "categorical"
    assert site.mode in {"A", "B", "C"}
    assert len(site.top_values) <= 20


def test_censoring_pair_detected() -> None:
    df = _synth()
    p = profile_dataframe(df)
    assert ("overall_survival_days", "overall_survival_event") in p.censoring_pairs


def test_high_correlation_detected() -> None:
    df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100) * 2.0, "z": np.random.rand(100)})
    p = profile_dataframe(df)
    pairs = {(a, b) for a, b, _ in p.column_pairs_high_corr}
    assert ("x", "y") in pairs


def test_missing_rate_computed() -> None:
    df = _synth().copy()
    df.loc[: 60, "site"] = None
    p = profile_dataframe(df)
    assert p.column("site").missing_rate > 0.20


def test_flag_emission_from_profile() -> None:
    df = _synth()
    p = profile_dataframe(df)
    from causalrag.core.flags import DataFlag

    flags = emit_from_profile(p, treatment="treatment", outcome="overall_survival_days")
    assert DataFlag.BINARY_TREATMENT in flags
    # overall_survival_days is continuous → continuous outcome (we treat survival
    # as censored only when a paired event indicator is detected, which we add
    # to the flag set separately).
    assert DataFlag.RIGHT_CENSORED_OUTCOME in flags


def test_small_sample_flag() -> None:
    from causalrag.core.flags import DataFlag

    df = _synth(n=100)
    flags = emit_from_profile(profile_dataframe(df))
    assert DataFlag.SMALL_SAMPLE in flags
