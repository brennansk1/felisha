"""Tests for the missing-data diagnostic stage.

Each test pins one branch of the recommendation logic so a regression that
silently downgrades (e.g., from "refuse" to "consider_mice") fails loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalrag.data.missingness import MissingnessReport, diagnose_missingness


def _base_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    # T depends mildly on X1 to give the bias proxy something to detect.
    T = (rng.random(n) < 1 / (1 + np.exp(-X1))).astype(int)
    Y = 0.5 * T + 0.3 * X1 + 0.2 * X2 + rng.normal(size=n) * 0.5
    return pd.DataFrame({"T": T, "Y": Y, "X1": X1, "X2": X2})


# ─────────────────────────── 0% missingness ─────────────────────────────────


def test_no_missingness_proceeds_complete_case() -> None:
    df = _base_frame()
    report = diagnose_missingness(df, treatment="T", outcome="Y")

    assert isinstance(report, MissingnessReport)
    assert report.recommendation == "proceed_complete_case"
    assert all(rate == 0.0 for rate in report.per_column_rate.values())
    assert report.complete_case_bias_score == 0.0
    assert report.mcar_test_p_value is None


# ─────────────────────── 30% missingness on confounder ──────────────────────


def test_moderate_missingness_recommends_mice_or_ipcw() -> None:
    df = _base_frame(n=400, seed=1)
    rng = np.random.default_rng(2)
    mask = rng.random(len(df)) < 0.30
    df.loc[mask, "X1"] = np.nan

    report = diagnose_missingness(df, treatment="T", outcome="Y")

    assert report.recommendation in {"consider_mice", "consider_ipcw"}
    assert 0.25 < report.per_column_rate["X1"] < 0.35


# ──────────────────────────── >50% missingness ──────────────────────────────


def test_severe_missingness_refuses() -> None:
    df = _base_frame(n=300, seed=3)
    rng = np.random.default_rng(4)
    mask = rng.random(len(df)) < 0.60
    df.loc[mask, "X1"] = np.nan

    report = diagnose_missingness(df, treatment="T", outcome="Y")

    assert report.recommendation == "refuse"
    assert report.per_column_rate["X1"] > 0.5
    assert any("50%" in note or "0.5" in note.lower() for note in report.notes)


# ───────────────── complete-case bias when MAR-on-treatment ─────────────────


def test_bias_score_high_when_missingness_correlates_with_treatment() -> None:
    """Construct missingness in X1 that occurs almost exclusively when T=1.

    Under that mechanism, dropping rows with missing X1 throws out mostly
    treated units, so E[T|complete] ≪ E[T|any-missing] and the standardized
    gap should comfortably exceed 0.5σ.
    """
    df = _base_frame(n=600, seed=5)
    rng = np.random.default_rng(6)
    # Missing X1 ~ 25% of the time among treated, ~2% among untreated.
    p_missing = np.where(df["T"] == 1, 0.55, 0.02)
    mask = rng.random(len(df)) < p_missing
    df.loc[mask, "X1"] = np.nan

    report = diagnose_missingness(df, treatment="T", outcome="Y")

    assert report.complete_case_bias_score > 0.5
    # With T-correlated missingness in X1, MICE (or IPCW) is the right call.
    assert report.recommendation in {"consider_mice", "consider_ipcw", "refuse"}


# ─────────────────────────── edge: empty frame ──────────────────────────────


def test_empty_frame_returns_default_report() -> None:
    report = diagnose_missingness(pd.DataFrame())
    assert report.recommendation == "proceed_complete_case"
    assert report.per_column_rate == {}


# ───────────── outcome with censoring indicator triggers IPCW ───────────────


def test_right_censoring_indicator_routes_to_ipcw() -> None:
    df = _base_frame(n=400, seed=7)
    rng = np.random.default_rng(8)
    df["event"] = rng.integers(0, 2, size=len(df))
    # Add modest missingness so the diagnostic has something to weigh in on.
    mask = rng.random(len(df)) < 0.10
    df.loc[mask, "X1"] = np.nan

    report = diagnose_missingness(df, treatment="T", outcome="Y")

    assert report.recommendation == "consider_ipcw"


# ───────────────── threshold note triggers per-column flag ──────────────────


@pytest.mark.parametrize("rate", [0.10, 0.15])
def test_threshold_note_emitted(rate: float) -> None:
    df = _base_frame(n=500, seed=9)
    rng = np.random.default_rng(10)
    mask = rng.random(len(df)) < rate
    df.loc[mask, "X2"] = np.nan

    report = diagnose_missingness(
        df, treatment="T", outcome="Y", threshold_pct=0.05
    )
    assert any("X2" in note for note in report.notes)
