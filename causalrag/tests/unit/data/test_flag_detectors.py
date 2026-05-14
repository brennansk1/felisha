"""Detector-level tests for the new DataFlags wired in Week 3.

Each new flag gets one positive + one negative case. The detectors run on top
of ``DatasetProfile`` objects (built from synthetic frames via the production
profiler), so failures here mean the detector logic — not the profiler — broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.data.flags import emit_from_profile
from causalrag.data.profiler import profile_dataframe
from causalrag.discovery.expert import (
    CandidateDAGSpec,
    DomainExpertBrief,
    OutcomeCandidate,
    TreatmentCandidate,
    UnmeasuredConfounder,
    flags_from_brief,
)
from causalrag.discovery.investigator import InvestigatorColumn, InvestigatorReport
from causalrag.core.roles import VariableRole


# ───────────────────────── RARE_OUTCOME ─────────────────────────────────────

def _binary_frame(n: int, prev: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "T": rng.integers(0, 2, size=n),
            "Y": (rng.random(size=n) < prev).astype(int),
            "X1": rng.normal(size=n),
            "X2": rng.normal(size=n),
        }
    )


def test_rare_outcome_positive() -> None:
    df = _binary_frame(2000, prev=0.02)
    profile = profile_dataframe(df)
    flags = emit_from_profile(profile, treatment="T", outcome="Y")
    assert DataFlag.RARE_OUTCOME in flags


def test_rare_outcome_negative() -> None:
    df = _binary_frame(2000, prev=0.45)
    profile = profile_dataframe(df)
    flags = emit_from_profile(profile, treatment="T", outcome="Y")
    assert DataFlag.RARE_OUTCOME not in flags


# ─────────────────────── IMBALANCED_TREATMENT ───────────────────────────────

def test_imbalanced_treatment_positive() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "T": (rng.random(2000) < 0.08).astype(int),
            "Y": rng.normal(size=2000),
        }
    )
    flags = emit_from_profile(profile_dataframe(df), treatment="T", outcome="Y")
    assert DataFlag.IMBALANCED_TREATMENT in flags


def test_imbalanced_treatment_negative() -> None:
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "T": (rng.random(2000) < 0.50).astype(int),
            "Y": rng.normal(size=2000),
        }
    )
    flags = emit_from_profile(profile_dataframe(df), treatment="T", outcome="Y")
    assert DataFlag.IMBALANCED_TREATMENT not in flags


# ─────────────────────── BOUNDED_OUTCOME ────────────────────────────────────

def test_bounded_outcome_positive() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "T": rng.integers(0, 2, size=500),
            "Y": rng.beta(2, 5, size=500),  # values in (0, 1)
        }
    )
    flags = emit_from_profile(profile_dataframe(df), treatment="T", outcome="Y")
    assert DataFlag.BOUNDED_OUTCOME in flags


def test_bounded_outcome_negative_unbounded() -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "T": rng.integers(0, 2, size=500),
            "Y": rng.normal(loc=10.0, scale=3.0, size=500),
        }
    )
    flags = emit_from_profile(profile_dataframe(df), treatment="T", outcome="Y")
    assert DataFlag.BOUNDED_OUTCOME not in flags


# ────────────────────── ZERO_INFLATED_OUTCOME ───────────────────────────────

def test_zero_inflated_outcome_positive() -> None:
    rng = np.random.default_rng(5)
    # 70% zeros, the rest in {1..6}
    counts = np.where(rng.random(1000) < 0.7, 0, rng.integers(1, 7, size=1000))
    df = pd.DataFrame({"T": rng.integers(0, 2, size=1000), "Y": counts.astype(int)})
    flags = emit_from_profile(profile_dataframe(df), treatment="T", outcome="Y", df=df)
    assert DataFlag.ZERO_INFLATED_OUTCOME in flags


def test_zero_inflated_outcome_negative() -> None:
    rng = np.random.default_rng(6)
    df = pd.DataFrame(
        {"T": rng.integers(0, 2, size=1000), "Y": rng.poisson(lam=5, size=1000)}
    )
    flags = emit_from_profile(profile_dataframe(df), treatment="T", outcome="Y", df=df)
    assert DataFlag.ZERO_INFLATED_OUTCOME not in flags


# ─────────────────── TIME_VARYING_TREATMENT + DiD ───────────────────────────

def _panel_frame(staggered: bool = False) -> pd.DataFrame:
    rows = []
    for sid in range(50):
        # treated cohort: subjects with even id; control: odd id.
        if sid % 2 == 0:
            onset = 3 if (not staggered or sid < 20) else (4 if sid < 35 else 5)
        else:
            onset = None
        for t in range(1, 7):
            tr = 1 if (onset is not None and t >= onset) else 0
            rows.append({"sid": sid, "t": t, "T": tr, "Y": float(tr) + 0.1 * t})
    return pd.DataFrame(rows)


def test_time_varying_treatment_positive() -> None:
    df = _panel_frame()
    profile = profile_dataframe(df)
    flags = emit_from_profile(
        profile, treatment="T", outcome="Y", df=df, subject_column="sid", time_column="t"
    )
    assert DataFlag.TIME_VARYING_TREATMENT in flags


def test_time_varying_treatment_negative_cross_section() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"sid": np.arange(200), "t": 1, "T": rng.integers(0, 2, 200), "Y": rng.normal(size=200)})
    profile = profile_dataframe(df)
    flags = emit_from_profile(
        profile, treatment="T", outcome="Y", df=df, subject_column="sid", time_column="t"
    )
    assert DataFlag.TIME_VARYING_TREATMENT not in flags


def test_diff_in_diff_candidate_positive() -> None:
    df = _panel_frame()
    flags = emit_from_profile(
        profile_dataframe(df),
        treatment="T",
        outcome="Y",
        df=df,
        subject_column="sid",
        time_column="t",
    )
    assert DataFlag.DIFF_IN_DIFF_CANDIDATE in flags


def test_diff_in_diff_candidate_negative() -> None:
    rng = np.random.default_rng(8)
    df = pd.DataFrame(
        {
            "sid": np.repeat(np.arange(50), 4),
            "t": np.tile([1, 2, 3, 4], 50),
            "T": rng.integers(0, 2, size=1)[0],  # constant for everyone, no switching
            "Y": rng.normal(size=200),
        }
    )
    df["T"] = 0  # everyone control, no DiD structure
    flags = emit_from_profile(
        profile_dataframe(df),
        treatment="T",
        outcome="Y",
        df=df,
        subject_column="sid",
        time_column="t",
    )
    assert DataFlag.DIFF_IN_DIFF_CANDIDATE not in flags


def test_staggered_adoption_positive() -> None:
    df = _panel_frame(staggered=True)
    flags = emit_from_profile(
        profile_dataframe(df),
        treatment="T",
        outcome="Y",
        df=df,
        subject_column="sid",
        time_column="t",
    )
    assert DataFlag.STAGGERED_ADOPTION in flags


def test_staggered_adoption_negative_uniform_onset() -> None:
    df = _panel_frame(staggered=False)
    flags = emit_from_profile(
        profile_dataframe(df),
        treatment="T",
        outcome="Y",
        df=df,
        subject_column="sid",
        time_column="t",
    )
    # All treated subjects share onset=3, so no stagger.
    assert DataFlag.STAGGERED_ADOPTION not in flags


# ────────────────── flags_from_brief: IV + MEDIATOR + CATE ──────────────────


def _minimal_brief(
    mediators: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
    unmeasured: list[UnmeasuredConfounder] | None = None,
) -> DomainExpertBrief:
    return DomainExpertBrief(
        domain_summary="synthetic",
        treatments=[TreatmentCandidate(column="T", rationale="r", suitability=0.5)],
        outcomes=[OutcomeCandidate(column="Y", rationale="r")],
        mediators=mediators or [],
        effect_modifiers=effect_modifiers or [],
        unmeasured_confounders=unmeasured or [],
        candidate_dags=[CandidateDAGSpec(rank=1, rationale="r", edges=[("T", "Y")])],
    )


def test_flags_from_brief_emits_mediator_proposed() -> None:
    brief = _minimal_brief(mediators=["M1"])
    flags = flags_from_brief(brief)
    assert DataFlag.MEDIATOR_PROPOSED in flags


def test_flags_from_brief_no_mediator_no_flag() -> None:
    brief = _minimal_brief()
    flags = flags_from_brief(brief)
    assert DataFlag.MEDIATOR_PROPOSED not in flags


def test_flags_from_brief_emits_effect_modification() -> None:
    brief = _minimal_brief(effect_modifiers=["age"])
    flags = flags_from_brief(brief)
    assert DataFlag.EFFECT_MODIFICATION_OF_INTEREST in flags


def test_flags_from_brief_emits_iv_from_unmeasured_reason() -> None:
    brief = _minimal_brief(
        unmeasured=[
            UnmeasuredConfounder(
                name="distance_to_clinic",
                reason="Plausible instrument for treatment uptake.",
            )
        ]
    )
    flags = flags_from_brief(brief)
    assert DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT in flags


def test_flags_from_brief_emits_iv_from_investigator_role() -> None:
    brief = _minimal_brief()
    investigator = InvestigatorReport(
        domain_tag="other",
        columns=[
            InvestigatorColumn(
                column="Z",
                domain_meaning="lottery assignment",
                temporal_position="pre_treatment",
                proposed_role=VariableRole.INSTRUMENT,
            ),
            InvestigatorColumn(
                column="T",
                domain_meaning="treatment uptake",
                temporal_position="treatment_era",
            ),
        ],
    )
    flags = flags_from_brief(brief, investigator)
    assert DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT in flags


def test_flags_from_brief_iv_negative() -> None:
    brief = _minimal_brief()
    flags = flags_from_brief(brief)
    assert DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT not in flags
