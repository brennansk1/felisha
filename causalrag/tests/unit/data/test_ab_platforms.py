"""Tests for the A/B-platform ingest schemas (Sprint 5.2).

Each test round-trips a synthetic vendor export through the canonical
container into a ``TargetTrialProtocol``, pinning both the canonical
shape (column names, arm counts, dedupe behaviour) and the seven
Hernán-Robins fields a downstream estimator will rely on.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from causalrag.core.estimand import TargetTrialProtocol
from causalrag.data.ab_platforms import (
    CanonicalABExperiment,
    ingest_eppo_export,
    ingest_growthbook_export,
    ingest_optimizely_export,
    ingest_statsig_export,
    to_target_trial,
)


# ─────────────────────────── synthetic fixtures ─────────────────────────────


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _eppo_payload() -> dict:
    return {
        "experiment": {
            "id": "exp_eppo_001",
            "variations": ["control", "treatment"],
            "primary_metric": "conversion",
            "secondary_metrics": ["revenue"],
            "start_time": "2026-04-01T00:00:00Z",
            "end_time": "2026-04-15T00:00:00Z",
        },
        "assignments": [
            {"subject": "u1", "variation": "control", "timestamp": "2026-04-02"},
            {"subject": "u2", "variation": "treatment", "timestamp": "2026-04-02"},
            {"subject": "u3", "variation": "treatment", "timestamp": "2026-04-03"},
            # duplicate exposure — last write wins.
            {"subject": "u1", "variation": "control", "timestamp": "2026-04-04"},
        ],
        "metrics": [
            {"subject": "u1", "metric": "conversion", "value": 0.0},
            {"subject": "u2", "metric": "conversion", "value": 1.0},
            {"subject": "u3", "metric": "conversion", "value": 1.0},
            {"subject": "u2", "metric": "revenue", "value": 12.5},
        ],
    }


def _statsig_payload() -> dict:
    return {
        "experiment": {
            "id": "exp_statsig_42",
            "groups": [{"name": "control"}, {"name": "variantA"}, {"name": "variantB"}],
            "primaryMetric": "click_through_rate",
            "secondaryMetrics": ["session_length"],
            "startTime": "2026-03-10",
            "endTime": None,
        },
        "exposures": [
            {"unitID": "s1", "groupID": "control"},
            {"unitID": "s2", "groupID": "variantA"},
            {"unitID": "s3", "groupID": "variantB"},
            {"unitID": "s4", "groupID": "variantA"},
        ],
        "metric_lift": [
            {"unitID": "s1", "metric": "click_through_rate", "value": 0.10},
            {"unitID": "s2", "metric": "click_through_rate", "value": 0.20},
            {"unitID": "s3", "metric": "click_through_rate", "value": 0.18},
            {"unitID": "s4", "metric": "click_through_rate", "value": 0.22},
            {"unitID": "s2", "metric": "session_length", "value": 45.0},
        ],
    }


def _growthbook_payload() -> dict:
    return {
        "experiment": {
            "trackingKey": "gb_homepage_cta",
            "name": "Homepage CTA copy",
            "variations": [{"key": "control"}, {"key": "bold_copy"}],
            "goalMetric": "signup",
            "secondary_metrics": ["bounce"],
            "dateStarted": "2026-02-01",
            "dateEnded": "2026-02-28",
        },
        "users": [
            {"id": "g1", "variation": 0},
            {"id": "g2", "variation": 1},
            {"id": "g3", "variation": 1},
            {"id": "g4", "variation": 0},
        ],
        "metrics": [
            {"user_id": "g1", "metric": "signup", "value": 0.0},
            {"user_id": "g2", "metric": "signup", "value": 1.0},
            {"user_id": "g3", "metric": "signup", "value": 0.0},
            {"user_id": "g4", "metric": "signup", "value": 1.0},
            {"user_id": "g2", "metric": "bounce", "value": 0.0},
        ],
    }


def _write_optimizely(tmp_path: Path) -> tuple[Path, Path]:
    csv_path = tmp_path / "opt.csv"
    manifest_path = tmp_path / "opt_manifest.json"

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["visitor_id", "variation", "event_name", "event_value"]
        )
        writer.writeheader()
        for row in [
            {"visitor_id": "v1", "variation": "control", "event_name": "purchase", "event_value": 0},
            {"visitor_id": "v2", "variation": "treatment", "event_name": "purchase", "event_value": 1},
            {"visitor_id": "v3", "variation": "treatment", "event_name": "purchase", "event_value": 1},
            {"visitor_id": "v3", "variation": "treatment", "event_name": "addtocart", "event_value": 1},
        ]:
            writer.writerow(row)

    manifest = {
        "id": "opt_xyz",
        "name": "Checkout button colour",
        "variations": ["control", "treatment"],
        "primary_metric": "purchase",
        "secondary_metrics": ["addtocart"],
        "start_time": "2026-01-15",
        "end_time": "2026-01-30",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return csv_path, manifest_path


# ─────────────────────────── Eppo ───────────────────────────────────────────


def test_eppo_round_trip(tmp_path: Path) -> None:
    p = _write_json(tmp_path / "eppo.json", _eppo_payload())
    exp = ingest_eppo_export(p)

    assert isinstance(exp, CanonicalABExperiment)
    assert exp.vendor == "eppo"
    assert exp.experiment_id == "exp_eppo_001"
    assert exp.treatment_arms == ["control", "treatment"]
    assert exp.primary_outcome == "conversion"
    assert exp.secondary_outcomes == ["revenue"]
    # 4 raw rows but u1 is duplicated → 3 unique units.
    assert exp.n_total == 3
    assert exp.per_arm_counts == {"control": 1, "treatment": 2}
    assert "duplicate_assignment_rows_collapsed" in exp.notes
    assert list(exp.assignment_df.columns) == ["unit_id", "arm"]
    assert list(exp.outcome_df.columns) == ["unit_id", "outcome", "value"]
    assert set(exp.outcome_df["outcome"]) == {"conversion", "revenue"}

    proto = to_target_trial(exp)
    assert isinstance(proto, TargetTrialProtocol)
    assert proto.treatment_strategies == ["control", "treatment"]
    assert proto.causal_contrast == "ATE"
    assert "conversion" in proto.outcome_definition
    assert "eppo" in proto.assignment_procedure
    assert "2026-04-01" in proto.followup_period


# ─────────────────────────── Statsig ────────────────────────────────────────


def test_statsig_round_trip(tmp_path: Path) -> None:
    p = _write_json(tmp_path / "statsig.json", _statsig_payload())
    exp = ingest_statsig_export(p)

    assert exp.vendor == "statsig"
    assert exp.experiment_id == "exp_statsig_42"
    assert exp.treatment_arms == ["control", "variantA", "variantB"]
    assert exp.primary_outcome == "click_through_rate"
    assert exp.secondary_outcomes == ["session_length"]
    assert exp.n_total == 4
    assert exp.per_arm_counts == {"control": 1, "variantA": 2, "variantB": 1}
    assert exp.ended_at is None
    assert exp.started_at == pd.Timestamp("2026-03-10")

    proto = to_target_trial(exp)
    assert proto.treatment_strategies == ["control", "variantA", "variantB"]
    assert "ongoing" in proto.followup_period
    assert "click_through_rate" in proto.outcome_definition
    assert "session_length" in proto.outcome_definition


# ─────────────────────────── Optimizely ─────────────────────────────────────


def test_optimizely_round_trip(tmp_path: Path) -> None:
    csv_path, manifest_path = _write_optimizely(tmp_path)
    exp = ingest_optimizely_export(csv_path, manifest_path)

    assert exp.vendor == "optimizely"
    assert exp.experiment_id == "opt_xyz"
    assert exp.treatment_arms == ["control", "treatment"]
    assert exp.primary_outcome == "purchase"
    assert exp.secondary_outcomes == ["addtocart"]
    # v3 appears twice but only one assignment row should result.
    assert exp.n_total == 3
    assert exp.per_arm_counts == {"control": 1, "treatment": 2}
    # All four event rows become outcome rows.
    assert len(exp.outcome_df) == 4
    assert set(exp.outcome_df["outcome"]) == {"purchase", "addtocart"}

    proto = to_target_trial(exp)
    assert proto.treatment_strategies == ["control", "treatment"]
    assert "optimizely" in proto.assignment_procedure
    assert proto.causal_contrast == "ATE"


# ─────────────────────────── GrowthBook ─────────────────────────────────────


def test_growthbook_round_trip(tmp_path: Path) -> None:
    p = _write_json(tmp_path / "gb.json", _growthbook_payload())
    exp = ingest_growthbook_export(p)

    assert exp.vendor == "growthbook"
    assert exp.experiment_id == "gb_homepage_cta"
    assert exp.treatment_arms == ["control", "bold_copy"]
    assert exp.primary_outcome == "signup"
    assert exp.secondary_outcomes == ["bounce"]
    assert exp.n_total == 4
    # Integer variation indices should be resolved to arm names.
    assert exp.per_arm_counts == {"control": 2, "bold_copy": 2}
    assert set(exp.assignment_df["arm"]) == {"control", "bold_copy"}

    proto = to_target_trial(exp)
    assert proto.treatment_strategies == ["control", "bold_copy"]
    assert "signup" in proto.outcome_definition
    assert "bounce" in proto.outcome_definition
    assert "growthbook" in proto.assignment_procedure
    assert proto.causal_contrast == "ATE"


# ─────────────────────────── shared invariants ──────────────────────────────


def test_zero_unit_arm_still_listed(tmp_path: Path) -> None:
    """Declared arms with zero exposures still appear in per_arm_counts."""
    payload = _eppo_payload()
    # Declare a third arm that no unit landed in.
    payload["experiment"]["variations"] = ["control", "treatment", "ghost"]
    p = _write_json(tmp_path / "eppo_ghost.json", payload)
    exp = ingest_eppo_export(p)
    assert exp.per_arm_counts["ghost"] == 0
    assert "ghost" in exp.treatment_arms


def test_lenient_field_renames(tmp_path: Path) -> None:
    """Renamed-but-equivalent vendor keys are still recognised."""
    payload = {
        "experiment": {
            "key": "exp_renamed",
            "arms": ["A", "B"],
            "primary_outcome": "clicks",
            "secondary_outcomes": [],
            "started_at": "2026-01-01",
            "ended_at": "2026-01-31",
        },
        "assignments": [
            {"subject_id": "x1", "variant": "A"},
            {"subject_id": "x2", "variant": "B"},
        ],
        "metrics": [
            {"subject_id": "x1", "metric_name": "clicks", "metric_value": 3},
            {"subject_id": "x2", "metric_name": "clicks", "metric_value": 7},
        ],
    }
    p = _write_json(tmp_path / "renamed.json", payload)
    exp = ingest_eppo_export(p)
    assert exp.experiment_id == "exp_renamed"
    assert exp.treatment_arms == ["A", "B"]
    assert exp.primary_outcome == "clicks"
    assert exp.n_total == 2
    assert float(exp.outcome_df.set_index("unit_id").loc["x2", "value"]) == 7.0
