"""A/B-platform ingest schemas (Sprint 5.2).

Canonicalizes per-unit assignment + outcome exports from the major
commercial experimentation platforms — Eppo, Statsig, Optimizely, and
GrowthBook — onto a single internal schema that maps cleanly onto the
target-trial protocol (Hernán-Robins 7 elements; see
``causalrag.core.estimand.TargetTrialProtocol``).

Each vendor ships exports in a slightly different shape:

* **Eppo** — JSON with ``experiment``/``assignments``/``metrics`` blocks.
  Assignment rows: ``{subject, variation, timestamp}``; metric rows:
  ``{subject, metric, value}``.
* **Statsig** — JSON with a top-level ``experiment`` object and parallel
  ``exposures`` / ``metric_lift`` arrays. Variant ids in ``groupID``.
* **Optimizely** — CSV per-unit + JSON manifest. CSV columns vary
  (``visitor_id``/``user_id``, ``variation``/``variation_name``,
  ``event_name``, ``event_value``); manifest carries experiment-level
  metadata.
* **GrowthBook** — JSON with ``experiment``/``users``/``metrics``.
  Users carry ``variation`` (index) which the experiment block maps to
  variation keys (``key``/``name``).

The ingestors are lenient on field renames within each format — vendors
quietly tweak column names across product versions — but require the
essentials (a unit identifier, an arm identifier, a primary outcome).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from causalrag.core.estimand import TargetTrialProtocol

Vendor = Literal["eppo", "statsig", "optimizely", "growthbook"]


# ─────────────────────────── Canonical container ────────────────────────────


@dataclass
class CanonicalABExperiment:
    """Vendor-agnostic experiment view.

    ``assignment_df`` has columns ``unit_id`` and ``arm``; ``outcome_df``
    has columns ``unit_id``, ``outcome`` (metric name), and ``value``.
    """

    experiment_id: str
    vendor: Vendor
    treatment_arms: list[str]
    primary_outcome: str
    secondary_outcomes: list[str]
    started_at: pd.Timestamp
    ended_at: pd.Timestamp | None
    n_total: int
    per_arm_counts: dict[str, int]
    assignment_df: pd.DataFrame
    outcome_df: pd.DataFrame
    notes: list[str] = field(default_factory=list)


# ─────────────────────────── shared helpers ─────────────────────────────────


def _first(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key — vendors rename fields between versions."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_ts(v: Any) -> pd.Timestamp | None:
    if v is None:
        return None
    try:
        return pd.Timestamp(v)
    except (ValueError, TypeError):
        return None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finalize(
    *,
    experiment_id: str,
    vendor: Vendor,
    treatment_arms: list[str],
    primary_outcome: str,
    secondary_outcomes: list[str],
    started_at: pd.Timestamp | None,
    ended_at: pd.Timestamp | None,
    assignments: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    notes: list[str] | None = None,
) -> CanonicalABExperiment:
    assignment_df = pd.DataFrame(assignments, columns=["unit_id", "arm"])
    outcome_df = pd.DataFrame(outcomes, columns=["unit_id", "outcome", "value"])

    # Deduplicate assignments (last-write-wins per unit) — vendors emit
    # one row per exposure event, but our canonical view is one row per
    # unit. A unit reassigned mid-experiment is a flag worth surfacing.
    if not assignment_df.empty and assignment_df["unit_id"].duplicated().any():
        notes = (notes or []) + ["duplicate_assignment_rows_collapsed"]
        assignment_df = assignment_df.drop_duplicates(
            subset=["unit_id"], keep="last"
        ).reset_index(drop=True)

    per_arm_counts: dict[str, int] = (
        assignment_df.groupby("arm").size().to_dict() if not assignment_df.empty else {}
    )
    # Ensure every declared arm is present (with zero if no units landed there).
    for arm in treatment_arms:
        per_arm_counts.setdefault(arm, 0)

    return CanonicalABExperiment(
        experiment_id=experiment_id,
        vendor=vendor,
        treatment_arms=treatment_arms,
        primary_outcome=primary_outcome,
        secondary_outcomes=secondary_outcomes,
        started_at=started_at if started_at is not None else pd.Timestamp("1970-01-01"),
        ended_at=ended_at,
        n_total=int(len(assignment_df)),
        per_arm_counts=per_arm_counts,
        assignment_df=assignment_df,
        outcome_df=outcome_df,
        notes=list(notes or []),
    )


# ─────────────────────────── Eppo ───────────────────────────────────────────


def ingest_eppo_export(json_path: str | Path) -> CanonicalABExperiment:
    """Ingest an Eppo experiment export.

    Expected shape (lenient on key names)::

        {
          "experiment": {"id"|"key", "variations"|"arms", "primary_metric",
                         "secondary_metrics", "start_time"|"started_at",
                         "end_time"|"ended_at"},
          "assignments": [{"subject"|"subject_id", "variation"|"variant",
                           "timestamp"}, ...],
          "metrics": [{"subject"|"subject_id", "metric"|"metric_name",
                       "value"}, ...]
        }
    """
    payload = _read_json(json_path)
    exp = payload.get("experiment", {}) or {}

    variations = _first(exp, "variations", "arms", default=[])
    treatment_arms = [str(v) for v in variations]
    primary_outcome = _first(exp, "primary_metric", "primary_outcome", default="")
    secondary_outcomes = list(
        _first(exp, "secondary_metrics", "secondary_outcomes", default=[]) or []
    )

    assignments = [
        {
            "unit_id": str(_first(a, "subject", "subject_id", "user_id")),
            "arm": str(_first(a, "variation", "variant", "arm")),
        }
        for a in payload.get("assignments", []) or []
    ]
    outcomes = [
        {
            "unit_id": str(_first(m, "subject", "subject_id", "user_id")),
            "outcome": str(_first(m, "metric", "metric_name", "name")),
            "value": float(_first(m, "value", "metric_value", default=0.0) or 0.0),
        }
        for m in payload.get("metrics", []) or []
    ]

    return _finalize(
        experiment_id=str(_first(exp, "id", "key", "experiment_id", default="")),
        vendor="eppo",
        treatment_arms=treatment_arms,
        primary_outcome=str(primary_outcome or ""),
        secondary_outcomes=[str(s) for s in secondary_outcomes],
        started_at=_to_ts(_first(exp, "start_time", "started_at")),
        ended_at=_to_ts(_first(exp, "end_time", "ended_at")),
        assignments=assignments,
        outcomes=outcomes,
    )


# ─────────────────────────── Statsig ────────────────────────────────────────


def ingest_statsig_export(json_path: str | Path) -> CanonicalABExperiment:
    """Ingest a Statsig Experiments/Gates export.

    Expected shape::

        {
          "experiment": {"id"|"name", "groups"|"variants", "primary_metric"
                         |"primaryMetric", "secondary_metrics", "startTime",
                         "endTime"},
          "exposures": [{"unitID"|"userID", "groupID"|"group", ...}],
          "metric_lift" | "metrics": [{"unitID", "metric", "value"}]
        }
    """
    payload = _read_json(json_path)
    exp = payload.get("experiment", {}) or {}

    groups = _first(exp, "groups", "variants", "arms", default=[]) or []
    treatment_arms = [
        str(g.get("name") if isinstance(g, dict) else g) for g in groups
    ]
    primary_outcome = _first(exp, "primary_metric", "primaryMetric", default="")
    secondary_outcomes = list(
        _first(exp, "secondary_metrics", "secondaryMetrics", default=[]) or []
    )

    assignments = [
        {
            "unit_id": str(_first(e, "unitID", "userID", "unit_id")),
            "arm": str(_first(e, "groupID", "group", "groupName")),
        }
        for e in payload.get("exposures", []) or []
    ]
    metric_rows = payload.get("metric_lift") or payload.get("metrics") or []
    outcomes = [
        {
            "unit_id": str(_first(m, "unitID", "userID", "unit_id")),
            "outcome": str(_first(m, "metric", "metric_name", "name")),
            "value": float(_first(m, "value", "metric_value", default=0.0) or 0.0),
        }
        for m in metric_rows
    ]

    return _finalize(
        experiment_id=str(_first(exp, "id", "name", "experiment_id", default="")),
        vendor="statsig",
        treatment_arms=treatment_arms,
        primary_outcome=str(primary_outcome or ""),
        secondary_outcomes=[str(s) for s in secondary_outcomes],
        started_at=_to_ts(_first(exp, "startTime", "start_time")),
        ended_at=_to_ts(_first(exp, "endTime", "end_time")),
        assignments=assignments,
        outcomes=outcomes,
    )


# ─────────────────────────── Optimizely ─────────────────────────────────────


def ingest_optimizely_export(
    csv_path: str | Path, manifest_path: str | Path
) -> CanonicalABExperiment:
    """Ingest an Optimizely per-unit CSV plus JSON manifest.

    The CSV is expected to have at least ``visitor_id`` (or ``user_id``)
    and ``variation`` (or ``variation_name``) columns. Outcome rows are
    inferred from any of ``event_name``+``event_value`` columns, or one
    column per metric matching names declared in the manifest.

    Manifest is JSON with experiment-level metadata mirroring the
    Optimizely Experiment Results API::

        {"id", "name", "variations": ["control","variant"],
         "primary_metric", "secondary_metrics", "start_time", "end_time"}
    """
    manifest = _read_json(manifest_path)
    treatment_arms = [
        str(v) for v in _first(manifest, "variations", "arms", default=[]) or []
    ]
    primary_outcome = str(_first(manifest, "primary_metric", default="") or "")
    secondary_outcomes = [
        str(s) for s in _first(manifest, "secondary_metrics", default=[]) or []
    ]
    declared_metrics = {primary_outcome, *secondary_outcomes} - {""}

    with Path(csv_path).open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assignments: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    seen_units: set[str] = set()

    for row in rows:
        unit = str(
            _first(row, "visitor_id", "user_id", "unit_id", "subject", default="")
        )
        arm = str(_first(row, "variation", "variation_name", "arm", default=""))
        if not unit:
            continue
        if unit not in seen_units:
            assignments.append({"unit_id": unit, "arm": arm})
            seen_units.add(unit)

        # Long-form outcome row.
        event_name = _first(row, "event_name", "metric", "metric_name")
        if event_name:
            try:
                v = float(_first(row, "event_value", "value", default=0.0) or 0.0)
            except (TypeError, ValueError):
                v = 0.0
            outcomes.append(
                {"unit_id": unit, "outcome": str(event_name), "value": v}
            )

        # Wide-form: one column per declared metric.
        for m in declared_metrics:
            if m in row and m not in {
                "visitor_id",
                "user_id",
                "variation",
                "variation_name",
            }:
                raw = row.get(m)
                if raw is None or raw == "":
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                outcomes.append({"unit_id": unit, "outcome": m, "value": v})

    return _finalize(
        experiment_id=str(_first(manifest, "id", "name", default="")),
        vendor="optimizely",
        treatment_arms=treatment_arms,
        primary_outcome=primary_outcome,
        secondary_outcomes=secondary_outcomes,
        started_at=_to_ts(_first(manifest, "start_time", "started_at")),
        ended_at=_to_ts(_first(manifest, "end_time", "ended_at")),
        assignments=assignments,
        outcomes=outcomes,
    )


# ─────────────────────────── GrowthBook ─────────────────────────────────────


def ingest_growthbook_export(json_path: str | Path) -> CanonicalABExperiment:
    """Ingest a GrowthBook experiment export.

    Expected shape::

        {
          "experiment": {"id"|"trackingKey", "name", "variations":
                         [{"key"|"name"}, ...], "primary_metric"|
                         "goalMetric", "secondary_metrics", "dateStarted",
                         "dateEnded"},
          "users": [{"id"|"user_id", "variation": <int index or key>}],
          "metrics": [{"user_id", "metric", "value"}]
        }
    """
    payload = _read_json(json_path)
    exp = payload.get("experiment", {}) or {}

    variations = _first(exp, "variations", "arms", default=[]) or []
    treatment_arms = [
        str(v.get("key") or v.get("name")) if isinstance(v, dict) else str(v)
        for v in variations
    ]

    primary_outcome = str(_first(exp, "primary_metric", "goalMetric", default="") or "")
    secondary_outcomes = [
        str(s)
        for s in _first(exp, "secondary_metrics", "guardrailMetrics", default=[]) or []
    ]

    assignments: list[dict[str, Any]] = []
    for u in payload.get("users", []) or []:
        unit = str(_first(u, "id", "user_id", "unitID", default=""))
        var = _first(u, "variation", "variant", "arm")
        if isinstance(var, int) and 0 <= var < len(treatment_arms):
            arm = treatment_arms[var]
        else:
            arm = str(var) if var is not None else ""
        assignments.append({"unit_id": unit, "arm": arm})

    outcomes = [
        {
            "unit_id": str(_first(m, "user_id", "id", "unitID", default="")),
            "outcome": str(_first(m, "metric", "metric_name", "name", default="")),
            "value": float(_first(m, "value", "metric_value", default=0.0) or 0.0),
        }
        for m in payload.get("metrics", []) or []
    ]

    return _finalize(
        experiment_id=str(
            _first(exp, "id", "trackingKey", "name", "experiment_id", default="")
        ),
        vendor="growthbook",
        treatment_arms=treatment_arms,
        primary_outcome=primary_outcome,
        secondary_outcomes=secondary_outcomes,
        started_at=_to_ts(_first(exp, "dateStarted", "start_time", "started_at")),
        ended_at=_to_ts(_first(exp, "dateEnded", "end_time", "ended_at")),
        assignments=assignments,
        outcomes=outcomes,
    )


# ─────────────────────────── target-trial mapper ────────────────────────────


def to_target_trial(exp: CanonicalABExperiment) -> TargetTrialProtocol:
    """Fill the Hernán-Robins 7 fields from the canonical experiment.

    Commercial A/B platforms randomize at the unit level on assignment,
    so we can populate the protocol confidently:

    * **eligibility** — units exposed during the experiment window.
    * **treatment_strategies** — the declared arms.
    * **assignment_procedure** — randomized assignment (vendor-managed).
    * **followup_period** — [started_at, ended_at).
    * **outcome_definition** — primary metric column.
    * **causal_contrast** — ATE (default for randomized between-subjects
      A/B tests).
    * **analysis_plan** — intent-to-treat by arm; vendor-side variance
      reduction (CUPED/sequential testing) is noted as a sensitivity
      analysis stub.
    """
    end = exp.ended_at.isoformat() if exp.ended_at is not None else "ongoing"
    eligibility = (
        f"Units exposed to experiment {exp.experiment_id} on the "
        f"{exp.vendor} platform between {exp.started_at.isoformat()} "
        f"and {end} (n={exp.n_total})."
    )
    followup = (
        f"Time origin = first exposure timestamp per unit; follow-up "
        f"window {exp.started_at.isoformat()} → {end}."
    )
    outcome_def = (
        f"Primary metric '{exp.primary_outcome}' as emitted by the "
        f"{exp.vendor} export"
        + (
            f"; secondary metrics: {', '.join(exp.secondary_outcomes)}."
            if exp.secondary_outcomes
            else "."
        )
    )
    analysis = (
        "Intent-to-treat contrast across arms; vendor-side variance "
        "reduction (CUPED / sequential testing) recorded as a "
        "sensitivity analysis."
    )

    return TargetTrialProtocol(
        eligibility=eligibility,
        treatment_strategies=list(exp.treatment_arms),
        assignment_procedure=(
            f"Randomized assignment by the {exp.vendor} platform "
            f"(per-unit hash bucketing)."
        ),
        followup_period=followup,
        outcome_definition=outcome_def,
        causal_contrast="ATE",
        analysis_plan=analysis,
    )


__all__ = [
    "CanonicalABExperiment",
    "ingest_eppo_export",
    "ingest_statsig_export",
    "ingest_optimizely_export",
    "ingest_growthbook_export",
    "to_target_trial",
]
