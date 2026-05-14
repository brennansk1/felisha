"""EstimationResult and MultiverseResult — PDD §10.7 / §11."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class EstimationResult(BaseModel):
    """Standardized output of every CausalEstimator (Python or R-bridged)."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    estimator_id: str = Field(..., description="Registry id, e.g. 'python.dml.linear'")
    estimand_class: str
    point_estimate: float
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    n_used: int
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    refutations: dict[str, Any] = Field(default_factory=dict)
    backend_version: str | None = None
    r_session_metadata: dict[str, Any] | None = None
    fit_seconds: float | None = None
    timestamp: datetime = Field(default_factory=_utcnow)


class MultiverseResult(BaseModel):
    """Aggregation across specification choices (PDD §11.1 multiverse)."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    specifications: list[EstimationResult]
    summary: dict[str, Any] = Field(default_factory=dict)
