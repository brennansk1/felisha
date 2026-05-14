"""Variable role taxonomy used throughout the discovery and Roadmap stages.

See PDD §7 (Phase 1 discovery agent) and §10 (Step 2 — Define the Causal Model).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class VariableRole(StrEnum):
    TREATMENT = "treatment"
    OUTCOME = "outcome"
    CONFOUNDER = "confounder"
    MEDIATOR = "mediator"
    COLLIDER = "collider"
    INSTRUMENT = "instrument"
    EFFECT_MODIFIER = "effect_modifier"
    NEGATIVE_CONTROL = "negative_control"
    POSITIVE_CONTROL = "positive_control"
    PROXY = "proxy"
    UNMEASURED_CONFOUNDER_CANDIDATE = "unmeasured_confounder_candidate"
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"
    CENSORING_INDICATOR = "censoring_indicator"
    AUXILIARY = "auxiliary"
    EXCLUDED = "excluded"


class VariableSpec(BaseModel):
    """Per-column metadata produced by Stage 1c (LLM investigator) and refined
    by Stage 1e (domain expert) and analyst overrides."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    name: str = Field(..., description="Column name in the source dataset")
    role: VariableRole = VariableRole.AUXILIARY
    dtype: str = Field(..., description="Canonical dtype: int64, float64, bool, str, datetime64")
    nullable: bool = True
    semantic_description: str | None = None
    unit: str | None = None
    measured_at: str | None = Field(
        default=None,
        description="Temporal-order tag (baseline, T0, T1, ...) used in Stage 1c.",
    )
    llm_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="LLM-reported confidence in this assignment"
    )
    analyst_override: bool = False
