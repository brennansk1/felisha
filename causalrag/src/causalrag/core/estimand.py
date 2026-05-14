"""Causal and statistical estimand objects (PDD §10.3, §10.6)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class EstimandClass(StrEnum):
    ATE = "ATE"
    ATT = "ATT"
    ATC = "ATC"
    CATE = "CATE"
    LATE = "LATE"
    RMST_CONTRAST = "RMST_CONTRAST"
    NDE = "NDE"
    NIE = "NIE"
    COUNTERFACTUAL_DISTRIBUTION = "COUNTERFACTUAL_DISTRIBUTION"
    MODIFIED_TREATMENT_POLICY = "MODIFIED_TREATMENT_POLICY"
    COUNTERFACTUAL_QUANTILE = "COUNTERFACTUAL_QUANTILE"


class CausalEstimand(BaseModel):
    """Typed counterfactual quantity — Step 3 output."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    klass: EstimandClass = Field(..., alias="class")
    treatment: str
    outcome: str
    modifiers: tuple[str, ...] = ()
    mediator: str | None = None
    instrument: str | None = None
    formal_expression: str = Field(
        ..., description="Plain-text math, e.g. 'E[Y(1) - Y(0)]'"
    )

    def is_full_counterfactual(self) -> bool:
        return self.klass in (
            EstimandClass.COUNTERFACTUAL_DISTRIBUTION,
            EstimandClass.MODIFIED_TREATMENT_POLICY,
            EstimandClass.COUNTERFACTUAL_QUANTILE,
        )


class StatisticalEstimand(BaseModel):
    """Canonical statistical functional — Step 6 output. Variables are resolved
    to dataset columns."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    causal_estimand: CausalEstimand
    canonical_form: str = Field(
        ..., description="Functional in canonical form (e.g., g-formula expression)"
    )
    adjustment_set: tuple[str, ...] = ()
    identification_strategy: str = Field(
        ..., description="One of: backdoor, frontdoor, iv, do-calculus, sequential-backdoor"
    )
