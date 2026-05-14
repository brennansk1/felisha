"""Causal and statistical estimand objects (PDD §10.3, §10.6).

ICH-E9(R1) estimand framework fields landed in Sprint 1.6 — see
`docs/SPRINT_PLAN_V1.md`. Every CausalEstimand now carries:

* population (which subjects)
* endpoint (which outcome operationalisation)
* intercurrent_event_strategy (treatment-policy / composite / hypothetical /
  principal-stratum / while-on-treatment)
* summary_measure (mean diff, ratio, hazard ratio, win odds, etc.)
* treatment_condition (the protocol-level treatment-vs-comparator
  contrast)

These default to permissive values so existing tests still pass, but
estimators that implement specific ICH-E9 strategies (per-protocol
weighting, principal-stratum analysis, etc.) check the
intercurrent_event_strategy and fail loudly on a mismatch.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


IntercurrentEventStrategy = Literal[
    "treatment_policy",
    "composite_endpoint",
    "hypothetical",
    "principal_stratum",
    "while_on_treatment",
    "unspecified",
]


class TargetTrialProtocol(BaseModel):
    """Hernán-Robins 7-element target-trial-emulation protocol.

    Filled in front of estimation rather than after; an estimator that
    implements per-protocol weighting / cloning-censoring-weighting
    will check this protocol before fitting.
    """

    model_config = ConfigDict(extra="forbid")

    eligibility: str | None = Field(
        default=None,
        description="Eligibility criteria — who is in the analysis cohort.",
    )
    treatment_strategies: list[str] = Field(
        default_factory=list,
        description="The treatment strategies being contrasted (named, atomic).",
    )
    assignment_procedure: str | None = Field(
        default=None,
        description="How subjects are assigned (e.g., 'observational' / 'natural experiment' / 'cluster-rand').",
    )
    followup_period: str | None = Field(
        default=None,
        description="Time origin + follow-up window definition (avoids immortal-time bias).",
    )
    outcome_definition: str | None = Field(
        default=None,
        description="How the outcome is measured (column + transformation + window).",
    )
    causal_contrast: str | None = Field(
        default=None,
        description="Population-level contrast — ATE / ATT / LATE / NDE / etc.",
    )
    analysis_plan: str | None = Field(
        default=None,
        description="Estimator + adjustment set + sensitivity analyses, pre-specified.",
    )


class CausalEstimand(BaseModel):
    """Typed counterfactual quantity — Step 3 output (ICH-E9(R1)-enriched)."""

    model_config = ConfigDict(frozen=False, extra="forbid")

    klass: EstimandClass = Field(..., alias="class")
    treatment: str
    outcome: str
    modifiers: tuple[str, ...] = ()
    # Multi-mediator chain support (Sprint 6.5.2). The legacy `mediator`
    # single-string field is preserved for back-compat and resolved as
    # the first entry of `mediators`.
    mediators: tuple[str, ...] = Field(
        default=(),
        description=(
            "Ordered mediators forming a chain T → M_1 → M_2 → ... → M_K → Y. "
            "For single-mediator analyses use a length-1 tuple."
        ),
    )
    mediator: str | None = Field(
        default=None,
        description="Back-compat alias for the first mediator. Prefer `mediators` for new code.",
    )
    instrument: str | None = None
    formal_expression: str = Field(
        ..., description="Plain-text math, e.g. 'E[Y(1) - Y(0)]'"
    )

    # ICH-E9(R1) fields — see module docstring.
    population: str | None = Field(
        default=None,
        description=(
            "Which subjects the estimand pertains to (e.g. 'all eligible "
            "patients meeting inclusion criteria as of t=0')."
        ),
    )
    endpoint: str | None = Field(
        default=None,
        description=(
            "How the outcome is operationalised — column + measurement window."
        ),
    )
    intercurrent_event_strategy: IntercurrentEventStrategy = Field(
        default="unspecified",
        description=(
            "Per ICH-E9(R1) Addendum: how intercurrent events (treatment "
            "switching, rescue medication, death-before-outcome, etc.) "
            "are handled in the target estimand."
        ),
    )
    summary_measure: str | None = Field(
        default=None,
        description=(
            "Population summary — 'mean difference', 'risk difference', "
            "'risk ratio', 'odds ratio', 'hazard ratio', 'RMST contrast', "
            "'win odds', 'quantile contrast', etc."
        ),
    )
    treatment_condition: str | None = Field(
        default=None,
        description=(
            "The protocol-level contrast — e.g., 'metformin monotherapy "
            "vs. sulfonylurea monotherapy at 12 months'."
        ),
    )
    target_trial: TargetTrialProtocol | None = Field(
        default=None,
        description="Optional Hernán-Robins TTE 7-element protocol.",
    )

    @model_validator(mode="after")
    def _reconcile_mediator_alias(self) -> "CausalEstimand":
        """Keep `mediator` (singular) and `mediators[0]` in sync."""
        if self.mediators and not self.mediator:
            object.__setattr__(self, "mediator", self.mediators[0])
        elif self.mediator and not self.mediators:
            object.__setattr__(self, "mediators", (self.mediator,))
        return self

    def is_full_counterfactual(self) -> bool:
        return self.klass in (
            EstimandClass.COUNTERFACTUAL_DISTRIBUTION,
            EstimandClass.MODIFIED_TREATMENT_POLICY,
            EstimandClass.COUNTERFACTUAL_QUANTILE,
        )

    @property
    def mediator_chain(self) -> tuple[str, ...]:
        """Ordered list of mediators forming the T → M1 → … → Mk → Y chain."""
        return self.mediators


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
