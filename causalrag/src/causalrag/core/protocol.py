"""StudyProtocol — the central, serializable study object (PDD §14.1).

Every CLI command reads from and updates this object. Reproducibility derives
entirely from it. Persisted as ``study.causalrag.yaml`` at the root of a project.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from causalrag.core.estimand import CausalEstimand, StatisticalEstimand
from causalrag.core.flags import DataFlag, validate_flag_set
from causalrag.core.graph import CausalGraph
from causalrag.core.result import EstimationResult
from causalrag.core.roles import VariableSpec


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


Tier = Literal["data-scientist", "academic", "domain-expert"]


class DatasetSpec(BaseModel):
    """Pointer to the input dataset plus reproducibility metadata."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="URI or relative path; csv://, parquet://, sql://, ...")
    sha256: str | None = Field(default=None, description="Hash of the file content if local")
    n_rows: int | None = None
    n_cols: int | None = None
    columns: tuple[VariableSpec, ...] = ()


class LLMConfig(BaseModel):
    """Model selection, digests, seed, hardware tier — every LLM call records
    these (PDD §10 design principle 10: 'Honest provenance')."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["ollama", "fake", "cloud"] = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    reasoning_model: str | None = None
    general_model: str | None = None
    model_digest: str | None = Field(default=None, description="SHA of pulled Ollama model")
    seed: int = 0
    temperature: float = 0.0
    hardware_tier: int | None = Field(default=None, ge=0, le=4)
    prompt_pack_version: str = "v0"


class Decision(BaseModel):
    """One entry in the analyst-decision ledger (PDD §12.3)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=_utcnow)
    phase: str
    decision: str
    chose: str
    source: Literal["analyst", "llm", "default", "auto"] = "default"
    note: str | None = None


class Override(BaseModel):
    """Diff of an analyst override against the LLM's original suggestion."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=_utcnow)
    site: str = Field(..., description="Module + field, e.g. 'discovery.roles.age'")
    llm_value: Any
    analyst_value: Any
    reason: str | None = None


class Hypothesis(BaseModel):
    """Element of the HypothesisQueue (PDD §9 — Phase 3 output)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    treatment: str
    outcome: str
    modifiers: tuple[str, ...] = ()
    counterfactual: bool = False
    rationale: str | None = None
    impact_score: float | None = None
    estimand: CausalEstimand | None = None
    flags: set[DataFlag] = Field(default_factory=set)

    @field_serializer("flags")
    def _serialize_flags(self, v: set[DataFlag]) -> list[str]:
        return sorted(f.value for f in v)


class DiscoveryReport(BaseModel):
    """Stage 1a–1e composite output (PDD §7)."""

    model_config = ConfigDict(extra="forbid")

    columns: tuple[VariableSpec, ...] = ()
    domain_brief: str | None = None
    candidate_graphs: tuple[CausalGraph, ...] = ()
    flags: set[DataFlag] = Field(default_factory=set)
    markov_boundaries: tuple[dict[str, Any], ...] = Field(
        default=(),
        description=(
            "Per-target Markov-boundary reports from bnlearn IAMB (or a "
            "Python fallback). Each entry: target, mb (columns), backend, "
            "disagreement_with_investigator. Used by the synthesis layer "
            "and the master loop's graph builder as a stats-vs-LLM "
            "cross-check against the investigator's CONFOUNDER labels."
        ),
    )

    @field_serializer("flags")
    def _serialize_flags(self, v: set[DataFlag]) -> list[str]:
        return sorted(f.value for f in v)


class FeasibilityReport(BaseModel):
    """Phase 2 output (PDD §8.5)."""

    model_config = ConfigDict(extra="forbid")

    admissible_pairs: tuple[tuple[str, str], ...] = ()
    n_floor: int = 200
    power_target: float = 0.8
    alpha: float = 0.05
    notes: str | None = None


class RoadmapWalk(BaseModel):
    """Per-hypothesis Phase 4 record — one of these per Hypothesis id."""

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str
    q1_question: str | None = None
    q2_model_graph_index: int | None = None
    q3_estimand: CausalEstimand | None = None
    q4_observed_data_spec: dict[str, Any] = Field(default_factory=dict)
    q5_identification: dict[str, Any] = Field(default_factory=dict)
    q6_statistical_estimand: StatisticalEstimand | None = None
    q7_estimates: tuple[EstimationResult, ...] = ()
    q8_interpretation: str | None = None

    # Master-loop chain bookkeeping.
    chain_id: str | None = Field(
        default=None,
        description="Foundation-chain root id. None for an independent root.",
    )
    parent_id: str | None = Field(
        default=None,
        description="Direct parent hypothesis_id when this walk was a foundation follow-up.",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Captured reason if this walk could not complete (unidentifiable, estimator error).",
    )
    sensitivity_verdict: str | None = Field(
        default=None,
        description="green / yellow / red / unknown / errored — surfaced separately from q8_interpretation prose.",
    )


class StudyProtocol(BaseModel):
    """Central, serializable study object (PDD §14.1).

    Every CLI command reads from and updates this. The YAML projection is the
    on-disk contract; the in-memory object is the source of truth during a run.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "0.1"
    created: datetime = Field(default_factory=_utcnow)
    updated: datetime = Field(default_factory=_utcnow)
    tier: Tier = "academic"

    research_question: str | None = None
    dataset: DatasetSpec | None = None

    discovery: DiscoveryReport | None = None
    feasibility: FeasibilityReport | None = None
    hypothesis_queue: tuple[Hypothesis, ...] = ()
    roadmap_walks: dict[str, RoadmapWalk] = Field(default_factory=dict)

    candidate_queue: tuple[dict[str, Any], ...] = Field(
        default=(),
        description=(
            "Master-loop scored candidate experiments. Each entry is a "
            "dict with at minimum: candidate_id, treatment, outcome, "
            "estimand_class, impact, identifiability, power_proxy, "
            "novelty, score, status ('pending'|'completed'|'vetoed')."
        ),
    )

    flags: set[DataFlag] = Field(default_factory=set)
    candidate_graphs: tuple[CausalGraph, ...] = ()
    selected_graph_index: int = 0

    multiple_testing: Literal["bh", "by", "bonferroni", "none"] = "bh"
    counterfactual_ratio: float = Field(default=0.30, ge=0.0, le=1.0)

    llm: LLMConfig = Field(default_factory=LLMConfig)
    decision_ledger: tuple[Decision, ...] = ()
    overrides: tuple[Override, ...] = ()

    @field_validator("flags")
    @classmethod
    def _validate_flags(cls, v: set[DataFlag]) -> set[DataFlag]:
        validate_flag_set(v)
        return v

    @field_serializer("flags")
    def _serialize_flags(self, v: set[DataFlag]) -> list[str]:
        return sorted(f.value for f in v)

    # --- YAML I/O -----------------------------------------------------------------

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", by_alias=True)
        return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)

    def write_yaml(self, path: Path) -> None:
        path.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def from_yaml(cls, text: str) -> StudyProtocol:
        data = yaml.safe_load(text) or {}
        return cls.model_validate(data)

    @classmethod
    def read_yaml(cls, path: Path) -> StudyProtocol:
        return cls.from_yaml(path.read_text(encoding="utf-8"))
