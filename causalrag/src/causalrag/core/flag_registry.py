"""YAML-driven flag registry (Sprint 1.1).

The `DataFlag` `StrEnum` in `core/flags.py` is the runtime symbol the
rest of the codebase uses — 30+ files reference `DataFlag.X` directly.
We do NOT replace the enum (the refactor surface is too large) and we
do NOT autogenerate it. Instead, the registry **shadows** the enum:
every `DataFlag` member is documented in a YAML manifest with full
metadata (group, parent, semver, deprecation, citations, detector
path, routes_to), and the registry exposes that metadata while leaving
the enum API untouched.

Used by:
- The Sprint 9.5.1 flow audit — checks every enum member has a YAML
  entry, every YAML entry maps to a real enum member, every routes_to
  reference is a real catalog id.
- The discovery / synthesis / HTML report layers — pull the canonical
  description + implication when rendering flag chips.
- Third parties — can ship their own YAML overlay via the
  `causalroadmap.flags` entry-point.

Schema (YAML):

```yaml
flags:
  binary_treatment:
    group: treatment
    parent: treatment_type
    introduced_in: "0.1"
    deprecated_in: null
    replaces: []
    description: "Treatment column has exactly two values."
    implication: "Standard ATE/ATT/RD landscape."
    citations: []
    detector: "causalrag.data.flags._binary_treatment"
    routes_to: ["python.dml.linear", "rbridge.matchit", "rbridge.weightit"]
    implies: []
    forbids: []
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.flags import DataFlag
from causalrag.core.flag_descriptions import _DESCRIPTIONS, describe_safe


FlagGroup = Literal[
    "treatment",
    "outcome",
    "structure",
    "design",
    "sample",
    "measurement",
    "selection",
    "discovery",
]


class FlagSpec(BaseModel):
    """Full metadata for one DataFlag — the YAML registry entry shape."""

    model_config = ConfigDict(extra="forbid")

    name: str  # snake_case matching the DataFlag value
    group: FlagGroup
    parent: str | None = None
    introduced_in: str = "0.1"
    deprecated_in: str | None = None
    replaces: list[str] = Field(default_factory=list)
    description: str
    implication: str
    citations: list[str] = Field(default_factory=list)
    detector: str | None = Field(
        default=None,
        description=(
            "Dotted path to the detector function (e.g. "
            "'causalrag.data.flags._binary_treatment'). Used by the "
            "flow audit to verify the flag has a producer."
        ),
    )
    routes_to: list[str] = Field(
        default_factory=list,
        description="Catalog estimator ids this flag activates / steers.",
    )
    implies: list[str] = Field(
        default_factory=list,
        description="Flags automatically set when this one is set (closure rule).",
    )
    forbids: list[str] = Field(
        default_factory=list,
        description="Flags that must not co-occur with this one.",
    )
    confidence_required: bool = False


@dataclass
class FlagRegistry:
    """In-memory registry indexed by flag name."""

    specs: dict[str, FlagSpec] = field(default_factory=dict)
    yaml_path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "FlagRegistry":
        """Load the canonical YAML registry, or build one from
        ``flag_descriptions`` when no YAML file is present yet (so the
        registry is usable on first install)."""
        registry = cls(yaml_path=path)
        if path is not None and path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for name, body in (raw.get("flags") or {}).items():
                spec = FlagSpec(name=name, **body)
                registry.specs[name] = spec
        else:
            registry.specs = registry._bootstrap_from_descriptions()
        return registry

    @staticmethod
    def _bootstrap_from_descriptions() -> dict[str, FlagSpec]:
        """Build a registry from the in-code `flag_descriptions` table
        when no YAML file is on disk. Lets the registry be usable
        immediately without forcing a separate config file."""
        specs: dict[str, FlagSpec] = {}
        for flag, fd in _DESCRIPTIONS.items():
            specs[flag.value] = FlagSpec(
                name=flag.value,
                group=_GROUP_BY_FLAG.get(flag, "structure"),
                parent=_PARENT_BY_FLAG.get(flag),
                introduced_in="0.1",
                description=fd.summary,
                implication=fd.implication,
                routes_to=list(fd.routes_to),
            )
        return specs

    def get(self, flag: DataFlag | str) -> FlagSpec | None:
        name = flag.value if isinstance(flag, DataFlag) else flag
        return self.specs.get(name)

    def all(self) -> tuple[FlagSpec, ...]:
        return tuple(self.specs.values())

    def by_group(self, group: FlagGroup) -> tuple[FlagSpec, ...]:
        return tuple(s for s in self.specs.values() if s.group == group)

    def closure(self, flags: set[DataFlag] | set[str]) -> set[str]:
        """Apply implication-closure: if A implies B, B is auto-added.

        Detects cycles in the implication graph and raises clearly."""
        names: set[str] = set()
        for f in flags:
            names.add(f.value if isinstance(f, DataFlag) else f)
        changed = True
        steps = 0
        while changed:
            changed = False
            steps += 1
            if steps > 100:
                raise RuntimeError(
                    f"Implication-closure didn't terminate after 100 steps "
                    f"— likely cycle in the registry. Current: {names}"
                )
            for n in list(names):
                spec = self.specs.get(n)
                if spec is None:
                    continue
                for implied in spec.implies:
                    if implied not in names:
                        names.add(implied)
                        changed = True
        return names

    def check_consistency(self) -> list[str]:
        """Validate registry against the runtime enum. Returns a list
        of human-readable problems (empty list = healthy)."""
        problems: list[str] = []
        enum_names = {f.value for f in DataFlag}
        registry_names = set(self.specs.keys())

        for name in enum_names - registry_names:
            problems.append(
                f"DataFlag.{name} has no registry entry — missing metadata"
            )
        for name in registry_names - enum_names:
            problems.append(
                f"Registry entry '{name}' has no DataFlag enum member"
            )

        # Implication targets must exist
        for spec in self.specs.values():
            for implied in spec.implies:
                if implied not in registry_names:
                    problems.append(
                        f"'{spec.name}' implies non-existent flag '{implied}'"
                    )
            for forbidden in spec.forbids:
                if forbidden not in registry_names:
                    problems.append(
                        f"'{spec.name}' forbids non-existent flag '{forbidden}'"
                    )

        # Cycle detection in the implication graph
        try:
            for spec in self.specs.values():
                if spec.implies:
                    self.closure({spec.name})
        except RuntimeError as e:
            problems.append(str(e))

        return problems

    def to_yaml(self) -> str:
        """Serialise the in-memory registry to a YAML string for
        canonical persistence."""
        body = {
            "flags": {
                name: {
                    k: v
                    for k, v in spec.model_dump().items()
                    if k != "name"
                }
                for name, spec in sorted(self.specs.items())
            }
        }
        return yaml.safe_dump(body, sort_keys=False, default_flow_style=False)

    def save(self, path: Path) -> None:
        path.write_text(self.to_yaml(), encoding="utf-8")


# ─── Default group + parent attribution per flag ─────────────────────


_GROUP_BY_FLAG: dict[DataFlag, FlagGroup] = {
    # treatment
    DataFlag.BINARY_TREATMENT: "treatment",
    DataFlag.CATEGORICAL_TREATMENT: "treatment",
    DataFlag.CONTINUOUS_TREATMENT: "treatment",
    DataFlag.MIXTURE_EXPOSURE: "treatment",
    DataFlag.TIME_VARYING_TREATMENT: "treatment",
    DataFlag.IMBALANCED_TREATMENT: "treatment",
    # outcome
    DataFlag.BINARY_OUTCOME: "outcome",
    DataFlag.CONTINUOUS_OUTCOME: "outcome",
    DataFlag.COUNT_OUTCOME: "outcome",
    DataFlag.RIGHT_CENSORED_OUTCOME: "outcome",
    DataFlag.RARE_OUTCOME: "outcome",
    DataFlag.BOUNDED_OUTCOME: "outcome",
    DataFlag.ZERO_INFLATED_OUTCOME: "outcome",
    DataFlag.COMPETING_RISKS: "outcome",
    DataFlag.REPEATED_OUTCOME: "outcome",
    # structure
    DataFlag.SMALL_SAMPLE: "sample",
    DataFlag.HIGH_DIMENSIONAL: "sample",
    DataFlag.POSITIVITY_VIOLATION: "sample",
    DataFlag.HEAVY_MISSINGNESS: "sample",
    DataFlag.HEAVY_CENSORING: "outcome",
    DataFlag.SUSPECTED_INFORMATIVE_CENSORING: "outcome",
    DataFlag.PANEL_STRUCTURE: "structure",
    DataFlag.LONGITUDINAL: "structure",
    DataFlag.CLUSTERED: "structure",
    DataFlag.NETWORK_INTERFERENCE: "structure",
    DataFlag.SINGLE_TREATED_UNIT: "structure",
    DataFlag.CROSS_SECTIONAL_SLICE: "structure",
    # design
    DataFlag.INSTRUMENTAL_CANDIDATE_PRESENT: "design",
    DataFlag.MEDIATOR_PROPOSED: "design",
    DataFlag.EFFECT_MODIFICATION_OF_INTEREST: "design",
    DataFlag.NEGATIVE_CONTROL_AVAILABLE: "design",
    DataFlag.DIFF_IN_DIFF_CANDIDATE: "design",
    DataFlag.STAGGERED_ADOPTION: "design",
    # discovery
    DataFlag.IDENTIFICATION_FAILED: "discovery",
}


_PARENT_BY_FLAG: dict[DataFlag, str] = {
    DataFlag.BINARY_TREATMENT: "treatment_type",
    DataFlag.CATEGORICAL_TREATMENT: "treatment_type",
    DataFlag.CONTINUOUS_TREATMENT: "treatment_type",
    DataFlag.MIXTURE_EXPOSURE: "treatment_type",
    DataFlag.BINARY_OUTCOME: "outcome_type",
    DataFlag.CONTINUOUS_OUTCOME: "outcome_type",
    DataFlag.COUNT_OUTCOME: "outcome_type",
    DataFlag.RIGHT_CENSORED_OUTCOME: "outcome_type",
    DataFlag.BOUNDED_OUTCOME: "outcome_type",
    DataFlag.ZERO_INFLATED_OUTCOME: "outcome_type",
}


# Module-level singleton bootstrapped from `flag_descriptions`.
_REGISTRY: FlagRegistry | None = None


def get_registry(reload: bool = False) -> FlagRegistry:
    """Lazy-construct and return the module-level registry singleton."""
    global _REGISTRY
    if _REGISTRY is None or reload:
        candidate = Path(__file__).parent / "flag_registry.yaml"
        _REGISTRY = FlagRegistry.load(candidate if candidate.exists() else None)
    return _REGISTRY


__all__ = [
    "FlagGroup",
    "FlagSpec",
    "FlagRegistry",
    "get_registry",
]
