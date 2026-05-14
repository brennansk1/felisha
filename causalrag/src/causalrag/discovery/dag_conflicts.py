"""DAG-mismatch alert layer (Sprint 6.5.10).

Three potentially-disagreeing sources name the variables that should be
in a (T, Y)'s adjustment set:

1. The **investigator** LLM, which labels each column with a role
   (CONFOUNDER / MEDIATOR / etc.) from semantics alone.
2. The **domain-expert brief**, which proposes candidate DAGs whose
   structure implies a confounder set.
3. The **Markov-boundary** statistical pass (`discovery/markov_boundary.py`),
   which returns the data-driven neighbourhood of the target.

When all three agree, the adjustment set is uncontroversial. When they
disagree, the analyst — and any downstream LLM call — should know.
This module produces a structured ``DAGMismatchReport`` that lists the
disagreements per target so the synthesis layer can surface them and
the master loop's graph builder can decide which source to trust on
each variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from causalrag.core.roles import VariableRole, VariableSpec


@dataclass(frozen=True)
class _SourceVote:
    source: str  # "investigator" | "expert_brief" | "markov_boundary"
    in_adjustment_set: bool
    rationale: str | None = None


class DAGConflict(BaseModel):
    """One column where the three sources disagree."""

    model_config = ConfigDict(extra="forbid")

    target: str  # the outcome (or treatment) the MB was computed for
    column: str
    investigator_says: bool
    brief_says: bool
    markov_says: bool
    severity: str  # "high" / "medium" / "low"
    notes: list[str] = Field(default_factory=list)

    @property
    def all_agree(self) -> bool:
        return self.investigator_says == self.brief_says == self.markov_says


class DAGMismatchReport(BaseModel):
    """Aggregate disagreements across all (target × column) pairs."""

    model_config = ConfigDict(extra="forbid")

    conflicts: list[DAGConflict] = Field(default_factory=list)
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    summary: str = ""

    def has_high_severity(self) -> bool:
        return self.high_severity_count > 0


def _confounder_columns_from_investigator(
    columns: Iterable[VariableSpec],
) -> set[str]:
    return {
        v.name for v in columns if v.role is VariableRole.CONFOUNDER
    }


def _confounder_columns_from_brief(brief_confounders: Iterable[Any]) -> set[str]:
    """Tolerant extraction — brief_confounders may be a list of dicts /
    Pydantic models / strings depending on caller."""
    out: set[str] = set()
    for c in brief_confounders or ():
        if isinstance(c, str):
            out.add(c)
        elif isinstance(c, dict):
            name = c.get("name") or c.get("variable") or c.get("column")
            if name:
                out.add(str(name))
        else:
            name = getattr(c, "name", None) or getattr(c, "variable", None)
            if name:
                out.add(str(name))
    return out


def _classify_severity(
    investigator: bool, brief: bool, markov: bool
) -> str:
    """Severity heuristic:

    * **high** — exactly one of three says "in adjustment set"; the other
      two disagree. This is the noisiest split and warrants a flag in
      the synthesis.
    * **medium** — two say in, one says out (or vice-versa).
    * **low** — all three agree (no conflict).
    """
    votes = (investigator, brief, markov)
    n_in = sum(votes)
    if n_in == 0 or n_in == 3:
        return "low"
    if n_in == 1:
        return "high"
    return "medium"


def detect_conflicts(
    *,
    target: str,
    columns: Iterable[VariableSpec],
    brief_confounders: Iterable[Any] | None = None,
    markov_boundary: Iterable[str] | None = None,
) -> DAGMismatchReport:
    """Build a DAGMismatchReport for one target.

    Compares the investigator's CONFOUNDER labels, the expert brief's
    explicit confounder list, and the statistical Markov boundary on
    the target. Every column mentioned by any source is checked.
    """
    inv_set = _confounder_columns_from_investigator(columns)
    brief_set = _confounder_columns_from_brief(brief_confounders or ())
    mb_set = set(markov_boundary or ())

    universe = sorted(inv_set | brief_set | mb_set)
    conflicts: list[DAGConflict] = []
    counts = {"high": 0, "medium": 0, "low": 0}

    for col in universe:
        if col == target:
            continue
        investigator_says = col in inv_set
        brief_says = col in brief_set
        markov_says = col in mb_set
        severity = _classify_severity(investigator_says, brief_says, markov_says)
        counts[severity] += 1
        if severity == "low":
            # All-agree cases aren't conflicts — skip them in the report
            # (they're recorded in the count but not the list).
            continue
        notes: list[str] = []
        if markov_says and not brief_says and not investigator_says:
            notes.append(
                "Statistical MB flagged this column but neither the LLM "
                "investigator nor the expert brief named it — possible "
                "spurious correlation or genuinely-missed confounder."
            )
        if investigator_says and not markov_says and not brief_says:
            notes.append(
                "Investigator labelled this as CONFOUNDER but the statistical "
                "MB does not include it — possible role-misassignment or "
                "weak association below the CI test threshold."
            )
        if brief_says and not markov_says and not investigator_says:
            notes.append(
                "Brief listed this as a confounder but neither the investigator "
                "labelling nor the statistical MB agrees — verify the brief's "
                "domain claim against the data."
            )
        conflicts.append(
            DAGConflict(
                target=target,
                column=col,
                investigator_says=investigator_says,
                brief_says=brief_says,
                markov_says=markov_says,
                severity=severity,
                notes=notes,
            )
        )

    return DAGMismatchReport(
        conflicts=conflicts,
        high_severity_count=counts["high"],
        medium_severity_count=counts["medium"],
        low_severity_count=counts["low"],
        summary=(
            f"{counts['high']} high-severity + {counts['medium']} medium-severity "
            f"disagreements across {len(universe)} candidate confounders for target {target!r}."
        ),
    )


__all__ = ["DAGConflict", "DAGMismatchReport", "detect_conflicts"]
