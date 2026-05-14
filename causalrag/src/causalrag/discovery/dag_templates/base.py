"""DAGTemplate base contract (Sprint 6.5.7).

A ``DAGTemplate`` is a Pydantic model that declares named *slots* — abstract
roles like ``treatment`` or ``outcome`` — plus the edges that connect them.
Subclasses implement :meth:`instantiate`, which takes a ``column_map`` mapping
slot names to real dataset columns and returns a fully built
:class:`~causalrag.core.graph.CausalGraph` with roles assigned.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from causalrag.core.graph import CausalGraph


class DAGTemplate(BaseModel):
    """A reusable DAG skeleton for a domain pattern.

    Subclasses define the structural slots (treatment, outcome, confounders,
    mediators, instruments) plus domain-specific nodes (washout window,
    eligibility, neighbour adjacency, service call edges, ...) and materialise
    them into a :class:`CausalGraph` via :meth:`instantiate`.
    """

    model_config = ConfigDict(extra="forbid")

    template_name: str
    domain: str  # "clinical" | "marketing" | "engineering" | "ecology"
    description: str
    slots: list[str]  # required column-name slots

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def instantiate(self, column_map: dict[str, str]) -> CausalGraph:  # pragma: no cover - abstract
        """Map slots -> real column names; build a CausalGraph.

        Raises ``ValueError`` if a required slot is missing from ``column_map``.
        Subclasses must override this.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _check_required_slots(self, column_map: dict[str, str]) -> None:
        """Validate that every slot in :attr:`slots` is present in ``column_map``.

        Raises ``ValueError`` listing every missing slot so the user can fix
        their mapping in one round-trip.
        """
        missing = [s for s in self.slots if s not in column_map or not column_map[s]]
        if missing:
            raise ValueError(
                f"DAGTemplate {self.template_name!r} is missing required slot "
                f"mappings: {missing}. Provide every slot in column_map."
            )
