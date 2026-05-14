"""Domain-specific DAG templates (Sprint 6.5.7).

Each template is a Pydantic model that defines a reusable DAG skeleton —
named structural slots plus the edges that wire them together. Calling
:meth:`DAGTemplate.instantiate` with a ``column_map`` mapping each slot to a
real dataset column produces a :class:`~causalrag.core.graph.CausalGraph` with
roles assigned, ready for downstream identification.

The templates are deliberately read-only: they do not mutate the study
protocol, do not call the LLM, and contain no I/O. They exist so that an
analyst (or an upstream LLM stage) can pick a recognised domain pattern —
target-trial emulation, MMM, sequential attribution, spatiotemporal
partial-interference, engineering-trace SLO — and skip straight to a
well-typed causal graph.
"""

from __future__ import annotations

from causalrag.discovery.dag_templates.attribution import AttributionTemplate
from causalrag.discovery.dag_templates.base import DAGTemplate
from causalrag.discovery.dag_templates.clinical_tte import ClinicalTTETemplate
from causalrag.discovery.dag_templates.engineering_trace import EngineeringTraceTemplate
from causalrag.discovery.dag_templates.mmm import MMMTemplate
from causalrag.discovery.dag_templates.spatiotemporal import SpatiotemporalTemplate

__all__ = [
    "AttributionTemplate",
    "ClinicalTTETemplate",
    "DAGTemplate",
    "EngineeringTraceTemplate",
    "MMMTemplate",
    "SpatiotemporalTemplate",
]
