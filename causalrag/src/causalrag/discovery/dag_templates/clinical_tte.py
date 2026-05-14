"""Clinical target-trial-emulation (TTE) DAG template (Hernan & Robins).

Encodes the canonical target-trial flow:

    eligibility -> treatment_strategy -> treatment_received -> adherence
                                                                  |
                                                                  v
                  baseline_confounders ----------------> outcome_observed
                                                                  ^
                  loss_to_followup_censoring -----------------------+

Baseline confounders point at both treatment assignment and the outcome (the
classical confounding fork). Time-varying confounders, if supplied, are
mediators between treatment and outcome and also influence later censoring.
"""

from __future__ import annotations

from pydantic import Field

from causalrag.core.graph import CausalEdge, CausalGraph
from causalrag.core.roles import VariableRole
from causalrag.discovery.dag_templates.base import DAGTemplate


_REQUIRED_SLOTS = [
    "eligibility_indicator",
    "washout_period_complete",
    "treatment_strategy",
    "treatment_received",
    "adherence",
    "loss_to_followup_censoring",
    "outcome_observed",
]


class ClinicalTTETemplate(DAGTemplate):
    """Target-trial-emulation skeleton for observational clinical data."""

    template_name: str = "clinical_tte"
    domain: str = "clinical"
    description: str = (
        "Hernan-Robins target-trial emulation: eligibility and washout gate "
        "treatment assignment; baseline confounders fork into treatment and "
        "outcome; adherence and censoring sit on the post-baseline path."
    )
    slots: list[str] = Field(default_factory=lambda: list(_REQUIRED_SLOTS))

    baseline_confounders: list[str] = Field(default_factory=list)
    time_varying_confounders: list[str] = Field(default_factory=list)

    def instantiate(self, column_map: dict[str, str]) -> CausalGraph:
        self._check_required_slots(column_map)

        eligibility = column_map["eligibility_indicator"]
        washout = column_map["washout_period_complete"]
        strategy = column_map["treatment_strategy"]
        received = column_map["treatment_received"]
        adherence = column_map["adherence"]
        censoring = column_map["loss_to_followup_censoring"]
        outcome = column_map["outcome_observed"]

        # Baseline confounders may be passed either as a list of *slot names*
        # (each looked up in column_map) or as already-resolved column names.
        baseline = [column_map.get(c, c) for c in self.baseline_confounders]
        time_varying = [column_map.get(c, c) for c in self.time_varying_confounders]

        edges: list[tuple[str, str]] = []
        # Target-trial backbone.
        edges.append((eligibility, strategy))
        edges.append((washout, strategy))
        edges.append((strategy, received))
        edges.append((received, adherence))
        edges.append((adherence, outcome))
        edges.append((censoring, outcome))
        edges.append((adherence, censoring))

        # Baseline confounders fork into both treatment and outcome.
        for c in baseline:
            edges.append((c, strategy))
            edges.append((c, outcome))

        # Time-varying confounders sit on the treatment -> outcome path and
        # also drive later censoring (classical time-dependent confounding).
        for tv in time_varying:
            edges.append((received, tv))
            edges.append((tv, outcome))
            edges.append((tv, censoring))

        roles: dict[str, VariableRole] = {
            eligibility: VariableRole.AUXILIARY,
            washout: VariableRole.AUXILIARY,
            strategy: VariableRole.TREATMENT,
            received: VariableRole.TREATMENT,
            adherence: VariableRole.MEDIATOR,
            censoring: VariableRole.CENSORING_INDICATOR,
            outcome: VariableRole.OUTCOME,
        }
        for c in baseline:
            roles.setdefault(c, VariableRole.CONFOUNDER)
        for tv in time_varying:
            roles.setdefault(tv, VariableRole.MEDIATOR)

        edge_objs = tuple(
            CausalEdge(source=u, target=v, note=f"{self.template_name} template")
            for u, v in edges
        )
        nodes: list[str] = []
        seen: set[str] = set()
        for u, v in edges:
            for n in (u, v):
                if n not in seen:
                    seen.add(n)
                    nodes.append(n)
        return CausalGraph(nodes=tuple(nodes), edges=edge_objs, roles=roles)
