"""Manual hypothesis path — analyst writes hypotheses directly.

Produces ``Hypothesis`` objects with an impact score = 1.0 (analyst-priority
overrides automated ranking). Used by ``causalrag hypothesize --manual ...``.
"""

from __future__ import annotations

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.protocol import Hypothesis


def from_pairs(
    pairs: list[tuple[str, str]],
    *,
    counterfactual: bool = False,
) -> list[Hypothesis]:
    """Construct one Hypothesis per (treatment, outcome) pair with an ATE
    estimand. Modifiers / mediator / instrument default to none — the analyst
    can edit the protocol YAML to refine."""
    out: list[Hypothesis] = []
    for i, (t, y) in enumerate(pairs):
        est = CausalEstimand.model_validate(
            {
                "class": EstimandClass.ATE,
                "treatment": t,
                "outcome": y,
                "formal_expression": "E[Y(1) - Y(0)]",
            }
        )
        out.append(
            Hypothesis(
                id=f"manual-{i + 1:02d}",
                treatment=t,
                outcome=y,
                estimand=est,
                counterfactual=counterfactual,
                rationale="Analyst-defined hypothesis (manual mode).",
                impact_score=1.0,
            )
        )
    return out


__all__ = ["from_pairs"]
