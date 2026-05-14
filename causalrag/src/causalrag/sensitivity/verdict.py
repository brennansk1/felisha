"""Sensitivity verdict aggregator — green / yellow / red triangulation
(PDD §11, §13 ``verdict.py``).

Different sensitivity methods answer related but distinct questions:

- E-value: how strong an unmeasured confounder is needed on both arms?
- Robustness value (Cinelli-Hazlett): partial R² strength benchmarked against
  observed covariates.
- Rosenbaum bounds (deferred to v0.5): worst-case Γ that flips the
  significance of a matched-pair test.

No single method is universally agreed-best; we run the ones available and
aggregate them into a single colored verdict the analyst can quote in the
report. The aggregation rule itself is contestable, so it is configurable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from causalrag.sensitivity.evalue import EValueResult
from causalrag.sensitivity.sensemakr_py import SensemakrResult

Color = Literal["green", "yellow", "red"]


class SensitivityVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    color: Color
    rationale: str
    components: dict[str, str] = Field(default_factory=dict)


def aggregate(
    *,
    evalue: EValueResult | None = None,
    sensemakr: SensemakrResult | None = None,
    rule: Literal["min", "average", "strict"] = "min",
) -> SensitivityVerdict:
    """Combine sensitivity outputs into a single verdict.

    Aggregation rules:

    - ``"min"``: pick the weakest component (most pessimistic). Default — the
      analyst should not be allowed to ignore a single red signal.
    - ``"average"``: ordinal mean (green=2, yellow=1, red=0) rounded.
    - ``"strict"``: only green if every component is green.
    """
    components: dict[str, Color] = {}
    if evalue is not None:
        components["evalue"] = _evalue_color(evalue)
    if sensemakr is not None:
        components["sensemakr"] = _sensemakr_color(sensemakr)

    if not components:
        return SensitivityVerdict(
            color="yellow",
            rationale="No sensitivity methods produced results.",
            components={},
        )

    if rule == "strict":
        color: Color = "green" if all(c == "green" for c in components.values()) else (
            "red" if "red" in components.values() else "yellow"
        )
    elif rule == "average":
        scores = {"green": 2, "yellow": 1, "red": 0}
        avg = sum(scores[c] for c in components.values()) / len(components)
        color = "green" if avg >= 1.5 else ("yellow" if avg >= 0.5 else "red")
    else:  # min
        order = ["red", "yellow", "green"]
        color = min(components.values(), key=lambda c: order.index(c))  # type: ignore[arg-type]

    rationale = "; ".join(f"{name}={c}" for name, c in components.items())
    return SensitivityVerdict(
        color=color,
        rationale=rationale,
        components={k: v for k, v in components.items()},
    )


def _evalue_color(r: EValueResult) -> Color:
    if r.e_value_ci is not None and r.e_value_ci <= 1.25:
        return "red"
    if r.e_value >= 3.0:
        return "green"
    if r.e_value >= 1.75:
        return "yellow"
    return "red"


def _sensemakr_color(r: SensemakrResult) -> Color:
    if r.robustness_value >= 0.2:
        return "green"
    if r.robustness_value >= 0.075:
        return "yellow"
    return "red"


__all__ = ["SensitivityVerdict", "aggregate"]
