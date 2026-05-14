"""Step 6 — translate the causal estimand to a canonical statistical functional.

PDD §10.6. The output is a :class:`StatisticalEstimand` that names the
adjustment set, the identification strategy, and a plain-text canonical form
(e.g. ``E[Y | T=1, X] − E[Y | T=0, X]`` for ATE under backdoor).

Step 6 is deterministic — no LLM. It reads the IdentificationResult from
Step 5 and emits the matching functional form.
"""

from __future__ import annotations

from causalrag.core.estimand import CausalEstimand, EstimandClass, StatisticalEstimand
from causalrag.roadmap.q5_identify import IdentificationResult


_CANONICAL_FORM_TEMPLATES: dict[EstimandClass, str] = {
    EstimandClass.ATE: "E_X[E[Y | T=1, X] − E[Y | T=0, X]]",
    EstimandClass.ATT: "E_X|T=1[E[Y | T=1, X] − E[Y | T=0, X]]",
    EstimandClass.ATC: "E_X|T=0[E[Y | T=1, X] − E[Y | T=0, X]]",
    EstimandClass.CATE: "E[Y | T=1, X=x] − E[Y | T=0, X=x]",
    EstimandClass.LATE: "Wald: Cov(Y, Z | X) / Cov(T, Z | X)",
    EstimandClass.RMST_CONTRAST: "E[min(T_surv, τ) | A=1] − E[min(T_surv, τ) | A=0]",
    EstimandClass.NDE: "E_X[E[Y(1, M(0)) − Y(0, M(0)) | X]]",
    EstimandClass.NIE: "E_X[E[Y(1, M(1)) − Y(1, M(0)) | X]]",
}


def derive_statistical_estimand(
    causal: CausalEstimand,
    identification: IdentificationResult,
) -> StatisticalEstimand:
    """Step 6 — emit a canonical statistical functional."""
    template = _CANONICAL_FORM_TEMPLATES.get(
        causal.klass, f"{causal.klass.value}_functional"
    )
    return StatisticalEstimand(
        causal_estimand=causal,
        canonical_form=template,
        adjustment_set=tuple(identification.adjustment_set),
        identification_strategy=identification.strategy,
    )


__all__ = ["derive_statistical_estimand"]
