"""``mediation`` wrapper — Imai-Keele-Yamamoto causal mediation.

Decomposes the total effect into Natural Direct (NDE) and Natural
Indirect (NIE) components through a named mediator. Uses
``mediation::mediate`` with bootstrap CIs.

Auto-routes when ``MEDIATOR_PROPOSED`` is flagged AND the requested
estimand is NDE / NIE / TE.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import EstimatorEntry, register
from causalrag.core.result import EstimationResult
from causalrag.estimators.rbridge._r import (
    converter,
    r_session,
    r_session_metadata,
    require,
)


class MediationEstimator:
    id: str = "rbridge.mediation"
    backend: Literal["python", "r"] = "r"
    supported_estimands: tuple[str, ...] = ("NDE", "NIE")
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.MEDIATOR_PROPOSED})
    excluded_flags: frozenset[DataFlag] = frozenset({DataFlag.RIGHT_CENSORED_OUTCOME})
    min_sample_size: int = 100
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        mediator: str,
        confounders: tuple[str, ...],
        modifiers: tuple[str, ...] = (),
        *,
        boot: int = 1000,
        seed: int = 42,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.mediator = mediator
        self.confounders = confounders
        self.modifiers = modifiers
        self.boot = boot
        self.seed = seed
        self._n_used = 0
        self._fit_seconds: float | None = None

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> "MediationEstimator":
        require("mediation")
        ro = r_session()
        cols = [self.outcome, self.mediator, self.treatment, *self.confounders, *self.modifiers]
        df = data[cols].dropna()
        self._n_used = len(df)
        if self._n_used < self.min_sample_size:
            raise ValueError(f"mediation needs ≥ {self.min_sample_size}; got {self._n_used}")
        with converter():
            ro.globalenv["df_"] = ro.conversion.py2rpy(df)
        adj = (" + " + " + ".join(self.confounders)) if self.confounders else ""
        # Mediator model and outcome model
        ro.r(f"m_model <- lm({self.mediator} ~ {self.treatment}{adj}, data = df_)")
        ro.r(
            f"y_model <- lm({self.outcome} ~ {self.treatment} + {self.mediator}{adj}, data = df_)"
        )
        start = time.perf_counter()
        ro.r(
            f"set.seed({self.seed});"
            f'res_ <- mediation::mediate(m_model, y_model, treat = "{self.treatment}", '
            f'mediator = "{self.mediator}", boot = TRUE, sims = {self.boot})'
        )
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        ro = r_session()
        # res_$d.avg = NIE estimate, $z.avg = NDE, with $.ci attributes
        nie = float(list(ro.r("res_$d.avg"))[0])
        nie_ci = list(ro.r("res_$d.avg.ci"))
        nde = float(list(ro.r("res_$z.avg"))[0])
        nde_ci = list(ro.r("res_$z.avg.ci"))
        # Total = direct + indirect
        te = float(list(ro.r("res_$tau.coef"))[0])

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="NIE",
            point_estimate=nie,
            ci_low=float(nie_ci[0]),
            ci_high=float(nie_ci[1]),
            n_used=self._n_used,
            diagnostics={
                "nde": nde,
                "nde_ci": [float(nde_ci[0]), float(nde_ci[1])],
                "total_effect": te,
                "mediator": self.mediator,
                "boot": self.boot,
            },
            backend_version=r_session_metadata().get("packages", {}).get("mediation", "?"),
            r_session_metadata=r_session_metadata(),
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {"n_used": self._n_used, "mediator": self.mediator}

    def refute(self) -> dict[str, Any]:
        return {}


def _register() -> None:
    register(
        EstimatorEntry(
            id=MediationEstimator.id,
            factory=MediationEstimator,
            backend=MediationEstimator.backend,
            supported_estimands=frozenset(MediationEstimator.supported_estimands),
            required_flags=MediationEstimator.required_flags,
            excluded_flags=MediationEstimator.excluded_flags,
            min_sample_size=MediationEstimator.min_sample_size,
            produces_cate=MediationEstimator.produces_cate,
            produces_full_counterfactual=MediationEstimator.produces_full_counterfactual,
            propensity_required=MediationEstimator.propensity_required,
        )
    )


_register()
