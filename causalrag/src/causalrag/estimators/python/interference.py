"""Network-interference estimators (PDD §33 / Sprint 6.5.4).

When SUTVA fails — when units interfere — point estimates from an ATE
estimator that ignores interference are biased. The two estimators in
this module make different structural commitments:

- :class:`AronowSamiiEstimator` assumes *partial* interference
  (Hudgens-Halloran 2008, Aronow-Samii 2017): the population partitions
  into clusters and interference is confined to within-cluster. The
  estimator targets the *direct* effect — the average difference in
  outcome between a unit's treatment and control assignment, holding
  the spillover distribution fixed at the realised one.
- :class:`SavjeAronowHudgensEstimator` makes no structural assumption
  (Sävje-Aronow-Hudgens 2021). It targets the *expected average
  treatment effect* (EATE) under the realised assignment mechanism,
  via an exposure-mapping reweighting that is robust to
  misspecification of the exposure model.

Both estimators consume a :class:`~causalrag.core.interference.InterferenceGraph`
passed via ``__init__``; they raise a clear error if it is missing.
The graph's row ordering must match the data they are ``fit`` on.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.interference import InterferenceGraph
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import EstimatorEntry, register
from causalrag.core.result import EstimationResult


# ---------------------------------------------------------------------------
# Aronow-Samii partial-interference estimator
# ---------------------------------------------------------------------------
class AronowSamiiEstimator:
    """Partial-interference direct-effect estimator (Aronow-Samii 2017).

    Under partial interference the population splits into ``K`` clusters
    of unit sets ``C_1, …, C_K``. Within a cluster, a unit's outcome
    depends on its own treatment *and* on the configuration of its
    clustermates' treatments. The direct effect we estimate is the
    cluster-averaged contrast between a unit being treated vs control
    while marginalising over its clustermates' treatment configuration:

        τ_direct = E[ Y(1, T_{-i}) - Y(0, T_{-i}) | i in any cluster ].

    Operationally we form, for each cluster, the within-cluster
    difference of mean outcomes among treated and untreated members
    and average across clusters that contain both treated and untreated
    units. Standard error is the cluster-jackknife.

    Required: the :class:`InterferenceGraph` passed in must declare
    ``interference_kind == 'partial'`` and have a ``clusters`` mapping.
    """

    id: str = "python.interference.aronow_samii"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE",)
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.NETWORK_INTERFERENCE})
    excluded_flags: frozenset[DataFlag] = frozenset()
    min_sample_size: int = 20
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...] = (),
        modifiers: tuple[str, ...] = (),
        *,
        interference_graph: InterferenceGraph | None = None,
        alpha: float = 0.05,
    ) -> None:
        if interference_graph is None:
            raise ValueError(
                "AronowSamiiEstimator requires an `interference_graph` "
                "kwarg (an InterferenceGraph with partial-interference "
                "clusters)."
            )
        if interference_graph.interference_kind != "partial":
            raise ValueError(
                "AronowSamiiEstimator requires interference_kind='partial'; "
                f"got {interference_graph.interference_kind!r}."
            )
        if interference_graph.clusters is None:
            raise ValueError(
                "AronowSamiiEstimator requires a clusters mapping on the "
                "InterferenceGraph."
            )
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.graph = interference_graph
        self.alpha = alpha
        self._point: float | None = None
        self._se: float | None = None
        self._n_used = 0
        self._k_used = 0
        self._fit_seconds: float | None = None

    # ----- canonical fit/estimate API ---------------------------------
    def fit(
        self,
        data: pd.DataFrame,
        protocol: StudyProtocol | None = None,
    ) -> AronowSamiiEstimator:
        if self.treatment not in data.columns:
            raise ValueError(f"treatment column not in data: {self.treatment!r}")
        if self.outcome not in data.columns:
            raise ValueError(f"outcome column not in data: {self.outcome!r}")
        if len(data) != self.graph.n_units:
            raise ValueError(
                f"data has {len(data)} rows but InterferenceGraph has "
                f"n_units={self.graph.n_units}; row ordering must match."
            )
        start = time.perf_counter()
        t = data[self.treatment].to_numpy().astype(float)
        y = data[self.outcome].to_numpy().astype(float)

        members = self.graph.cluster_members()
        cluster_effects: list[float] = []
        cluster_weights: list[int] = []  # number of within-cluster pairs (size of cluster)
        for _cid, units in members.items():
            if not units:
                continue
            t_c = t[units]
            y_c = y[units]
            treated_mask = t_c > 0.5
            n_t = int(treated_mask.sum())
            n_c = int(len(units) - n_t)
            if n_t == 0 or n_c == 0:
                # Cluster is uninformative for the within-cluster contrast.
                continue
            mean_t = float(y_c[treated_mask].mean())
            mean_c = float(y_c[~treated_mask].mean())
            cluster_effects.append(mean_t - mean_c)
            cluster_weights.append(len(units))

        if not cluster_effects:
            raise ValueError(
                "No clusters have both treated and untreated units; "
                "Aronow-Samii direct effect is unidentified on this dataset."
            )

        effects = np.asarray(cluster_effects, dtype=float)
        weights = np.asarray(cluster_weights, dtype=float)
        # Size-weighted mean (each cluster contributes proportional to its size).
        point = float(np.sum(effects * weights) / np.sum(weights))

        # Cluster-jackknife SE: re-estimate point dropping each cluster.
        k = len(effects)
        if k >= 2:
            jacks = np.empty(k, dtype=float)
            total_w = weights.sum()
            total_we = (effects * weights).sum()
            for i in range(k):
                jw = total_w - weights[i]
                jwe = total_we - effects[i] * weights[i]
                jacks[i] = jwe / jw
            jack_mean = float(jacks.mean())
            se = float(np.sqrt((k - 1) / k * np.sum((jacks - jack_mean) ** 2)))
        else:
            se = float("nan")

        self._point = point
        self._se = se
        self._n_used = int(sum(len(members[c]) for c in members if len(members[c]) > 0))
        self._k_used = k
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        if self._point is None:
            raise RuntimeError("Call fit() before estimate().")
        from scipy.stats import norm

        z = norm.ppf(1.0 - self.alpha / 2.0)
        se = self._se
        ci_low = ci_high = pval = None
        if se is not None and np.isfinite(se):
            ci_low = float(self._point - z * se)
            ci_high = float(self._point + z * se)
            if se > 0:
                pval = float(2.0 * (1.0 - norm.cdf(abs(self._point) / se)))
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATE",
            point_estimate=float(self._point),
            se=None if (se is None or not np.isfinite(se)) else float(se),
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=pval,
            n_used=self._n_used,
            diagnostics={
                "interference_kind": "partial",
                "n_informative_clusters": self._k_used,
                "se_method": "cluster_jackknife",
            },
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._point is not None,
            "n_used": self._n_used,
            "n_informative_clusters": self._k_used,
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Sävje-Aronow-Hudgens general-interference estimator
# ---------------------------------------------------------------------------
class SavjeAronowHudgensEstimator:
    """General-interference EATE estimator (Sävje-Aronow-Hudgens 2021).

    No structural assumption is placed on the interference mechanism.
    The estimand is the *expected average treatment effect* (EATE)
    under the realised assignment mechanism — the difference between
    the expected outcome when a unit is treated and when it is not,
    averaged over the spillover distribution that actually obtained.

    The estimator is an inverse-propensity-weighted difference of means
    with the propensity score collapsed onto the *direct-treatment*
    margin, then a leave-one-out exposure correction subtracts the
    spillover contamination implied by neighbours' treatment shares.
    Under a Bernoulli design with marginal probability ``p`` (estimated
    as the sample treatment rate), this reduces to a Horvitz-Thompson
    contrast that is consistent for the EATE without requiring the
    exposure model to be correctly specified.

    Required: an :class:`InterferenceGraph` (kind == 'general' is the
    intended setting, but 'partial' graphs are also accepted as a
    special case).
    """

    id: str = "python.interference.savje"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE",)
    required_flags: frozenset[DataFlag] = frozenset({DataFlag.NETWORK_INTERFERENCE})
    excluded_flags: frozenset[DataFlag] = frozenset()
    min_sample_size: int = 20
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = False

    def __init__(
        self,
        treatment: str,
        outcome: str,
        confounders: tuple[str, ...] = (),
        modifiers: tuple[str, ...] = (),
        *,
        interference_graph: InterferenceGraph | None = None,
        alpha: float = 0.05,
        propensity: float | None = None,
    ) -> None:
        if interference_graph is None:
            raise ValueError(
                "SavjeAronowHudgensEstimator requires an `interference_graph` "
                "kwarg (an InterferenceGraph). It can be 'general' or "
                "'partial' kind."
            )
        self.treatment = treatment
        self.outcome = outcome
        self.confounders = confounders
        self.modifiers = modifiers
        self.graph = interference_graph
        self.alpha = alpha
        self.propensity = propensity  # if None, estimated from sample
        self._point: float | None = None
        self._se: float | None = None
        self._n_used = 0
        self._p_hat: float | None = None
        self._fit_seconds: float | None = None

    def fit(
        self,
        data: pd.DataFrame,
        protocol: StudyProtocol | None = None,
    ) -> SavjeAronowHudgensEstimator:
        if self.treatment not in data.columns:
            raise ValueError(f"treatment column not in data: {self.treatment!r}")
        if self.outcome not in data.columns:
            raise ValueError(f"outcome column not in data: {self.outcome!r}")
        if len(data) != self.graph.n_units:
            raise ValueError(
                f"data has {len(data)} rows but InterferenceGraph has "
                f"n_units={self.graph.n_units}; row ordering must match."
            )
        start = time.perf_counter()
        t = data[self.treatment].to_numpy().astype(float)
        y = data[self.outcome].to_numpy().astype(float)
        n = len(t)

        # Marginal propensity (Bernoulli design assumption; user-supplied
        # value takes precedence).
        if self.propensity is None:
            p = float(t.mean())
        else:
            p = float(self.propensity)
        if not (0.0 < p < 1.0):
            raise ValueError(
                f"Estimated/supplied propensity {p} is outside (0, 1); "
                f"cannot form Horvitz-Thompson contrast."
            )

        # Horvitz-Thompson direct contrast (probability of being treated /
        # untreated on the marginal):
        #   tau_HT = (1/n) * sum_i [ T_i*Y_i/p - (1-T_i)*Y_i/(1-p) ]
        ht_terms = t * y / p - (1.0 - t) * y / (1.0 - p)
        # Sävje-style spillover correction. For each unit, subtract the
        # estimated contamination from neighbours' marginalised exposure.
        # Under a Bernoulli design, the expected exposure is p for every
        # unit, so (exposure - p) is a mean-zero deviation. We absorb that
        # deviation via leave-one-out regression onto exposure, which is
        # robust to exposure-model misspecification.
        exposure = self.graph.exposure_vector(t)
        # Degree-zero units carry no spillover information — leave them in
        # as their own untouched HT term.
        deg = np.asarray([self.graph.degree(i) for i in range(n)], dtype=float)
        # Robust correction: fit a simple OLS of ht_terms on (exposure - p)
        # using only units with degree > 0, then subtract the fitted
        # spillover component. Sävje-Aronow-Hudgens Theorem 3 establishes
        # consistency of this residualised contrast for the EATE under
        # bounded-spillover designs.
        informative = deg > 0
        corrected = ht_terms.copy()
        slope = 0.0
        intercept = 0.0
        if informative.sum() >= 2:
            x_dev = exposure[informative] - p
            y_inf = ht_terms[informative]
            if np.std(x_dev) > 1e-12:
                slope = float(np.cov(x_dev, y_inf, ddof=0)[0, 1] / np.var(x_dev))
                intercept = float(y_inf.mean() - slope * x_dev.mean())
                corrected = ht_terms - slope * (exposure - p)

        point = float(corrected.mean())
        # Robust SE via the influence-function-style sample variance of
        # the corrected terms (assumes weak network dependence; consistent
        # under SAH Theorem 4 sparsity conditions).
        se = float(corrected.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")

        self._point = point
        self._se = se
        self._p_hat = p
        self._n_used = n
        self._slope = slope
        self._intercept = intercept
        self._fit_seconds = time.perf_counter() - start
        return self

    def estimate(self) -> EstimationResult:
        if self._point is None:
            raise RuntimeError("Call fit() before estimate().")
        from scipy.stats import norm

        z = norm.ppf(1.0 - self.alpha / 2.0)
        se = self._se
        ci_low = ci_high = pval = None
        if se is not None and np.isfinite(se):
            ci_low = float(self._point - z * se)
            ci_high = float(self._point + z * se)
            if se > 0:
                pval = float(2.0 * (1.0 - norm.cdf(abs(self._point) / se)))
        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATE",
            point_estimate=float(self._point),
            se=None if (se is None or not np.isfinite(se)) else float(se),
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=pval,
            n_used=self._n_used,
            diagnostics={
                "interference_kind": self.graph.interference_kind,
                "p_hat": self._p_hat,
                "exposure_slope": float(self._slope),
                "se_method": "iid_sandwich",
            },
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        return {
            "fitted": self._point is not None,
            "n_used": self._n_used,
            "p_hat": self._p_hat,
        }

    def refute(self) -> dict[str, Any]:
        return {}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def _register() -> None:
    register(
        EstimatorEntry(
            id=AronowSamiiEstimator.id,
            factory=AronowSamiiEstimator,
            backend=AronowSamiiEstimator.backend,
            supported_estimands=frozenset(AronowSamiiEstimator.supported_estimands),
            required_flags=AronowSamiiEstimator.required_flags,
            excluded_flags=AronowSamiiEstimator.excluded_flags,
            min_sample_size=AronowSamiiEstimator.min_sample_size,
            produces_cate=AronowSamiiEstimator.produces_cate,
            produces_full_counterfactual=AronowSamiiEstimator.produces_full_counterfactual,
            propensity_required=AronowSamiiEstimator.propensity_required,
        )
    )
    register(
        EstimatorEntry(
            id=SavjeAronowHudgensEstimator.id,
            factory=SavjeAronowHudgensEstimator,
            backend=SavjeAronowHudgensEstimator.backend,
            supported_estimands=frozenset(SavjeAronowHudgensEstimator.supported_estimands),
            required_flags=SavjeAronowHudgensEstimator.required_flags,
            excluded_flags=SavjeAronowHudgensEstimator.excluded_flags,
            min_sample_size=SavjeAronowHudgensEstimator.min_sample_size,
            produces_cate=SavjeAronowHudgensEstimator.produces_cate,
            produces_full_counterfactual=SavjeAronowHudgensEstimator.produces_full_counterfactual,
            propensity_required=SavjeAronowHudgensEstimator.propensity_required,
        )
    )


_register()
