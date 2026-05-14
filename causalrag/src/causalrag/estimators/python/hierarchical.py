"""HierarchicalDMLEstimator — cluster-aware DML (Sprint 6.5.5).

First-class support for clustered / multilevel data: units nested in
clusters (schools, hospitals, firms). Treatment may live at the unit
level (e.g., patient-level prescription) or at the cluster level (e.g.,
hospital-level policy); the estimator detects this from within-cluster
treatment variance and routes accordingly.

Architecture:

- Nuisance models: GradientBoosting regressor (outcome) and classifier
  (propensity). Confounders include unit-level + cluster-level covariates
  (cluster-level columns are taken at face value — they are constant
  within a cluster and the GBM will simply learn the cluster offset).
- Cross-fitting: ``GroupKFold`` keyed on the cluster column so units in
  the same cluster always land in the same fold. This is the canonical
  fix for the leakage that plain KFold introduces under clustering.
- ATE: AIPW (doubly-robust) score, averaged over units. When treatment
  is cluster-level, the AIPW score is averaged within each cluster first
  (so every cluster contributes one independent draw to the influence
  function), then over clusters.
- Standard errors: cluster-robust sandwich estimator on the influence
  function (the Liang-Zeger CR0 form). Bootstrap-of-clusters
  (Cameron-Gelbach-Miller 2011) resamples whole clusters with
  replacement, refits, and reports empirical-quantile CIs alongside.

The ``CLUSTERED`` DataFlag is *not* declared as a hard requirement: the
estimator is well-defined and informative on single-level data too (it
just collapses to a plain AIPW with degenerate cluster structure), and
the registry's hard-requirement semantics would otherwise hide it on
ordinary datasets. Discovery + scoring can still prefer it when the
flag is set.

Diagnostics surfaced:

- ``icc``: intra-class correlation of the outcome (one-way ANOVA ratio
  ``between_var / (between_var + within_var)``).
- ``n_clusters`` and ``units_per_cluster_p50``.
- ``treatment_level``: ``"unit"`` or ``"cluster"``, set from the mean of
  per-cluster within-cluster treatment variance.
- ``naive_se`` and ``cluster_robust_se`` so callers can see the gap
  between the iid and clustered standard errors.
- ``bootstrap_se``, ``bootstrap_ci_low``, ``bootstrap_ci_high``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.registry import EstimatorEntry, register
from causalrag.core.result import EstimationResult

_Z_975 = 1.959963984540054
_MIN_N_CLUSTERS = 10


@dataclass
class _ClusteredPrepared:
    y: np.ndarray
    t: np.ndarray
    x: np.ndarray  # all confounders (unit + cluster level), possibly empty (n, 0)
    groups: np.ndarray  # int-coded cluster ids per row
    cluster_ids: np.ndarray  # unique cluster labels, sorted
    n: int
    n_clusters: int


class HierarchicalDMLEstimator:
    """Cluster-aware DML with cluster-robust standard errors.

    Two-level structure: units nested in clusters. Treatment can be at
    unit level (e.g., patient-level prescription) or cluster level
    (e.g., hospital-level policy). The estimator detects the level
    automatically based on the variance of treatment within clusters
    and routes to the appropriate cross-fit strategy.

    Cross-fitting respects cluster boundaries — units in the same
    cluster always end up in the same fold (cluster-CV).

    Standard errors are computed via the cluster-robust sandwich
    estimator + bootstrap-of-clusters (resamples whole clusters, not
    rows, per Cameron-Gelbach-Miller 2011).
    """

    id: str = "python.hierarchical.dml"
    backend: Literal["python", "r"] = "python"
    supported_estimands: tuple[str, ...] = ("ATE", "ATT")
    # CLUSTERED is allowed to drive routing preference but is not a hard
    # requirement: the estimator degrades gracefully on flat data.
    required_flags: frozenset[DataFlag] = frozenset()
    excluded_flags: frozenset[DataFlag] = frozenset(
        {
            DataFlag.RIGHT_CENSORED_OUTCOME,
            DataFlag.TIME_VARYING_TREATMENT,
            DataFlag.LONGITUDINAL,
            DataFlag.PANEL_STRUCTURE,
        }
    )
    min_sample_size: int = 100  # AND at least _MIN_N_CLUSTERS clusters
    produces_cate: bool = False
    produces_full_counterfactual: bool = False
    propensity_required: bool = True

    def __init__(
        self,
        treatment: str,
        outcome: str,
        cluster_column: str,
        confounders: tuple[str, ...] = (),
        cluster_confounders: tuple[str, ...] = (),
        modifiers: tuple[str, ...] = (),
        *,
        n_folds: int = 5,
        bootstrap_iterations: int = 200,
        seed: int = 42,
    ) -> None:
        self.treatment = treatment
        self.outcome = outcome
        self.cluster_column = cluster_column
        self.confounders = tuple(confounders)
        self.cluster_confounders = tuple(cluster_confounders)
        self.modifiers = tuple(modifiers)
        self.n_folds = int(n_folds)
        self.bootstrap_iterations = int(bootstrap_iterations)
        self.seed = int(seed)

        # Fit-time state.
        self._prep: _ClusteredPrepared | None = None
        self._mu0: np.ndarray | None = None  # E[Y|X, T=0]
        self._mu1: np.ndarray | None = None  # E[Y|X, T=1]
        self._ps: np.ndarray | None = None  # P(T=1|X)
        self._phi: np.ndarray | None = None  # AIPW influence-function values
        self._point: float | None = None
        self._naive_se: float | None = None
        self._cluster_se: float | None = None
        self._treatment_level: str | None = None
        self._icc: float | None = None
        self._fit_seconds: float | None = None
        self._backend_version: str | None = None

    # ------------------------------------------------------------------
    # Data prep
    # ------------------------------------------------------------------
    def _prepare(self, data: pd.DataFrame) -> _ClusteredPrepared:
        feature_cols = tuple(self.confounders) + tuple(self.cluster_confounders)
        cols = [self.outcome, self.treatment, self.cluster_column, *feature_cols]
        for c in cols:
            if c not in data.columns:
                raise ValueError(f"Column not in data: {c!r}")
        df = data[cols].dropna()
        n = len(df)
        if n < self.min_sample_size:
            raise ValueError(
                f"HierarchicalDML requires at least {self.min_sample_size} rows "
                f"after dropna; got {n}"
            )
        unique_clusters = np.array(sorted(df[self.cluster_column].unique().tolist()))
        n_clusters = len(unique_clusters)
        if n_clusters < _MIN_N_CLUSTERS:
            raise ValueError(
                f"HierarchicalDML requires at least {_MIN_N_CLUSTERS} clusters; "
                f"got {n_clusters}"
            )
        # Int-code clusters for fast groupby downstream.
        cluster_index = {c: i for i, c in enumerate(unique_clusters.tolist())}
        groups = np.array(
            [cluster_index[c] for c in df[self.cluster_column].tolist()], dtype=np.int64
        )
        y = df[self.outcome].to_numpy().astype(np.float64)
        t = df[self.treatment].to_numpy().astype(np.float64)
        unique_t = set(np.unique(t).tolist())
        if not unique_t.issubset({0.0, 1.0}):
            raise ValueError(
                f"HierarchicalDML requires binary {{0, 1}} treatment; got {unique_t}"
            )
        if feature_cols:
            x = df[list(feature_cols)].to_numpy().astype(np.float64)
        else:
            x = np.zeros((n, 0), dtype=np.float64)
        return _ClusteredPrepared(
            y=y,
            t=t,
            x=x,
            groups=groups,
            cluster_ids=unique_clusters,
            n=n,
            n_clusters=n_clusters,
        )

    # ------------------------------------------------------------------
    # Treatment-level detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_treatment_level(t: np.ndarray, groups: np.ndarray) -> str:
        # Mean within-cluster variance of T. Exactly 0 ⇒ cluster-level.
        total = 0.0
        n_groups = 0
        for g in np.unique(groups):
            tg = t[groups == g]
            if tg.size <= 1:
                continue
            total += float(np.var(tg))
            n_groups += 1
        if n_groups == 0:
            return "unit"  # degenerate; fall through to unit-level path
        mean_within_var = total / n_groups
        return "cluster" if mean_within_var == 0.0 else "unit"

    # ------------------------------------------------------------------
    # ICC of the outcome
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_icc(y: np.ndarray, groups: np.ndarray) -> float:
        grand = float(np.mean(y))
        between_num = 0.0
        within_num = 0.0
        n_groups = 0
        for g in np.unique(groups):
            yg = y[groups == g]
            ng = len(yg)
            mean_g = float(np.mean(yg))
            between_num += ng * (mean_g - grand) ** 2
            within_num += float(np.sum((yg - mean_g) ** 2))
            n_groups += 1
        n_total = len(y)
        if n_groups <= 1 or n_total <= n_groups:
            return 0.0
        between_ms = between_num / (n_groups - 1)
        within_ms = within_num / (n_total - n_groups)
        # ANOVA-style ICC(1,1): (MSB - MSW) / (MSB + (k-1) * MSW) with k = avg group size.
        k = n_total / n_groups
        denom = between_ms + (k - 1) * within_ms
        if denom <= 0:
            return 0.0
        icc = (between_ms - within_ms) / denom
        # Clip into [0, 1] for reporting; negative values just mean essentially zero.
        return float(max(0.0, min(1.0, icc)))

    # ------------------------------------------------------------------
    # Nuisance fitting with cluster-aware cross-fitting
    # ------------------------------------------------------------------
    def _fit_nuisances(
        self, prep: _ClusteredPrepared
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.model_selection import GroupKFold, KFold

        n = prep.n
        mu0 = np.zeros(n, dtype=np.float64)
        mu1 = np.zeros(n, dtype=np.float64)
        ps = np.full(n, 0.5, dtype=np.float64)

        # When working with a small dataset (e.g., collapsed cluster-level
        # rows), the GBM nuisance models overfit and push propensities to
        # 0/1. Switch to regularized linear nuisances under that regime —
        # they're the appropriate fallback at n ≲ 100.
        small_n = n < 100

        # GroupKFold needs ≥ n_folds distinct groups. With very few clusters,
        # fall back to ordinary KFold to keep tests/synthetic data feasible.
        n_folds = min(self.n_folds, prep.n_clusters)
        if n_folds < 2:
            n_folds = 2
        if prep.n_clusters >= n_folds and len(np.unique(prep.groups)) >= n_folds:
            splitter = GroupKFold(n_splits=n_folds).split(prep.x, prep.y, prep.groups)
        else:
            splitter = KFold(
                n_splits=n_folds, shuffle=True, random_state=self.seed
            ).split(prep.x)

        x_for_fit = prep.x if prep.x.shape[1] > 0 else np.zeros((n, 1))

        def _make_reg() -> Any:
            if small_n:
                return Ridge(alpha=1.0, random_state=self.seed)
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.seed,
            )

        def _make_clf() -> Any:
            if small_n:
                return LogisticRegression(
                    C=1.0, max_iter=1000, random_state=self.seed
                )
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=self.seed,
            )

        for train_idx, test_idx in splitter:
            y_tr, t_tr = prep.y[train_idx], prep.t[train_idx]
            x_tr = x_for_fit[train_idx]
            x_te = x_for_fit[test_idx]
            # Outcome models, fit on treated/control subsets of the train fold.
            for arm in (0, 1):
                mask = t_tr == arm
                if mask.sum() < 2:
                    pred = (
                        float(np.mean(y_tr[mask])) if mask.sum() == 1 else float(np.mean(y_tr))
                    )
                    if arm == 0:
                        mu0[test_idx] = pred
                    else:
                        mu1[test_idx] = pred
                    continue
                reg = _make_reg()
                reg.fit(x_tr[mask], y_tr[mask])
                pred = reg.predict(x_te)
                if arm == 0:
                    mu0[test_idx] = pred
                else:
                    mu1[test_idx] = pred
            # Propensity model — only fit if both arms present in the train fold.
            if len(np.unique(t_tr)) < 2:
                ps[test_idx] = float(np.mean(t_tr)) if t_tr.size else 0.5
            else:
                clf = _make_clf()
                clf.fit(x_tr, t_tr.astype(int))
                ps[test_idx] = clf.predict_proba(x_te)[:, 1]
        # Clip propensities — tighter under small n where extreme values
        # are almost certainly overfitting artifacts rather than signal.
        lo, hi = (0.05, 0.95) if small_n else (1e-3, 1 - 1e-3)
        ps = np.clip(ps, lo, hi)
        return mu0, mu1, ps

    # ------------------------------------------------------------------
    # AIPW influence function + cluster-robust SE
    # ------------------------------------------------------------------
    @staticmethod
    def _aipw_phi(
        y: np.ndarray, t: np.ndarray, mu0: np.ndarray, mu1: np.ndarray, ps: np.ndarray
    ) -> np.ndarray:
        return (mu1 - mu0) + t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps)

    @staticmethod
    def _cluster_robust_se(phi: np.ndarray, groups: np.ndarray) -> float:
        n = len(phi)
        psi = phi - float(np.mean(phi))
        # Sum of cluster-sums squared, divided by n^2 — this is the
        # standard CR0 variance for the sample mean of phi.
        ss = 0.0
        for g in np.unique(groups):
            ss += float(np.sum(psi[groups == g])) ** 2
        var = ss / (n * n)
        return float(np.sqrt(max(var, 0.0)))

    @staticmethod
    def _naive_se_estimate(phi: np.ndarray) -> float:
        n = len(phi)
        if n < 2:
            return 0.0
        return float(np.std(phi, ddof=1) / np.sqrt(n))

    # ------------------------------------------------------------------
    # Bootstrap-of-clusters
    # ------------------------------------------------------------------
    def _bootstrap_clusters(
        self, prep: _ClusteredPrepared, alpha: float = 0.05
    ) -> tuple[float | None, float | None, float | None]:
        B = max(1, int(self.bootstrap_iterations))
        rng = np.random.default_rng(np.random.SeedSequence(self.seed))
        cluster_ix = np.arange(prep.n_clusters)
        # Pre-index rows by cluster for O(1) lookup per draw.
        cluster_rows: list[np.ndarray] = [
            np.where(prep.groups == g)[0] for g in cluster_ix
        ]
        cluster_level = self._treatment_level == "cluster"
        replicates: list[float] = []
        for _ in range(B):
            sampled = rng.integers(0, prep.n_clusters, size=prep.n_clusters)
            row_idx = np.concatenate([cluster_rows[c] for c in sampled])
            if len(np.unique(prep.t[row_idx])) < 2:
                continue
            sub_prep = _ClusteredPrepared(
                y=prep.y[row_idx],
                t=prep.t[row_idx],
                x=prep.x[row_idx] if prep.x.shape[1] > 0 else np.zeros((len(row_idx), 0)),
                groups=np.repeat(np.arange(len(sampled)), [len(cluster_rows[c]) for c in sampled]),
                cluster_ids=np.arange(len(sampled)),
                n=len(row_idx),
                n_clusters=len(sampled),
            )
            try:
                if cluster_level:
                    sub_prep = self._collapse_to_cluster(sub_prep)
                mu0_b, mu1_b, ps_b = self._fit_nuisances(sub_prep)
                phi_b = self._aipw_phi(sub_prep.y, sub_prep.t, mu0_b, mu1_b, ps_b)
                replicates.append(float(np.mean(phi_b)))
            except Exception:
                continue
        if len(replicates) < 2:
            return None, None, None
        arr = np.asarray(replicates, dtype=np.float64)
        lo = float(np.quantile(arr, alpha / 2))
        hi = float(np.quantile(arr, 1 - alpha / 2))
        se = float(np.std(arr, ddof=1))
        return lo, hi, se

    # ------------------------------------------------------------------
    # Estimator Protocol
    # ------------------------------------------------------------------
    def _collapse_to_cluster(
        self, prep: _ClusteredPrepared
    ) -> _ClusteredPrepared:
        """Average to one row per cluster — the right unit-of-analysis when
        treatment is assigned at the cluster level."""
        unique_g = np.unique(prep.groups)
        n_c = len(unique_g)
        y_c = np.zeros(n_c, dtype=np.float64)
        t_c = np.zeros(n_c, dtype=np.float64)
        x_c = np.zeros((n_c, prep.x.shape[1]), dtype=np.float64)
        for i, g in enumerate(unique_g):
            mask = prep.groups == g
            y_c[i] = float(np.mean(prep.y[mask]))
            t_c[i] = float(prep.t[mask][0])  # constant within cluster
            if prep.x.shape[1] > 0:
                x_c[i] = prep.x[mask].mean(axis=0)
        return _ClusteredPrepared(
            y=y_c,
            t=t_c,
            x=x_c,
            groups=np.arange(n_c, dtype=np.int64),
            cluster_ids=unique_g,
            n=n_c,
            n_clusters=n_c,
        )

    def fit(
        self, data: pd.DataFrame, protocol: StudyProtocol
    ) -> HierarchicalDMLEstimator:
        prep = self._prepare(data)
        start = time.perf_counter()
        treatment_level = self._detect_treatment_level(prep.t, prep.groups)

        if treatment_level == "cluster":
            # Collapse to one row per cluster — every cluster becomes the
            # unit of analysis. AIPW is computed at the cluster level and
            # the SE is just the iid SE of cluster-level scores (which IS
            # the cluster-robust SE in this regime).
            prep_eff = self._collapse_to_cluster(prep)
            mu0, mu1, ps = self._fit_nuisances(prep_eff)
            phi = self._aipw_phi(prep_eff.y, prep_eff.t, mu0, mu1, ps)
            point = float(np.mean(phi))
            naive_se = self._naive_se_estimate(phi)
            cluster_se = naive_se  # already at cluster level
        else:
            mu0, mu1, ps = self._fit_nuisances(prep)
            phi = self._aipw_phi(prep.y, prep.t, mu0, mu1, ps)
            point = float(np.mean(phi))
            naive_se = self._naive_se_estimate(phi)
            cluster_se = self._cluster_robust_se(phi, prep.groups)

        self._prep = prep
        self._mu0 = mu0
        self._mu1 = mu1
        self._ps = ps
        self._phi = phi
        self._point = point
        self._naive_se = naive_se
        self._cluster_se = cluster_se
        self._treatment_level = treatment_level
        self._icc = self._compute_icc(prep.y, prep.groups)
        self._fit_seconds = time.perf_counter() - start
        try:
            import sklearn

            self._backend_version = f"sklearn {sklearn.__version__}"
        except Exception:
            self._backend_version = None
        return self

    def estimate(self) -> EstimationResult:
        if self._prep is None or self._point is None:
            raise RuntimeError("Call fit() before estimate().")
        prep = self._prep

        # Cluster-bootstrap CI/SE.
        b_lo, b_hi, b_se = self._bootstrap_clusters(prep, alpha=0.05)

        # Prefer cluster-robust analytic CI for the headline interval; fall
        # back to the bootstrap when the sandwich SE collapses to 0.
        se = self._cluster_se if self._cluster_se and self._cluster_se > 0 else b_se
        if se is not None and se > 0:
            ci_low = self._point - _Z_975 * se
            ci_high = self._point + _Z_975 * se
            from math import erfc, sqrt

            p_value = float(erfc(abs(self._point) / (se * sqrt(2.0))))
        else:
            ci_low = b_lo
            ci_high = b_hi
            p_value = None

        # Sizes for diagnostics.
        sizes = [int(np.sum(prep.groups == g)) for g in np.unique(prep.groups)]
        units_per_cluster_p50 = float(np.median(sizes)) if sizes else 0.0

        diagnostics: dict[str, Any] = {
            "icc": self._icc,
            "n_clusters": prep.n_clusters,
            "units_per_cluster_p50": units_per_cluster_p50,
            "treatment_level": self._treatment_level,
            "naive_se": self._naive_se,
            "cluster_robust_se": self._cluster_se,
            "bootstrap_se": b_se,
            "bootstrap_ci_low": b_lo,
            "bootstrap_ci_high": b_hi,
            "bootstrap_iterations": self.bootstrap_iterations,
            "n_folds": min(self.n_folds, prep.n_clusters),
        }

        return EstimationResult(
            estimator_id=self.id,
            estimand_class="ATE",
            point_estimate=self._point,
            se=se,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_used=prep.n,
            diagnostics=diagnostics,
            backend_version=self._backend_version,
            fit_seconds=self._fit_seconds,
        )

    def diagnose(self) -> dict[str, Any]:
        if self._prep is None:
            return {"fitted": False}
        prep = self._prep
        sizes = [int(np.sum(prep.groups == g)) for g in np.unique(prep.groups)]
        return {
            "fitted": True,
            "icc": self._icc,
            "n_clusters": prep.n_clusters,
            "units_per_cluster_p50": float(np.median(sizes)) if sizes else 0.0,
            "units_per_cluster_min": int(min(sizes)) if sizes else 0,
            "units_per_cluster_max": int(max(sizes)) if sizes else 0,
            "treatment_level": self._treatment_level,
            "naive_se": self._naive_se,
            "cluster_robust_se": self._cluster_se,
        }

    def refute(self) -> dict[str, Any]:
        # The cluster bootstrap is itself the primary refutation: it
        # stress-tests the dependence structure that the sandwich SE
        # assumes. Surface the gap between naive and cluster SE as a
        # cheap sanity check on whether clustering matters here.
        if self._naive_se is None or self._cluster_se is None:
            return {}
        ratio = (
            self._cluster_se / self._naive_se if self._naive_se > 0 else None
        )
        return {
            "cluster_se_over_naive_se": ratio,
            "clustering_matters": (ratio is not None and ratio > 1.25),
        }


def _register() -> None:
    register(
        EstimatorEntry(
            id=HierarchicalDMLEstimator.id,
            factory=HierarchicalDMLEstimator,
            backend=HierarchicalDMLEstimator.backend,
            supported_estimands=frozenset(HierarchicalDMLEstimator.supported_estimands),
            required_flags=HierarchicalDMLEstimator.required_flags,
            excluded_flags=HierarchicalDMLEstimator.excluded_flags,
            min_sample_size=HierarchicalDMLEstimator.min_sample_size,
            produces_cate=HierarchicalDMLEstimator.produces_cate,
            produces_full_counterfactual=HierarchicalDMLEstimator.produces_full_counterfactual,
            propensity_required=HierarchicalDMLEstimator.propensity_required,
        )
    )


_register()
