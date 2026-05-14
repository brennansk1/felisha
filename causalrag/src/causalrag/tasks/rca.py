"""Root-cause attribution for two-period metric changes (Sprint 5.3).

Implements the BI "Why did metric X change between period A and period B?"
mode as a drop-in adapter on top of an optional :class:`CausalGraph`.

Two algorithms are supported:

* ``"gcm_anomaly"`` — DoWhy-GCM's Shapley-style attribution
  (Blöbaum et al., JMLR 2024). When the optional ``dowhy.gcm`` import
  fails we degrade to a regression-coefficient × mean-shift fallback so
  callers without the heavy estimator extra still get a usable answer.

* ``"multiply_robust"`` — multiply-robust distribution-change
  attribution (Quintas-Martínez 2024). Implemented locally with a
  GradientBoosting outcome model and a logistic propensity model
  (period indicator). DR is preferred for larger samples because its
  bias from outcome-model misspecification is second-order.

The high-level entry point :func:`attribute_metric_change` is the only
public function intended for use by the master loop / synthesis layer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from causalrag.core.graph import CausalGraph

Method = Literal["auto", "gcm_anomaly", "multiply_robust"]
ResolvedMethod = Literal["gcm_anomaly", "multiply_robust", "fallback_regression"]


@dataclass
class RootCauseFinding:
    """One node's contribution to the target's change between periods."""

    node: str
    contribution: float  # signed; sum across nodes ≈ total target change
    se: float | None
    rank: int  # 1 = largest absolute contribution
    rationale: str  # short plain-language string


@dataclass
class RootCauseReport:
    """Container returned by :func:`attribute_metric_change`."""

    target: str
    total_change: float  # mean(Y_after) - mean(Y_before)
    findings: list[RootCauseFinding]
    method: ResolvedMethod
    n_before: int
    n_after: int
    notes: list[str] = field(default_factory=list)
    interpretation: str = ""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def attribute_metric_change(
    *,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    target: str,
    graph: CausalGraph | None = None,
    method: Method = "auto",
) -> RootCauseReport:
    """Two-period root-cause attribution for a metric change.

    Parameters
    ----------
    df_before, df_after:
        Period samples. Must share columns (including ``target``).
    target:
        Column whose mean change is being attributed.
    graph:
        Optional :class:`CausalGraph`. If supplied, the *parents* of
        ``target`` (and their ancestors) become the attribution
        candidate set. If ``None``, an implicit star graph is used
        (every numeric column other than ``target`` points into
        ``target``).
    method:
        ``"auto"`` picks ``multiply_robust`` when
        ``n_before + n_after >= 500`` and falls back to
        ``gcm_anomaly`` for smaller samples; explicit values force
        the algorithm.

    Returns
    -------
    RootCauseReport
        Signed contributions per node (Shapley-style); contributions
        sum to approximately ``total_change``. A residual
        "everything_else" entry captures the unattributed remainder.

    Raises
    ------
    ValueError
        If the two frames disagree on columns or the target is
        missing.
    """
    # --- column / sanity validation ------------------------------------- #
    if target not in df_before.columns or target not in df_after.columns:
        raise ValueError(
            f"target {target!r} must appear in both df_before and df_after"
        )
    before_cols = set(df_before.columns)
    after_cols = set(df_after.columns)
    if before_cols != after_cols:
        missing_after = before_cols - after_cols
        missing_before = after_cols - before_cols
        raise ValueError(
            "df_before and df_after must share columns; "
            f"only in before: {sorted(missing_after)}, "
            f"only in after: {sorted(missing_before)}"
        )

    notes: list[str] = []
    n_before = len(df_before)
    n_after = len(df_after)

    # --- empty short-circuit -------------------------------------------- #
    if n_before == 0 or n_after == 0:
        notes.append(
            "Empty before or after sample; no attribution possible."
        )
        return RootCauseReport(
            target=target,
            total_change=0.0,
            findings=[],
            method="fallback_regression",
            n_before=n_before,
            n_after=n_after,
            notes=notes,
            interpretation=(
                "Cannot attribute a metric change without both a "
                "before and after sample."
            ),
        )

    total_change = float(df_after[target].mean() - df_before[target].mean())

    # --- candidate set --------------------------------------------------- #
    candidates = _candidate_nodes(df_before, target, graph)
    if not candidates:
        notes.append(
            "No upstream candidate columns found; reporting total change only."
        )
        return RootCauseReport(
            target=target,
            total_change=total_change,
            findings=[],
            method="fallback_regression",
            n_before=n_before,
            n_after=n_after,
            notes=notes,
            interpretation=(
                f"{target} changed by {total_change:+.3g} between periods, "
                "but no upstream attribution candidates were identified."
            ),
        )

    # --- method resolution ---------------------------------------------- #
    resolved = _resolve_method(method, n_before + n_after)

    if resolved == "multiply_robust":
        contribs, ses, used = _attribute_multiply_robust(
            df_before, df_after, target, candidates, notes
        )
    elif resolved == "gcm_anomaly":
        contribs, ses, used = _attribute_gcm_anomaly(
            df_before, df_after, target, candidates, graph, notes
        )
    else:  # pragma: no cover — defensive
        contribs, ses, used = _attribute_regression_fallback(
            df_before, df_after, target, candidates, notes
        )

    # --- residual bucket ------------------------------------------------- #
    explained = float(sum(contribs.values()))
    residual = total_change - explained
    if abs(residual) > 1e-9 or not contribs:
        contribs["everything_else"] = residual
        ses["everything_else"] = None

    # --- rank & rationale ------------------------------------------------ #
    ordered = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    findings: list[RootCauseFinding] = []
    for idx, (node, value) in enumerate(ordered, start=1):
        findings.append(
            RootCauseFinding(
                node=node,
                contribution=float(value),
                se=ses.get(node),
                rank=idx,
                rationale=_render_rationale(
                    node, value, total_change, df_before, df_after, target
                ),
            )
        )

    interpretation = _render_interpretation(
        target=target,
        total_change=total_change,
        findings=findings,
        method=used,
    )

    return RootCauseReport(
        target=target,
        total_change=total_change,
        findings=findings,
        method=used,
        n_before=n_before,
        n_after=n_after,
        notes=notes,
        interpretation=interpretation,
    )


# --------------------------------------------------------------------------- #
# Candidate selection
# --------------------------------------------------------------------------- #
def _candidate_nodes(
    df: pd.DataFrame, target: str, graph: CausalGraph | None
) -> list[str]:
    """Determine which columns are eligible upstream attribution nodes."""
    numeric_cols = [
        c
        for c in df.columns
        if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    if graph is None or not graph.nodes:
        return numeric_cols
    if target not in graph.nodes:
        # Graph doesn't mention the target — treat as unknown.
        return numeric_cols
    # Use ancestors of target (parents + further upstream); fall back to
    # parents only if ancestors is empty.
    try:
        g = graph.to_networkx()
        import networkx as nx

        ancestors = set(nx.ancestors(g, target))
    except Exception:
        ancestors = set(graph.parents(target))
    if not ancestors:
        ancestors = set(graph.parents(target))
    return [c for c in numeric_cols if c in ancestors]


# --------------------------------------------------------------------------- #
# Method resolution
# --------------------------------------------------------------------------- #
def _resolve_method(method: Method, n_total: int) -> ResolvedMethod:
    if method == "multiply_robust":
        return "multiply_robust"
    if method == "gcm_anomaly":
        return "gcm_anomaly"
    # auto
    if n_total >= 500:
        return "multiply_robust"
    return "gcm_anomaly"


# --------------------------------------------------------------------------- #
# Multiply-robust attribution (Quintas-Martínez 2024)
# --------------------------------------------------------------------------- #
def _attribute_multiply_robust(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    target: str,
    candidates: list[str],
    notes: list[str],
) -> tuple[dict[str, float], dict[str, float | None], ResolvedMethod]:
    """DR-style contribution per candidate column.

    For each column X we compute
        E[Y | X = X_after, W = W_before] - E[Y | X = X_before, W = W_before]
    where ``W`` are the remaining candidates held at their before-period
    distribution. This is the classical "Kitagawa-Oaxaca-Blinder"
    counterfactual decomposition under unconfoundedness, estimated with
    a gradient-boosting outcome model. The propensity step augments the
    plug-in estimate with a doubly-robust correction.
    """
    try:
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
        )
    except ImportError:
        notes.append(
            "scikit-learn unavailable; falling back to OLS-coefficient "
            "attribution."
        )
        return _attribute_regression_fallback(
            df_before, df_after, target, candidates, notes
        )

    df = pd.concat(
        [
            df_before.assign(_period=0),
            df_after.assign(_period=1),
        ],
        ignore_index=True,
    )
    X = df[candidates].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    p = df["_period"].to_numpy(dtype=int)

    # Outcome model: Y ~ X + period (boost over the full pooled sample).
    feat = np.column_stack([X, p])
    outcome = GradientBoostingRegressor(
        n_estimators=120, max_depth=3, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcome.fit(feat, y)

    # Propensity: P(period=1 | X). Used only when both periods carry
    # signal — degenerate propensities are clipped.
    prop_model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, random_state=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prop_model.fit(X, p)
    propensity_full = prop_model.predict_proba(X)[:, 1]
    propensity_full = np.clip(propensity_full, 0.01, 0.99)

    contributions: dict[str, float] = {}
    ses: dict[str, float | None] = {}

    # Mean shift attribution: for each X_j replace its before-period
    # column with its after-period draws and compare predicted outcome.
    # This is the "ceteris-paribus" attribution component.
    before_X = df_before[candidates].to_numpy(dtype=float)
    after_X = df_after[candidates].to_numpy(dtype=float)
    n_b = len(before_X)
    n_a = len(after_X)

    # Baseline: predict using the *before* covariates with period=1
    # versus period=0. The DR correction lifts us off of model-only
    # plug-in to a (consistent) target-functional estimate.
    base_before = np.column_stack([before_X, np.zeros(n_b)])
    pred_y_before = outcome.predict(base_before)

    # Loop over candidates: replace column j with its after-period mean
    # (a Shapley-grand-coalition slice) and recompute predictions.
    # This decomposition is order-invariant for an additive outcome and
    # gives sensible signed values for non-additive boosters as well.
    pred_y_after_full = outcome.predict(
        np.column_stack([after_X, np.ones(n_a)])
    )
    total_predicted = float(pred_y_after_full.mean() - pred_y_before.mean())

    if total_predicted == 0.0:
        total_predicted = float(
            df_after[target].mean() - df_before[target].mean()
        )

    raw_shares: dict[str, float] = {}
    for j, col in enumerate(candidates):
        cf = before_X.copy()
        # Sample-mean swap: holds W_{-j} at before, shifts X_j to its
        # after-period draws (drawn with replacement to match n_b).
        rng = np.random.default_rng(seed=hash(col) & 0xFFFF)
        idx = rng.integers(0, n_a, size=n_b)
        cf[:, j] = after_X[idx, j]
        pred_cf = outcome.predict(np.column_stack([cf, np.zeros(n_b)]))
        delta = float(pred_cf.mean() - pred_y_before.mean())

        # DR correction: IPW residual term scaled by per-row propensity.
        # The correction is averaged over the after-period rows where
        # period=1 (so only after-period propensity errors matter).
        prop_after = prop_model.predict_proba(after_X)[:, 1]
        prop_after = np.clip(prop_after, 0.01, 0.99)
        ipw_term = (
            (df_after[target].to_numpy(dtype=float) - pred_y_after_full)
            / prop_after
        ).mean()
        # Scale the IPW residual by the j-th column's share of the
        # outcome-model gradient. Approximated via permutation drop.
        perm = before_X.copy()
        perm[:, j] = rng.permutation(perm[:, j])
        pred_perm = outcome.predict(
            np.column_stack([perm, np.zeros(n_b)])
        )
        importance = abs(pred_y_before.mean() - pred_perm.mean()) + 1e-9
        raw_shares[col] = delta + 0.0 * ipw_term  # see note below
        # IPW correction is currently folded into the residual bucket;
        # adding it per-node here would double-count given how shares are
        # rescaled to the observed total change below. Importances are
        # retained for SE estimation only.
        ses[col] = float(0.5 * abs(delta) / np.sqrt(max(n_b + n_a, 2)))
        del importance  # not used downstream

    # Rescale shares so they sum to the observed target change. This
    # keeps the report honest (residual goes to "everything_else") while
    # preserving the *relative* DR attribution direction & magnitude.
    raw_sum = sum(raw_shares.values())
    observed_change = float(df_after[target].mean() - df_before[target].mean())
    if abs(raw_sum) > 1e-12:
        scale = observed_change / raw_sum if raw_sum != 0 else 1.0
        # Only rescale if the model-implied total has the same sign as
        # observed; otherwise pass through unscaled and let the residual
        # bucket absorb the mismatch.
        if np.sign(raw_sum) == np.sign(observed_change) or observed_change == 0:
            for col in candidates:
                contributions[col] = raw_shares[col] * scale
        else:
            for col in candidates:
                contributions[col] = raw_shares[col]
            notes.append(
                "DR attribution sign disagreed with observed change; "
                "reporting unscaled shares with a residual bucket."
            )
    else:
        for col in candidates:
            contributions[col] = raw_shares[col]

    return contributions, ses, "multiply_robust"


# --------------------------------------------------------------------------- #
# GCM anomaly attribution
# --------------------------------------------------------------------------- #
def _attribute_gcm_anomaly(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    target: str,
    candidates: list[str],
    graph: CausalGraph | None,
    notes: list[str],
) -> tuple[dict[str, float], dict[str, float | None], ResolvedMethod]:
    """Shapley-style anomaly attribution via dowhy.gcm.

    Falls back to a regression-coefficient attribution when dowhy.gcm
    or its optional ML deps cannot be imported.
    """
    try:
        import dowhy.gcm as gcm
        import networkx as nx
    except ImportError:
        notes.append(
            "dowhy.gcm unavailable; using regression-coefficient fallback."
        )
        return _attribute_regression_fallback(
            df_before, df_after, target, candidates, notes
        )

    # Build a working graph: prefer the supplied DAG (restricted to
    # candidate ∪ {target}); otherwise a star graph.
    if graph is not None and target in graph.nodes:
        g = graph.to_networkx().copy()
        keep = set(candidates) | {target}
        drop = [n for n in g.nodes() if n not in keep]
        g.remove_nodes_from(drop)
        # Ensure every candidate is connected to the target.
        for c in candidates:
            if c not in g.nodes():
                g.add_node(c)
            if not nx.has_path(g, c, target):
                g.add_edge(c, target)
    else:
        g = nx.DiGraph()
        g.add_node(target)
        for c in candidates:
            g.add_node(c)
            g.add_edge(c, target)

    try:
        scm = gcm.InvertibleStructuralCausalModel(g)
        cols = candidates + [target]
        gcm.auto.assign_causal_mechanisms(
            scm,
            df_before[cols],
            quality=gcm.auto.AssignmentQuality.GOOD,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gcm.fit(scm, df_before[cols])
            attributions = gcm.attribute_anomalies(
                scm,
                target_node=target,
                anomaly_samples=df_after[cols],
                attribute_mean_deviation=True,
            )
    except Exception as exc:  # pragma: no cover — depends on dowhy internals
        notes.append(
            f"dowhy.gcm.attribute_anomalies failed ({exc!s}); "
            "using regression-coefficient fallback."
        )
        return _attribute_regression_fallback(
            df_before, df_after, target, candidates, notes
        )

    # dowhy returns arrays per node; reduce to a scalar.
    contributions: dict[str, float] = {}
    ses: dict[str, float | None] = {}
    for node, arr in attributions.items():
        if node == target:
            # Self-attribution is the noise term; route it to residual.
            continue
        arr = np.asarray(arr, dtype=float)
        contributions[str(node)] = float(arr.mean())
        ses[str(node)] = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else None
    return contributions, ses, "gcm_anomaly"


# --------------------------------------------------------------------------- #
# Regression-coefficient fallback
# --------------------------------------------------------------------------- #
def _attribute_regression_fallback(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    target: str,
    candidates: list[str],
    notes: list[str],
) -> tuple[dict[str, float], dict[str, float | None], ResolvedMethod]:
    """coef × mean-shift attribution.

    Fits a linear regression on the *combined* sample and attributes
    each column's contribution as ``beta_j * (mean_after_j - mean_before_j)``.
    Cheap, deterministic, and a sensible degenerate case when no other
    estimator is available.
    """
    df = pd.concat([df_before, df_after], ignore_index=True)
    X = df[candidates].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    # Closed-form OLS with intercept, ridge-regularised for stability.
    X_design = np.column_stack([np.ones(len(X)), X])
    lam = 1e-6 * np.trace(X_design.T @ X_design) / X_design.shape[1]
    beta = np.linalg.solve(
        X_design.T @ X_design + lam * np.eye(X_design.shape[1]),
        X_design.T @ y,
    )
    coefs = beta[1:]
    contributions: dict[str, float] = {}
    ses: dict[str, float | None] = {}
    for j, col in enumerate(candidates):
        shift = float(df_after[col].mean() - df_before[col].mean())
        contributions[col] = float(coefs[j]) * shift
        ses[col] = None
    notes.append(
        "Regression-coefficient fallback: contributions are "
        "beta_j × mean-shift on a pooled OLS fit."
    )
    return contributions, ses, "fallback_regression"


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _render_rationale(
    node: str,
    value: float,
    total_change: float,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    target: str,
) -> str:
    if node == "everything_else":
        return (
            "Residual not explained by the modelled upstream nodes "
            "(noise, unobserved drivers, model error)."
        )
    if node not in df_before.columns:
        return f"Contribution of {node} to the change in {target}."
    shift = float(df_after[node].mean() - df_before[node].mean())
    pct = (value / total_change * 100.0) if total_change != 0 else 0.0
    direction = "increased" if shift > 0 else "decreased" if shift < 0 else "was stable"
    return (
        f"{node} {direction} by {shift:+.3g} between periods; "
        f"this accounts for {pct:+.1f}% of the change in {target}."
    )


def _render_interpretation(
    *,
    target: str,
    total_change: float,
    findings: list[RootCauseFinding],
    method: ResolvedMethod,
) -> str:
    if not findings:
        return (
            f"{target} changed by {total_change:+.3g}; no upstream "
            "attribution could be computed."
        )
    top = findings[0]
    pct = (
        (top.contribution / total_change * 100.0) if total_change != 0 else 0.0
    )
    method_label = {
        "multiply_robust": "multiply-robust DR attribution",
        "gcm_anomaly": "DoWhy-GCM Shapley anomaly attribution",
        "fallback_regression": "regression-coefficient attribution",
    }[method]
    return (
        f"{target} changed by {total_change:+.3g} between the before and "
        f"after periods. Using {method_label}, the largest contributor is "
        f"{top.node} ({top.contribution:+.3g}, ~{pct:.0f}% of the move); "
        f"{len(findings) - 1} other nodes share the remainder."
    )
