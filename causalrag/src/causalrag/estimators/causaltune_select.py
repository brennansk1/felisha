"""Ground-truth-free CV metrics for causal estimator selection.

CausalTune-style selection without ATE supervision: pick among fitted
causal estimators using the *energy score* (Székely-Rizzo 2013) and
*ERUPT* (Estimated Response Under Proposed Treatments). Self-contained
re-implementations — no CausalTune dependency.

Both metrics consume any fitted estimator that exposes one of
``predict_cate(X)`` / ``effect(X)`` / ``predict(X)``. ``select_best_estimator``
then picks the Pareto frontier (lowest energy score AND ERUPT in the
top quartile) and emits a leaderboard.

Definitions (PDD-internal, see Sprint 2.7 spec):

* Two-sample energy distance::

      E(X, Y) = 2 * mean|X - Y| - mean|X - X'| - mean|Y - Y'|

  We aggregate the per-arm energy distance between the estimator's
  predicted counterfactual outcome distribution and the empirical
  observed outcome distribution. Lower = better.

* ERUPT (binary treatment, IPW form)::

      ERUPT(pi) = E[ Y * 1{A == pi(X)} / e(X, A) ]

  where ``pi(X) = 1{CATE(X) > threshold}`` and ``e(X, A)`` is the
  estimated propensity of the observed assignment. Higher = better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CATE extraction helper
# ---------------------------------------------------------------------------


def _predict_cate(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Best-effort CATE extraction from any estimator-like object.

    Tries ``predict_cate``, ``effect``, ``predict`` in that order. Raises
    ``AttributeError`` if none are present.
    """
    if hasattr(estimator, "predict_cate"):
        out = estimator.predict_cate(X)
    elif hasattr(estimator, "effect"):
        out = estimator.effect(X)
    elif hasattr(estimator, "predict"):
        out = estimator.predict(X)
    else:
        raise AttributeError(
            f"Estimator {type(estimator).__name__} has none of "
            "predict_cate / effect / predict."
        )
    arr = np.asarray(out, dtype=np.float64).reshape(-1)
    return arr


def _predict_outcome(
    estimator: Any, X: np.ndarray, t: np.ndarray
) -> np.ndarray | None:
    """Return E[Y | X, T=t] if the estimator exposes such a hook.

    Looks for ``predict_outcome(X, t)`` or ``mu(X, t)``. Returns ``None``
    if neither is available — callers must then fall back to CATE-based
    counterfactual reconstruction.
    """
    if hasattr(estimator, "predict_outcome"):
        return np.asarray(
            estimator.predict_outcome(X, t), dtype=np.float64
        ).reshape(-1)
    if hasattr(estimator, "mu"):
        return np.asarray(estimator.mu(X, t), dtype=np.float64).reshape(-1)
    return None


# ---------------------------------------------------------------------------
# Energy distance / score
# ---------------------------------------------------------------------------


def energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sample energy distance (Székely 2013).

    E(X, Y) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|. Zero iff distributions match
    (in 1-D, identical samples up to permutation). Lower = closer.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("inf")
    xx = float(np.abs(x[:, None] - x[None, :]).mean())
    yy = float(np.abs(y[:, None] - y[None, :]).mean())
    xy = float(np.abs(x[:, None] - y[None, :]).mean())
    return 2.0 * xy - xx - yy


def _fit_propensity(
    X: np.ndarray, t: np.ndarray, clip: tuple[float, float] = (0.05, 0.95)
) -> np.ndarray:
    """Logistic propensity e(X) = P(T=1|X), clipped to ``clip``.

    Falls back to the empirical marginal P(T=1) if sklearn isn't available
    or the logistic fit fails (e.g. zero treatment variance).
    """
    p_marginal = float(np.mean(t)) if len(t) else 0.5
    if len(np.unique(t)) < 2:
        return np.full(len(t), np.clip(p_marginal, *clip), dtype=np.float64)
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000)
        lr.fit(X, t)
        e = lr.predict_proba(X)[:, 1]
    except Exception:
        e = np.full(len(t), p_marginal, dtype=np.float64)
    return np.clip(e, clip[0], clip[1])


def _common_support_mask(
    e: np.ndarray, t: np.ndarray, trim: float = 0.05
) -> np.ndarray:
    """Trim units with propensity outside ``[trim, 1-trim]``."""
    return (e >= trim) & (e <= 1.0 - trim)


def energy_score(
    estimator: Any,
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...] | Sequence[str],
) -> float:
    """Energy-score selection metric — sum of per-arm energy distances.

    For each treatment arm a in {0, 1}:
      1. Build the estimator's predicted counterfactual outcome for the
         opposite arm using CATE: hat_Y(a' | X) = Y_obs - sign * CATE(X)
         where sign = +1 if a=1 (subtract effect to get untreated potential)
         and sign = -1 if a=0 (add effect to get treated potential).
      2. Compare hat_Y_counterfactual against the empirical outcome
         distribution of the *opposite* arm (on common support).

    Returns the sum of the two arm-wise energy distances. Lower = better.
    Returns ``inf`` on empty / degenerate inputs (no common support, no
    treatment variance, etc.).
    """
    confounders = tuple(confounders)
    if len(df) == 0 or treatment not in df.columns or outcome not in df.columns:
        return float("inf")
    cols = [outcome, treatment, *confounders]
    sub = df[cols].dropna()
    if len(sub) == 0:
        return float("inf")
    t = sub[treatment].to_numpy().astype(np.float64)
    y = sub[outcome].to_numpy().astype(np.float64)
    if len(np.unique(t)) < 2:
        return float("inf")
    X = (
        sub[list(confounders)].to_numpy().astype(np.float64)
        if confounders
        else np.zeros((len(sub), 1), dtype=np.float64)
    )

    # Propensity-trim to common support.
    e = _fit_propensity(X, t)
    mask = _common_support_mask(e, t)
    if mask.sum() == 0:
        return float("inf")
    X_cs = X[mask]
    y_cs = y[mask]
    t_cs = t[mask]
    try:
        cate = _predict_cate(estimator, X_cs)
    except Exception:
        return float("inf")
    if cate.shape[0] != X_cs.shape[0]:
        return float("inf")

    total = 0.0
    for arm in (0.0, 1.0):
        in_arm = t_cs == arm
        in_opp = t_cs == (1.0 - arm)
        if in_arm.sum() == 0 or in_opp.sum() == 0:
            return float("inf")
        # Counterfactual: shift observed-arm outcomes by CATE to estimate
        # the *opposite-arm* potential outcome. If arm=1, hat_Y(0) = Y - CATE.
        # If arm=0, hat_Y(1) = Y + CATE.
        sign = 1.0 if arm == 1.0 else -1.0
        y_cf = y_cs[in_arm] - sign * cate[in_arm]
        # Compare to the empirical opposite-arm outcomes.
        total += energy_distance(y_cf, y_cs[in_opp])
    return float(total)


# ---------------------------------------------------------------------------
# ERUPT
# ---------------------------------------------------------------------------


def erupt(
    estimator: Any,
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...] | Sequence[str],
    threshold: float = 0.0,
) -> float:
    """ERUPT — policy value under ``pi(X) = 1{CATE(X) > threshold}``.

    Inverse-propensity weighted::

        ERUPT = mean_i [ Y_i * 1{A_i == pi(X_i)} / e_hat(X_i, A_i) ]

    where ``e_hat(X, A) = P(T=A|X)``. Higher = better. Returns ``-inf``
    on empty / degenerate input.
    """
    confounders = tuple(confounders)
    if len(df) == 0 or treatment not in df.columns or outcome not in df.columns:
        return float("-inf")
    cols = [outcome, treatment, *confounders]
    sub = df[cols].dropna()
    if len(sub) == 0:
        return float("-inf")
    t = sub[treatment].to_numpy().astype(np.float64)
    y = sub[outcome].to_numpy().astype(np.float64)
    if len(np.unique(t)) < 2:
        return float("-inf")
    X = (
        sub[list(confounders)].to_numpy().astype(np.float64)
        if confounders
        else np.zeros((len(sub), 1), dtype=np.float64)
    )
    try:
        cate = _predict_cate(estimator, X)
    except Exception:
        return float("-inf")
    if cate.shape[0] != X.shape[0]:
        return float("-inf")

    pi = (cate > threshold).astype(np.float64)
    e1 = _fit_propensity(X, t)  # P(T=1|X)
    # P(T=A_i|X_i) — flip for control units.
    e_obs = np.where(t == 1.0, e1, 1.0 - e1)
    match = (t == pi).astype(np.float64)
    contributions = y * match / e_obs
    return float(np.mean(contributions))


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    name: str
    energy_score: float
    erupt: float


def _coerce_candidates(
    candidates: Iterable[Any] | Mapping[str, Any],
) -> list[tuple[str, Any]]:
    if isinstance(candidates, Mapping):
        return list(candidates.items())
    out: list[tuple[str, Any]] = []
    for i, est in enumerate(candidates):
        name = getattr(est, "id", None) or getattr(
            est, "name", None
        ) or f"est_{i}"
        out.append((str(name), est))
    return out


def select_best_estimator(
    candidates: Iterable[Any] | Mapping[str, Any],
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: tuple[str, ...] | Sequence[str],
    threshold: float = 0.0,
) -> dict[str, Any]:
    """Score every candidate on (energy_score, erupt) and pick a Pareto winner.

    Selection rule: take the candidate with the *lowest* energy score
    among those whose ERUPT lies in the top quartile of the candidate
    set. If that filter empties (e.g. fewer than 4 candidates), fall
    back to the candidate with the lowest energy score overall.

    Returns ``{"best": <estimator>, "best_name": str, "leaderboard":
    list[LeaderboardEntry]}``.
    """
    pairs = _coerce_candidates(candidates)
    if not pairs:
        return {"best": None, "best_name": None, "leaderboard": []}

    confounders = tuple(confounders)
    leaderboard: list[LeaderboardEntry] = []
    for name, est in pairs:
        es = energy_score(est, df, treatment, outcome, confounders)
        er = erupt(est, df, treatment, outcome, confounders, threshold=threshold)
        leaderboard.append(LeaderboardEntry(name=name, energy_score=es, erupt=er))

    erupts = np.array([e.erupt for e in leaderboard], dtype=np.float64)
    # ERUPT top-quartile threshold (75th percentile of finite values).
    finite_erupts = erupts[np.isfinite(erupts)]
    if len(finite_erupts) >= 4:
        q75 = float(np.quantile(finite_erupts, 0.75))
        eligible = [e for e in leaderboard if e.erupt >= q75]
    else:
        eligible = list(leaderboard)
    if not eligible:
        eligible = list(leaderboard)

    # Pick lowest energy score among the eligible set.
    best_entry = min(eligible, key=lambda e: e.energy_score)
    best_est = dict(pairs)[best_entry.name]

    return {
        "best": best_est,
        "best_name": best_entry.name,
        "leaderboard": leaderboard,
    }


__all__ = [
    "energy_distance",
    "energy_score",
    "erupt",
    "select_best_estimator",
    "LeaderboardEntry",
]
