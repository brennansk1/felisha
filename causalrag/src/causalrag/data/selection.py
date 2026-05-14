"""Principled variable selection for the adjustment set.

Methods exposed (no single best — situation-dependent):

- ``post_double_selection`` (Belloni, Chernozhukov & Hansen 2014, REStud) —
  Lasso(Y ~ W) ∪ Lasso(T ~ W). The canonical method for high-dim adjustment;
  gives valid inference under DML's sparsity assumption. **Default** when
  ``HIGH_DIMENSIONAL`` flag is set or |W| > 20.
- ``lasso_intersection`` — more conservative variant: intersection rather
  than union. Use when the analyst suspects many spurious variables.
- ``correlation_pruning`` — drop variables with |r| > threshold to another
  retained variable; among each near-collinear pair keep the one with less
  missingness. Cheap and theory-neutral.
- ``none`` — pass through. Use when the analyst has hand-curated the
  adjustment set or when n >> p.
- ``auto`` — picks ``post_double_selection`` for HIGH_DIMENSIONAL or |W|>20,
  ``correlation_pruning`` for 5 ≤ |W| ≤ 20, ``none`` for |W| < 5.

All methods return both the selected variable list AND a structured
:class:`SelectionResult` that records *why* each variable was kept or dropped,
suitable for the analyst-decision ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Method = Literal[
    "auto",
    "post_double_selection",
    "lasso_intersection",
    "correlation_pruning",
    "none",
]


@dataclass
class SelectionResult:
    method: Method
    selected: tuple[str, ...]
    dropped: tuple[str, ...] = ()
    reasons: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "selected": list(self.selected),
            "dropped": list(self.dropped),
            "reasons": dict(self.reasons),
            "notes": list(self.notes),
        }


def resolve_method(method: Method, *, n_candidates: int, high_dimensional: bool) -> Method:
    if method != "auto":
        return method
    if high_dimensional or n_candidates > 20:
        return "post_double_selection"
    if n_candidates >= 5:
        return "correlation_pruning"
    return "none"


def select_variables(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    candidates: tuple[str, ...],
    *,
    method: Method = "auto",
    high_dimensional: bool = False,
    pinned: tuple[str, ...] = (),
    random_state: int = 42,
    correlation_threshold: float = 0.9,
) -> SelectionResult:
    """Select the adjustment set from ``candidates``.

    Pinned variables are always retained regardless of method (caller
    promises they are theoretically required).
    """
    resolved = resolve_method(method, n_candidates=len(candidates), high_dimensional=high_dimensional)

    if resolved == "none" or not candidates:
        return SelectionResult(method=resolved, selected=candidates)

    if resolved == "correlation_pruning":
        return _correlation_pruning(
            df, candidates, threshold=correlation_threshold, pinned=pinned
        )

    if resolved in ("post_double_selection", "lasso_intersection"):
        return _post_double_selection(
            df,
            treatment=treatment,
            outcome=outcome,
            candidates=candidates,
            intersect=(resolved == "lasso_intersection"),
            pinned=pinned,
            random_state=random_state,
        )

    return SelectionResult(method=resolved, selected=candidates)


# --- correlation pruning ----------------------------------------------------


def _correlation_pruning(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    *,
    threshold: float,
    pinned: tuple[str, ...],
) -> SelectionResult:
    cols = [c for c in candidates if c in df.columns]
    work = df[cols].copy()
    numeric = work.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return SelectionResult(method="correlation_pruning", selected=tuple(cols))

    corr = numeric.corr().abs()
    missing = work.isna().mean()

    kept: set[str] = set(pinned)
    dropped: dict[str, str] = {}
    order = sorted(
        numeric.columns,
        key=lambda c: (c not in pinned, missing.get(c, 0.0)),
    )
    for c in order:
        if c in dropped:
            continue
        if c in kept:
            keep_decision = True
        else:
            keep_decision = True
        if keep_decision:
            kept.add(c)
            for other in numeric.columns:
                if other in kept or other in dropped:
                    continue
                if other in pinned:
                    continue
                if corr.loc[c, other] >= threshold:
                    dropped[other] = f"|r|={corr.loc[c, other]:.3f} with kept variable {c}"
    # Non-numeric candidates flow through unchanged
    non_numeric = [c for c in cols if c not in numeric.columns]
    selected = [c for c in cols if c in kept or c in non_numeric]
    return SelectionResult(
        method="correlation_pruning",
        selected=tuple(selected),
        dropped=tuple(dropped),
        reasons=dropped,
        notes=[f"threshold |r| ≥ {threshold}"],
    )


# --- post-double-selection ---------------------------------------------------


def _post_double_selection(
    df: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    candidates: tuple[str, ...],
    intersect: bool,
    pinned: tuple[str, ...],
    random_state: int,
) -> SelectionResult:
    """Belloni-Chernozhukov-Hansen 2014.

    Run Lasso(Y ~ W) and Lasso(T ~ W) on the candidate W, take union (default)
    or intersection of the non-zero coefficients, plus any pinned variables.
    """
    cols = [c for c in candidates if c in df.columns]
    work = df[[treatment, outcome, *cols]].dropna()
    if work.empty or not cols:
        return SelectionResult(method="post_double_selection", selected=tuple(cols))

    numeric_cols = [
        c for c in cols if pd.api.types.is_numeric_dtype(work[c]) and work[c].nunique() > 1
    ]
    non_numeric = [c for c in cols if c not in numeric_cols]
    if not numeric_cols:
        return SelectionResult(
            method="post_double_selection",
            selected=tuple(cols),
            notes=["No numeric candidates; selection skipped."],
        )

    w = work[numeric_cols].to_numpy().astype(float)
    y = work[outcome].to_numpy().astype(float)
    t = work[treatment].to_numpy().astype(float)

    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    w_std = scaler.fit_transform(w)

    # Y ~ W (Lasso)
    try:
        lasso_y = LassoCV(cv=5, random_state=random_state, max_iter=10_000)
        lasso_y.fit(w_std, y)
        y_selected = {numeric_cols[i] for i, coef in enumerate(lasso_y.coef_) if coef != 0}
    except Exception:
        y_selected = set(numeric_cols)

    # T ~ W (Lasso for continuous, Logistic-Lasso for binary). Magnitude
    # threshold for "selected" is 0.05 on standardized features — coefficients
    # smaller than this are noise even when CV doesn't penalize them to exact
    # zero. This matches Belloni-Chernozhukov-Hansen's plug-in penalty spirit
    # without requiring the user to tune a theoretical lambda.
    try:
        if set(np.unique(t).tolist()).issubset({0.0, 1.0}):
            lasso_t = LogisticRegressionCV(
                cv=5,
                random_state=random_state,
                max_iter=5000,
                penalty="l1",
                solver="liblinear",
                Cs=20,
            )
            lasso_t.fit(w_std, t.astype(int))
            coefs = np.abs(lasso_t.coef_).ravel()
            threshold = 0.05
        else:
            lasso_t = LassoCV(cv=5, random_state=random_state, max_iter=10_000)
            lasso_t.fit(w_std, t)
            coefs = np.abs(lasso_t.coef_).ravel()
            threshold = 1e-8
        t_selected = {numeric_cols[i] for i, c in enumerate(coefs) if c > threshold}
    except Exception:
        t_selected = set(numeric_cols)

    if intersect:
        chosen = y_selected & t_selected
    else:
        chosen = y_selected | t_selected
    chosen.update(pinned)

    selected = [c for c in cols if c in chosen or c in non_numeric]
    dropped = {
        c: f"zero-coef in both Y-Lasso ({c not in y_selected}) and T-Lasso ({c not in t_selected})"
        for c in numeric_cols
        if c not in chosen and c not in pinned
    }
    return SelectionResult(
        method="lasso_intersection" if intersect else "post_double_selection",
        selected=tuple(selected),
        dropped=tuple(dropped),
        reasons=dropped,
        notes=[
            f"Y-Lasso selected {len(y_selected)} / {len(numeric_cols)}",
            f"T-Lasso selected {len(t_selected)} / {len(numeric_cols)}",
            f"|pinned| = {len(pinned)}",
        ],
    )


__all__ = ["SelectionResult", "Method", "select_variables", "resolve_method"]
