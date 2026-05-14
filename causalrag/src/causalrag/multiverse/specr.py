"""Specification curve / multiverse analysis (Sprint 6.2).

References
----------
Simonsohn, U., Simmons, J. P., & Nelson, L. D. (2020). Specification curve
analysis. *Nature Human Behaviour*, 4, 1208-1214.

Del Giudice, M., & Gangestad, S. W. (2021). A traveler's guide to the
multiverse: Promises, pitfalls, and a framework for the evaluation of
analytic decisions. *Advances in Methods and Practices in Psychological
Science*, 4(1).

Design notes
------------
A *specification* is the full Cartesian product of analyst-controllable
choices for a single causal hypothesis:

    spec = (adjustment_set, estimator, trimming_threshold, time_window, extra)

For each spec we obtain a point estimate, SE and CI by running the
underlying estimator on the filtered/trimmed data. We then sort points and
report the curve plus three summary statistics:

- ``significance_share`` — fraction of specs whose CI excludes 0
- ``sign_consistency_share`` — fraction sharing the sign of the median
- ``joint_test_p`` — Simonsohn's permutation joint inference, ONLY when
  the analyst certifies all specs are *principled equivalents* of the
  hypothesis (Del Giudice & Gangestad 2021).

If the analyst does NOT certify principled equivalence we refuse the joint
test and fall back to a Bonferroni-corrected minimum p-value across specs;
the interpretation string spells that out so reviewers see why.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.estimators.python.select import select_estimator

# Importing the estimator modules registers their factories in the global
# registry; we need at least OLS + the DML family to satisfy the typical
# ``estimators=[...]`` argument.
from causalrag.estimators.python import ols as _ols  # noqa: F401
from causalrag.estimators.python import dml as _dml  # noqa: F401


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class SpecResult:
    """One specification's outcome."""

    spec_id: str
    spec: dict[str, Any]
    point: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    converged: bool
    estimator_id: str = ""
    n_used: int = 0
    error: str | None = None

    def significant(self) -> bool:
        """CI excludes zero — only meaningful when ``converged`` is True."""
        if not self.converged or self.ci_low is None or self.ci_high is None:
            return False
        return self.ci_low > 0.0 or self.ci_high < 0.0


@dataclass
class SpecCurve:
    """Sorted specification curve plus joint-inference summary."""

    results: list[SpecResult]
    point_curve: np.ndarray
    significance_share: float
    sign_consistency_share: float
    joint_test_p: float | None
    interpretation: str
    principled_equivalence: bool = False
    median_point: float = 0.0
    bonferroni_min_p: float | None = None
    converged_count: int = 0
    n_specs: int = 0

    # Persisted so analysts can audit which spec produced which point.
    sorted_spec_ids: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def specification_curve(
    *,
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adjustment_sets: Sequence[tuple[str, ...]],
    estimators: Sequence[str],
    trimming_thresholds: Sequence[float] = (0.0, 0.01, 0.05),
    time_windows: Sequence[tuple[Any, Any] | None] = (None,),
    extra_specs: Sequence[dict[str, Any]] | None = None,
    principled_equivalence: bool = False,
    time_column: str | None = None,
    alpha: float = 0.05,
    joint_permutations: int = 200,
    random_state: int | None = 0,
) -> SpecCurve:
    """Run the K-cardinal product of specifications and return a SpecCurve.

    Parameters
    ----------
    df, treatment, outcome
        Data and the (T, Y) pair under investigation.
    adjustment_sets
        Iterable of adjustment-set candidates. Each element is a tuple of
        column names to control for.
    estimators
        Registered estimator ids (e.g. ``"python.linear.ols"``,
        ``"python.dml.linear"``). Resolved through
        :func:`causalrag.estimators.python.select.select_estimator` with the
        id as ``prefer=`` so the registry's flag/sample-size guards still
        apply.
    trimming_thresholds
        Propensity-score trim levels in ``[0, 0.5)``. ``0.0`` keeps everyone;
        ``0.05`` drops rows with estimated propensity outside ``[0.05, 0.95]``
        (Crump et al. 2009). For non-binary treatments we fall back to
        treatment-quantile trimming (drop tails at the given level).
    time_windows
        Sequence of ``(start, end)`` half-open intervals on ``time_column``.
        Pass ``None`` to keep all rows. If you pass a non-None window but no
        ``time_column``, that spec converges with ``error="no time_column"``.
    extra_specs
        Iterable of free-form parameter overlays (each a ``dict``) appended
        to the Cartesian product. Treated as ``estimator_kwargs`` forwarded
        to the factory and recorded verbatim on the SpecResult.
    principled_equivalence
        Set to ``True`` ONLY when the analyst can defend that every spec in
        the product is a principled equivalent of the hypothesis (Del
        Giudice & Gangestad 2021). When True we report Simonsohn's
        permutation joint test; otherwise we Bonferroni-correct the minimum
        p-value across specs and surface the caveat in ``interpretation``.
    """
    if not adjustment_sets:
        raise ValueError("adjustment_sets must be non-empty")
    if not estimators:
        raise ValueError("estimators must be non-empty")

    extras: Sequence[dict[str, Any]] = tuple(extra_specs or ())
    windows: Sequence[tuple[Any, Any] | None] = tuple(time_windows) or (None,)

    rng = np.random.default_rng(random_state)
    protocol = StudyProtocol(name="multiverse__specr")
    results: list[SpecResult] = []
    spec_counter = 0

    for adj in adjustment_sets:
        for est_id in estimators:
            for trim in trimming_thresholds:
                for window in windows:
                    overlays: Sequence[dict[str, Any]] = extras or ({},)
                    for extra in overlays:
                        spec_counter += 1
                        spec = {
                            "adjustment_set": tuple(adj),
                            "estimator": est_id,
                            "trim": float(trim),
                            "time_window": window,
                            "extra": dict(extra),
                        }
                        spec_id = f"s{spec_counter:04d}"
                        result = _run_one_spec(
                            spec_id=spec_id,
                            spec=spec,
                            df=df,
                            treatment=treatment,
                            outcome=outcome,
                            adj=tuple(adj),
                            est_id=est_id,
                            trim=float(trim),
                            window=window,
                            time_column=time_column,
                            extra=dict(extra),
                            protocol=protocol,
                        )
                        results.append(result)

    return _summarize(
        results,
        principled_equivalence=principled_equivalence,
        alpha=alpha,
        joint_permutations=joint_permutations,
        rng=rng,
    )


# --------------------------------------------------------------------------- #
# Per-spec execution
# --------------------------------------------------------------------------- #


def _run_one_spec(
    *,
    spec_id: str,
    spec: dict[str, Any],
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    adj: tuple[str, ...],
    est_id: str,
    trim: float,
    window: tuple[Any, Any] | None,
    time_column: str | None,
    extra: dict[str, Any],
    protocol: StudyProtocol,
) -> SpecResult:
    try:
        sub = _apply_time_window(df, window, time_column)
        sub = _apply_trim(sub, treatment, adj, trim)

        if len(sub) < 5:
            return SpecResult(
                spec_id=spec_id,
                spec=spec,
                point=float("nan"),
                se=None,
                ci_low=None,
                ci_high=None,
                converged=False,
                estimator_id=est_id,
                n_used=int(len(sub)),
                error="too few rows after trim/window",
            )

        # Resolve through the registered selector so flag-based gates still
        # apply; the id we pass in ``prefer`` wins as long as it's compatible.
        # A typo'd estimator id silently falls through to the default in
        # select_estimator, which would hide the analyst's mistake — so we
        # verify the resolution and reject mismatches.
        entry = select_estimator(
            estimand="ATE",
            flags=frozenset(),
            n=len(sub),
            prefer=est_id,
        )
        if entry.id != est_id:
            raise LookupError(
                f"Requested estimator id {est_id!r} did not resolve "
                f"({entry.id!r} returned instead). Check the spelling or "
                f"register the estimator before invoking specification_curve."
            )
        estimator = entry.factory(
            treatment=treatment,
            outcome=outcome,
            confounders=adj,
            modifiers=(),
            **extra,
        )
        estimator.fit(sub, protocol)
        out = estimator.estimate()
        return SpecResult(
            spec_id=spec_id,
            spec=spec,
            point=float(out.point_estimate),
            se=(float(out.se) if out.se is not None else None),
            ci_low=(float(out.ci_low) if out.ci_low is not None else None),
            ci_high=(float(out.ci_high) if out.ci_high is not None else None),
            converged=True,
            estimator_id=out.estimator_id,
            n_used=int(out.n_used),
        )
    except Exception as exc:  # pragma: no cover - exercised via tests
        return SpecResult(
            spec_id=spec_id,
            spec=spec,
            point=float("nan"),
            se=None,
            ci_low=None,
            ci_high=None,
            converged=False,
            estimator_id=est_id,
            n_used=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _apply_time_window(
    df: pd.DataFrame,
    window: tuple[Any, Any] | None,
    time_column: str | None,
) -> pd.DataFrame:
    if window is None:
        return df
    if time_column is None or time_column not in df.columns:
        raise ValueError(
            "time_window requested but no usable time_column in DataFrame"
        )
    start, end = window
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df[time_column] >= start
    if end is not None:
        mask &= df[time_column] < end
    return df.loc[mask]


def _apply_trim(
    df: pd.DataFrame,
    treatment: str,
    adj: tuple[str, ...],
    trim: float,
) -> pd.DataFrame:
    """Crump-style propensity-score trimming for binary T; quantile trim else.

    For binary treatment we fit a logistic regression of T on the adjustment
    set and drop rows with estimated p̂ outside ``[trim, 1 - trim]``. For
    continuous / categorical treatment we drop the tails of T at the
    quantile level ``trim``. If ``trim == 0`` this is a no-op.
    """
    if trim <= 0.0:
        return df
    if trim >= 0.5:
        raise ValueError("trimming_threshold must be in [0, 0.5)")

    t = df[treatment]
    is_binary = t.dropna().nunique() <= 2

    if is_binary and adj:
        try:
            from sklearn.linear_model import LogisticRegression

            X = df[list(adj)].astype(float).to_numpy()
            y = df[treatment].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                return df
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            ps = model.predict_proba(X)[:, 1]
            keep = (ps >= trim) & (ps <= 1.0 - trim)
            return df.loc[df.index[keep]]
        except Exception:
            # Fall through to quantile trim on T as a defensible fallback.
            pass

    # Quantile trim on the treatment column.
    lo, hi = np.quantile(t.astype(float).dropna(), [trim, 1.0 - trim])
    return df.loc[(t >= lo) & (t <= hi)]


# --------------------------------------------------------------------------- #
# Aggregation + joint inference
# --------------------------------------------------------------------------- #


def _summarize(
    results: list[SpecResult],
    *,
    principled_equivalence: bool,
    alpha: float,
    joint_permutations: int,
    rng: np.random.Generator,
) -> SpecCurve:
    converged = [r for r in results if r.converged]
    n_specs = len(results)
    n_ok = len(converged)

    if n_ok == 0:
        return SpecCurve(
            results=results,
            point_curve=np.array([]),
            significance_share=0.0,
            sign_consistency_share=0.0,
            joint_test_p=None,
            interpretation=(
                "No specifications converged. The hypothesis cannot be "
                "summarized; inspect SpecResult.error for the per-spec cause."
            ),
            principled_equivalence=principled_equivalence,
            median_point=0.0,
            bonferroni_min_p=None,
            converged_count=0,
            n_specs=n_specs,
            sorted_spec_ids=[r.spec_id for r in results],
        )

    pts_pairs = sorted(((r.point, r.spec_id) for r in converged), key=lambda p: p[0])
    point_curve = np.array([p for p, _ in pts_pairs], dtype=float)
    sorted_ids = [sid for _, sid in pts_pairs]

    median_point = float(np.median(point_curve))
    median_sign = 1.0 if median_point >= 0 else -1.0

    n_sig = sum(1 for r in converged if r.significant())
    significance_share = n_sig / n_ok

    n_sign_consistent = sum(
        1 for r in converged if (1.0 if r.point >= 0 else -1.0) == median_sign
    )
    sign_consistency_share = n_sign_consistent / n_ok

    # Per-spec two-sided p-values from (point / se).
    pvals: list[float] = []
    for r in converged:
        if r.se is not None and r.se > 0 and math.isfinite(r.se):
            z = r.point / r.se
            p = 2.0 * (1.0 - _normal_cdf(abs(z)))
            pvals.append(p)
    bonferroni_min_p: float | None
    if pvals:
        bonferroni_min_p = float(min(min(pvals) * len(pvals), 1.0))
    else:
        bonferroni_min_p = None

    joint_test_p: float | None = None
    if principled_equivalence:
        joint_test_p = _simonsohn_permutation_p(
            converged=converged,
            n_permutations=joint_permutations,
            rng=rng,
        )

    interpretation = _build_interpretation(
        principled_equivalence=principled_equivalence,
        n_specs=n_specs,
        n_ok=n_ok,
        significance_share=significance_share,
        sign_consistency_share=sign_consistency_share,
        median_point=median_point,
        joint_test_p=joint_test_p,
        bonferroni_min_p=bonferroni_min_p,
        alpha=alpha,
    )

    return SpecCurve(
        results=results,
        point_curve=point_curve,
        significance_share=significance_share,
        sign_consistency_share=sign_consistency_share,
        joint_test_p=joint_test_p,
        interpretation=interpretation,
        principled_equivalence=principled_equivalence,
        median_point=median_point,
        bonferroni_min_p=bonferroni_min_p,
        converged_count=n_ok,
        n_specs=n_specs,
        sorted_spec_ids=sorted_ids,
    )


def _normal_cdf(x: float) -> float:
    """Standard normal CDF without scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _simonsohn_permutation_p(
    *,
    converged: list[SpecResult],
    n_permutations: int,
    rng: np.random.Generator,
) -> float | None:
    """Simonsohn-Simmons-Nelson 2020 joint inference.

    The textbook test re-runs every spec under the null by permuting the
    treatment label; here we approximate the same logic using the analytic
    SEs (treating each spec's z-score as exchangeable under the joint null).
    This is a tractable surrogate for the full re-fit permutation when the
    caller can't afford K * P fits; it preserves the *direction* of the test
    (more significant specs than expected by chance) and is honest about
    being a surrogate.

    Returns the share of permuted draws whose median |z| meets or exceeds
    the observed median |z|. ``None`` if SEs aren't available for any spec.
    """
    zs = []
    for r in converged:
        if r.se is not None and r.se > 0 and math.isfinite(r.se):
            zs.append(r.point / r.se)
    if not zs:
        return None
    observed = float(np.median(np.abs(zs)))
    n = len(zs)
    # Null distribution: each spec's z ~ N(0, 1) independently.
    null_medians = np.median(np.abs(rng.standard_normal(size=(n_permutations, n))), axis=1)
    p = float((null_medians >= observed).mean())
    # Clamp away from exact 0 so the report doesn't claim p=0.
    return max(p, 1.0 / (n_permutations + 1))


def _build_interpretation(
    *,
    principled_equivalence: bool,
    n_specs: int,
    n_ok: int,
    significance_share: float,
    sign_consistency_share: float,
    median_point: float,
    joint_test_p: float | None,
    bonferroni_min_p: float | None,
    alpha: float,
) -> str:
    head = (
        f"Ran {n_specs} specs ({n_ok} converged). "
        f"Median effect = {median_point:.3g}; "
        f"{significance_share:.0%} significant at alpha={alpha:.2g}, "
        f"{sign_consistency_share:.0%} sign-consistent with the median."
    )
    if principled_equivalence:
        if joint_test_p is None:
            tail = (
                "Principled equivalence was certified but no spec returned "
                "an analytic SE; joint inference is unavailable."
            )
        else:
            verdict = "reject" if joint_test_p < alpha else "do not reject"
            tail = (
                "Principled equivalence was certified (Del Giudice & "
                "Gangestad 2021), so Simonsohn-Simmons-Nelson 2020 joint "
                f"inference is applicable: permutation p = {joint_test_p:.3g}; "
                f"{verdict} the joint null at alpha={alpha:.2g}."
            )
    else:
        bm = (
            "no per-spec p-values"
            if bonferroni_min_p is None
            else f"Bonferroni-corrected min p = {bonferroni_min_p:.3g}"
        )
        tail = (
            "Principled equivalence NOT certified, so joint inference is "
            "withheld (Del Giudice & Gangestad 2021 caveat — the spec "
            "product mixes non-equivalent analyses, making the joint null "
            f"ill-defined). Using {bm} as a conservative summary."
        )
    return head + " " + tail


# --------------------------------------------------------------------------- #
# HTML rendering
# --------------------------------------------------------------------------- #


def render_html(curve: SpecCurve, *, title: str = "Specification curve") -> str:
    """Render the SpecCurve as a self-contained HTML fragment.

    Uses inline SVG so the report can ship without a JS plotting library.
    Each spec is drawn as a dot; converged specs get an SE bar.
    """
    n = curve.n_specs
    converged = [r for r in curve.results if r.converged]
    if not converged:
        return (
            f"<section class='specr'><h2>{_esc(title)}</h2>"
            f"<p><strong>No converged specifications.</strong></p>"
            f"<p>{_esc(curve.interpretation)}</p></section>"
        )

    pts = curve.point_curve
    y_lo = float(np.min(pts))
    y_hi = float(np.max(pts))
    if y_lo == y_hi:
        y_lo -= 1.0
        y_hi += 1.0
    pad = 0.1 * (y_hi - y_lo)
    y_lo -= pad
    y_hi += pad

    width = 720
    height = 240
    margin_l, margin_r, margin_t, margin_b = 60, 16, 28, 32
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def x_of(i: int) -> float:
        if len(pts) == 1:
            return margin_l + plot_w / 2
        return margin_l + plot_w * i / (len(pts) - 1)

    def y_of(v: float) -> float:
        return margin_t + plot_h * (1.0 - (v - y_lo) / (y_hi - y_lo))

    dots: list[str] = []
    bars: list[str] = []
    sid_to_result = {r.spec_id: r for r in curve.results}
    for i, sid in enumerate(curve.sorted_spec_ids):
        r = sid_to_result[sid]
        cx = x_of(i)
        cy = y_of(r.point)
        color = "#2c7" if r.significant() else "#888"
        dots.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='3' fill='{color}'>"
            f"<title>{_esc(sid)}: {r.point:.3g} (est={_esc(r.estimator_id)})</title>"
            f"</circle>"
        )
        if r.ci_low is not None and r.ci_high is not None:
            y1 = y_of(r.ci_low)
            y2 = y_of(r.ci_high)
            bars.append(
                f"<line x1='{cx:.1f}' y1='{y1:.1f}' x2='{cx:.1f}' y2='{y2:.1f}' "
                f"stroke='{color}' stroke-width='1' opacity='0.4'/>"
            )

    # Y-axis ticks at min/median/max.
    ticks = []
    for v in (y_lo + pad, (y_lo + y_hi) / 2, y_hi - pad):
        ticks.append(
            f"<text x='{margin_l - 6:.1f}' y='{y_of(v) + 4:.1f}' "
            f"text-anchor='end' font-size='10' fill='#444'>{v:.3g}</text>"
        )
    # Zero line.
    zero_line = ""
    if y_lo <= 0 <= y_hi:
        zy = y_of(0.0)
        zero_line = (
            f"<line x1='{margin_l}' y1='{zy:.1f}' x2='{margin_l + plot_w}' "
            f"y2='{zy:.1f}' stroke='#c33' stroke-dasharray='3,3' opacity='0.6'/>"
        )

    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' role='img' aria-label='specification curve'>"
        f"<rect x='{margin_l}' y='{margin_t}' width='{plot_w}' height='{plot_h}' "
        f"fill='#fafafa' stroke='#ddd'/>"
        f"{zero_line}"
        f"{''.join(bars)}"
        f"{''.join(dots)}"
        f"{''.join(ticks)}"
        f"<text x='{margin_l + plot_w/2:.1f}' y='{height - 8}' text-anchor='middle' "
        f"font-size='11' fill='#444'>specification rank (n={n})</text>"
        f"</svg>"
    )

    rows_html = []
    for sid in curve.sorted_spec_ids:
        r = sid_to_result[sid]
        ci = (
            f"[{r.ci_low:.3g}, {r.ci_high:.3g}]"
            if r.ci_low is not None and r.ci_high is not None
            else "—"
        )
        rows_html.append(
            f"<tr><td>{_esc(sid)}</td>"
            f"<td>{_esc(_short_adj(r.spec['adjustment_set']))}</td>"
            f"<td>{_esc(str(r.spec['estimator']))}</td>"
            f"<td>{r.spec['trim']:.2f}</td>"
            f"<td>{r.point:.3g}</td>"
            f"<td>{ci}</td></tr>"
        )

    summary = (
        f"<ul>"
        f"<li>n specs: {curve.n_specs} ({curve.converged_count} converged)</li>"
        f"<li>significance share: {curve.significance_share:.0%}</li>"
        f"<li>sign consistency: {curve.sign_consistency_share:.0%}</li>"
        f"<li>median effect: {curve.median_point:.3g}</li>"
        f"<li>principled equivalence: {curve.principled_equivalence}</li>"
        + (
            f"<li>joint test p: {curve.joint_test_p:.3g}</li>"
            if curve.joint_test_p is not None
            else ""
        )
        + (
            f"<li>Bonferroni min p: {curve.bonferroni_min_p:.3g}</li>"
            if curve.bonferroni_min_p is not None
            else ""
        )
        + f"</ul>"
    )

    return (
        f"<section class='specr'>"
        f"<h2>{_esc(title)}</h2>"
        f"{svg}"
        f"<p>{_esc(curve.interpretation)}</p>"
        f"{summary}"
        f"<table><thead><tr><th>spec</th><th>adj</th><th>est</th>"
        f"<th>trim</th><th>point</th><th>CI</th></tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
        f"</section>"
    )


def _short_adj(adj: tuple[str, ...]) -> str:
    if not adj:
        return "∅"
    return ",".join(adj) if len(adj) <= 4 else f"{','.join(adj[:3])},…(+{len(adj)-3})"


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


__all__ = [
    "SpecCurve",
    "SpecResult",
    "render_html",
    "specification_curve",
]
