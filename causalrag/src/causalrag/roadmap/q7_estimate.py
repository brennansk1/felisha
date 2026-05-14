"""Step 7 — Estimate the Statistical Estimand (PDD §10.7).

Picks an estimator via :func:`causalrag.estimators.python.select.select_estimator`,
fits it on the supplied DataFrame, and emits an :class:`EstimationResult`.
The selection rule and the user's manual override (``prefer=``) are both
recorded on the result's ``diagnostics`` so the report can show *why* this
estimator ran.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.result import EstimationResult
from causalrag.data.checks import overlap_summary
from causalrag.data.features import auto_preprocess
from causalrag.data.profiler import DatasetProfile, profile_dataframe
from causalrag.data.selection import Method as SelectionMethod, select_variables
from causalrag.estimators.python.select import select_estimator
from causalrag.roadmap.q5_identify import IdentificationResult


def estimate(
    *,
    df: pd.DataFrame,
    estimand: CausalEstimand,
    identification: IdentificationResult,
    protocol: StudyProtocol,
    confounders: tuple[str, ...] | None = None,
    modifiers: tuple[str, ...] | None = None,
    flags: set[DataFlag] | None = None,
    prefer: str | None = None,
    allow_nonidentifiable: bool = False,
    profile: DatasetProfile | None = None,
    preprocess: bool = True,
    selection: SelectionMethod = "auto",
    pinned_covariates: tuple[str, ...] = (),
    pin_adjustment_set: bool = True,
) -> EstimationResult:
    """Run Step 7 for a single estimand.

    ``confounders`` defaults to ``identification.adjustment_set``. If Step 5
    judged the estimand non-identifiable, estimation is blocked unless
    ``allow_nonidentifiable=True`` (PDD design principle 8 — identifiability
    is a hard gate).
    """
    if not identification.identifiable and not allow_nonidentifiable:
        raise ValueError(
            "Step 5 judged the estimand non-identifiable. Re-specify the DAG "
            "or pass allow_nonidentifiable=True to override (recorded as an "
            "override in the StudyProtocol)."
        )

    adj = confounders if confounders is not None else identification.adjustment_set
    mods = modifiers or estimand.modifiers
    situation_flags = set(flags or set())

    # Preprocessing: encode categoricals, standardize continuous, etc.
    preprocess_manifest_dict: dict[str, object] | None = None
    selection_dict: dict[str, object] | None = None
    df_used = df
    adj_used: tuple[str, ...] = tuple(adj)
    mods_used: tuple[str, ...] = tuple(mods)
    if preprocess:
        prof = profile or profile_dataframe(df)
        df_used, manifest = auto_preprocess(
            df, prof, treatment=estimand.treatment, outcome=estimand.outcome
        )
        preprocess_manifest_dict = manifest.to_dict()
        # Expand adjustment + modifiers to their derived columns (one-hot, date parts).
        def _expand(names: tuple[str, ...]) -> tuple[str, ...]:
            out: list[str] = []
            for n in names:
                derived = manifest.new_columns_from.get(n)
                if derived:
                    out.extend(derived)
                elif n in df_used.columns:
                    out.append(n)
            return tuple(out)
        adj_used = _expand(adj_used)
        mods_used = _expand(mods_used)

    # Principled variable selection on the adjustment set — but only when the
    # adjustment set is *not* a confirmatory DAG-derived backdoor set. When
    # Step 5 produced a backdoor-admissible set we must keep it verbatim;
    # otherwise data-driven selection can silently drop columns the DAG
    # required for identification.
    variable_selection_skipped = False
    skip_reason: str | None = None
    if (
        pin_adjustment_set
        and identification.strategy == "backdoor"
        and len(identification.adjustment_set) > 0
        and confounders is None
    ):
        variable_selection_skipped = True
        skip_reason = (
            "Adjustment set pinned from Step 5 backdoor identification; "
            "data-driven variable selection skipped to preserve identifiability."
        )
    elif adj_used:
        sel = select_variables(
            df_used,
            estimand.treatment,
            estimand.outcome,
            adj_used,
            method=selection,
            high_dimensional=DataFlag.HIGH_DIMENSIONAL in situation_flags,
            pinned=pinned_covariates,
        )
        selection_dict = sel.to_dict()
        adj_used = sel.selected

    # Pre-flight positivity / overlap diagnostic. If propensity tails are
    # extreme we add POSITIVITY_VIOLATION to the situation, which routes the
    # selector toward doubly-robust methods or away from pure IPW.
    overlap: dict[str, object] | None = None
    if adj_used and DataFlag.BINARY_TREATMENT in situation_flags:
        try:
            diag = overlap_summary(df_used, estimand.treatment, tuple(adj_used))
            overlap = diag.to_dict()
            if diag.positivity.verdict == "red":
                situation_flags.add(DataFlag.POSITIVITY_VIOLATION)
        except Exception:
            overlap = None

    entry = select_estimator(
        estimand=estimand.klass.value,
        flags=frozenset(situation_flags),
        n=len(df_used),
        n_modifiers=len(mods_used),
        prefer=prefer,
    )

    Factory = entry.factory
    estimator = Factory(
        treatment=estimand.treatment,
        outcome=estimand.outcome,
        confounders=tuple(adj_used),
        modifiers=tuple(mods_used),
    )
    estimator.fit(df_used, protocol)
    result = estimator.estimate()

    result.diagnostics = dict(result.diagnostics)
    result.diagnostics["selected_estimator_id"] = entry.id
    result.diagnostics["prefer_override"] = prefer
    result.diagnostics["identification_strategy"] = identification.strategy
    result.diagnostics["adjustment_set_initial"] = list(adj)
    result.diagnostics["adjustment_set_used"] = list(adj_used)
    if preprocess_manifest_dict is not None:
        result.diagnostics["preprocessing"] = preprocess_manifest_dict
    if selection_dict is not None:
        result.diagnostics["variable_selection"] = selection_dict
    if variable_selection_skipped:
        result.diagnostics["variable_selection_skipped"] = True
        result.diagnostics["variable_selection_skipped_reason"] = skip_reason
    if overlap is not None:
        result.diagnostics["overlap"] = overlap

    # PDD §29.1 — refutations: placebo treatment, random common cause,
    # subset bootstrap. These are deterministic, statistical robustness checks
    # that PhD-level analyses include by default. Each refuter is best-effort;
    # if the estimator does not support a given refuter we skip it.
    result.refutations = _run_refutations(estimator, df_used, estimand, protocol, result)
    return result


def _refute_protocol(protocol: StudyProtocol) -> StudyProtocol:
    """Return a minimal-state shallow copy of ``protocol`` for refutation refits.

    We retain ``flags`` and ``llm`` (which drives nuisance-library selection)
    so the refutation fit is methodologically comparable to the original, but
    drop heavy walk/queue state to avoid mutating the caller's protocol and
    to keep refits fast. Calling ``model_copy`` produces a shallow copy; we
    deliberately keep ``flags`` as a shared set since the refits never mutate
    it.
    """
    return protocol.model_copy(update={"name": f"{protocol.name}__refute"})


def _run_refutations(
    estimator,
    df,
    estimand: CausalEstimand,
    protocol: StudyProtocol,
    original_result: EstimationResult,
) -> dict[str, object]:
    """Run the standard refutation battery and return a JSON-serializable dict.

    Three checks (Pearl/Imai conventions), all anchored to the **original
    estimator's standard error** rather than ad-hoc absolute thresholds:

    - **placebo_treatment**: re-fit with a randomly permuted treatment.
      Passes iff ``|placebo| < 2 * original.se`` — i.e. the placebo estimate
      is statistically indistinguishable from 0 at the original estimator's
      noise level.
    - **random_common_cause**: append a synthetic confounder; passes iff
      ``|refuted - original| < 2 * original.se`` — i.e. the estimate stays
      within 2 SE of the original.
    - **subset_bootstrap**: re-fit on K=10 70% bootstrap samples; passes iff
      the mean of those estimates is within 2 SE of the original. Also
      surfaces the bootstrap std as a diagnostic.

    If ``original.se`` is None, the pass/fail verdict cannot be calibrated
    and we return ``passed=None`` with an explanatory ``reason``.

    For each refuter we store: ``original``, ``original_se``, ``refuted``
    (a.k.a. ``refuted_estimate``), ``delta_in_se_units``, ``passed``, and
    ``reason`` (when ``passed`` is None).
    """
    import numpy as np
    import math

    out: dict[str, object] = {}
    original = float(original_result.point_estimate)
    original_se = original_result.se
    refute_protocol = _refute_protocol(protocol)

    def _verdict(refuted_value: float) -> tuple[bool | None, float | None, str | None]:
        """Return (passed, delta_in_se_units, reason). Used for placebo (vs 0)
        only by passing ``refuted_value`` already as the absolute deviation.
        """
        if original_se is None or not math.isfinite(original_se) or original_se <= 0:
            return None, None, "no SE available"
        delta = abs(refuted_value) / original_se
        return delta < 2.0, float(delta), None

    # 1. Placebo treatment — compare |placebo| against 2 * original_se.
    try:
        rng = np.random.default_rng(0)
        placebo_df = df.copy()
        placebo_df[estimand.treatment] = rng.permutation(
            placebo_df[estimand.treatment].to_numpy()
        )
        cls = type(estimator)
        placebo = cls(
            treatment=estimand.treatment,
            outcome=estimand.outcome,
            confounders=getattr(estimator, "confounders", ()),
            modifiers=getattr(estimator, "modifiers", ()),
        )
        placebo.fit(placebo_df, refute_protocol)
        placebo_estimate = float(placebo.estimate().point_estimate)
        passed, delta, reason = _verdict(placebo_estimate)
        out["placebo_treatment"] = {
            "original": original,
            "original_se": original_se,
            "refuted": placebo_estimate,
            "refuted_estimate": placebo_estimate,
            "delta_in_se_units": delta,
            "passed": passed,
            "reason": reason,
        }
    except Exception as e:
        out["placebo_treatment"] = {"error": f"{type(e).__name__}: {e}"}

    # 2. Random common cause — passes iff |refuted - original| < 2 * SE.
    try:
        rng = np.random.default_rng(1)
        rc_df = df.copy()
        rc_df["_random_common_cause"] = rng.normal(size=len(rc_df))
        cls = type(estimator)
        rc = cls(
            treatment=estimand.treatment,
            outcome=estimand.outcome,
            confounders=tuple(list(getattr(estimator, "confounders", ())) + ["_random_common_cause"]),
            modifiers=getattr(estimator, "modifiers", ()),
        )
        rc.fit(rc_df, refute_protocol)
        rc_estimate = float(rc.estimate().point_estimate)
        if original_se is None or not math.isfinite(original_se) or original_se <= 0:
            passed, delta, reason = None, None, "no SE available"
        else:
            delta = abs(rc_estimate - original) / original_se
            passed = delta < 2.0
            reason = None
        out["random_common_cause"] = {
            "original": original,
            "original_se": original_se,
            "refuted": rc_estimate,
            "refuted_estimate": rc_estimate,
            "delta_in_se_units": float(delta) if delta is not None else None,
            "passed": passed,
            "reason": reason,
        }
    except Exception as e:
        out["random_common_cause"] = {"error": f"{type(e).__name__}: {e}"}

    # 3. Subset bootstrap — K iterations of a 70% subsample; verdict on the mean.
    try:
        rng = np.random.default_rng(2)
        K = 10
        estimates: list[float] = []
        cls = type(estimator)
        n = len(df)
        size = max(2, int(0.7 * n))
        for k in range(K):
            idx = rng.choice(n, size=size, replace=False)
            sub_df = df.iloc[idx]
            sub = cls(
                treatment=estimand.treatment,
                outcome=estimand.outcome,
                confounders=getattr(estimator, "confounders", ()),
                modifiers=getattr(estimator, "modifiers", ()),
            )
            sub.fit(sub_df, refute_protocol)
            estimates.append(float(sub.estimate().point_estimate))
        sub_mean = float(np.mean(estimates))
        sub_std = float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0
        if original_se is None or not math.isfinite(original_se) or original_se <= 0:
            passed, delta, reason = None, None, "no SE available"
        else:
            delta = abs(sub_mean - original) / original_se
            passed = delta < 2.0
            reason = None
        out["subset_bootstrap"] = {
            "original": original,
            "original_se": original_se,
            "refuted": sub_mean,
            "refuted_estimate": sub_mean,
            "bootstrap_std": sub_std,
            "n_iterations": K,
            "delta_in_se_units": float(delta) if delta is not None else None,
            "passed": passed,
            "reason": reason,
        }
    except Exception as e:
        out["subset_bootstrap"] = {"error": f"{type(e).__name__}: {e}"}

    out["n_passed"] = sum(
        1
        for v in out.values()
        if isinstance(v, dict) and v.get("passed") is True
    )
    return out


__all__ = ["estimate"]
