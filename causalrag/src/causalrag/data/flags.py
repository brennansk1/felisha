"""DataFlag emission from the deterministic profile — Stage 1d (PDD §7.4, §15.1).

The profiler is the **veto** source: any flag emitted here cannot be overridden
by the LLM. The LLM's Stage 1c suggestions are merged in via a separate
emitter (``discovery/investigator.py``).

This module only emits flags that can be decided structurally from the data —
treatment / outcome dtype, censoring, sample size, missingness, dimensionality,
positivity-violation hints. Semantic flags (mediator, IV candidate, negative
control) require domain knowledge and live downstream.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.data.profiler import ColumnProfile, DatasetProfile


def emit_from_profile(
    profile: DatasetProfile,
    *,
    treatment: str | None = None,
    outcome: str | None = None,
    df: pd.DataFrame | None = None,
    time_column: str | None = None,
    subject_column: str | None = None,
) -> set[DataFlag]:
    """Emit deterministic flags. When ``treatment`` / ``outcome`` are known,
    emit the corresponding type flags; otherwise emit only structural flags.

    When ``df`` is provided alongside treatment/outcome, additional
    structural-on-data flags (DiD candidate, staggered adoption,
    time-varying treatment, zero-inflated count outcome) may be emitted.
    """
    flags: set[DataFlag] = set()

    if treatment:
        t = profile.column(treatment)
        flags |= _treatment_flags(t)
        if t.is_binary_01 or t.logical_dtype == "binary":
            if _imbalanced_treatment(t):
                flags.add(DataFlag.IMBALANCED_TREATMENT)
    if outcome:
        y = profile.column(outcome)
        flags |= _outcome_flags(y)
        if y.is_binary_01 or y.logical_dtype == "binary":
            if _rare_outcome(y, profile):
                flags.add(DataFlag.RARE_OUTCOME)
        if _bounded_outcome(y):
            flags.add(DataFlag.BOUNDED_OUTCOME)
        if df is not None and outcome in df.columns and _zero_inflated_outcome(y, df[outcome]):
            flags.add(DataFlag.ZERO_INFLATED_OUTCOME)

    flags |= _structural_flags(profile)

    if profile.censoring_pairs:
        flags.add(DataFlag.RIGHT_CENSORED_OUTCOME)

    if df is not None and treatment is not None:
        subj = subject_column or _guess_subject_column(profile, df, treatment)
        tcol = time_column or _guess_time_column(profile)
        if subj is not None and tcol is not None and subj in df.columns and tcol in df.columns:
            if _time_varying_treatment(df, subj, treatment):
                flags.add(DataFlag.TIME_VARYING_TREATMENT)
            if _diff_in_diff_candidate(df, subj, tcol, treatment):
                flags.add(DataFlag.DIFF_IN_DIFF_CANDIDATE)
                if _staggered_adoption(df, subj, tcol, treatment):
                    flags.add(DataFlag.STAGGERED_ADOPTION)

    return flags


def _treatment_flags(col: ColumnProfile) -> set[DataFlag]:
    if col.is_binary_01 or col.logical_dtype == "binary":
        return {DataFlag.BINARY_TREATMENT}
    if col.logical_dtype in {"continuous", "count"}:
        return {DataFlag.CONTINUOUS_TREATMENT}
    if col.logical_dtype == "categorical":
        return {DataFlag.CATEGORICAL_TREATMENT}
    return set()


def _outcome_flags(col: ColumnProfile) -> set[DataFlag]:
    if col.is_binary_01 or col.logical_dtype == "binary":
        return {DataFlag.BINARY_OUTCOME}
    if col.logical_dtype == "count":
        return {DataFlag.COUNT_OUTCOME}
    if col.logical_dtype == "continuous":
        return {DataFlag.CONTINUOUS_OUTCOME}
    return set()


def _structural_flags(profile: DatasetProfile) -> set[DataFlag]:
    flags: set[DataFlag] = set()

    if profile.n_rows < 200:
        flags.add(DataFlag.SMALL_SAMPLE)

    # p > sqrt(n) under naive one-hot expansion bound: count distinct levels.
    p_effective = 0
    for c in profile.columns:
        if c.logical_dtype == "categorical":
            p_effective += max(c.cardinality - 1, 0)
        elif c.logical_dtype in {"continuous", "count", "ordinal", "binary"}:
            p_effective += 1
    if profile.n_rows > 0 and p_effective**2 > profile.n_rows:
        flags.add(DataFlag.HIGH_DIMENSIONAL)

    if any(c.missing_rate > 0.20 for c in profile.columns):
        flags.add(DataFlag.HEAVY_MISSINGNESS)

    # Censoring heaviness: an event column with mean < 0.30 indicates >70%
    # censored — only meaningful if a censoring pair was detected.
    for _, ev_name in profile.censoring_pairs:
        ev = profile.column(ev_name)
        # event-rate proxy: use mode frequency for the binary event indicator
        if ev.top_values:
            ones = next((v for k, v in ev.top_values if k in {"1", "1.0", "True", "true"}), 0)
            if ev.n_total > 0 and (ones / ev.n_total) < 0.30:
                flags.add(DataFlag.HEAVY_CENSORING)
                break

    return flags


def _rare_outcome(col: ColumnProfile, profile: DatasetProfile) -> bool:
    """Binary outcome with prevalence <5% OR event-per-covariate ratio <10.

    The event-per-covariate (EPC) check uses the structural-flag p_effective
    proxy so we only count covariates whose dummy expansion would actually
    enter a model.
    """
    prev = _binary_prevalence(col)
    if prev is None:
        return False
    if min(prev, 1.0 - prev) < 0.05:
        return True
    # EPC: number of events / number of covariates < 10.
    n_events = min(prev, 1.0 - prev) * col.n_total
    p_eff = 0
    for c in profile.columns:
        if c.name == col.name:
            continue
        if c.logical_dtype == "categorical":
            p_eff += max(c.cardinality - 1, 0)
        elif c.logical_dtype in {"continuous", "count", "ordinal", "binary"}:
            p_eff += 1
    if p_eff > 0 and (n_events / p_eff) < 10:
        return True
    return False


def _imbalanced_treatment(col: ColumnProfile) -> bool:
    """Binary treatment prevalence outside [0.15, 0.85]."""
    prev = _binary_prevalence(col)
    if prev is None:
        return False
    return prev < 0.15 or prev > 0.85


def _bounded_outcome(col: ColumnProfile) -> bool:
    """Numeric Y bounded to [0, 1] with ≥5 distinct values (avoids binary)."""
    if col.logical_dtype not in {"continuous", "ordinal"}:
        return False
    if col.is_binary_01:
        return False
    if col.min is None or col.max is None:
        return False
    if col.min < 0.0 or col.max > 1.0:
        return False
    if col.cardinality < 5:
        return False
    return True


def _zero_inflated_outcome(col: ColumnProfile, series: pd.Series) -> bool:
    """Integer Y with ≥50% zeros and >2 unique non-zero values.

    The profiler labels integer columns as ``count`` or ``ordinal`` depending
    on cardinality; we accept either, provided the values are integers ≥0.
    """
    if col.logical_dtype not in {"count", "ordinal"}:
        return False
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return False
    if not float(s.min()) >= 0:
        return False
    # Must look integer-typed.
    if not bool((s == s.astype(int)).all()):
        return False
    zero_share = float((s == 0).mean())
    if zero_share < 0.50:
        return False
    nonzero = s[s != 0]
    return int(nonzero.nunique()) > 2


def _binary_prevalence(col: ColumnProfile) -> float | None:
    """Return the prevalence (share of 1/True) for a binary column, or None."""
    if not (col.is_binary_01 or col.logical_dtype == "binary"):
        return None
    if not col.top_values or col.n_total <= 0:
        return None
    ones = next(
        (v for k, v in col.top_values if k in {"1", "1.0", "True", "true"}),
        None,
    )
    if ones is None:
        # Fall back to: 1 - (share of the most-common value if mode != "1")
        total = sum(v for _, v in col.top_values)
        if total <= 0:
            return None
        # If we can't find a 1 key, treat the *minority* as the event share.
        sorted_counts = sorted((v for _, v in col.top_values), reverse=True)
        minority = sorted_counts[-1] if len(sorted_counts) >= 2 else 0
        return minority / col.n_total
    return ones / col.n_total


def _guess_subject_column(
    profile: DatasetProfile, df: pd.DataFrame, treatment: str
) -> str | None:
    """Best-effort subject-id detection: integer/string column where each
    value repeats ≥2 times AND that correlates with treatment switches.

    Returns the first viable candidate, or ``None``.
    """
    n = profile.n_rows
    for c in profile.columns:
        if c.name == treatment:
            continue
        if c.name not in df.columns:
            continue
        if c.cardinality < 2 or c.cardinality >= n:
            continue
        # Each value should repeat ≥2 times on average.
        if n / max(c.cardinality, 1) < 2:
            continue
        if c.logical_dtype not in {"categorical", "ordinal", "count"} and not c.suspected_identifier:
            continue
        return c.name
    return None


def _guess_time_column(profile: DatasetProfile) -> str | None:
    for c in profile.columns:
        if c.suspected_time_column:
            return c.name
    for c in profile.columns:
        if c.logical_dtype == "datetime":
            return c.name
    return None


def _time_varying_treatment(df: pd.DataFrame, subject: str, treatment: str) -> bool:
    """Same subject_id has multiple distinct treatment values."""
    if subject not in df.columns or treatment not in df.columns:
        return False
    sub = df[[subject, treatment]].dropna()
    if sub.empty:
        return False
    per_subject = sub.groupby(subject)[treatment].nunique()
    return bool((per_subject > 1).any())


def _diff_in_diff_candidate(
    df: pd.DataFrame, subject: str, time_col: str, treatment: str
) -> bool:
    """Panel + pre/post + treated/control structure.

    Heuristic: at least 2 distinct time periods, at least 2 subjects, and at
    least one subject whose treatment switches over time while at least one
    subject's treatment stays constant.
    """
    needed = {subject, time_col, treatment}
    if not needed.issubset(df.columns):
        return False
    sub = df[[subject, time_col, treatment]].dropna()
    if sub.empty:
        return False
    if sub[time_col].nunique() < 2:
        return False
    if sub[subject].nunique() < 2:
        return False
    switches = sub.groupby(subject)[treatment].nunique()
    has_switcher = bool((switches > 1).any())
    has_constant = bool((switches == 1).any())
    return has_switcher and has_constant


def _staggered_adoption(
    df: pd.DataFrame, subject: str, time_col: str, treatment: str
) -> bool:
    """DiD where treated subjects adopt the treatment at variable times."""
    sub = df[[subject, time_col, treatment]].dropna()
    if sub.empty:
        return False
    # Treatment must be binary-ish for the "onset" idea to make sense.
    treat_vals = sub[treatment].unique()
    if len(treat_vals) > 5:
        return False
    # First-treated period per subject (subjects that ever take treatment==1).
    treated_rows = sub[sub[treatment].astype(float) > 0]
    if treated_rows.empty:
        return False
    onset = treated_rows.groupby(subject)[time_col].min()
    return onset.nunique() >= 2


def positivity_violation(
    df: pd.DataFrame,
    treatment: str,
    confounders: Iterable[str],
    threshold: float = 0.05,
) -> bool:
    """Detect crude positivity violation: any (confounder-level × treatment-arm)
    cell with empirical probability below ``threshold``.

    The full check happens later in Step 5 via DoWhy's positivity diagnostics;
    this is the Stage 1b veto signal so the routing brain can already exclude
    methods that require positivity (e.g., IPW).
    """
    if treatment not in df.columns:
        return False
    arms = df[treatment].dropna().unique()
    if len(arms) < 2:
        return False
    for c in confounders:
        if c not in df.columns:
            continue
        if df[c].nunique() > 20:
            continue
        joint = pd.crosstab(df[c], df[treatment], normalize="index")
        if (joint < threshold).any().any():
            return True
    return False
