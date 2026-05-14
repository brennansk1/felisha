"""Deterministic statistical profiler — Stage 1b (PDD §7.2).

Runs before any LLM call. Produces a compact ``DatasetProfile`` (~5–20 KB JSON
for typical datasets) consumed by both Stage 1c prompts and the feasibility
filter. Designed to complete in under 10 seconds on 10M rows on commodity
hardware; we don't optimize aggressively in v0.1 but the structure is in place.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

LogicalDType = Literal[
    "binary",
    "ordinal",
    "categorical",
    "continuous",
    "count",
    "time_to_event",
    "date",
    "datetime",
    "identifier",
    "text",
]


_IDENTIFIER_NAME_PAT = re.compile(r"(^id$|_id$|^uuid$|^guid$|patient|subject_no)", re.I)
_EVENT_NAME_PAT = re.compile(r"(event|status|censored|indicator)$", re.I)
_TIME_NAME_PAT = re.compile(r"(time|days|months|years|duration|tenure)$", re.I)


class ColumnProfile(BaseModel):
    """Compact per-column profile (PDD §7.2).

    Fields are chosen so the JSON projection stays under ~1 KB per column.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: str  # pyarrow/pandas dtype string
    logical_dtype: LogicalDType
    n_total: int
    n_missing: int
    missing_rate: float
    cardinality: int

    # Continuous-only
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    p5: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p95: float | None = None
    skew: float | None = None
    kurtosis: float | None = None
    n_tukey_outliers: int | None = None

    # Categorical-only
    mode: str | None = None
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    entropy: float | None = None

    # Heuristic flags
    suspected_identifier: bool = False
    suspected_event_indicator: bool = False
    suspected_time_column: bool = False
    is_binary_01: bool = False
    constant: bool = False


class DatasetProfile(BaseModel):
    """Output of Stage 1b (PDD §7.2)."""

    model_config = ConfigDict(extra="forbid")

    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    column_pairs_high_corr: list[tuple[str, str, float]] = Field(default_factory=list)
    censoring_pairs: list[tuple[str, str]] = Field(default_factory=list)
    missingness_clusters: list[list[str]] = Field(
        default_factory=list,
        description="Groups of columns that co-occur as missing in >5% of rows — "
        "informative for distinguishing MCAR (no clusters) from MAR/MNAR "
        "(structured clusters).",
    )
    n_exact_duplicate_rows: int = Field(
        default=0,
        description="Count of rows that are exact duplicates of another row (after "
        "dropping identifier columns). Informative for clinical claims data where "
        "the same encounter shows up multiple times.",
    )
    string_formats: dict[str, str] = Field(
        default_factory=dict,
        description="Per-text-column inferred format: ``date_like``, "
        "``identifier_like``, ``categorical_like``, ``free_text``, ``email_like``, "
        "``url_like``, ``zip_like``. Helps the investigator distinguish identifiers "
        "from analytic variables.",
    )

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)


# --- Heuristics --------------------------------------------------------------


def _infer_logical_dtype(series: pd.Series, name: str) -> LogicalDType:
    s = series.dropna()
    n = len(s)
    if n == 0:
        return "categorical"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s):
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        unique = s.nunique()
        if unique == 2 and set(s.unique().tolist()).issubset({0, 1, 0.0, 1.0, True, False}):
            return "binary"
        if pd.api.types.is_integer_dtype(s) and unique <= 10:
            return "ordinal"
        if pd.api.types.is_integer_dtype(s) and (s >= 0).all() and unique <= 50:
            return "count"
        if pd.api.types.is_integer_dtype(s) and unique > 0 and unique / max(n, 1) > 0.95:
            return "identifier" if _IDENTIFIER_NAME_PAT.search(name) else "count"
        return "continuous"
    # Object / string
    unique = s.nunique()
    if _IDENTIFIER_NAME_PAT.search(name) and unique / max(n, 1) > 0.9:
        return "identifier"
    if unique <= 50 or unique / max(n, 1) < 0.5:
        return "categorical"
    # Try date parsing for the first 100 non-null values
    sample = s.astype(str).head(100)
    parsed = pd.to_datetime(sample, errors="coerce")
    if parsed.notna().mean() > 0.8:
        return "date"
    return "text"


def _continuous_stats(series: pd.Series) -> dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if s.empty:
        return {}
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    iqr = q[0.75] - q[0.25]
    lo, hi = q[0.25] - 1.5 * iqr, q[0.75] + 1.5 * iqr
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "p5": float(q[0.05]),
        "p25": float(q[0.25]),
        "p50": float(q[0.50]),
        "p75": float(q[0.75]),
        "p95": float(q[0.95]),
        "skew": float(s.skew()) if len(s) > 2 else 0.0,
        "kurtosis": float(s.kurtosis()) if len(s) > 3 else 0.0,
        "n_tukey_outliers": int(((s < lo) | (s > hi)).sum()),
    }


def _categorical_stats(series: pd.Series) -> dict[str, Any]:
    s = series.dropna().astype(str)
    if s.empty:
        return {"top_values": [], "mode": None, "entropy": 0.0}
    vc = s.value_counts().head(20)
    probs = vc.values / vc.values.sum()
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    return {
        "mode": str(vc.index[0]),
        "top_values": [(str(k), int(v)) for k, v in vc.items()],
        "entropy": entropy,
    }


def _profile_column(name: str, series: pd.Series) -> ColumnProfile:
    n_total = len(series)
    n_missing = int(series.isna().sum())
    cardinality = int(series.nunique(dropna=True))
    logical = _infer_logical_dtype(series, name)

    cont = _continuous_stats(series) if logical in {"continuous", "count", "ordinal"} else {}
    cat = _categorical_stats(series) if logical in {"categorical", "binary", "ordinal"} else {}

    unique_set: set[Any] = set()
    if pd.api.types.is_numeric_dtype(series):
        unique_set = set(series.dropna().unique().tolist())

    return ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        logical_dtype=logical,
        n_total=n_total,
        n_missing=n_missing,
        missing_rate=round(n_missing / max(n_total, 1), 4),
        cardinality=cardinality,
        suspected_identifier=(logical == "identifier") or bool(_IDENTIFIER_NAME_PAT.search(name)),
        suspected_event_indicator=bool(_EVENT_NAME_PAT.search(name))
        and unique_set.issubset({0, 1, 0.0, 1.0}),
        suspected_time_column=bool(_TIME_NAME_PAT.search(name)),
        is_binary_01=unique_set.issubset({0, 1, 0.0, 1.0}) and 0 < cardinality <= 2,
        constant=cardinality <= 1,
        **cont,
        **cat,
    )


def _find_censoring_pairs(profiles: list[ColumnProfile]) -> list[tuple[str, str]]:
    """Detect paired (time, event) columns (PDD §7.2).

    Matching is permissive: a pair is detected when (a) the bases match
    exactly after stripping time/event suffixes, (b) one base is a prefix of
    the other (handles ``overall_time_days`` + ``overall_event``), or (c) the
    bases share a non-trivial token prefix.
    """
    times = [c.name for c in profiles if c.suspected_time_column]
    events = [c.name for c in profiles if c.suspected_event_indicator]
    pairs: list[tuple[str, str]] = []

    time_suffix = re.compile(r"(_days|_time|_months|_years|_duration|_tenure)$", re.I)
    event_suffix = re.compile(r"(_event|_status|_censored|_indicator)$", re.I)

    def _stem(name: str, suffix_pat: re.Pattern[str]) -> str:
        # Strip the matched suffix repeatedly (handles ``overall_time_days`` →
        # ``overall_time`` → ``overall``).
        prev: str | None = None
        out = name
        while prev != out:
            prev = out
            out = suffix_pat.sub("", out)
        return out.lower()

    for t in times:
        t_stem = _stem(t, time_suffix)
        for e in events:
            e_stem = _stem(e, event_suffix)
            if not t_stem or not e_stem:
                continue
            match = (
                t_stem == e_stem
                or t_stem.startswith(e_stem)
                or e_stem.startswith(t_stem)
            )
            if match:
                pairs.append((t, e))
    return pairs


def _find_high_correlations(
    df: pd.DataFrame, threshold: float = 0.9
) -> list[tuple[str, str, float]]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr().abs()
    out: list[tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if pd.notna(v) and v >= threshold:
                out.append((a, b, float(round(v, 4))))
    return out


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """Run Stage 1b on a pandas DataFrame."""
    columns = [_profile_column(name, df[name]) for name in df.columns]
    return DatasetProfile(
        n_rows=len(df),
        n_cols=df.shape[1],
        columns=columns,
        column_pairs_high_corr=_find_high_correlations(df),
        censoring_pairs=_find_censoring_pairs(columns),
        missingness_clusters=_find_missingness_clusters(df),
        n_exact_duplicate_rows=_count_duplicate_rows(df, columns),
        string_formats=_infer_string_formats(df, columns),
    )


# --- Additional Stage 1b signals --------------------------------------------


def _find_missingness_clusters(
    df: pd.DataFrame, min_overlap_fraction: float = 0.05
) -> list[list[str]]:
    """Identify groups of columns that go missing together.

    Two columns ``a, b`` belong to the same cluster if the rate of rows where
    both are missing exceeds ``min_overlap_fraction`` AND that overlap is at
    least 70% of the union of their individual missing rows (i.e., they almost
    always go missing together).
    """
    miss = df.isna()
    nullable = [c for c in miss.columns if miss[c].any() and miss[c].mean() > 0.02]
    if len(nullable) < 2:
        return []

    n = len(df)
    adj: dict[str, set[str]] = {c: set() for c in nullable}
    for i, a in enumerate(nullable):
        for b in nullable[i + 1 :]:
            both = (miss[a] & miss[b]).sum()
            either = (miss[a] | miss[b]).sum()
            if either == 0:
                continue
            if both / n >= min_overlap_fraction and both / either >= 0.7:
                adj[a].add(b)
                adj[b].add(a)

    seen: set[str] = set()
    clusters: list[list[str]] = []
    for c in nullable:
        if c in seen:
            continue
        stack, group = [c], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            group.append(v)
            stack.extend(adj[v] - seen)
        if len(group) >= 2:
            clusters.append(sorted(group))
    return clusters


def _count_duplicate_rows(df: pd.DataFrame, columns: list[ColumnProfile]) -> int:
    """Count exact-duplicate rows after dropping identifier columns.

    PhD-level claims data routinely contains the same encounter multiple times;
    leaving duplicates in inflates effective sample size and breaks CI coverage.
    """
    id_cols = [c.name for c in columns if c.suspected_identifier]
    work = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")
    if work.empty or work.shape[1] == 0:
        return 0
    return int(work.duplicated().sum())


_DATE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}",
    r"^\d{2}/\d{2}/\d{4}",
    r"^\d{2}-\d{2}-\d{4}",
    r"^\d{4}/\d{2}/\d{2}",
]
_EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
_URL_PATTERN = r"^https?://"
_ZIP_PATTERN = r"^\d{5}(-\d{4})?$"
_UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"


def _infer_string_formats(
    df: pd.DataFrame, columns: list[ColumnProfile]
) -> dict[str, str]:
    """Per-text-column format inference (PDD §7.2).

    Tags each non-numeric, non-datetime column with one of:
    ``date_like``, ``email_like``, ``url_like``, ``zip_like``,
    ``identifier_like``, ``categorical_like``, ``free_text``.
    """
    out: dict[str, str] = {}
    for col in columns:
        if col.logical_dtype not in {"text", "categorical", "identifier"}:
            continue
        series = df[col.name].dropna().astype(str)
        if series.empty:
            continue
        sample = series.head(200)

        def _rate(pat: str) -> float:
            return float(sample.str.match(pat).mean())

        if _rate(_UUID_PATTERN) > 0.8:
            out[col.name] = "identifier_like"
            continue
        if _rate(_EMAIL_PATTERN) > 0.8:
            out[col.name] = "email_like"
            continue
        if _rate(_URL_PATTERN) > 0.8:
            out[col.name] = "url_like"
            continue
        if _rate(_ZIP_PATTERN) > 0.8:
            out[col.name] = "zip_like"
            continue
        if any(_rate(p) > 0.8 for p in _DATE_PATTERNS):
            out[col.name] = "date_like"
            continue
        # Free-text check runs BEFORE the high-cardinality identifier check:
        # 400 rows of unique long sentences would otherwise be tagged as
        # identifiers and dropped, but they're free text and should be flagged
        # accordingly so auto_preprocess drops them for the right reason.
        avg_len = float(sample.str.len().mean()) if not sample.empty else 0.0
        if avg_len > 30:
            out[col.name] = "free_text"
            continue
        if col.cardinality / max(col.n_total - col.n_missing, 1) > 0.95:
            out[col.name] = "identifier_like"
            continue
        out[col.name] = "categorical_like"
    return out
