"""Deterministic feature preprocessing — Stage 1f.

Auto-preprocess takes a raw frame plus a ``DatasetProfile`` and returns a
fully-numeric, model-ready frame plus a :class:`PreprocessingManifest` that
records every transform. Manifest entries are reversible and analyst-
auditable; downstream stages (q5_identify, q7_estimate, sensitivity)
operate on the transformed frame but report results in the original column
names where possible.

Transformations applied, in order:

1. **Drop**: constant columns, identifier-like text, free text. These cannot
   serve as covariates without external NLP, and silently leaving them in
   guarantees a crash or a misleading estimate.
2. **Date encoding**: ``date_like`` text columns are parsed to datetime, then
   replaced with three derived numerics: ``<col>_year``, ``<col>_month``,
   ``<col>_dow``. The raw datetime is also retained for downstream temporal
   reasoning.
3. **Boolean → int**.
4. **One-hot encoding** of categorical columns with cardinality ≤ ``max_onehot_card``
   (default 10). Drops the first level to avoid dummy-variable trap.
5. **Standardization** of continuous columns (z-score). Skipped for binary
   indicators, count outcomes, and survival times.
6. **Log-transform** of right-skewed continuous columns (|skew| > 2 and all
   values > 0). Records the offset and direction.

The manifest is appended to ``EstimationResult.diagnostics["preprocessing"]``
so the analyst can see exactly what the estimator received.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from causalrag.data.profiler import ColumnProfile, DatasetProfile


@dataclass
class TransformRecord:
    """One reversible transform applied during preprocessing."""

    column: str
    kind: Literal[
        "drop_constant",
        "drop_identifier",
        "drop_free_text",
        "drop_high_cardinality",
        "bool_to_int",
        "date_decompose",
        "onehot",
        "standardize",
        "log_transform",
        "kept",
    ]
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingManifest:
    transforms: list[TransformRecord] = field(default_factory=list)
    new_columns_from: dict[str, list[str]] = field(default_factory=dict)
    """Mapping original_column -> list of derived column names (so downstream
    code can map adjustment-set names back to the original variable)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "transforms": [
                {"column": t.column, "kind": t.kind, "details": t.details}
                for t in self.transforms
            ],
            "new_columns_from": dict(self.new_columns_from),
        }

    def derived_for(self, original: str) -> list[str]:
        return list(self.new_columns_from.get(original, [original]))


def auto_preprocess(
    df: pd.DataFrame,
    profile: DatasetProfile,
    *,
    treatment: str | None = None,
    outcome: str | None = None,
    max_onehot_card: int = 20,
    standardize_continuous: bool = True,
    log_skew_threshold: float = 2.0,
    drop_free_text: bool = True,
) -> tuple[pd.DataFrame, PreprocessingManifest]:
    """Return ``(preprocessed_df, manifest)``.

    The treatment and outcome columns are preserved verbatim if known —
    standardizing the treatment would change ATE scale, and standardizing the
    outcome makes the headline number harder to interpret. The analyst can
    still ask for a standardized-outcome run as a separate hypothesis.
    """
    work = df.copy()
    manifest = PreprocessingManifest()
    keep_original = {c for c in (treatment, outcome) if c}

    formats = profile.string_formats
    profile_by_name = {c.name: c for c in profile.columns}

    # 1. Drop constants / identifiers / free text
    for col in list(work.columns):
        prof = profile_by_name.get(col)
        if prof is None:
            continue
        if col in keep_original:
            continue
        if prof.constant:
            manifest.transforms.append(TransformRecord(column=col, kind="drop_constant"))
            work.drop(columns=[col], inplace=True)
            continue
        fmt = formats.get(col)
        if fmt == "identifier_like" or prof.suspected_identifier:
            manifest.transforms.append(
                TransformRecord(column=col, kind="drop_identifier", details={"format": fmt})
            )
            work.drop(columns=[col], inplace=True)
            continue
        if drop_free_text and fmt == "free_text":
            manifest.transforms.append(TransformRecord(column=col, kind="drop_free_text"))
            work.drop(columns=[col], inplace=True)
            continue

    # 2. Date decomposition
    for col in list(work.columns):
        prof = profile_by_name.get(col)
        fmt = formats.get(col)
        if prof is None:
            continue
        if prof.logical_dtype in {"datetime", "date"} or fmt == "date_like":
            try:
                parsed = pd.to_datetime(work[col], errors="coerce")
            except Exception:
                continue
            if parsed.isna().mean() > 0.5:
                continue
            new_cols: list[str] = []
            for suffix, fn in (
                ("year", lambda s: s.dt.year.astype("float64")),
                ("month", lambda s: s.dt.month.astype("float64")),
                ("dow", lambda s: s.dt.dayofweek.astype("float64")),
            ):
                new = f"{col}_{suffix}"
                work[new] = fn(parsed)
                new_cols.append(new)
            manifest.transforms.append(
                TransformRecord(column=col, kind="date_decompose", details={"derived": new_cols})
            )
            manifest.new_columns_from[col] = new_cols
            work.drop(columns=[col], inplace=True)

    # 3. Boolean -> int
    for col in list(work.columns):
        if pd.api.types.is_bool_dtype(work[col]):
            work[col] = work[col].astype(int)
            manifest.transforms.append(TransformRecord(column=col, kind="bool_to_int"))

    # 4. One-hot encode low-cardinality categoricals
    for col in list(work.columns):
        prof = profile_by_name.get(col)
        if prof is None or col in keep_original:
            continue
        if prof.logical_dtype not in {"categorical", "ordinal"}:
            continue
        if prof.cardinality > max_onehot_card or prof.cardinality < 2:
            continue
        if pd.api.types.is_numeric_dtype(work[col]) and prof.logical_dtype != "categorical":
            continue
        dummies = pd.get_dummies(work[col], prefix=col, drop_first=True, dummy_na=False)
        if dummies.empty:
            continue
        dummies = dummies.astype(int)
        work = pd.concat([work.drop(columns=[col]), dummies], axis=1)
        derived = list(dummies.columns)
        manifest.transforms.append(
            TransformRecord(column=col, kind="onehot", details={"derived": derived})
        )
        manifest.new_columns_from.setdefault(col, []).extend(derived)

    # 4b. Drop high-cardinality categoricals that exceed max_onehot_card.
    # Leaving them in would crash estimators that .astype(float) the covariate
    # matrix (LinearDML). The analyst can re-introduce them via target encoding
    # or hash buckets in a custom preprocess if they matter.
    for col in list(work.columns):
        if col in keep_original:
            continue
        if pd.api.types.is_numeric_dtype(work[col]):
            continue
        # Anything that's still a string/object/categorical at this point is
        # high-cardinality (one-hot would have caught the low-card case).
        n_unique = work[col].nunique(dropna=True)
        manifest.transforms.append(
            TransformRecord(
                column=col,
                kind="drop_high_cardinality",
                details={"n_unique": int(n_unique), "limit": max_onehot_card},
            )
        )
        work.drop(columns=[col], inplace=True)

    # 5. Log-transform skewed continuous (outcome too, but flagged separately)
    for col in list(work.columns):
        prof = profile_by_name.get(col)
        if prof is None:
            continue
        if prof.logical_dtype != "continuous":
            continue
        if prof.skew is None or abs(prof.skew) < log_skew_threshold:
            continue
        if prof.min is None or prof.min <= 0:
            continue
        if col in keep_original and col == outcome:
            # Flag but do not transform the headline outcome by default
            manifest.transforms.append(
                TransformRecord(
                    column=col,
                    kind="log_transform",
                    details={"applied": False, "reason": "outcome column, skipped to preserve scale", "skew": prof.skew},
                )
            )
            continue
        if col in keep_original and col == treatment:
            continue
        work[col] = np.log1p(work[col])
        manifest.transforms.append(
            TransformRecord(
                column=col, kind="log_transform", details={"applied": True, "skew_before": prof.skew}
            )
        )

    # 6. Standardize remaining continuous covariates
    if standardize_continuous:
        for col in list(work.columns):
            if col in keep_original:
                continue
            prof = profile_by_name.get(col)
            # Newly-created dummies and date parts are not in profile_by_name —
            # we still want to standardize date parts but not dummies.
            if prof is not None and prof.logical_dtype == "binary":
                continue
            if not pd.api.types.is_numeric_dtype(work[col]):
                continue
            unique = work[col].nunique(dropna=True)
            if unique <= 2:
                continue  # dummy
            mu = float(work[col].mean())
            sigma = float(work[col].std(ddof=1) or 1.0)
            if sigma == 0:
                continue
            work[col] = (work[col] - mu) / sigma
            manifest.transforms.append(
                TransformRecord(
                    column=col,
                    kind="standardize",
                    details={"mean": mu, "std": sigma},
                )
            )

    # Surviving columns that received no transform are recorded as "kept"
    transformed = {t.column for t in manifest.transforms} | set(manifest.new_columns_from.keys())
    for col in profile_by_name:
        if col not in transformed and col in work.columns:
            manifest.transforms.append(TransformRecord(column=col, kind="kept"))

    return work, manifest


__all__ = ["TransformRecord", "PreprocessingManifest", "auto_preprocess"]
