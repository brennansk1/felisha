"""Tests for ``catalog_markdown(flags=...)`` filtering.

The catalog table that is injected into LLM prompts should be filtered to
the estimators that are actually compatible with the active DataFlag set,
so the LLM never sees methods that don't apply to the current protocol.
"""

from __future__ import annotations

from causalrag.core.flags import DataFlag
from causalrag.estimators.catalog import CATALOG, catalog_markdown


def _rendered_ids(markdown: str) -> set[str]:
    """Pull the estimator ids out of a catalog markdown table."""
    ids: set[str] = set()
    for line in markdown.splitlines():
        # Each data row starts with "| `<estimator_id>` |"
        if not line.startswith("| `"):
            continue
        # take the token between the first backticks
        first = line.find("`")
        last = line.find("`", first + 1)
        if first == -1 or last == -1:
            continue
        ids.add(line[first + 1 : last])
    return ids


def test_no_flags_returns_full_catalog() -> None:
    """Default (flags=None) is backward compatible — every spec is rendered."""
    rendered = _rendered_ids(catalog_markdown())
    expected = {spec.estimator_id for spec in CATALOG}
    assert rendered == expected


def test_empty_flag_set_returns_full_catalog_minus_required_only_specs() -> None:
    """An explicitly empty active set keeps only entries with no required flags.

    Estimators that require a specific flag (BINARY_TREATMENT, CONTINUOUS_TREATMENT,
    RIGHT_CENSORED_OUTCOME, etc.) should drop out; unconditional ones stay.
    """
    rendered = _rendered_ids(catalog_markdown(flags=set()))
    expected = {spec.estimator_id for spec in CATALOG if not spec.required_flags}
    assert rendered == expected
    # Sanity: linear OLS has no required flags — kept.
    assert "python.linear.ols" in rendered
    # Sanity: survRM2 requires BINARY_TREATMENT+RIGHT_CENSORED — dropped.
    assert "rbridge.survrm2" not in rendered


def test_binary_treatment_only_excludes_rmst_estimators() -> None:
    """With only BINARY_TREATMENT active, RMST/survival-only specs are filtered out.

    survRM2 *requires* RIGHT_CENSORED_OUTCOME, so it must not appear.
    causal_survival_forest also requires RIGHT_CENSORED_OUTCOME — also out.
    """
    flags = {DataFlag.BINARY_TREATMENT}
    rendered = _rendered_ids(catalog_markdown(flags=flags))

    # survRM2 needs RIGHT_CENSORED_OUTCOME → must be excluded
    assert "rbridge.survrm2" not in rendered
    # causal_survival_forest needs RIGHT_CENSORED_OUTCOME → must be excluded
    assert "rbridge.grf.causal_survival_forest" not in rendered

    # No row in the filtered table should produce the RMST_CONTRAST estimand
    # exclusively (i.e., survRM2-style entries).
    for spec in CATALOG:
        if spec.estimands == ("RMST_CONTRAST",):
            assert spec.estimator_id not in rendered

    # Sanity: matchit DOES require BINARY_TREATMENT and excludes RIGHT_CENSORED.
    # With BINARY_TREATMENT active and RIGHT_CENSORED not active, it should stay.
    assert "rbridge.matchit" in rendered


def test_right_censored_outcome_keeps_survival_forest_drops_matchit() -> None:
    """RIGHT_CENSORED_OUTCOME active:
    - grf.causal_survival_forest (requires it) must appear.
    - rbridge.matchit (excludes RIGHT_CENSORED_OUTCOME) must NOT appear.
    """
    flags = {DataFlag.RIGHT_CENSORED_OUTCOME}
    rendered = _rendered_ids(catalog_markdown(flags=flags))

    assert "rbridge.grf.causal_survival_forest" in rendered
    assert "rbridge.matchit" not in rendered

    # Also: python.linear.ols excludes RIGHT_CENSORED_OUTCOME → must be filtered.
    assert "python.linear.ols" not in rendered


def test_backends_filter_still_works_with_flags() -> None:
    """The existing ``backends`` filter composes cleanly with ``flags``."""
    rendered = _rendered_ids(
        catalog_markdown(backends=("python",), flags={DataFlag.BINARY_TREATMENT})
    )
    # Every rendered id must come from the python backend.
    by_id = {spec.estimator_id: spec for spec in CATALOG}
    for est_id in rendered:
        assert by_id[est_id].backend == "python"


def test_filter_logic_matches_spec() -> None:
    """Property check: an entry is rendered iff (required ⊆ active) AND
    (excluded ∩ active == ∅)."""
    active: set[DataFlag] = {
        DataFlag.BINARY_TREATMENT,
        DataFlag.RIGHT_CENSORED_OUTCOME,
    }
    rendered = _rendered_ids(catalog_markdown(flags=active))
    for spec in CATALOG:
        req_ok = (not spec.required_flags) or set(spec.required_flags).issubset(active)
        exc_ok = not (set(spec.excluded_flags) & active)
        should_be_in = req_ok and exc_ok
        assert (spec.estimator_id in rendered) is should_be_in, spec.estimator_id
