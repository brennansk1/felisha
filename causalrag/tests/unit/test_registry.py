from __future__ import annotations

import pytest

from causalrag.core.flags import DataFlag
from causalrag.core.registry import EstimatorEntry, Registry


def _entry(
    estimator_id: str,
    *,
    backend: str = "python",
    estimands: tuple[str, ...] = ("ATE",),
    required: frozenset[DataFlag] = frozenset(),
    excluded: frozenset[DataFlag] = frozenset(),
    min_n: int = 100,
) -> EstimatorEntry:
    return EstimatorEntry(
        id=estimator_id,
        factory=object,
        backend=backend,
        supported_estimands=frozenset(estimands),
        required_flags=required,
        excluded_flags=excluded,
        min_sample_size=min_n,
        produces_cate=False,
        produces_full_counterfactual=False,
        propensity_required=True,
    )


def test_register_and_get() -> None:
    r = Registry()
    r.register(_entry("a"))
    assert r.get("a").id == "a"


def test_duplicate_registration_raises() -> None:
    r = Registry()
    r.register(_entry("a"))
    with pytest.raises(ValueError, match="already registered"):
        r.register(_entry("a"))


def test_candidates_filter_by_estimand() -> None:
    r = Registry()
    r.register(_entry("ate-only", estimands=("ATE",)))
    r.register(_entry("rmst-only", estimands=("RMST_CONTRAST",)))
    out = r.candidates_for("ATE")
    assert {e.id for e in out} == {"ate-only"}


def test_required_flags_must_be_subset_of_situation() -> None:
    r = Registry()
    r.register(
        _entry(
            "needs-cens",
            required=frozenset({DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.BINARY_TREATMENT}),
        )
    )
    # Situation lacks BINARY_TREATMENT — should not qualify.
    assert (
        r.candidates_for("ATE", required={DataFlag.RIGHT_CENSORED_OUTCOME}) == ()
    )
    # Now both flags present.
    out = r.candidates_for(
        "ATE", required={DataFlag.RIGHT_CENSORED_OUTCOME, DataFlag.BINARY_TREATMENT}
    )
    assert {e.id for e in out} == {"needs-cens"}


def test_excluded_flags_disqualify() -> None:
    r = Registry()
    r.register(
        _entry("no-longitudinal", excluded=frozenset({DataFlag.LONGITUDINAL}))
    )
    assert (
        r.candidates_for("ATE", required={DataFlag.LONGITUDINAL}) == ()
    )
    assert {e.id for e in r.candidates_for("ATE")} == {"no-longitudinal"}


def test_backend_filter() -> None:
    r = Registry()
    r.register(_entry("py", backend="python"))
    r.register(_entry("r", backend="r"))
    assert {e.id for e in r.candidates_for("ATE", backends=("python",))} == {"py"}
    assert {e.id for e in r.candidates_for("ATE", backends=("r",))} == {"r"}


def test_min_sample_size_filter() -> None:
    r = Registry()
    r.register(_entry("big", min_n=500))
    r.register(_entry("small", min_n=50))
    assert {e.id for e in r.candidates_for("ATE", n=100)} == {"small"}
    assert {e.id for e in r.candidates_for("ATE", n=1000)} == {"big", "small"}


def test_bundled_linear_dml_is_registered() -> None:
    # Importing the estimators package triggers side-effect registration.
    import causalrag.estimators  # noqa: F401
    from causalrag.core.registry import get_registry

    reg = get_registry()
    ids = {e.id for e in reg.all()}
    assert "python.dml.linear" in ids
