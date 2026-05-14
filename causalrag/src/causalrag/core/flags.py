"""DataFlag enum — routing brain for estimator dispatch.

See PDD §14.3 and §15. Flag emission is the responsibility of the profiler and
discovery agent; flag consumption is the responsibility of the estimator registry.
"""

from __future__ import annotations

from enum import StrEnum


class DataFlag(StrEnum):
    # Treatment properties
    BINARY_TREATMENT = "binary_treatment"
    CATEGORICAL_TREATMENT = "categorical_treatment"
    CONTINUOUS_TREATMENT = "continuous_treatment"
    MIXTURE_EXPOSURE = "mixture_exposure"
    TIME_VARYING_TREATMENT = "time_varying_treatment"

    # Outcome properties
    BINARY_OUTCOME = "binary_outcome"
    CONTINUOUS_OUTCOME = "continuous_outcome"
    COUNT_OUTCOME = "count_outcome"
    RIGHT_CENSORED_OUTCOME = "right_censored_outcome"
    COMPETING_RISKS = "competing_risks"
    REPEATED_OUTCOME = "repeated_outcome"
    RARE_OUTCOME = "rare_outcome"
    BOUNDED_OUTCOME = "bounded_outcome"
    ZERO_INFLATED_OUTCOME = "zero_inflated_outcome"

    # Treatment-balance refinements
    IMBALANCED_TREATMENT = "imbalanced_treatment"

    # Data structure
    PANEL_STRUCTURE = "panel_structure"
    LONGITUDINAL = "longitudinal"
    CLUSTERED = "clustered"
    NETWORK_INTERFERENCE = "network_interference"
    SINGLE_TREATED_UNIT = "single_treated_unit"
    DIFF_IN_DIFF_CANDIDATE = "diff_in_diff_candidate"
    STAGGERED_ADOPTION = "staggered_adoption"

    # Design hints
    INSTRUMENTAL_CANDIDATE_PRESENT = "instrumental_candidate_present"
    MEDIATOR_PROPOSED = "mediator_proposed"
    NEGATIVE_CONTROL_AVAILABLE = "negative_control_available"
    EFFECT_MODIFICATION_OF_INTEREST = "effect_modification_of_interest"

    # Sample properties
    SMALL_SAMPLE = "small_sample"
    HIGH_DIMENSIONAL = "high_dimensional"
    HEAVY_MISSINGNESS = "heavy_missingness"
    HEAVY_CENSORING = "heavy_censoring"
    POSITIVITY_VIOLATION = "positivity_violation"
    SUSPECTED_INFORMATIVE_CENSORING = "suspected_informative_censoring"

    # Hypothesis-local refinements (§15.3)
    CROSS_SECTIONAL_SLICE = "cross_sectional_slice"

    # Phase 4 Step-5 sentinel
    IDENTIFICATION_FAILED = "identification_failed"


TREATMENT_FLAGS: frozenset[DataFlag] = frozenset(
    {
        DataFlag.BINARY_TREATMENT,
        DataFlag.CATEGORICAL_TREATMENT,
        DataFlag.CONTINUOUS_TREATMENT,
        DataFlag.MIXTURE_EXPOSURE,
        DataFlag.TIME_VARYING_TREATMENT,
    }
)

OUTCOME_FLAGS: frozenset[DataFlag] = frozenset(
    {
        DataFlag.BINARY_OUTCOME,
        DataFlag.CONTINUOUS_OUTCOME,
        DataFlag.COUNT_OUTCOME,
        DataFlag.RIGHT_CENSORED_OUTCOME,
        DataFlag.REPEATED_OUTCOME,
    }
)


def validate_flag_set(flags: set[DataFlag] | frozenset[DataFlag]) -> None:
    """Raise ValueError if the set is inconsistent (e.g., multiple treatment types).

    Inheritance refinements (e.g., LONGITUDINAL → CROSS_SECTIONAL_SLICE) are
    handled elsewhere; this function only blocks mutually-exclusive primary kinds.
    """
    tx = flags & TREATMENT_FLAGS
    if len(tx) > 1:
        raise ValueError(
            f"At most one treatment-type flag is allowed; got {sorted(f.value for f in tx)}"
        )
    out = flags & OUTCOME_FLAGS
    if len(out) > 1:
        raise ValueError(
            f"At most one outcome-type flag is allowed; got {sorted(f.value for f in out)}"
        )
