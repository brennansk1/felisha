from __future__ import annotations

import pytest

from causalrag.core.flags import DataFlag, validate_flag_set


def test_dataflag_values_are_lowercase_snake_case() -> None:
    for member in DataFlag:
        assert member.value == member.value.lower()
        assert " " not in member.value


def test_validate_flag_set_accepts_consistent_combo() -> None:
    validate_flag_set({DataFlag.BINARY_TREATMENT, DataFlag.RIGHT_CENSORED_OUTCOME})


def test_validate_flag_set_rejects_double_treatment() -> None:
    with pytest.raises(ValueError, match="treatment-type"):
        validate_flag_set({DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_TREATMENT})


def test_validate_flag_set_rejects_double_outcome() -> None:
    with pytest.raises(ValueError, match="outcome-type"):
        validate_flag_set({DataFlag.BINARY_OUTCOME, DataFlag.CONTINUOUS_OUTCOME})
