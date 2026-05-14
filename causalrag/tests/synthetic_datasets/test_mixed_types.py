"""Mixed-type dataset — auto_preprocess pipeline stress test.

Strings, dates, booleans, free text, identifiers, skewed continuous — the
pipeline must drop what it can't use, encode what it can, and never crash
the downstream estimator on a string column.
"""

from __future__ import annotations

import pytest

from causalrag.core.estimand import CausalEstimand, EstimandClass
from causalrag.core.flags import DataFlag
from causalrag.core.graph import CausalGraph
from causalrag.core.protocol import StudyProtocol
from causalrag.core.roles import VariableRole
from causalrag.data.features import auto_preprocess
from causalrag.data.profiler import profile_dataframe
from causalrag.roadmap.q5_identify import IdentificationResult
from causalrag.roadmap.q7_estimate import estimate

pytestmark = pytest.mark.integration


def test_preprocessing_handles_strings_dates_booleans(mixed_types_dataset) -> None:
    p = profile_dataframe(mixed_types_dataset)
    out, manifest = auto_preprocess(mixed_types_dataset, p, treatment="treat", outcome="y")
    kinds = {t.kind for t in manifest.transforms}
    # Identifier dropped
    assert "drop_identifier" in kinds
    # Boolean → int
    assert "bool_to_int" in kinds
    # One-hot for the site column
    onehot_records = [t for t in manifest.transforms if t.kind == "onehot"]
    assert any(t.column == "site" for t in onehot_records)
    # All remaining columns are numeric (i.e., no string columns survived)
    assert all(out[c].dtype.kind in {"i", "u", "f", "b"} for c in out.columns), (
        f"Non-numeric columns survived preprocessing: "
        f"{[c for c in out.columns if out[c].dtype.kind not in 'iufb']}"
    )


def test_estimate_doesnt_crash_on_mixed_types(mixed_types_dataset) -> None:
    """End-to-end: estimate must run without error on a frame that originally
    contained strings and dates."""
    covariates = ("age", "income", "site", "active")
    estimand = CausalEstimand.model_validate(
        {
            "class": EstimandClass.ATE,
            "treatment": "treat",
            "outcome": "y",
            "formal_expression": "E[Y(1)-Y(0)]",
        }
    )
    graph = CausalGraph.from_edge_list(
        [(c, "treat") for c in covariates] + [(c, "y") for c in covariates] + [("treat", "y")],
        roles={
            **{c: VariableRole.CONFOUNDER for c in covariates},
            "treat": VariableRole.TREATMENT,
            "y": VariableRole.OUTCOME,
        },
    )
    ident = IdentificationResult(identifiable=True, strategy="backdoor")
    result = estimate(
        df=mixed_types_dataset,
        estimand=estimand,
        identification=ident,
        protocol=StudyProtocol(name="mixed"),
        confounders=covariates,
        flags={DataFlag.BINARY_TREATMENT, DataFlag.CONTINUOUS_OUTCOME},
        selection="none",
    )
    assert result.estimator_id  # something ran
    assert result.diagnostics.get("preprocessing") is not None
    # Identifier was dropped, free-text 'notes' was dropped
    transforms = result.diagnostics["preprocessing"]["transforms"]
    kinds = {t["kind"] for t in transforms}
    assert "drop_identifier" in kinds
    assert "drop_free_text" in kinds
