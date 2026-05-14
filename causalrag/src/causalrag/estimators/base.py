"""CausalEstimator Protocol — every estimator (Python or R-bridged) honors this.

PDD §14.2. Concrete implementations live under ``estimators/python/`` and
``estimators/rbridge/``.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from causalrag.core.flags import DataFlag
from causalrag.core.protocol import StudyProtocol
from causalrag.core.result import EstimationResult


@runtime_checkable
class CausalEstimator(Protocol):
    id: str
    backend: Literal["python", "r"]
    supported_estimands: tuple[str, ...]
    required_flags: frozenset[DataFlag]
    excluded_flags: frozenset[DataFlag]
    min_sample_size: int
    produces_cate: bool
    produces_full_counterfactual: bool
    propensity_required: bool

    def fit(self, data: pd.DataFrame, protocol: StudyProtocol) -> CausalEstimator: ...

    def estimate(self) -> EstimationResult: ...

    def diagnose(self) -> dict: ...

    def refute(self) -> dict: ...
