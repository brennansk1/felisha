"""HypothesisQueue — ranked, scoped list of testable hypotheses (PDD §9).

The queue is a thin container with helpers for ranking, pinning, and
filtering. Persisted as ``protocol.hypothesis_queue``.
"""

from __future__ import annotations

from causalrag.core.protocol import Hypothesis


def rank_by_impact(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    return sorted(
        hypotheses,
        key=lambda h: -(h.impact_score or 0.0),
    )


def pin(hypotheses: list[Hypothesis], ids: set[str]) -> list[Hypothesis]:
    pinned = [h for h in hypotheses if h.id in ids]
    rest = [h for h in hypotheses if h.id not in ids]
    return pinned + rest


__all__ = ["rank_by_impact", "pin"]
