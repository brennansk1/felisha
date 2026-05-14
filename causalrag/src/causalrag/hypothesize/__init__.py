"""Phase 3 — hypothesis generation (PDD §9)."""

from causalrag.hypothesize.automated import (
    HypothesisProposal,
    HypothesisQueue,
    deterministic_proposals,
    proposals_to_hypotheses,
    run_automated,
)
from causalrag.hypothesize.manual import from_pairs
from causalrag.hypothesize.queue import pin, rank_by_impact

__all__ = [
    "HypothesisProposal",
    "HypothesisQueue",
    "deterministic_proposals",
    "from_pairs",
    "pin",
    "proposals_to_hypotheses",
    "rank_by_impact",
    "run_automated",
]
