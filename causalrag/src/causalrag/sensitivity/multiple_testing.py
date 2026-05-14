"""Multiple-testing adjustment across the K experiments the master loop runs.

The :class:`StudyProtocol` declares a ``multiple_testing`` field — one of
``bh`` / ``by`` / ``bonferroni`` / ``none`` — but no other module reads it.
This helper closes that gap: after the master loop completes, call
:func:`adjust_protocol_p_values` to write an ``adjusted_p_value`` and
``adjustment_method`` entry onto each walk's last
:class:`EstimationResult`'s ``diagnostics`` dict, plus a per-walk summary
the reporting layer can surface to the LLM.

The function is **idempotent** — calling it twice produces the same
adjusted values and does not double-log to the decision ledger.
"""

from __future__ import annotations

from typing import Any

from causalrag.core.ledger import record_decision
from causalrag.core.protocol import StudyProtocol


_METHOD_TO_STATSMODELS: dict[str, str] = {
    "bh": "fdr_bh",
    "by": "fdr_by",
    "bonferroni": "bonferroni",
}


def _walk_p_values(protocol: StudyProtocol) -> list[tuple[str, float]]:
    """Collect (walk_id, raw_p) for walks with a non-None last p_value."""
    pairs: list[tuple[str, float]] = []
    for walk_id, walk in protocol.roadmap_walks.items():
        if not walk.q7_estimates:
            continue
        est = walk.q7_estimates[-1]
        if est.p_value is None:
            continue
        pairs.append((walk_id, float(est.p_value)))
    return pairs


def _ledger_already_records(protocol: StudyProtocol, method: str, k: int) -> bool:
    """Return True if a prior identical multiple-testing decision was logged."""
    for entry in protocol.decision_ledger:
        if (
            entry.phase == "multiple_testing"
            and entry.chose == method
            and entry.note is not None
            and f"k={k}" in entry.note
        ):
            return True
    return False


def adjust_protocol_p_values(
    protocol: StudyProtocol,
) -> tuple[StudyProtocol, dict[str, dict[str, float]]]:
    """Apply ``protocol.multiple_testing`` across all walks.

    Returns the same protocol (mutated) plus a per-walk dict of
    ``{walk_id: {"raw_p": float, "adjusted_p": float}}`` for reporting.

    Behaviour:

    * ``"none"`` is a pass-through — adjusted equals raw.
    * Other methods dispatch to ``statsmodels.stats.multitest.multipletests``
      with method ``fdr_bh`` / ``fdr_by`` / ``bonferroni``.
    * Each adjusted walk's last ``EstimationResult.diagnostics`` gains two
      keys: ``adjusted_p_value`` and ``adjustment_method``.
    * A single ``Decision`` entry is appended to ``protocol.decision_ledger``
      recording method + number of comparisons. Re-invocation does not
      append a duplicate entry.
    """
    method = protocol.multiple_testing
    pairs = _walk_p_values(protocol)
    summary: dict[str, dict[str, float]] = {}

    if not pairs:
        # Nothing to do, but still record a ledger entry (idempotently)
        # so the audit trail reflects that adjustment was considered.
        if not _ledger_already_records(protocol, method, 0):
            record_decision(
                protocol,
                phase="multiple_testing",
                decision=f"multiple-testing adjustment ({method})",
                chose=method,
                source="auto",
                note="k=0 comparisons — no p-values to adjust",
            )
        return protocol, summary

    walk_ids = [w for w, _ in pairs]
    raw_ps = [p for _, p in pairs]

    if method == "none":
        adjusted = list(raw_ps)
    else:
        sm_method = _METHOD_TO_STATSMODELS.get(method)
        if sm_method is None:  # pragma: no cover — pydantic Literal guards this
            raise ValueError(f"unknown multiple_testing method: {method!r}")
        # Import locally so the rest of the package doesn't pay the cost
        # at import time.
        from statsmodels.stats.multitest import multipletests

        _, adjusted_arr, _, _ = multipletests(raw_ps, method=sm_method)
        adjusted = [float(x) for x in adjusted_arr]

    for walk_id, raw_p, adj_p in zip(walk_ids, raw_ps, adjusted, strict=True):
        walk = protocol.roadmap_walks[walk_id]
        est = walk.q7_estimates[-1]
        diag: dict[str, Any] = dict(est.diagnostics or {})
        diag["adjusted_p_value"] = float(adj_p)
        diag["adjustment_method"] = method
        est.diagnostics = diag
        summary[walk_id] = {"raw_p": float(raw_p), "adjusted_p": float(adj_p)}

    if not _ledger_already_records(protocol, method, len(pairs)):
        record_decision(
            protocol,
            phase="multiple_testing",
            decision=f"multiple-testing adjustment ({method})",
            chose=method,
            source="auto",
            note=f"k={len(pairs)} comparisons adjusted",
        )

    return protocol, summary


__all__ = ["adjust_protocol_p_values"]
