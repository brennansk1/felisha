"""Decision-ledger helpers — append entries without races.

Both the CLI and TUI mutate ``protocol.decision_ledger``. Bundling the logic
in one place keeps the audit trail consistent across surfaces.
"""

from __future__ import annotations

from datetime import UTC, datetime

from causalrag.core.protocol import Decision, Override, StudyProtocol


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def record_decision(
    protocol: StudyProtocol,
    phase: str,
    decision: str,
    chose: str,
    source: str = "default",
    note: str | None = None,
) -> None:
    entry = Decision(
        timestamp=_now(),
        phase=phase,
        decision=decision,
        chose=chose,
        source=source,  # type: ignore[arg-type]
        note=note,
    )
    protocol.decision_ledger = tuple(list(protocol.decision_ledger) + [entry])


def record_override(
    protocol: StudyProtocol,
    site: str,
    llm_value,
    analyst_value,
    reason: str | None = None,
) -> None:
    entry = Override(
        timestamp=_now(),
        site=site,
        llm_value=llm_value,
        analyst_value=analyst_value,
        reason=reason,
    )
    protocol.overrides = tuple(list(protocol.overrides) + [entry])


__all__ = ["record_decision", "record_override"]
