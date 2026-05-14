"""ChainForestPanel — live view of completed walks grouped by chain.

Listens for ``LoopEvent(kind="card")`` payloads (one per completed
experiment). Groups by ``chain_id``, ordering children beneath their
parent via ``parent_id`` with indentation reflecting depth. Each node
renders as ``[hypothesis_id] T → Y · sensitivity_color``.
"""

from __future__ import annotations

from typing import Any

from rich.console import Group, RenderableType
from rich.text import Text
from textual.widgets import Static


_SENSITIVITY_STYLE: dict[str, str] = {
    "green": "#7ed2e6",
    "yellow": "#a3b6da",
    "red": "#e08877",
    "errored": "#e08877",
    "unknown": "#4d5773",
}


def _sensitivity_style(verdict: str | None) -> str:
    return _SENSITIVITY_STYLE.get((verdict or "").lower(), "#9aa3b5")


class ChainForestPanel(Static):
    """Indented forest of completed walks grouped by chain."""

    DEFAULT_CSS = """
    ChainForestPanel {
        height: auto;
        max-height: 16;
        padding: 0 1;
        border: round #4d5773;
    }
    """

    def __init__(self) -> None:
        super().__init__("", markup=False)
        # List of rows preserves insertion order (deterministic, matches
        # the master loop's emission order — roots first, children next).
        self._rows: list[dict[str, Any]] = []

    def on_mount(self) -> None:
        self._refresh()

    # --- Public API ------------------------------------------------------

    def update_panel(self, payload: dict[str, Any]) -> None:
        """Apply one ``LoopEvent(kind="card")`` payload."""
        if not isinstance(payload, dict):
            return
        hid = payload.get("id") or payload.get("hypothesis_id")
        if not hid:
            return
        # Replace if same id already present (idempotent on re-emits).
        for i, r in enumerate(self._rows):
            if r.get("id") == hid:
                self._rows[i] = payload
                self._refresh()
                return
        self._rows.append(dict(payload))
        self._refresh()

    def update(  # type: ignore[override]
        self, renderable: RenderableType | dict[str, Any] | None = None
    ) -> None:
        if isinstance(renderable, dict):
            self.update_panel(renderable)
            return
        super().update(renderable if renderable is not None else "")

    def render_panel(self) -> RenderableType:
        if not self._rows:
            return Text(
                "chains · waiting for first result …",
                style="#4d5773",
            )
        # Group rows by chain_id (None / falsy → standalone chain rooted at the row itself).
        chains: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for r in self._rows:
            cid = str(r.get("chain_id") or r.get("id") or "?")
            if cid not in chains:
                chains[cid] = []
                order.append(cid)
            chains[cid].append(r)

        out_lines: list[Text] = []
        for cid in order:
            members = chains[cid]
            # Compute depth per row by walking parent_id back inside this chain.
            id_to_parent: dict[str, str | None] = {
                str(m.get("id")): (str(m["parent_id"]) if m.get("parent_id") else None)
                for m in members
            }

            def _depth(row_id: str) -> int:
                d = 0
                cur = id_to_parent.get(row_id)
                seen: set[str] = set()
                while cur and cur not in seen:
                    seen.add(cur)
                    d += 1
                    cur = id_to_parent.get(cur)
                return d

            for m in members:
                hid = str(m.get("id") or "?")
                depth = _depth(hid)
                indent = "  " * depth
                connector = "└─ " if depth > 0 else ""
                verdict = m.get("sensitivity_verdict")
                line = Text()
                line.append(indent, style="#4d5773")
                if connector:
                    line.append(connector, style="#4d5773")
                line.append(f"[{hid}] ", style="#9ec2ff bold")
                line.append(
                    f"{m.get('treatment', '?')} → {m.get('outcome', '?')}",
                    style="#cfd6e4",
                )
                line.append("  ·  ", style="#4d5773")
                line.append(
                    f"● {verdict or 'unknown'}",
                    style=_sensitivity_style(verdict),
                )
                out_lines.append(line)
        return Group(*out_lines)

    # --- Internal --------------------------------------------------------

    def _refresh(self) -> None:
        super().update(self.render_panel())


__all__ = ["ChainForestPanel"]
