"""JSON / JSONL connector — PDD §7.1.

Uses :mod:`pyarrow.json` for line-delimited JSON (one record per line). For
non-line-delimited JSON arrays we fall back to pandas. Auto-flattens nested
records by reading via pandas json_normalize when nesting is detected; we
warn rather than auto-flatten silently for cassette-driven reproducibility.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa


def _looks_like_ndjson(p: Path) -> bool:
    """Heuristically decide whether a JSON file is line-delimited (ndjson).

    A pretty-printed single object also starts with ``{`` but spans multiple
    lines, so the leading character alone is ambiguous. Instead we require
    that the first two non-empty lines each parse as a complete JSON value —
    the defining property of ndjson. A top-level array (``[``) is never
    ndjson.
    """
    head = p.read_text(encoding="utf-8")[:8192].lstrip()
    if not head or head[0] == "[":
        return False

    non_empty = [ln for ln in head.splitlines() if ln.strip()]
    if not non_empty:
        return False

    # A single complete JSON value spanning one line is ndjson; if the first
    # line alone fails to parse, it is almost certainly a pretty-printed
    # (multi-line) single object/array, not ndjson.
    try:
        json.loads(non_empty[0])
    except ValueError:
        return False
    # If there is a second line, it should also parse on its own.
    if len(non_empty) >= 2:
        try:
            json.loads(non_empty[1])
        except ValueError:
            return False
    return True


@dataclass
class JSONConnector:
    path: str | Path
    lines: bool | None = None  # auto-detect by default

    def to_arrow(self) -> pa.Table:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {p}")

        is_lines = self.lines
        if is_lines is None:
            is_lines = _looks_like_ndjson(p)

        if is_lines:
            import pyarrow.json as pa_json

            return pa_json.read_json(p)

        # Array-of-records JSON: load via pandas + Arrow conversion
        import pandas as pd

        with p.open() as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        # Flatten one level of nesting
        if data and isinstance(data[0], dict) and any(
            isinstance(v, (dict, list)) for v in data[0].values()
        ):
            warnings.warn(
                "JSONConnector: nested records detected; flattening one level via json_normalize.",
                stacklevel=2,
            )
            df = pd.json_normalize(data)
        else:
            df = pd.DataFrame(data)
        return pa.Table.from_pandas(df)

    def describe(self) -> dict[str, Any]:
        p = Path(self.path)
        return {
            "source": f"json://{p}",
            "size_bytes": p.stat().st_size if p.exists() else None,
            "lines": self.lines,
        }

    def supports_lazy(self) -> bool:
        return False
