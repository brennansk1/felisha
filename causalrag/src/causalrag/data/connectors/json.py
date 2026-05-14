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
            first = p.read_text(encoding="utf-8")[:1024].lstrip()
            is_lines = first.startswith("{")  # ndjson starts with {, arrays with [

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
