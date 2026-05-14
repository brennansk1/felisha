"""Parquet connector — PDD §7.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pa_pq


@dataclass
class ParquetConnector:
    path: str | Path

    def to_arrow(self) -> pa.Table:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Parquet not found: {p}")
        return pa_pq.read_table(p)

    def describe(self) -> dict[str, Any]:
        p = Path(self.path)
        if not p.exists():
            return {"source": f"parquet://{p}", "exists": False}
        meta = pa_pq.read_metadata(p)
        return {
            "source": f"parquet://{p}",
            "num_rows": meta.num_rows,
            "num_columns": meta.num_columns,
            "size_bytes": p.stat().st_size,
        }

    def supports_lazy(self) -> bool:
        return True
