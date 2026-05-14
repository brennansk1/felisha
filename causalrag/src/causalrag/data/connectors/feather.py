"""Feather / Arrow IPC connector — PDD §7.1.

Native zero-copy read via :mod:`pyarrow.feather`. The connector treats
``.feather`` and ``.arrow`` extensions interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.feather as pa_feather


@dataclass
class FeatherConnector:
    path: str | Path

    def to_arrow(self) -> pa.Table:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Feather/Arrow file not found: {p}")
        return pa_feather.read_table(p)

    def describe(self) -> dict[str, Any]:
        p = Path(self.path)
        if not p.exists():
            return {"source": f"feather://{p}", "exists": False}
        # Feather files don't expose metadata without reading the schema
        schema = pa_feather.read_table(p, columns=None).schema
        return {
            "source": f"feather://{p}",
            "num_columns": len(schema.names),
            "size_bytes": p.stat().st_size,
        }

    def supports_lazy(self) -> bool:
        return False
