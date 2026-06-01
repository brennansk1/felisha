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
        # Read only the schema/footer, not the full table. Feather V2 is
        # Arrow IPC file format, so open_file exposes the schema without
        # materializing any record batches. Fall back to a zero-column read
        # for legacy V1 files that open_file can't parse.
        try:
            import pyarrow.ipc as pa_ipc

            with pa.memory_map(str(p), "r") as source:
                schema = pa_ipc.open_file(source).schema
        except (pa.ArrowInvalid, OSError):
            schema = pa_feather.read_table(p, columns=[]).schema
        return {
            "source": f"feather://{p}",
            "num_columns": len(schema.names),
            "size_bytes": p.stat().st_size,
        }

    def supports_lazy(self) -> bool:
        return False
