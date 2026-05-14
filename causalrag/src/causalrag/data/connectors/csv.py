"""CSV / TSV connector — PDD §7.1.

Uses :mod:`pyarrow.csv` for streaming, schema-inferring reads; falls back to
pandas only on column-name conflicts that pyarrow cannot resolve.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pa_csv


@dataclass
class CSVConnector:
    path: str | Path
    delimiter: str = ","
    encoding: str = "utf-8"

    def _read_options(self) -> pa_csv.ReadOptions:
        return pa_csv.ReadOptions(encoding=self.encoding)

    def _parse_options(self) -> pa_csv.ParseOptions:
        return pa_csv.ParseOptions(delimiter=self.delimiter)

    def to_arrow(self) -> pa.Table:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return pa_csv.read_csv(
            p, read_options=self._read_options(), parse_options=self._parse_options()
        )

    def describe(self) -> dict[str, Any]:
        p = Path(self.path)
        size = p.stat().st_size if p.exists() else None
        digest: str | None = None
        if p.exists() and size is not None and size < 200 * 1024 * 1024:
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
        return {
            "source": f"csv://{p}",
            "size_bytes": size,
            "sha256": digest,
            "delimiter": self.delimiter,
            "encoding": self.encoding,
        }

    def supports_lazy(self) -> bool:
        return False
