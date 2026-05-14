"""Excel connector — PDD §7.1.

Loads .xlsx / .xls workbooks via pandas (openpyxl or calamine). Supports
sheet selection via ``ExcelConnector(path, sheet=...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa


@dataclass
class ExcelConnector:
    path: str | Path
    sheet: str | int = 0

    def to_arrow(self) -> pa.Table:
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("ExcelConnector requires pandas") from e
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Excel file not found: {p}")
        try:
            df = pd.read_excel(p, sheet_name=self.sheet)
        except ImportError as e:
            raise RuntimeError(
                "Reading Excel needs an engine — pip install openpyxl"
            ) from e
        return pa.Table.from_pandas(df)

    def describe(self) -> dict[str, Any]:
        p = Path(self.path)
        return {
            "source": f"excel://{p}",
            "sheet": self.sheet,
            "size_bytes": p.stat().st_size if p.exists() else None,
        }

    def supports_lazy(self) -> bool:
        return False
