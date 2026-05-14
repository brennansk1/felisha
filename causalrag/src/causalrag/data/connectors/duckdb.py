"""DuckDB connector — read tables and arbitrary SQL queries via DuckDB.

Two usage modes:

1. **Database file**: point at a ``.duckdb`` file and read a named table.
2. **Ad-hoc SQL**: pass a ``query=`` string. DuckDB can read CSV / Parquet /
   JSON / Arrow / S3 / HTTP transparently — the user's SQL is run inside an
   in-memory DuckDB session that has all of those readers available.

Examples::

    DuckDBConnector("warehouse.duckdb", table="patients")
    DuckDBConnector(query="SELECT * FROM 'data/*.parquet' WHERE year >= 2020")
    DuckDBConnector(query="SELECT * FROM read_csv_auto('https://...')")

DuckDB is a soft dependency; the connector raises a clear ``RuntimeError``
when it isn't installed, naming the ``duckdb`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa


@dataclass
class DuckDBConnector:
    path: str | Path | None = None
    table: str | None = None
    query: str | None = None

    def __post_init__(self) -> None:
        if self.query is None and self.table is None:
            raise ValueError(
                "DuckDBConnector needs either query=... or (path=..., table=...)"
            )
        if self.query is None and self.path is None:
            raise ValueError("table mode requires path= pointing to a .duckdb file")

    def _connect(self):
        try:
            import duckdb
        except ImportError as e:
            raise RuntimeError(
                "DuckDB connector requires the 'duckdb' extra: pip install 'causalrag[duckdb]'"
            ) from e
        # An empty/None path opens an in-memory DB; the query may reference
        # files via DuckDB's read_csv / read_parquet / read_json functions.
        return duckdb.connect(str(self.path) if self.path else ":memory:")

    def to_arrow(self) -> pa.Table:
        con = self._connect()
        try:
            if self.query is not None:
                rel = con.sql(self.query)
            else:
                # Validate table name to avoid SQL-injection on the metadata path.
                if self.table is None or not self.table.replace("_", "").isalnum():
                    raise ValueError(
                        f"Unsafe table name: {self.table!r}. Use only alphanumerics + underscore "
                        "or pass an explicit query=..."
                    )
                rel = con.sql(f'SELECT * FROM "{self.table}"')
            arrow_obj = rel.arrow()
            # DuckDB 1.x's .arrow() returns a RecordBatchReader rather than a
            # Table; .read_all() materializes the full table.
            if hasattr(arrow_obj, "read_all"):
                return arrow_obj.read_all()
            return arrow_obj
        finally:
            con.close()

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "source": f"duckdb://{self.path or ':memory:'}",
            "mode": "query" if self.query else "table",
        }
        if self.table:
            info["table"] = self.table
        if self.query:
            info["query"] = self.query[:200]
        if self.path:
            p = Path(self.path)
            info["size_bytes"] = p.stat().st_size if p.exists() else None
        return info

    def supports_lazy(self) -> bool:
        return True  # DuckDB pushes predicates down naturally
