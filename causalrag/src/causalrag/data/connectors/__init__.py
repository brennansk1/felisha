"""Data ingestion connectors (PDD §7.1, Stage 1a).

Every connector implements the :class:`Connector` Protocol. Connectors are
lazy-imported so optional dependencies (DuckDB, SQLAlchemy, openpyxl) surface
as actionable errors only when the user actually tries that source.

URI dispatch via :func:`from_uri`:

- ``csv://<path>``, ``*.csv``, ``*.tsv``       — CSVConnector
- ``parquet://<path>``, ``*.parquet``, ``*.pq`` — ParquetConnector
- ``feather://<path>``, ``*.feather``, ``*.arrow`` — FeatherConnector
- ``json://<path>``, ``*.json``, ``*.jsonl``     — JSONConnector
- ``excel://<path>``, ``*.xlsx``, ``*.xls``      — ExcelConnector
- ``duckdb://<path>?table=...``                 — DuckDBConnector (file mode)
- ``duckdb-sql://<base64-or-raw-query>``        — DuckDBConnector (query mode)
- ``sqlite:///...``, ``postgresql://...``, ``mysql://...``, ``mssql://...``,
  ``snowflake://...``                            — SQLConnector

For programmatic use, instantiate the connector class directly with its
parameters; ``from_uri`` is a convenience for the CLI path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

import pyarrow as pa

from causalrag.data.connectors.csv import CSVConnector
from causalrag.data.connectors.duckdb import DuckDBConnector
from causalrag.data.connectors.excel import ExcelConnector
from causalrag.data.connectors.feather import FeatherConnector
from causalrag.data.connectors.json import JSONConnector
from causalrag.data.connectors.parquet import ParquetConnector
from causalrag.data.connectors.sql import SQLConnector


@runtime_checkable
class Connector(Protocol):
    """Uniform interface for data sources (PDD §7.1)."""

    def to_arrow(self) -> pa.Table: ...
    def describe(self) -> dict[str, Any]: ...
    def supports_lazy(self) -> bool: ...


_SQL_SCHEMES = (
    "sqlite",
    "postgresql",
    "postgres",
    "mysql",
    "mysql+pymysql",
    "mariadb",
    "mssql",
    "mssql+pyodbc",
    "snowflake",
    "bigquery",
)


def from_uri(source: str | Path) -> Connector:
    """Dispatch a string URI / path / SQLAlchemy URL to the right connector."""
    s = str(source)

    # SQLAlchemy-style URLs (sqlite:///, postgresql://, mysql://, etc.)
    scheme = s.split("://", 1)[0] if "://" in s else ""
    if "+" in scheme:
        base = scheme.split("+", 1)[0]
    else:
        base = scheme
    if base in _SQL_SCHEMES:
        # If a #table=... fragment is provided, split it out
        if "#table=" in s:
            url, table = s.rsplit("#table=", 1)
            return SQLConnector(url=url, table=table)
        return SQLConnector(url=s, query="SELECT 1")  # caller should pass real query

    # DuckDB explicit schemes
    if s.startswith("duckdb://"):
        path = s[len("duckdb://") :]
        if "?" in path:
            db_path, _, qs = path.partition("?")
            params = parse_qs(qs)
            table = params.get("table", [None])[0]
            query = params.get("query", [None])[0]
            return DuckDBConnector(path=db_path or None, table=table, query=query)
        return DuckDBConnector(path=path or None, table="__default")
    if s.startswith("duckdb-sql://"):
        return DuckDBConnector(query=s[len("duckdb-sql://") :])

    # Other explicit URIs
    if s.startswith("csv://"):
        return CSVConnector(s[len("csv://") :])
    if s.startswith("parquet://"):
        return ParquetConnector(s[len("parquet://") :])
    if s.startswith("feather://"):
        return FeatherConnector(s[len("feather://") :])
    if s.startswith("json://"):
        return JSONConnector(s[len("json://") :])
    if s.startswith("excel://"):
        return ExcelConnector(s[len("excel://") :])

    # Suffix-based dispatch for bare paths
    suffix = Path(s).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return CSVConnector(s, delimiter="\t" if suffix == ".tsv" else ",")
    if suffix in {".parquet", ".pq"}:
        return ParquetConnector(s)
    if suffix in {".feather", ".arrow", ".ipc"}:
        return FeatherConnector(s)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return JSONConnector(s, lines=(suffix in {".jsonl", ".ndjson"}))
    if suffix in {".xlsx", ".xls"}:
        return ExcelConnector(s)
    if suffix == ".duckdb":
        raise ValueError(
            f"DuckDB file {s} requires a table or query — use "
            f"DuckDBConnector(path='{s}', table='...') programmatically or pass "
            f"duckdb://{s}?table=<name>."
        )

    raise ValueError(
        f"Unrecognized data source: {source!r}. Supported: csv, parquet, feather, "
        "arrow, json, jsonl, xlsx, xls, duckdb://, sqlite:///, postgresql://, "
        "mysql://, mssql://, snowflake://, bigquery://"
    )


__all__ = [
    "Connector",
    "CSVConnector",
    "DuckDBConnector",
    "ExcelConnector",
    "FeatherConnector",
    "JSONConnector",
    "ParquetConnector",
    "SQLConnector",
    "from_uri",
]
