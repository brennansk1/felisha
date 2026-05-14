"""Generic SQL connector via SQLAlchemy — PDD §7.1.

Supports any database with a SQLAlchemy driver: PostgreSQL, MySQL, SQLite,
MSSQL, Snowflake, BigQuery (via sqlalchemy-bigquery), etc. Pass either:

- ``query=`` for ad-hoc SQL, or
- ``table=`` for a quick ``SELECT * FROM <table> LIMIT n``.

The ``url`` is a SQLAlchemy URL like ``postgresql+psycopg://user:pw@host/db``
or ``sqlite:///path/to/db.sqlite``.

A row-limit is enforced by default to prevent surprise multi-GB pulls;
pass ``limit=None`` to disable.

Optional fast path: when ``connectorx`` is installed the read takes the
columnar fast path (5-20× faster than pandas on big pulls). We fall back to
``pandas.read_sql`` otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa


@dataclass
class SQLConnector:
    url: str  # SQLAlchemy URL: postgresql://..., sqlite:///..., mysql+pymysql://..., etc.
    query: str | None = None
    table: str | None = None
    limit: int | None = 1_000_000

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("SQLConnector needs a SQLAlchemy URL")
        if self.query is None and self.table is None:
            raise ValueError("SQLConnector needs either query= or table=")

    def _resolved_query(self) -> str:
        if self.query:
            q = self.query
        else:
            assert self.table is not None
            if not self.table.replace("_", "").replace(".", "").isalnum():
                raise ValueError(
                    f"Unsafe table name: {self.table!r}. Pass query=... for non-trivial cases."
                )
            q = f"SELECT * FROM {self.table}"
        if self.limit and "limit" not in q.lower():
            q = f"{q} LIMIT {int(self.limit)}"
        return q

    def to_arrow(self) -> pa.Table:
        q = self._resolved_query()
        # Try connectorx fast path
        try:
            import connectorx as cx  # type: ignore[import]

            return cx.read_sql(self.url, q, return_type="arrow")
        except ImportError:
            pass
        # Fallback: SQLAlchemy + pandas
        try:
            import pandas as pd
            from sqlalchemy import create_engine
        except ImportError as e:
            raise RuntimeError(
                "SQLConnector requires SQLAlchemy: pip install 'causalrag[sql]'"
            ) from e
        engine = create_engine(self.url)
        try:
            df = pd.read_sql(q, engine)
        finally:
            engine.dispose()
        return pa.Table.from_pandas(df)

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.url.split("@")[-1] if "@" in self.url else self.url,
            "mode": "query" if self.query else "table",
            "table": self.table,
            "query": (self.query or "")[:200],
            "limit": self.limit,
        }

    def supports_lazy(self) -> bool:
        return True
