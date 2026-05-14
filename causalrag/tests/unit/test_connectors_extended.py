"""Tests for the v0.1 connector matrix beyond CSV/Parquet:
Feather, JSON, Excel, DuckDB, SQLite (via SQLAlchemy)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from causalrag.data.connectors import (
    DuckDBConnector,
    ExcelConnector,
    FeatherConnector,
    JSONConnector,
    SQLConnector,
    from_uri,
)


# --- Feather / Arrow ---------------------------------------------------------


def test_feather_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "tiny.feather"
    import pyarrow.feather as pa_feather

    pa_feather.write_feather(pa.Table.from_pandas(df), p)
    c = FeatherConnector(p)
    t = c.to_arrow()
    assert t.num_rows == 3
    assert c.supports_lazy() is False


def test_feather_from_uri_suffix(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2]})
    p = tmp_path / "tiny.feather"
    import pyarrow.feather as pa_feather

    pa_feather.write_feather(pa.Table.from_pandas(df), p)
    c = from_uri(p)
    assert isinstance(c, FeatherConnector)


# --- JSON / JSONL ------------------------------------------------------------


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "tiny.jsonl"
    p.write_text(
        '{"a": 1, "b": "x"}\n{"a": 2, "b": "y"}\n{"a": 3, "b": "z"}\n', encoding="utf-8"
    )
    c = JSONConnector(p, lines=True)
    t = c.to_arrow()
    assert t.num_rows == 3
    assert set(t.column_names) == {"a", "b"}


def test_json_array_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "tiny.json"
    p.write_text('[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]', encoding="utf-8")
    c = JSONConnector(p, lines=False)
    t = c.to_arrow()
    assert t.num_rows == 2


def test_json_from_uri_dispatch(tmp_path: Path) -> None:
    p = tmp_path / "tiny.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    c = from_uri(p)
    assert isinstance(c, JSONConnector)
    assert c.lines is True


# --- Excel -------------------------------------------------------------------


def test_excel_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "tiny.xlsx"
    df.to_excel(p, index=False)
    c = ExcelConnector(p)
    t = c.to_arrow()
    assert t.num_rows == 3
    assert isinstance(from_uri(p), ExcelConnector)


# --- DuckDB ------------------------------------------------------------------


def test_duckdb_query_in_memory_against_csv(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    csv_path = tmp_path / "people.csv"
    pd.DataFrame({"age": [25, 40, 60], "income": [30_000, 70_000, 90_000]}).to_csv(
        csv_path, index=False
    )
    q = f"SELECT * FROM read_csv_auto('{csv_path}') WHERE age >= 40"
    c = DuckDBConnector(query=q)
    t = c.to_arrow()
    assert t.num_rows == 2
    assert c.supports_lazy() is True


def test_duckdb_file_table_roundtrip(tmp_path: Path) -> None:
    import duckdb

    db_path = tmp_path / "warehouse.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE patients (id INT, age INT, treat INT)")
    con.execute("INSERT INTO patients VALUES (1, 30, 1), (2, 50, 0), (3, 65, 1)")
    con.close()

    c = DuckDBConnector(path=db_path, table="patients")
    t = c.to_arrow()
    assert t.num_rows == 3
    assert set(t.column_names) == {"id", "age", "treat"}


def test_duckdb_url_dispatch(tmp_path: Path) -> None:
    db = tmp_path / "x.duckdb"
    db.touch()
    c = from_uri(f"duckdb://{db}?table=patients")
    assert isinstance(c, DuckDBConnector)
    assert c.table == "patients"


def test_duckdb_query_uri_dispatch() -> None:
    c = from_uri("duckdb-sql://SELECT 1 AS x")
    assert isinstance(c, DuckDBConnector)
    assert c.query == "SELECT 1 AS x"


def test_duckdb_rejects_unsafe_table_name() -> None:
    pytest.importorskip("duckdb")
    c = DuckDBConnector(path="ignored", table="x; DROP TABLE users;")
    with pytest.raises(ValueError, match="Unsafe table name"):
        c.to_arrow()


# --- SQLite via SQLAlchemy ---------------------------------------------------


def test_sqlite_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine

    db = tmp_path / "tiny.sqlite"
    eng = create_engine(f"sqlite:///{db}")
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df.to_sql("tbl", eng, index=False)
    eng.dispose()

    c = SQLConnector(url=f"sqlite:///{db}", table="tbl")
    t = c.to_arrow()
    assert t.num_rows == 3


def test_sql_query_with_limit_added(tmp_path: Path) -> None:
    pytest.importorskip("sqlalchemy")
    from sqlalchemy import create_engine

    db = tmp_path / "tiny.sqlite"
    eng = create_engine(f"sqlite:///{db}")
    pd.DataFrame({"a": list(range(100))}).to_sql("tbl", eng, index=False)
    eng.dispose()
    c = SQLConnector(url=f"sqlite:///{db}", query="SELECT * FROM tbl", limit=10)
    t = c.to_arrow()
    assert t.num_rows == 10


def test_from_uri_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognized"):
        from_uri("zfs://not-a-thing")
