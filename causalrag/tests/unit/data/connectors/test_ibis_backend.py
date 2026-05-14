"""Unit tests for the ibis-framework warehouse adapter (Sprint 5.1)."""

from __future__ import annotations

import sys

import pyarrow as pa
import pytest

from causalrag.data.connectors.ibis_backend import (
    IbisConnector,
    ibis_connector_from_table,
    register_ibis_uri_scheme,
)


# ---------------------------------------------------------------------------
# URI parsing tests — no ibis runtime required
# ---------------------------------------------------------------------------


def test_unsupported_scheme_raises() -> None:
    with pytest.raises(ValueError, match="Not an ibis URI"):
        IbisConnector.from_uri("postgresql://x/y/z")


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported ibis backend"):
        IbisConnector.from_uri("ibis+oracle:///x/y")


def test_bigquery_uri_parses() -> None:
    c = IbisConnector.from_uri(
        "ibis+bigquery://proj-a/marketing/events?credentials=/tmp/key.json"
    )
    assert c.backend == "bigquery"
    assert c.table_name == "events"
    assert c.connect_kwargs["project_id"] == "proj-a"
    assert c.connect_kwargs["dataset_id"] == "marketing"
    assert c.connect_kwargs["credentials"] == "/tmp/key.json"


def test_snowflake_uri_parses_with_env_password(monkeypatch) -> None:
    monkeypatch.setenv("SF_PW", "s3cret")
    c = IbisConnector.from_uri(
        "ibis+snowflake://acct1/wh1/db1/sch1/orders?user=alice&password_env=SF_PW"
    )
    assert c.backend == "snowflake"
    assert c.table_name == "orders"
    assert c.connect_kwargs["account"] == "acct1"
    assert c.connect_kwargs["warehouse"] == "wh1"
    assert c.connect_kwargs["user"] == "alice"
    assert c.connect_kwargs["password"] == "s3cret"


def test_postgres_uri_parses() -> None:
    c = IbisConnector.from_uri("ibis+postgres://alice:pw@db.example.com:5432/mydb/clicks")
    assert c.backend == "postgres"
    assert c.connect_kwargs["host"] == "db.example.com"
    assert c.connect_kwargs["port"] == 5432
    assert c.connect_kwargs["user"] == "alice"
    assert c.connect_kwargs["password"] == "pw"
    assert c.connect_kwargs["database"] == "mydb"
    assert c.table_name == "clicks"


def test_databricks_uri_uses_token_env(monkeypatch) -> None:
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxx")
    c = IbisConnector.from_uri(
        "ibis+databricks://host.databricks.com/catA/schB/tableC"
        "?token_env=DATABRICKS_TOKEN&http_path=/sql/1.0/warehouses/abc"
    )
    assert c.backend == "databricks"
    assert c.connect_kwargs["server_hostname"] == "host.databricks.com"
    assert c.connect_kwargs["catalog"] == "catA"
    assert c.connect_kwargs["schema"] == "schB"
    assert c.connect_kwargs["access_token"] == "dapi-xxx"
    assert c.connect_kwargs["http_path"] == "/sql/1.0/warehouses/abc"
    assert c.table_name == "tableC"


def test_redshift_aliased_to_postgres() -> None:
    c = IbisConnector.from_uri("ibis+redshift://u:p@h:5439/db/t")
    assert c.backend == "postgres"
    assert c.table_name == "t"


def test_missing_env_var_raises_clearly(monkeypatch) -> None:
    monkeypatch.delenv("NOT_SET_PW", raising=False)
    with pytest.raises(RuntimeError, match="NOT_SET_PW"):
        IbisConnector.from_uri(
            "ibis+snowflake://a/w/d/s/t?user=x&password_env=NOT_SET_PW"
        )


def test_malformed_uri_raises() -> None:
    with pytest.raises(ValueError, match="Malformed ibis URI"):
        IbisConnector.from_uri("ibis+duckdb")  # no scheme separator
    with pytest.raises(ValueError, match="must include a table"):
        IbisConnector.from_uri("ibis+duckdb:///:memory:")
    with pytest.raises(ValueError, match="ibis\\+bigquery"):
        IbisConnector.from_uri("ibis+bigquery://just-a-project")


def test_uri_parsing_works_without_ibis_installed(monkeypatch) -> None:
    """Parsing must not import ibis — only :meth:`_connect` does.

    We simulate "ibis not installed" by shadowing the import.
    """
    # Hide any cached ibis module
    saved = sys.modules.pop("ibis", None)
    monkeypatch.setitem(sys.modules, "ibis", None)  # makes `import ibis` fail
    try:
        c = IbisConnector.from_uri("ibis+postgres://u:p@h:5432/db/t")
        assert c.backend == "postgres"
        assert c.table_name == "t"
    finally:
        if saved is not None:
            sys.modules["ibis"] = saved


# ---------------------------------------------------------------------------
# Runtime tests — gated on ibis-framework being installed
# ---------------------------------------------------------------------------


ibis = pytest.importorskip("ibis")


@pytest.fixture
def synthetic_duckdb_table():
    """Materialize a small synthetic table in an in-memory DuckDB and
    yield the ibis Table expression."""
    pytest.importorskip("duckdb")
    con = ibis.duckdb.connect(":memory:")
    tbl = pa.table(
        {
            "x": list(range(500)),
            "y": [float(i) * 0.5 for i in range(500)],
            "label": [(i % 2) for i in range(500)],
        }
    )
    con.create_table("synthetic", tbl)
    yield con, con.table("synthetic")


def test_duckdb_uri_roundtrip_to_arrow(synthetic_duckdb_table) -> None:
    """``to_arrow`` on an ibis+duckdb URI returns the full table."""
    con, _ = synthetic_duckdb_table

    # We can't pass the live `con` through a URI, so build a connector
    # against the same in-memory DB and seed it through that connector's
    # own connection. Easiest: use ``ibis_connector_from_table``.
    table_expr = con.table("synthetic")
    c = ibis_connector_from_table(table_expr, describe_str="duckdb://:memory:/synthetic")

    out = c.to_arrow()
    assert isinstance(out, pa.Table)
    assert out.num_rows == 500
    assert set(out.column_names) == {"x", "y", "label"}


def test_sample_to_arrow_caps_at_n(synthetic_duckdb_table) -> None:
    con, _ = synthetic_duckdb_table
    c = ibis_connector_from_table(
        con.table("synthetic"), describe_str="duckdb://:memory:/synthetic"
    )
    sample = c.sample_to_arrow(n=100, seed=7)
    assert isinstance(sample, pa.Table)
    assert sample.num_rows <= 100
    assert sample.num_rows > 0


def test_sample_to_arrow_when_table_smaller_than_n(synthetic_duckdb_table) -> None:
    """If n exceeds total rows, the full table is returned."""
    con, _ = synthetic_duckdb_table
    c = ibis_connector_from_table(
        con.table("synthetic"), describe_str="duckdb://:memory:/synthetic"
    )
    sample = c.sample_to_arrow(n=10_000)
    assert sample.num_rows == 500


def test_sample_to_arrow_rejects_zero(synthetic_duckdb_table) -> None:
    con, _ = synthetic_duckdb_table
    c = ibis_connector_from_table(
        con.table("synthetic"), describe_str="duckdb://:memory:/synthetic"
    )
    with pytest.raises(ValueError):
        c.sample_to_arrow(n=0)


def test_describe_returns_backend_table_rowcount(synthetic_duckdb_table) -> None:
    con, _ = synthetic_duckdb_table
    c = ibis_connector_from_table(
        con.table("synthetic"), describe_str="duckdb://:memory:/synthetic"
    )
    info = c.describe()
    assert "backend" in info
    assert "table" in info
    assert info["row_count_estimate"] == 500
    assert info["source"] == "duckdb://:memory:/synthetic"


def test_supports_lazy_is_true() -> None:
    c = IbisConnector.from_uri("ibis+duckdb://:memory:/whatever")
    assert c.supports_lazy() is True


def test_register_ibis_uri_scheme_is_idempotent() -> None:
    """Calling the registrar twice should not double-wrap the dispatcher."""
    from causalrag.data import connectors as conn_pkg

    register_ibis_uri_scheme()
    first = conn_pkg.from_uri
    register_ibis_uri_scheme()
    second = conn_pkg.from_uri
    assert first is second


def test_register_ibis_uri_scheme_dispatches_ibis_prefix() -> None:
    register_ibis_uri_scheme()
    from causalrag.data import connectors as conn_pkg

    c = conn_pkg.from_uri("ibis+postgres://u:p@h:5432/db/t")
    assert isinstance(c, IbisConnector)
    assert c.backend == "postgres"
