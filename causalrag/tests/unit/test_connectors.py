from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pa_pq
import pytest

from causalrag.data.connectors import CSVConnector, ParquetConnector, from_uri


def test_csv_connector_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_csv(p, index=False)
    c = CSVConnector(p)
    t = c.to_arrow()
    assert t.num_rows == 3
    assert set(t.column_names) == {"a", "b"}
    info = c.describe()
    assert info["sha256"] is not None
    assert c.supports_lazy() is False


def test_csv_connector_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CSVConnector(tmp_path / "missing.csv").to_arrow()


def test_parquet_connector_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / "tiny.parquet"
    pa_pq.write_table(__import__("pyarrow").Table.from_pandas(df), p)
    c = ParquetConnector(p)
    t = c.to_arrow()
    assert t.num_rows == 3
    info = c.describe()
    assert info["num_rows"] == 3
    assert c.supports_lazy() is True


def test_from_uri_dispatch(tmp_path: Path) -> None:
    csv_path = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv_path, index=False)
    c = from_uri(csv_path)
    assert isinstance(c, CSVConnector)

    pq_path = tmp_path / "x.parquet"
    pa_pq.write_table(__import__("pyarrow").Table.from_pandas(pd.DataFrame({"a": [1]})), pq_path)
    p = from_uri(pq_path)
    assert isinstance(p, ParquetConnector)

    # Explicit URIs
    assert isinstance(from_uri(f"csv://{csv_path}"), CSVConnector)
    assert isinstance(from_uri(f"parquet://{pq_path}"), ParquetConnector)


def test_from_uri_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unrecognized"):
        from_uri("redshift://foo")
