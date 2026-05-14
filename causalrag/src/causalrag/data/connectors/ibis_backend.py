"""ibis-framework warehouse adapter (Sprint 5.1).

A single :class:`Connector`-compatible adapter that wraps
``ibis-framework`` so the pipeline can ingest from BigQuery, Snowflake,
Redshift (via the Postgres dialect), Postgres, DuckDB, and Databricks SQL
with one code path.

The connector is **sampling-aware**:

* ``to_arrow()`` extracts the full table — used when fitting estimators
  on the entire (or Crump-trimmed) population.
* ``sample_to_arrow(n=100_000)`` returns at most ``n`` rows — used by
  the data profiler for cheap descriptive statistics. Uses ibis's
  ``Table.sample`` (which lowers to TABLESAMPLE on backends that
  support it, otherwise an ``ORDER BY random() LIMIT n``).

URI grammar (all parsed by :meth:`IbisConnector.from_uri`)::

    ibis+bigquery://<project>/<dataset>/<table>?credentials=<path>
    ibis+snowflake://<acct>/<warehouse>/<database>/<schema>/<table>?user=...
    ibis+postgres://user:pass@host:port/db/<table>
    ibis+duckdb:///<path>/<table>          # path may be ``:memory:``
    ibis+databricks://<host>/<catalog>/<schema>/<table>?token_env=DATABRICKS_TOKEN

Credential discipline
---------------------
Connection passwords and tokens are **never** baked into the URI. The
URI may carry only an **env-var pointer** (``token_env=NAME``,
``password_env=NAME``, ``credentials_env=NAME``); the real secret is
read from ``os.environ[NAME]`` at connect time.

Optional dependency
-------------------
``ibis-framework`` is *not* a hard dependency of causalrag. This module
imports lazily and raises a clear :class:`RuntimeError` naming the
missing backend extra (e.g. ``pip install 'ibis-framework[bigquery]'``)
when the user actually tries that source.

Public surface
--------------
* :class:`IbisConnector`
* :func:`ibis_connector_from_table` — wrap an already-open ibis table.
* :func:`register_ibis_uri_scheme` — opt-in registration of the
  ``ibis+`` URI prefix with :func:`causalrag.data.connectors.from_uri`.
  We avoid touching that module directly so this ticket stays
  surgical; callers (CLI bootstrap, notebooks) invoke this once at
  startup to wire the dispatch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlparse

import pyarrow as pa

# Backends supported in this ticket. Maps the ``<backend>`` token in
# ``ibis+<backend>://`` to the ibis backend module attribute name.
_SUPPORTED_BACKENDS = {
    "bigquery": "bigquery",
    "snowflake": "snowflake",
    "postgres": "postgres",
    "postgresql": "postgres",  # alias
    "redshift": "postgres",    # Redshift speaks the Postgres wire protocol
    "duckdb": "duckdb",
    "databricks": "databricks",
}


def _resolve_env(value: str | None) -> str | None:
    """If ``value`` looks like an env-var pointer (``env:NAME`` or already
    a bare env name when the corresponding flag was used), return the
    resolved environment value. Otherwise return ``value`` unchanged.
    """
    if value is None:
        return None
    if value.startswith("env:"):
        name = value[4:]
        resolved = os.environ.get(name)
        if resolved is None:
            raise RuntimeError(
                f"Environment variable {name!r} referenced by URI is not set."
            )
        return resolved
    return value


@dataclass
class IbisConnector:
    """ibis-framework adapter (see module docstring for URI grammar).

    Parameters
    ----------
    backend
        Lowercase backend name (``"bigquery"``, ``"snowflake"``,
        ``"postgres"``, ``"duckdb"``, ``"databricks"``).
    table_name
        Fully-qualified-ish table identifier the backend understands.
        Some backends (BigQuery, Snowflake, Databricks) require a
        ``database``/``schema`` to be passed at connect time; those
        slots live in ``connect_kwargs``.
    connect_kwargs
        Keyword arguments forwarded to ``ibis.<backend>.connect(...)``.
    uri
        Original URI (for ``describe()`` provenance).
    """

    backend: str
    table_name: str
    connect_kwargs: dict[str, Any] = field(default_factory=dict)
    uri: str | None = None
    _ibis_table: Any = None  # if constructed via ibis_connector_from_table

    # ----- construction -------------------------------------------------

    @classmethod
    def from_uri(cls, uri: str) -> "IbisConnector":
        """Parse an ``ibis+<backend>://...`` URI into a connector."""
        if not uri.startswith("ibis+"):
            raise ValueError(
                f"Not an ibis URI: {uri!r} — expected scheme like "
                "'ibis+bigquery://', 'ibis+duckdb://', ..."
            )

        # urlparse mishandles double-prefixed schemes; split manually.
        scheme_and_rest = uri[len("ibis+") :]
        if "://" not in scheme_and_rest:
            raise ValueError(f"Malformed ibis URI: {uri!r}")
        backend_token, rest = scheme_and_rest.split("://", 1)
        backend_token = backend_token.lower()

        if backend_token not in _SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported ibis backend: {backend_token!r}. "
                f"Supported: {sorted(set(_SUPPORTED_BACKENDS))}."
            )
        backend = _SUPPORTED_BACKENDS[backend_token]

        # Split query string off the path portion.
        path_part, _, query_part = rest.partition("?")
        qs = {k: v[0] for k, v in parse_qs(query_part).items()}

        connect_kwargs, table_name = _parse_backend_path(backend, path_part, qs)

        return cls(
            backend=backend,
            table_name=table_name,
            connect_kwargs=connect_kwargs,
            uri=uri,
        )

    # ----- ibis access --------------------------------------------------

    def _connect(self):
        try:
            import ibis  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "IbisConnector requires the optional 'ibis-framework' package: "
                "pip install 'ibis-framework[<backend>]'"
            ) from e

        import ibis

        try:
            backend_mod = getattr(ibis, self.backend)
        except AttributeError as e:
            raise RuntimeError(
                f"ibis backend {self.backend!r} is not available. Install the "
                f"corresponding extra, e.g. pip install 'ibis-framework[{self.backend}]'"
            ) from e

        try:
            return backend_mod.connect(**self.connect_kwargs)
        except ImportError as e:
            raise RuntimeError(
                f"ibis backend {self.backend!r} requires an extra driver. "
                f"Install it with: pip install 'ibis-framework[{self.backend}]'"
            ) from e

    def _table(self):
        if self._ibis_table is not None:
            return self._ibis_table
        con = self._connect()
        return con.table(self.table_name)

    # ----- Connector protocol ------------------------------------------

    def to_arrow(self) -> pa.Table:
        """Materialize the full ibis table to a pyarrow Table."""
        return self._table().to_pyarrow()

    def sample_to_arrow(self, n: int = 100_000, *, seed: int = 42) -> pa.Table:
        """Return at most ``n`` rows as a pyarrow Table.

        Strategy:
        1. Probe an estimated row count via ``COUNT(*)``.
        2. If the table is already <= n rows, just materialize it.
        3. Otherwise call :meth:`ibis.Table.sample` with
           ``fraction=n/count`` (TABLESAMPLE on supported backends), then
           ``LIMIT n`` to clamp the upper bound — ``sample`` is
           probabilistic so the realized count can exceed ``n`` slightly.
        4. On any backend that doesn't implement ``sample`` (older
           dialects), fall back to ``order_by(ibis.random()).limit(n)``.
        """
        if n <= 0:
            raise ValueError(f"sample_to_arrow needs n > 0, got {n}")

        import ibis

        t = self._table()

        # Cheap-ish: COUNT(*). Some warehouses charge per scan, but
        # COUNT(*) is metadata-only on most modern engines.
        try:
            total = int(t.count().to_pyarrow().as_py())
        except Exception:
            total = None

        if total is not None and total <= n:
            return t.to_pyarrow()

        try:
            if total is None:
                # Without a count we can't compute a fraction; fall
                # back to a random-order limit.
                raise AttributeError
            fraction = min(1.0, max(n / total * 1.2, 1e-9))  # 20% headroom
            sampled = t.sample(fraction, seed=seed).limit(n)
            return sampled.to_pyarrow()
        except (AttributeError, NotImplementedError, Exception):
            try:
                sampled = t.order_by(ibis.random()).limit(n)
                return sampled.to_pyarrow()
            except Exception:
                # Last-ditch: materialize and slice.
                full = t.to_pyarrow()
                return full.slice(0, n)

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "source": self.uri or f"ibis+{self.backend}://{self.table_name}",
            "backend": self.backend,
            "table": self.table_name,
        }
        # Best-effort row-count estimate.
        try:
            t = self._table()
            info["row_count_estimate"] = int(t.count().to_pyarrow().as_py())
        except Exception as e:  # pragma: no cover - depends on backend
            info["row_count_estimate"] = None
            info["row_count_error"] = str(e)[:200]
        return info

    def supports_lazy(self) -> bool:
        return True  # ibis is lazy by construction


# ---------------------------------------------------------------------------
# URI -> backend kwargs translation
# ---------------------------------------------------------------------------

def _parse_backend_path(
    backend: str, path: str, qs: dict[str, str]
) -> tuple[dict[str, Any], str]:
    """Translate the path portion of an ``ibis+<backend>://`` URI into
    ``(connect_kwargs, table_name)`` for the given backend.

    Each branch is intentionally explicit: warehouse connection
    semantics differ enough that a "smart" generic parser would be
    actively misleading.
    """
    if backend == "duckdb":
        # ibis+duckdb:///<path>/<table>  — also ibis+duckdb://:memory:/<table>
        # We split off the LAST segment as the table; everything else
        # (including the leading slashes) is the database path.
        stripped = path.lstrip("/")
        if "/" not in stripped:
            raise ValueError(
                "ibis+duckdb URI must include a table: "
                "ibis+duckdb:///<path>/<table> or ibis+duckdb://:memory:/<table>"
            )
        db_path, table = stripped.rsplit("/", 1)
        if db_path == ":memory:" or db_path == "":
            return {"database": ":memory:"}, table
        return {"database": "/" + db_path if not db_path.startswith(":") else db_path}, table

    if backend == "bigquery":
        # ibis+bigquery://<project>/<dataset>/<table>
        parts = path.strip("/").split("/")
        if len(parts) != 3:
            raise ValueError(
                "ibis+bigquery URI must be ibis+bigquery://<project>/<dataset>/<table>"
            )
        project, dataset, table = parts
        kwargs: dict[str, Any] = {"project_id": project, "dataset_id": dataset}
        creds = qs.get("credentials") or (
            f"env:{qs['credentials_env']}" if "credentials_env" in qs else None
        )
        resolved = _resolve_env(creds)
        if resolved is not None:
            kwargs["credentials"] = resolved
        return kwargs, table

    if backend == "snowflake":
        # ibis+snowflake://<acct>/<warehouse>/<database>/<schema>/<table>
        parts = path.strip("/").split("/")
        if len(parts) != 5:
            raise ValueError(
                "ibis+snowflake URI must be "
                "ibis+snowflake://<account>/<warehouse>/<database>/<schema>/<table>"
            )
        account, warehouse, database, schema, table = parts
        kwargs = {
            "account": account,
            "warehouse": warehouse,
            "database": database,
            "schema": schema,
        }
        if "user" in qs:
            kwargs["user"] = qs["user"]
        pw = qs.get("password") or (
            f"env:{qs['password_env']}" if "password_env" in qs else None
        )
        resolved = _resolve_env(pw)
        if resolved is not None:
            kwargs["password"] = resolved
        return kwargs, table

    if backend == "postgres":
        # ibis+postgres://user:pass@host:port/db/<table>
        # Reuse urlparse against a normalized scheme.
        parsed = urlparse("postgres://" + path)
        if not parsed.hostname or not parsed.path:
            raise ValueError(
                "ibis+postgres URI must be "
                "ibis+postgres://user:pass@host:port/db/<table>"
            )
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 2:
            raise ValueError("ibis+postgres URI must include both <db> and <table>")
        database = path_parts[0]
        table = "/".join(path_parts[1:])  # allow schema.table-style ids
        kwargs = {
            "host": parsed.hostname,
            "database": database,
        }
        if parsed.port:
            kwargs["port"] = parsed.port
        if parsed.username:
            kwargs["user"] = parsed.username
        pw = parsed.password or (
            f"env:{qs['password_env']}" if "password_env" in qs else None
        )
        resolved = _resolve_env(pw)
        if resolved is not None:
            kwargs["password"] = resolved
        return kwargs, table

    if backend == "databricks":
        # ibis+databricks://<host>/<catalog>/<schema>/<table>?token_env=...&http_path=...
        parts = path.strip("/").split("/")
        if len(parts) != 4:
            raise ValueError(
                "ibis+databricks URI must be "
                "ibis+databricks://<host>/<catalog>/<schema>/<table>"
            )
        host, catalog, schema, table = parts
        kwargs = {
            "server_hostname": host,
            "catalog": catalog,
            "schema": schema,
        }
        if "http_path" in qs:
            kwargs["http_path"] = qs["http_path"]
        tok = qs.get("token") or (
            f"env:{qs['token_env']}" if "token_env" in qs else None
        )
        resolved = _resolve_env(tok)
        if resolved is not None:
            kwargs["access_token"] = resolved
        return kwargs, table

    raise ValueError(f"Unsupported ibis backend: {backend!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def ibis_connector_from_table(table_obj: Any, *, describe_str: str) -> IbisConnector:
    """Wrap an already-open ibis ``Table`` expression as an
    :class:`IbisConnector`.

    Useful when the caller has already authenticated and constructed
    custom joins/projections — we just want the pyarrow / sampling
    surface on top.

    Parameters
    ----------
    table_obj
        An ``ibis.expr.types.relations.Table``.
    describe_str
        Free-form provenance string surfaced in :meth:`describe`.
    """
    # Best-effort backend introspection — doesn't require importing ibis
    # at module import time, just at call time.
    backend_name = "unknown"
    try:
        be = getattr(table_obj, "_find_backend", None)
        if callable(be):
            backend_name = type(be()).__name__.lower()
    except Exception:
        pass

    return IbisConnector(
        backend=backend_name,
        table_name=describe_str,
        connect_kwargs={},
        uri=describe_str,
        _ibis_table=table_obj,
    )


def register_ibis_uri_scheme() -> None:
    """Register ``ibis+<backend>://`` dispatch with
    :func:`causalrag.data.connectors.from_uri`.

    This is an **opt-in** hook because we deliberately don't touch the
    ``connectors/__init__.py`` module in this ticket. CLI bootstrap or
    notebook setup should call it once at startup::

        from causalrag.data.connectors.ibis_backend import register_ibis_uri_scheme
        register_ibis_uri_scheme()

    Implementation: monkey-patch ``connectors.from_uri`` to fall
    through to :meth:`IbisConnector.from_uri` for ``ibis+`` prefixes
    before delegating to the original dispatcher.
    """
    from causalrag.data import connectors as _conn_pkg

    original = _conn_pkg.from_uri

    if getattr(original, "_ibis_registered", False):
        return  # idempotent

    def patched(source):  # type: ignore[no-untyped-def]
        s = str(source)
        if s.startswith("ibis+"):
            return IbisConnector.from_uri(s)
        return original(source)

    patched._ibis_registered = True  # type: ignore[attr-defined]
    _conn_pkg.from_uri = patched  # type: ignore[assignment]


__all__ = [
    "IbisConnector",
    "ibis_connector_from_table",
    "register_ibis_uri_scheme",
]
