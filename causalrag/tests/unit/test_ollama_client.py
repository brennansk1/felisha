from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from causalrag.llm.cassette import CassetteMiss
from causalrag.llm.ollama_client import (
    FakeOllamaTransport,
    OllamaClient,
    SchemaValidationFailed,
)


class _ColumnFact(BaseModel):
    column: str
    domain_meaning: str
    temporal_position: str


def _client(transport: FakeOllamaTransport, tmp_path: Path) -> OllamaClient:
    return OllamaClient(
        model="qwen3:14b-q4_K_M",
        seed=42,
        cassette_dir=tmp_path,
        transport=transport,
        allow_live=True,
    )


def test_happy_path_parses_and_caches(tmp_path: Path) -> None:
    transport = FakeOllamaTransport(
        {
            "age": {
                "column": "age",
                "domain_meaning": "subject age in years",
                "temporal_position": "baseline",
            }
        }
    )
    client = _client(transport, tmp_path)
    resp = client.parse(prompt="describe column 'age'", schema=_ColumnFact)
    assert isinstance(resp.parsed, _ColumnFact)
    assert resp.parsed.column == "age"
    assert resp.source == "live"
    assert resp.retries == 0
    assert resp.seed == 42
    assert resp.model_digest is not None
    assert (tmp_path / f"{resp.cassette_key}.json").exists()


def test_second_call_replays_from_cassette(tmp_path: Path) -> None:
    transport = FakeOllamaTransport(
        {
            "age": {
                "column": "age",
                "domain_meaning": "subject age in years",
                "temporal_position": "baseline",
            }
        }
    )
    c1 = _client(transport, tmp_path)
    r1 = c1.parse(prompt="describe column 'age'", schema=_ColumnFact)
    assert r1.source == "live"

    # New client, transport that would fail if invoked — proves replay.
    c2 = OllamaClient(
        model="qwen3:14b-q4_K_M",
        seed=42,
        cassette_dir=tmp_path,
        transport=FakeOllamaTransport({"unrelated": {}}),
        allow_live=False,
    )
    r2 = c2.parse(prompt="describe column 'age'", schema=_ColumnFact)
    assert r2.source == "cassette"
    assert r2.parsed.column == "age"


def test_cassette_miss_when_replay_only(tmp_path: Path) -> None:
    client = OllamaClient(
        model="m",
        seed=1,
        cassette_dir=tmp_path,
        transport=FakeOllamaTransport({"": {}}),
        allow_live=False,
    )
    with pytest.raises(CassetteMiss):
        client.parse(prompt="anything", schema=_ColumnFact)


def test_schema_retry_recovers(tmp_path: Path) -> None:
    """First response is malformed; the retry response is valid."""
    calls = {"n": 0}

    class _Counting(FakeOllamaTransport):
        def generate(self, **kwargs):  # type: ignore[override]
            calls["n"] += 1
            if calls["n"] == 1:
                return '{"column": "age"}'  # missing required fields
            return (
                '{"column": "age", "domain_meaning": "years", "temporal_position": "baseline"}'
            )

    transport = _Counting({"": {}})
    client = _client(transport, tmp_path)
    resp = client.parse(prompt="describe age", schema=_ColumnFact)
    assert resp.retries == 1
    assert resp.parsed.column == "age"
    assert calls["n"] == 2


def test_schema_failure_after_retries(tmp_path: Path) -> None:
    transport = FakeOllamaTransport({"": '{"not_a_column": "x"}'})
    client = _client(transport, tmp_path)
    with pytest.raises(SchemaValidationFailed) as excinfo:
        client.parse(prompt="describe", schema=_ColumnFact)
    assert excinfo.value.errors  # at least one captured attempt


def test_deterministic_options_passed_to_transport(tmp_path: Path) -> None:
    transport = FakeOllamaTransport(
        {
            "": {
                "column": "x",
                "domain_meaning": "y",
                "temporal_position": "baseline",
            }
        }
    )
    client = OllamaClient(
        model="m",
        seed=7,
        temperature=0.0,
        num_ctx=2048,
        cassette_dir=tmp_path,
        transport=transport,
        allow_live=True,
    )
    client.parse(prompt="x", schema=_ColumnFact)
    assert transport.calls[0]["options"]["seed"] == 7
    assert transport.calls[0]["options"]["temperature"] == 0.0
    assert transport.calls[0]["options"]["num_ctx"] == 2048
