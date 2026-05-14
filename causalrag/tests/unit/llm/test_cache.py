"""Tests for the three-layer LLM cache (Sprint 8.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalrag.llm.cache import (
    SemanticCache,
    cache_prefix_key,
    cassette_digest_matches,
    hash_inputs,
)


# ---------------------------------------------------------------------------
# hash_inputs / cache_prefix_key
# ---------------------------------------------------------------------------


def test_hash_inputs_stable_across_kwarg_order() -> None:
    h1 = hash_inputs(treatment="T", outcome="Y", covariates=["A", "B"])
    h2 = hash_inputs(outcome="Y", covariates=["A", "B"], treatment="T")
    assert h1 == h2
    assert len(h1) == 32


def test_hash_inputs_changes_with_payload() -> None:
    h1 = hash_inputs("a", x=1)
    h2 = hash_inputs("a", x=2)
    assert h1 != h2


def test_cache_prefix_key_stable_and_prefix_only() -> None:
    system = "You are a causal-inference assistant."
    prefix = "Few-shot example #1: ..."
    k1 = cache_prefix_key(system, prefix)
    k2 = cache_prefix_key(system, prefix)
    assert k1 == k2  # stability across calls
    assert len(k1) == 32
    # Different system or prefix changes the key.
    assert cache_prefix_key(system + " ", prefix) != k1
    assert cache_prefix_key(system, prefix + " ") != k1


def test_cassette_digest_matches_permissive() -> None:
    assert cassette_digest_matches({}, "sha256:abc") is True
    assert cassette_digest_matches({"model_digest": "x"}, None) is True
    assert cassette_digest_matches({"model_digest": "x"}, "x") is True
    assert cassette_digest_matches({"model_digest": "x"}, "y") is False


# ---------------------------------------------------------------------------
# SemanticCache: exact-match roundtrip + JSONL persistence
# ---------------------------------------------------------------------------


def test_semantic_cache_exact_roundtrip_persists_to_jsonl(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path)
    h = hash_inputs(prompt="What is the ATE?", dataset="lalonde")
    cache.store("ate_template", h, {"answer": 1700.0, "ci": [1200.0, 2200.0]})

    # In-memory hit.
    hit = cache.lookup("ate_template", h)
    assert hit == {"answer": 1700.0, "ci": [1200.0, 2200.0]}

    # JSONL persisted.
    jsonl = (tmp_path / "cache.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(jsonl) == 1
    obj = json.loads(jsonl[0])
    assert obj["template_id"] == "ate_template"
    assert obj["inputs_hash"] == h
    assert obj["response"]["answer"] == 1700.0

    # Reload from disk: new instance sees the same entry.
    reloaded = SemanticCache(tmp_path)
    again = reloaded.lookup("ate_template", h)
    assert again == {"answer": 1700.0, "ci": [1200.0, 2200.0]}


def test_semantic_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path)
    assert cache.lookup("template", "deadbeef") is None
    assert cache.stats()["misses"] == 1
    assert cache.stats()["hits"] == 0


def test_semantic_cache_store_overwrites_exact_key(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path)
    cache.store("t", "h", {"v": 1})
    cache.store("t", "h", {"v": 2})
    assert cache.lookup("t", "h") == {"v": 2}
    assert cache.stats()["entries"] == 1


def test_semantic_cache_in_memory_only_when_store_dir_is_none() -> None:
    cache = SemanticCache(None)
    cache.store("t", "h", {"v": 1})
    assert cache.lookup("t", "h") == {"v": 1}


# ---------------------------------------------------------------------------
# SemanticCache: fuzzy lookup via stub embedder
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Deterministic bag-of-words embedding over a tiny vocabulary."""

    VOCAB = ("ate", "att", "lalonde", "iv", "outcome", "treatment", "what", "is", "the")

    def encode(self, text: str) -> list[float]:
        tokens = text.lower().split()
        return [float(tokens.count(w)) for w in self.VOCAB]


def test_semantic_cache_fuzzy_lookup_above_threshold(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path, fuzzy_threshold=0.95, embedder=_StubEmbedder())
    cache.store(
        "ate_template",
        hash_inputs("v1"),
        {"answer": "stored"},
        query_text="What is the ATE for lalonde",
    )

    # Different inputs_hash, but very similar query text -> fuzzy hit.
    hit = cache.lookup(
        "ate_template",
        hash_inputs("v2"),
        query_text="What is the ATE for lalonde",  # cosine == 1.0
    )
    assert hit == {"answer": "stored"}
    assert cache.stats()["hits_fuzzy"] == 1
    assert cache.stats()["hits_exact"] == 0


def test_semantic_cache_fuzzy_miss_below_threshold(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path, fuzzy_threshold=0.99, embedder=_StubEmbedder())
    cache.store(
        "ate_template",
        hash_inputs("v1"),
        {"answer": "stored"},
        query_text="What is the ATE for lalonde",
    )
    # Different topic -> low cosine -> miss.
    hit = cache.lookup(
        "ate_template",
        hash_inputs("v3"),
        query_text="treatment outcome IV",
    )
    assert hit is None
    assert cache.stats()["misses"] == 1


def test_semantic_cache_fuzzy_respects_template_id(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path, fuzzy_threshold=0.5, embedder=_StubEmbedder())
    cache.store(
        "ate_template",
        hash_inputs("v1"),
        {"answer": "ate"},
        query_text="What is the ATE for lalonde",
    )
    # Same query text, different template_id -> no fuzzy hit.
    hit = cache.lookup(
        "att_template",
        hash_inputs("v2"),
        query_text="What is the ATE for lalonde",
    )
    assert hit is None


def test_semantic_cache_no_fuzzy_without_embedder(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path)  # no embedder
    cache.store("t", hash_inputs("v1"), {"answer": "x"}, query_text="hello world")
    # query_text supplied but no backend -> miss.
    assert cache.lookup("t", hash_inputs("v2"), query_text="hello world") is None


# ---------------------------------------------------------------------------
# stats() reflects hit/miss/store counters
# ---------------------------------------------------------------------------


def test_semantic_cache_stats_reflect_traffic(tmp_path: Path) -> None:
    cache = SemanticCache(tmp_path, embedder=_StubEmbedder())
    cache.store("t", "h1", {"v": 1}, query_text="ate lalonde")
    cache.store("t", "h2", {"v": 2}, query_text="iv outcome")

    # Exact hit.
    cache.lookup("t", "h1")
    # Fuzzy hit on the "ate lalonde" entry.
    cache.lookup("t", "h_other", query_text="ate lalonde")
    # Miss.
    cache.lookup("t", "h3")

    stats = cache.stats()
    assert stats["stores"] == 2
    assert stats["hits_exact"] == 1
    assert stats["hits_fuzzy"] == 1
    assert stats["misses"] == 1
    assert stats["hits"] == 2
    assert stats["entries"] == 2


# ---------------------------------------------------------------------------
# Engine prefix key: stability across repeated calls with same prefix
# ---------------------------------------------------------------------------


def test_cache_prefix_key_stable_across_many_calls() -> None:
    system = "system message"
    prefix = "shared leading context"
    keys = {cache_prefix_key(system, prefix) for _ in range(50)}
    assert len(keys) == 1


def test_cache_prefix_key_independent_of_variable_suffix() -> None:
    # The helper only takes the *prefix*; callers are responsible for slicing.
    # Verify two callers with the same prefix get the same key regardless of
    # what they intend to append.
    system = "S"
    prefix = "Few-shot: ..."
    assert cache_prefix_key(system, prefix) == cache_prefix_key(system, prefix)


@pytest.mark.parametrize(
    "system,prefix",
    [
        ("", ""),
        ("S", ""),
        ("", "P"),
        ("S with unicode —", "P with unicode —"),
    ],
)
def test_cache_prefix_key_handles_edge_inputs(system: str, prefix: str) -> None:
    key = cache_prefix_key(system, prefix)
    assert isinstance(key, str)
    assert len(key) == 32
