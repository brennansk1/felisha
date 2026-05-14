"""Three-layer LLM cache (PDD §33 / Sprint 8.5).

This module surfaces the three caches that flank :mod:`causalrag.llm.ollama_client`:

1. **Engine prefix-KV cache.** Modern serving engines (vLLM, llama.cpp,
   ``ollama``'s ``num_keep`` knob) automatically reuse the KV-cache for the
   leading tokens of a prompt when those tokens are *byte-identical* across
   calls. We can't manage that cache directly, but we can give every call a
   stable digest of ``(system, prompt_prefix)`` so telemetry can confirm that
   the engine is in fact reusing it. See :func:`cache_prefix_key`.

2. **Semantic cache.** A local JSONL-backed key/value store rooted at
   ``.causalrag/llm_cache/``, keyed by ``(prompt_template_id, hash(inputs))``.
   If ``sentence-transformers`` is installed (and a ``query_text`` is supplied)
   it also performs an embedding cosine-similarity lookup above
   ``fuzzy_threshold``. Without the optional dep it falls back to exact-match
   only. See :class:`SemanticCache`.

3. **Cassette replay.** Already implemented in
   :mod:`causalrag.llm.cassette` / :class:`OllamaClient`; this module does not
   touch it but :func:`cassette_digest_matches` helps callers verify a cassette
   was recorded against the same model digest they're running against today.

The module is deliberately self-contained: no imports from ``ollama_client``,
no monkeypatching. It is consumed by the master loop and other LLM call sites,
not by the client itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Stable hashing helpers
# ---------------------------------------------------------------------------


def _canonical(obj: Any) -> Any:
    """Project ``obj`` onto JSON-serialisable, order-stable primitives."""
    if isinstance(obj, dict):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [_canonical(x) for x in sorted(obj, key=repr)]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback: stringify (e.g. Path, Decimal, custom types).
    return repr(obj)


def hash_inputs(*args: Any, **kwargs: Any) -> str:
    """Stable 32-char sha256 hex digest of arbitrary call inputs.

    Suitable for use as the ``inputs_hash`` key in :class:`SemanticCache`.
    Order of positional args matters; keyword args are sorted.
    """
    payload = {"args": [_canonical(a) for a in args], "kwargs": _canonical(kwargs)}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def cache_prefix_key(system: str, prompt_prefix: str) -> str:
    """Stable digest of ``(system, prompt_prefix)`` for engine prefix-KV reuse.

    vLLM and llama.cpp transparently reuse the KV-cache for leading tokens that
    are byte-identical between calls. By emitting a single deterministic digest
    for the prefix portion of every call, telemetry can confirm that successive
    calls with the same system message + few-shot exemplars are landing on the
    same engine-level cache shard.

    The digest is independent of the variable suffix of the prompt; callers
    should pass *only* the stable leading portion.
    """
    blob = json.dumps(
        {"system": system, "prefix": prompt_prefix},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def cassette_digest_matches(payload: dict[str, Any], expected_digest: str | None) -> bool:
    """Layer-3 sanity check: confirm a cassette was recorded against the same
    model digest we're running against. ``None`` on either side is permissive
    (older cassettes lack the field)."""
    recorded = payload.get("model_digest")
    if recorded is None or expected_digest is None:
        return True
    return recorded == expected_digest


# ---------------------------------------------------------------------------
# Optional embedding backend
# ---------------------------------------------------------------------------


class EmbeddingBackend(Protocol):
    def encode(self, text: str) -> list[float]: ...


def _cosine(a: Iterable[float], b: Iterable[float]) -> float:
    av = list(a)
    bv = list(b)
    if not av or not bv or len(av) != len(bv):
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    na = math.sqrt(sum(x * x for x in av))
    nb = math.sqrt(sum(y * y for y in bv))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _default_embedder() -> EmbeddingBackend | None:
    try:  # pragma: no cover - import path exercised only when extra installed
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")

        class _ST:
            def encode(self, text: str) -> list[float]:
                vec = model.encode(text, normalize_embeddings=True)
                return [float(x) for x in vec]

        return _ST()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Semantic cache
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    template_id: str
    inputs_hash: str
    response: dict[str, Any]
    query_text: str | None = None
    embedding: list[float] | None = None


@dataclass
class CacheStats:
    hits_exact: int = 0
    hits_fuzzy: int = 0
    misses: int = 0
    stores: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "hits_exact": self.hits_exact,
            "hits_fuzzy": self.hits_fuzzy,
            "misses": self.misses,
            "stores": self.stores,
            "hits": self.hits_exact + self.hits_fuzzy,
        }


class SemanticCache:
    """``(prompt_template_id, hash(inputs))`` -> cached response.

    Entries are persisted one-per-line as JSON to ``store_dir/cache.jsonl``.
    Exact-match lookup hashes ``(template_id, inputs_hash)`` and scans the in-
    memory index. Fuzzy lookup (optional) embeds ``query_text`` with a
    :class:`EmbeddingBackend` and returns the entry with the highest cosine
    similarity at or above ``fuzzy_threshold``, restricted to entries sharing
    the same ``template_id``.

    The cache is deliberately small and append-only; rotation is the caller's
    responsibility (delete the file to reset).
    """

    FILENAME = "cache.jsonl"

    def __init__(
        self,
        store_dir: Path | None = None,
        *,
        fuzzy_threshold: float = 0.95,
        embedder: EmbeddingBackend | None = None,
        load_embedder: bool = False,
    ) -> None:
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self.fuzzy_threshold = float(fuzzy_threshold)
        self._entries: list[_Entry] = []
        self._exact_index: dict[tuple[str, str], _Entry] = {}
        self._stats = CacheStats()

        if embedder is not None:
            self._embedder: EmbeddingBackend | None = embedder
        elif load_embedder:
            self._embedder = _default_embedder()
        else:
            self._embedder = None

        if self.store_dir is not None:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # -- persistence ----------------------------------------------------------

    @property
    def _path(self) -> Path | None:
        if self.store_dir is None:
            return None
        return self.store_dir / self.FILENAME

    def _load(self) -> None:
        path = self._path
        if path is None or not path.exists():
            return
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            entry = _Entry(
                template_id=str(obj["template_id"]),
                inputs_hash=str(obj["inputs_hash"]),
                response=obj.get("response", {}),
                query_text=obj.get("query_text"),
                embedding=obj.get("embedding"),
            )
            self._append_in_memory(entry)

    def _append_in_memory(self, entry: _Entry) -> None:
        # Last-writer-wins: replace any prior entry sharing the same exact key.
        key = (entry.template_id, entry.inputs_hash)
        existing = self._exact_index.get(key)
        if existing is not None:
            try:
                self._entries.remove(existing)
            except ValueError:
                pass
        self._exact_index[key] = entry
        self._entries.append(entry)

    def _persist(self, entry: _Entry) -> None:
        path = self._path
        if path is None:
            return
        record = {
            "template_id": entry.template_id,
            "inputs_hash": entry.inputs_hash,
            "response": entry.response,
        }
        if entry.query_text is not None:
            record["query_text"] = entry.query_text
        if entry.embedding is not None:
            record["embedding"] = entry.embedding
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    # -- API ------------------------------------------------------------------

    def lookup(
        self,
        template_id: str,
        inputs_hash: str,
        *,
        query_text: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a cached response or ``None``.

        Exact match on ``(template_id, inputs_hash)`` always wins. If no exact
        match exists and ``query_text`` is provided *and* an embedding backend
        is configured, we fall back to fuzzy cosine-similarity lookup over
        entries with the same ``template_id``.
        """
        key = (template_id, inputs_hash)
        entry = self._exact_index.get(key)
        if entry is not None:
            self._stats.hits_exact += 1
            return dict(entry.response)

        if query_text is not None and self._embedder is not None:
            query_vec = self._embedder.encode(query_text)
            best_sim = -1.0
            best_entry: _Entry | None = None
            for cand in self._entries:
                if cand.template_id != template_id or cand.embedding is None:
                    continue
                sim = _cosine(query_vec, cand.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = cand
            if best_entry is not None and best_sim >= self.fuzzy_threshold:
                self._stats.hits_fuzzy += 1
                return dict(best_entry.response)

        self._stats.misses += 1
        return None

    def store(
        self,
        template_id: str,
        inputs_hash: str,
        response: dict[str, Any],
        *,
        query_text: str | None = None,
    ) -> None:
        """Persist ``response`` under ``(template_id, inputs_hash)``.

        If ``query_text`` is provided and an embedding backend is configured,
        its embedding is computed and stored to enable later fuzzy lookups.
        """
        embedding: list[float] | None = None
        if query_text is not None and self._embedder is not None:
            embedding = list(self._embedder.encode(query_text))

        entry = _Entry(
            template_id=str(template_id),
            inputs_hash=str(inputs_hash),
            response=dict(response),
            query_text=query_text,
            embedding=embedding,
        )
        self._append_in_memory(entry)
        self._persist(entry)
        self._stats.stores += 1

    def stats(self) -> dict[str, int]:
        """Return hit/miss counters plus the current entry count."""
        out = self._stats.as_dict()
        out["entries"] = len(self._entries)
        return out


__all__ = [
    "SemanticCache",
    "CacheStats",
    "EmbeddingBackend",
    "cache_prefix_key",
    "cassette_digest_matches",
    "hash_inputs",
]
