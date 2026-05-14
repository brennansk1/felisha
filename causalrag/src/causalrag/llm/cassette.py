"""Cassette persistence for LLM calls (PDD §16.7).

Every LLM call is keyed by ``sha256(model + system + prompt + format + options)``
and persisted under ``.causalrag/cassettes/<hash>.json``. Replay is the default
in tests; the live path is taken when no cassette exists *and* the caller passes
``allow_live=True`` (or sets ``CAUSALRAG_REFRESH_LLM=1``).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CassetteHit:
    key: str
    payload: dict[str, Any]
    source: str  # "disk" | "live"


def _stable_key(model: str, system: str, prompt: str, fmt: str, options: dict[str, Any]) -> str:
    """Hash the input tuple. Options are sorted for stability."""
    blob = json.dumps(
        {
            "model": model,
            "system": system,
            "prompt": prompt,
            "format": fmt,
            "options": dict(sorted(options.items())),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def refresh_requested() -> bool:
    return os.environ.get("CAUSALRAG_REFRESH_LLM") == "1"


class CassetteStore:
    """File-backed cassette store rooted at ``.causalrag/cassettes/``.

    Two modes:

    - **Replay** (default): if a cassette for the request key exists, return its
      payload. Otherwise raise :class:`CassetteMiss` so the caller can decide
      whether to allow a live request.
    - **Live capture**: the caller passes a live result back via :meth:`save`,
      which writes a new cassette atomically.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key_for(
        self, model: str, system: str, prompt: str, fmt: str, options: dict[str, Any]
    ) -> str:
        return _stable_key(model, system, prompt, fmt, options)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def load(self, key: str) -> dict[str, Any] | None:
        p = self.path_for(key)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def save(self, key: str, payload: dict[str, Any]) -> Path:
        p = self.path_for(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
        return p


class CassetteMiss(Exception):
    """Raised when replay-only mode has no cached response for a call."""

    def __init__(self, key: str) -> None:
        super().__init__(f"No cassette for key {key} (set CAUSALRAG_REFRESH_LLM=1 to record)")
        self.key = key
