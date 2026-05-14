"""Ollama HTTP client with cassette replay + JSON-schema retry loop.

Implements the four-layer hallucination guard's Layer 1 and Layer 2 (PDD §16.6):

- **Prevention**: ``format="json"`` is sent on every call; an optional JSON
  schema can be passed for Ollama 0.4+ structured-output enforcement.
- **Schema validation**: every response is parsed by a Pydantic model; failures
  retry once with the validation error fed back as a corrective message.
  Persistent failure raises :class:`SchemaValidationFailed` so the caller can
  fall back to safe defaults (PDD §16.6 Layer 2).

Deterministic seeding (``options.seed``) and model-digest capture
(``/api/show``) are recorded on every successful response and surfaced via
:class:`LLMResponse`. Cassette replay (``.causalrag/cassettes/``) is the
default; live calls require ``CAUSALRAG_REFRESH_LLM=1`` or
``Client(..., allow_live=True)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from causalrag.llm.cassette import CassetteMiss, CassetteStore, refresh_requested

T = TypeVar("T", bound=BaseModel)


class SchemaValidationFailed(Exception):
    def __init__(self, errors: list[dict[str, Any]], last_response: str) -> None:
        super().__init__(
            f"Pydantic schema validation failed after retries; {len(errors)} attempt(s)"
        )
        self.errors = errors
        self.last_response = last_response


@dataclass
class LLMResponse:
    """Wraps a parsed LLM response with reproducibility metadata."""

    parsed: BaseModel
    raw: str
    model: str
    model_digest: str | None
    seed: int
    cassette_key: str
    source: str  # "cassette" | "live"
    retries: int = 0
    options: dict[str, Any] = field(default_factory=dict)


class _Transport(Protocol):
    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str,
        options: dict[str, Any],
        fmt: str | dict,
    ) -> str: ...

    def show(self, model: str) -> dict[str, Any]: ...


class HttpxTransport:
    """Real Ollama transport using httpx (sync)."""

    def __init__(self, base_url: str, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _client(self):
        import httpx

        return httpx.Client(timeout=self.timeout)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str,
        options: dict[str, Any],
        fmt: str | dict,
    ) -> str:
        with self._client() as client:
            r = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "format": fmt,
                    "options": options,
                },
            )
            r.raise_for_status()
            return str(r.json()["response"])

    def show(self, model: str) -> dict[str, Any]:
        with self._client() as client:
            r = client.post(f"{self.base_url}/api/show", json={"name": model})
            r.raise_for_status()
            data = r.json()
            assert isinstance(data, dict)
            return data


class FakeOllamaTransport:
    """In-memory transport that returns scripted responses keyed by prompt.

    Use in tests; never hits the network. Construct with a mapping from prompt
    *substrings* (longest match wins) to response payloads. Each payload may be
    a raw JSON string or a Python object that will be ``json.dumps``-ed.
    """

    def __init__(
        self,
        responses: dict[str, str | dict | list],
        *,
        model_digest: str = "sha256:fake_digest_000000000000000000000000000000000000000000000000000000000000",
    ) -> None:
        self._responses = responses
        self._digest = model_digest
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str,
        options: dict[str, Any],
        fmt: str | dict,
    ) -> str:
        self.calls.append(
            {"model": model, "prompt": prompt, "system": system, "options": options, "fmt": fmt}
        )
        match = ""
        for key in self._responses:
            if key and key in prompt and len(key) > len(match):
                match = key
        if not match:
            if "" in self._responses:
                match = ""
            else:
                raise KeyError(f"No FakeOllamaTransport response matches prompt: {prompt[:80]!r}")
        value = self._responses[match]
        if isinstance(value, str):
            return value
        return json.dumps(value)

    def show(self, model: str) -> dict[str, Any]:
        return {"digest": self._digest, "modelfile": f"# fake model {model}"}


class OllamaClient:
    """High-level Ollama client used by the discovery, hypothesis, and
    interpretation stages.

    Construction parameters:

    - ``base_url``: Ollama HTTP endpoint.
    - ``model``: the model name (e.g. ``qwen3:14b-q4_K_M``).
    - ``seed``: deterministic seed forwarded to every ``/api/generate`` call.
    - ``cassette_dir``: directory for record-replay cassettes. Pass
      ``.causalrag/cassettes/`` for project-scoped persistence.
    - ``transport``: dependency-injected transport. Defaults to httpx; pass
      :class:`FakeOllamaTransport` in tests.
    - ``allow_live``: if False, missing cassettes raise :class:`CassetteMiss`
      instead of hitting the network.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        seed: int = 0,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        cassette_dir: Path | None = None,
        transport: _Transport | None = None,
        allow_live: bool | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.seed = seed
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_retries = max_retries

        self.cassettes = CassetteStore(cassette_dir) if cassette_dir is not None else None
        self.transport: _Transport = transport or HttpxTransport(base_url)
        self.allow_live = refresh_requested() if allow_live is None else allow_live
        self._digest_cache: dict[str, str | None] = {}

    # --- Public API -----------------------------------------------------------

    def parse(
        self,
        *,
        prompt: str,
        schema: type[T],
        system: str = "",
        json_schema: dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a response, parse it via ``schema``, retry on validation error.

        ``json_schema`` is forwarded to Ollama 0.4+'s structured-output
        enforcement when provided; falls back to ``format="json"`` otherwise.
        """
        options = self._merge_options(extra_options)
        fmt: str | dict[str, Any] = json_schema if json_schema is not None else "json"
        fmt_key = "json" if isinstance(fmt, str) else json.dumps(fmt, sort_keys=True)
        key = (
            self.cassettes.key_for(self.model, system, prompt, fmt_key, options)
            if self.cassettes
            else "no-cassette"
        )

        cached = self.cassettes.load(key) if self.cassettes else None
        if cached is not None:
            parsed, raw, retries, errs = self._validate_cached(cached["response"], schema, key)
            if parsed is not None:
                return LLMResponse(
                    parsed=parsed,
                    raw=raw,
                    model=self.model,
                    model_digest=cached.get("model_digest"),
                    seed=self.seed,
                    cassette_key=key,
                    source="cassette",
                    retries=retries,
                    options=options,
                )
            if not self.allow_live:
                raise SchemaValidationFailed(errs, raw)

        if not self.allow_live:
            raise CassetteMiss(key)

        digest = self._fetch_digest()
        raw = self.transport.generate(
            model=self.model, prompt=prompt, system=system, options=options, fmt=fmt
        )
        errors: list[dict[str, Any]] = []
        parsed: BaseModel | None
        parsed, errs = self._try_parse(raw, schema)
        retries = 0
        last_raw = raw
        while parsed is None and retries < self.max_retries:
            retries += 1
            errors.append({"attempt": retries, "errors": errs, "raw": last_raw})
            corrective = self._corrective_prompt(prompt, last_raw, errs)
            last_raw = self.transport.generate(
                model=self.model,
                prompt=corrective,
                system=system,
                options=options,
                fmt=fmt,
            )
            parsed, errs = self._try_parse(last_raw, schema)

        if parsed is None:
            errors.append({"attempt": retries + 1, "errors": errs, "raw": last_raw})
            raise SchemaValidationFailed(errors, last_raw)

        if self.cassettes:
            self.cassettes.save(
                key,
                {
                    "model": self.model,
                    "model_digest": digest,
                    "seed": self.seed,
                    "system": system,
                    "prompt": prompt,
                    "options": options,
                    "format": fmt_key,
                    "response": last_raw,
                    "retries": retries,
                },
            )

        return LLMResponse(
            parsed=parsed,
            raw=last_raw,
            model=self.model,
            model_digest=digest,
            seed=self.seed,
            cassette_key=key,
            source="live",
            retries=retries,
            options=options,
        )

    # --- Internal helpers -----------------------------------------------------

    def _merge_options(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        opts = {
            "seed": self.seed,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            # Allow up to 4096 output tokens — the discovery + expert responses
            # for medium-sized datasets routinely exceed Ollama's default of
            # ~128. Without this the model truncates and we get partial JSON
            # that fails Pydantic validation.
            "num_predict": 4096,
        }
        if extra:
            opts.update(extra)
        return opts

    def _try_parse(
        self, raw: str, schema: type[T]
    ) -> tuple[T | None, list[dict[str, Any]]]:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            return None, [{"type": "json_decode", "msg": str(e)}]
        try:
            return schema.model_validate(obj), []
        except ValidationError as e:
            return None, e.errors()

    def _validate_cached(
        self, raw: str, schema: type[T], _key: str
    ) -> tuple[T | None, str, int, list[dict[str, Any]]]:
        parsed, errs = self._try_parse(raw, schema)
        return parsed, raw, 0, errs

    def _corrective_prompt(
        self, original: str, bad_raw: str, errs: list[dict[str, Any]]
    ) -> str:
        return (
            f"{original}\n\n"
            "Your previous response failed schema validation. "
            "Return ONLY a corrected JSON object — no prose, no markdown fences.\n\n"
            f"PREVIOUS RESPONSE:\n{bad_raw}\n\n"
            f"VALIDATION ERRORS:\n{json.dumps(errs, indent=2)}\n"
        )

    def _fetch_digest(self) -> str | None:
        if self.model in self._digest_cache:
            return self._digest_cache[self.model]
        digest: str | None = None
        try:
            info = self.transport.show(self.model)
            digest = info.get("digest") or info.get("modelfile_digest")
            if isinstance(digest, str) and not digest:
                digest = None
        except Exception:
            digest = None
        self._digest_cache[self.model] = digest
        return digest


__all__ = [
    "OllamaClient",
    "FakeOllamaTransport",
    "HttpxTransport",
    "LLMResponse",
    "SchemaValidationFailed",
    "CassetteMiss",
]
