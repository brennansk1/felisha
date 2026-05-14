"""llama.cpp HTTP-server adapter.

Targets the official ``llama-server`` (and the Python build's
``python -m llama_cpp.server``) which both expose an OpenAI-compatible
``/v1/chat/completions`` endpoint. llama.cpp historically has the best
single-stream throughput and lowest inter-token latency at high load.

Structured output: llama.cpp supports GBNF grammars natively but also
honours the OpenAI ``response_format={"type": "json_object"}`` /
``json_schema`` hints. We send ``json_schema`` when the caller provides
one, and fall back to ``json_object`` otherwise — matching the contract
:class:`OllamaClient.parse` already exposes.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from causalrag.llm.engines.base import EngineNotAvailable, register_engine
from causalrag.llm.ollama_client import LLMResponse, SchemaValidationFailed

T = TypeVar("T", bound=BaseModel)


class LlamaCppServerAdapter:
    """OpenAI-compatible client for llama.cpp's HTTP server.

    Best single-stream throughput; lowest ITL at high load. Use this when
    a single user is hammering one box and you want the fastest
    interactive feel.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        seed: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 600.0,
        api_key: str | None = None,
        transport: Any = None,
        max_retries: int = 2,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = (base_url or "http://127.0.0.1:8080").rstrip("/")
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.api_key = api_key
        self.max_retries = max_retries
        # ``transport`` lets tests inject a fake without needing httpx; in
        # production we lazily import httpx on first use.
        self._transport = transport

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
        body = self._build_body(
            prompt=prompt,
            system=system,
            json_schema=json_schema,
            extra_options=extra_options,
        )

        raw = self._chat(body)
        parsed, errs = _try_parse(raw, schema)
        retries = 0
        errors: list[dict[str, Any]] = []
        last_raw = raw
        while parsed is None and retries < self.max_retries:
            retries += 1
            errors.append({"attempt": retries, "errors": errs, "raw": last_raw})
            body["messages"] = _corrective_messages(system, prompt, last_raw, errs)
            last_raw = self._chat(body)
            parsed, errs = _try_parse(last_raw, schema)

        if parsed is None:
            errors.append({"attempt": retries + 1, "errors": errs, "raw": last_raw})
            raise SchemaValidationFailed(errors, last_raw)

        return LLMResponse(
            parsed=parsed,
            raw=last_raw,
            model=self.model,
            model_digest=None,  # llama.cpp doesn't expose a stable digest
            seed=self.seed,
            cassette_key=f"llamacpp:{uuid4().hex}",
            source="live",
            retries=retries,
            options=body.get("metadata", {}),
        )

    def healthcheck(self) -> bool:
        """Hit ``/health`` (llama.cpp's standard probe). Returns False on error."""
        try:
            client = self._http()
        except EngineNotAvailable:
            return False
        try:
            r = client.get(f"{self.base_url}/health")
            return r.status_code == 200
        except Exception:
            return False
        finally:
            try:
                client.close()
            except Exception:
                pass

    # --- Internals -----------------------------------------------------------

    def _build_body(
        self,
        *,
        prompt: str,
        system: str,
        json_schema: dict[str, Any] | None,
        extra_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if json_schema is not None:
            response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": "causalrag_schema", "schema": json_schema},
            }
        else:
            response_format = {"type": "json_object"}

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "response_format": response_format,
            "stream": False,
        }
        if extra_options:
            body.update(extra_options)
        return body

    def _chat(self, body: dict[str, Any]) -> str:
        if self._transport is not None:
            return str(self._transport.chat(body))
        client = self._http()
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            r = client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
            r.raise_for_status()
            data = r.json()
            return str(data["choices"][0]["message"]["content"])
        finally:
            client.close()

    def _http(self) -> Any:
        try:
            import httpx
        except ImportError as exc:
            raise EngineNotAvailable("llamacpp", "httpx not installed") from exc
        return httpx.Client(timeout=self.timeout)


def _try_parse(
    raw: str, schema: type[T]
) -> tuple[T | None, list[dict[str, Any]]]:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, [{"type": "json_decode", "msg": str(e)}]
    try:
        return schema.model_validate(obj), []
    except ValidationError as e:
        return None, e.errors()


def _corrective_messages(
    system: str, prompt: str, bad_raw: str, errs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    msgs.append({"role": "assistant", "content": bad_raw})
    msgs.append(
        {
            "role": "user",
            "content": (
                "Your previous response failed schema validation. Return ONLY a "
                "corrected JSON object — no prose, no markdown fences.\n\n"
                f"VALIDATION ERRORS:\n{json.dumps(errs, indent=2)}"
            ),
        }
    )
    return msgs


register_engine("llamacpp", LlamaCppServerAdapter)


__all__ = ["LlamaCppServerAdapter"]
