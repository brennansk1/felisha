"""Ollama adapter — thin wrapper around the existing :class:`OllamaClient`.

This adapter is intentionally a pass-through: it preserves all of
``OllamaClient``'s behaviour (cassette replay, JSON-schema retry loop,
model-digest capture) so existing call-sites keep working when the master
loop is switched to the engine-abstraction API. The only thing it adds is
a uniform ``healthcheck()`` and the ``EngineNotAvailable`` contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from causalrag.llm.engines.base import EngineNotAvailable, register_engine
from causalrag.llm.ollama_client import LLMResponse, OllamaClient

T = TypeVar("T", bound=BaseModel)


class OllamaAdapter:
    """Wraps :class:`OllamaClient` to satisfy the ``InferenceEngine`` protocol."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        seed: int = 0,
        temperature: float = 0.0,
        num_ctx: int = 8192,
        cassette_dir: Path | None = None,
        transport: Any = None,
        allow_live: bool | None = None,
        max_retries: int = 2,
        **_: Any,  # tolerate kwargs from select_engine() the adapter doesn't use
    ) -> None:
        self.model = model
        self.base_url = base_url or "http://127.0.0.1:11434"
        try:
            self._client = OllamaClient(
                model=model,
                base_url=self.base_url,
                seed=seed,
                temperature=temperature,
                num_ctx=num_ctx,
                cassette_dir=cassette_dir,
                transport=transport,
                allow_live=allow_live,
                max_retries=max_retries,
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise EngineNotAvailable("ollama", f"client construction failed: {exc}") from exc

    def parse(
        self,
        *,
        prompt: str,
        schema: type[T],
        system: str = "",
        json_schema: dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        return self._client.parse(
            prompt=prompt,
            schema=schema,
            system=system,
            json_schema=json_schema,
            extra_options=extra_options,
        )

    def healthcheck(self) -> bool:
        """Probe ``/api/tags`` on the underlying Ollama server.

        Returns ``False`` (rather than raising) so ``select_engine("auto")``
        can fall through to another backend. Eager construction failures
        still raise :class:`EngineNotAvailable`.
        """
        try:
            import httpx
        except ImportError:
            return False
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(f"{self.base_url.rstrip('/')}/api/tags")
            return r.status_code == 200
        except Exception:
            return False


register_engine("ollama", OllamaAdapter)


__all__ = ["OllamaAdapter"]
