"""Inference-engine abstraction layer (Sprint 8.1).

CausalRAG was originally hard-wired to Ollama (see
:mod:`causalrag.llm.ollama_client`). This module introduces a thin
``InferenceEngine`` Protocol that every backend adapter implements so the
master loop can swap between Ollama, llama.cpp's HTTP server, vLLM, and
Apple's MLX-LM with a single config flag.

Every adapter MUST:

  * mirror :class:`causalrag.llm.ollama_client.OllamaClient.parse`'s
    signature exactly — keyword-only, returning an
    :class:`~causalrag.llm.ollama_client.LLMResponse` whose ``parsed`` is
    a validated Pydantic model;
  * expose a cheap ``healthcheck()`` for hardware/serve detection;
  * raise :class:`EngineNotAvailable` *eagerly* (at construction time)
    when the backend isn't reachable or its Python package isn't
    installed — silent degradation hides production misconfiguration.

The ``select_engine`` factory is the only public entry point external
callers should use. It defers heavy imports until the chosen engine is
actually requested so importing ``causalrag.llm.engines`` is cheap on
machines where, say, ``mlx-lm`` isn't installed.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from causalrag.llm.ollama_client import LLMResponse

T = TypeVar("T", bound=BaseModel)


class EngineNotAvailable(RuntimeError):
    """Raised when an engine's backend (server or Python package) is missing.

    Adapters MUST raise this — never silently fall back to another engine.
    The master loop catches it and surfaces a clear configuration error so
    users notice that, e.g., they asked for vLLM on a machine without CUDA.
    """

    def __init__(self, engine: str, reason: str) -> None:
        super().__init__(f"Inference engine {engine!r} is not available: {reason}")
        self.engine = engine
        self.reason = reason


@runtime_checkable
class InferenceEngine(Protocol):
    """Common interface every engine adapter implements.

    The signature is intentionally identical to
    :meth:`OllamaClient.parse` so existing call-sites work unchanged.
    """

    model: str

    def parse(
        self,
        *,
        prompt: str,
        schema: type[T],
        system: str = "",
        json_schema: dict[str, Any] | None = None,
        extra_options: dict[str, Any] | None = None,
    ) -> LLMResponse: ...

    def healthcheck(self) -> bool: ...


# Registry of engine name -> lazy factory. Populated by adapter modules at
# import time so ``select_engine("ollama", ...)`` works without forcing the
# user to import the adapter class directly.
_REGISTRY: dict[str, Any] = {}


def register_engine(name: str, factory: Any) -> None:
    """Register an adapter factory under a short config name (idempotent)."""
    _REGISTRY[name.lower()] = factory


def available_engines() -> list[str]:
    """Names of all registered engines (irrespective of runtime availability)."""
    return sorted(_REGISTRY)


def select_engine(
    name: str,
    model: str,
    base_url: str | None = None,
    **kwargs: Any,
) -> InferenceEngine:
    """Construct an engine adapter by short name.

    Parameters
    ----------
    name :
        One of ``"ollama"``, ``"llamacpp"``, ``"vllm"``, ``"mlx"``, or
        ``"auto"`` to pick the best available backend for this machine.
    model :
        Model identifier (interpretation is backend-specific).
    base_url :
        HTTP endpoint for server-style backends. Ignored by MLX.
    **kwargs :
        Forwarded to the adapter constructor verbatim.
    """
    # Trigger registration side-effects.
    from causalrag.llm.engines import (  # noqa: F401  (import-for-side-effects)
        llamacpp_adapter,
        mlx_adapter,
        ollama_adapter,
        vllm_adapter,
    )

    key = name.lower()
    if key == "auto":
        return _auto_select(model=model, base_url=base_url, **kwargs)

    if key not in _REGISTRY:
        raise EngineNotAvailable(
            name, f"unknown engine; choose from {available_engines() + ['auto']}"
        )
    factory = _REGISTRY[key]
    return factory(model=model, base_url=base_url, **kwargs)  # type: ignore[no-any-return]


def _auto_select(
    *, model: str, base_url: str | None, **kwargs: Any
) -> InferenceEngine:
    """Try MLX (Apple Silicon native) then Ollama then llama.cpp then vLLM.

    The order matches "fewest moving parts wins": MLX runs in-process on
    M-series; Ollama is the project's historical default; llama.cpp and
    vLLM only win when the user has already stood up a server.
    """
    last_reason: list[str] = []
    for candidate in ("mlx", "ollama", "llamacpp", "vllm"):
        if candidate not in _REGISTRY:
            continue
        try:
            engine = _REGISTRY[candidate](model=model, base_url=base_url, **kwargs)
            if engine.healthcheck():
                return engine  # type: ignore[no-any-return]
            last_reason.append(f"{candidate}: healthcheck failed")
        except EngineNotAvailable as e:
            last_reason.append(f"{candidate}: {e.reason}")
    raise EngineNotAvailable("auto", "; ".join(last_reason) or "no engines registered")


__all__ = [
    "EngineNotAvailable",
    "InferenceEngine",
    "available_engines",
    "register_engine",
    "select_engine",
]
