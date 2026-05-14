"""LLM inference-engine adapters (Sprint 8.1).

Public surface:

  * :class:`InferenceEngine` — Protocol every adapter implements.
  * :class:`EngineNotAvailable` — raised when a backend is missing.
  * :func:`select_engine` — factory keyed by short config name.

Importing this package is cheap; the heavy backends (mlx-lm, vllm, …) are
imported lazily inside their adapters' constructors, so machines without
those packages installed pay nothing for the abstraction.
"""

from causalrag.llm.engines.base import (
    EngineNotAvailable,
    InferenceEngine,
    available_engines,
    register_engine,
    select_engine,
)
from causalrag.llm.engines.llamacpp_adapter import LlamaCppServerAdapter
from causalrag.llm.engines.mlx_adapter import MlxLmAdapter
from causalrag.llm.engines.ollama_adapter import OllamaAdapter
from causalrag.llm.engines.vllm_adapter import VllmAdapter

__all__ = [
    "EngineNotAvailable",
    "InferenceEngine",
    "LlamaCppServerAdapter",
    "MlxLmAdapter",
    "OllamaAdapter",
    "VllmAdapter",
    "available_engines",
    "register_engine",
    "select_engine",
]
