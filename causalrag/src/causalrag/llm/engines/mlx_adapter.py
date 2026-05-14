"""MLX-LM adapter — Apple-Silicon native inference (M-series only).

`mlx-lm <https://github.com/ml-explore/mlx-examples/tree/main/llms>`_
runs models directly against Apple's Metal stack with no HTTP server in
between — the lowest setup overhead on Mac dev boxes. Useful when the
researcher is iterating locally on an M-series laptop.

Structured output: ``mlx-lm`` does not (as of this writing) ship native
JSON-schema enforcement. We rely on prompt-level instruction plus the
adapter-level retry loop (same as the OpenAI-compatible adapters above).
"""

from __future__ import annotations

import json
import platform
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from causalrag.llm.engines.base import EngineNotAvailable, register_engine
from causalrag.llm.ollama_client import LLMResponse, SchemaValidationFailed

T = TypeVar("T", bound=BaseModel)


class MlxLmAdapter:
    """In-process MLX-LM runner.

    Construction eagerly attempts to import ``mlx_lm`` and raises
    :class:`EngineNotAvailable` when the package isn't installed or when
    we're not on Apple Silicon — *never* silently degrade.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,  # accepted for signature parity, unused
        seed: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        runner: Any = None,
        max_retries: int = 2,
        skip_hardware_check: bool = False,
        **_: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url  # kept for diagnostics
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        if not skip_hardware_check and platform.system() != "Darwin":
            raise EngineNotAvailable(
                "mlx", f"requires macOS / Apple Silicon, got {platform.system()}"
            )
        if not skip_hardware_check and platform.machine() not in {"arm64", "aarch64"}:
            raise EngineNotAvailable(
                "mlx", f"requires arm64 CPU, got {platform.machine()}"
            )

        if runner is not None:
            self._runner = runner
        else:
            self._runner = _load_runner(model)

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
        full_prompt = _compose_prompt(system, prompt, json_schema)
        opts = self._merge_options(extra_options)

        raw = self._runner.generate(prompt=full_prompt, **opts)
        parsed, errs = _try_parse(raw, schema)
        retries = 0
        errors: list[dict[str, Any]] = []
        last_raw = raw
        while parsed is None and retries < self.max_retries:
            retries += 1
            errors.append({"attempt": retries, "errors": errs, "raw": last_raw})
            corrective = (
                f"{full_prompt}\n\nYour previous response failed schema "
                "validation. Return ONLY a corrected JSON object — no prose, "
                "no markdown fences.\n\n"
                f"PREVIOUS RESPONSE:\n{last_raw}\n\n"
                f"VALIDATION ERRORS:\n{json.dumps(errs, indent=2)}"
            )
            last_raw = self._runner.generate(prompt=corrective, **opts)
            parsed, errs = _try_parse(last_raw, schema)

        if parsed is None:
            errors.append({"attempt": retries + 1, "errors": errs, "raw": last_raw})
            raise SchemaValidationFailed(errors, last_raw)

        return LLMResponse(
            parsed=parsed,
            raw=last_raw,
            model=self.model,
            model_digest=None,
            seed=self.seed,
            cassette_key=f"mlx:{uuid4().hex}",
            source="live",
            retries=retries,
            options=opts,
        )

    def healthcheck(self) -> bool:
        """The runner is in-process; if construction succeeded, we're alive."""
        return self._runner is not None

    # --- Internals -----------------------------------------------------------

    def _merge_options(self, extra: dict[str, Any] | None) -> dict[str, Any]:
        opts = {
            "max_tokens": self.max_tokens,
            "temp": self.temperature,
            "seed": self.seed,
        }
        if extra:
            opts.update(extra)
        return opts


def _compose_prompt(
    system: str, prompt: str, json_schema: dict[str, Any] | None
) -> str:
    """MLX-LM ``generate`` takes a single string — we synthesise it here.

    The JSON-schema hint is appended to the system block so the model has
    a concrete shape to target even though MLX cannot enforce it.
    """
    parts: list[str] = []
    if system:
        parts.append(system.strip())
    if json_schema is not None:
        parts.append(
            "Return a JSON object matching exactly this schema:\n"
            f"{json.dumps(json_schema)}"
        )
    else:
        parts.append("Return a single JSON object. No prose, no fences.")
    parts.append(prompt)
    return "\n\n".join(parts)


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


def _load_runner(model: str) -> Any:
    """Import mlx_lm and return an object with a ``.generate(prompt, **kw)`` method.

    Wrapping the real ``mlx_lm.generate`` call in a tiny shim keeps the
    test seam clean (tests pass in a fake ``runner`` instead).
    """
    try:
        from mlx_lm import generate as mlx_generate  # type: ignore[import-not-found]
        from mlx_lm import load as mlx_load  # type: ignore[import-not-found]
    except ImportError as exc:
        raise EngineNotAvailable("mlx", "mlx-lm not installed") from exc

    try:
        model_obj, tokenizer = mlx_load(model)
    except Exception as exc:
        raise EngineNotAvailable("mlx", f"failed to load model {model!r}: {exc}") from exc

    class _Runner:
        def generate(self, *, prompt: str, **kw: Any) -> str:
            return str(mlx_generate(model_obj, tokenizer, prompt=prompt, **kw))

    return _Runner()


register_engine("mlx", MlxLmAdapter)


__all__ = ["MlxLmAdapter"]
