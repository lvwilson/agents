"""
Cerebras backend.

Implements :class:`LLMBackend` by deriving from
:class:`.openai_compat_backend.OpenAICompatBackend` and pointing it at the
Cerebras Inference API via the official ``cerebras_cloud_sdk``
(``pip install cerebras_cloud_sdk``).

Cerebras specifics (per https://inference-docs.cerebras.ai)
-----------------------------------------------------------
* OpenAI-compatible **chat completions** endpoint at
  ``https://api.cerebras.ai`` (overridable via ``base_url`` or the
  ``CEREBRAS_BASE_URL`` env var the SDK already honours).
* API key via the ``CEREBRAS_API_KEY`` environment variable.
* Reasoning is configurable per call with ``reasoning_effort``
  (``none`` / ``low`` / ``medium`` / ``high``) and returned separately in
  ``choices[0].delta.reasoning`` / ``message.reasoning``.  This backend
  streams those reasoning tokens to the UI (dimmed) and never lets them
  leak into the conversation context — see the shared base class.
* ``qwen-3.8-27b`` reasons at ``high`` effort by default; this backend
  keeps that default (it can be lowered/removed per model below).

Pricing (per https://inference-docs.cerebras.ai/support/pricing/, $/1M
tokens):

* ``qwen-3.8-27b`` — input $0.99, output $1.49; context 64k (free) /
  128k (paid); max output 32k (free) / 40k (paid); ~3000 tok/s.
* ``gpt-oss-120b`` — input $0.35, output $0.75; context 65k (free) /
  131k (paid); max output 32k (free) / 40k (paid).

Cerebras' prompt caching is automatic and a *latency* feature — cached
tokens are billed at the full input price (no hit discount), so
``cache_read_cost`` equals ``input_token_cost`` in the table above.
"""

from __future__ import annotations

import os

from ..llm_backend import StreamHandler
from .openai_compat_backend import OpenAICompatBackend


class CerebrasBackend(OpenAICompatBackend):
    """Cerebras Inference backend over the official SDK."""

    DEFAULT_BASE_URL = "https://api.cerebras.ai"
    DEFAULT_MODEL = "qwen-3.8-27b"

    # $/1M tokens — https://inference-docs.cerebras.ai/support/pricing/
    #
    # Cerebras' prompt caching is a *latency* feature, not a discount:
    # "Cached tokens are priced the same whether or not the key is set."
    # So cache_read_cost equals the input cost (no hit discount).
    MODEL_PRICING: dict[str, dict[str, float]] = {
        "qwen-3.8-27b": {
            "input_token_cost": 0.99,
            "output_token_cost": 1.49,
            "cache_read_cost": 0.99,
        },
        "gpt-oss-120b": {
            "input_token_cost": 0.35,
            "output_token_cost": 0.75,
            "cache_read_cost": 0.35,
        },
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "qwen-3.8-27b": "Cerebras Qwen 3.8 27B",
        "gpt-oss-120b": "Cerebras GPT-OSS 120B",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "qwen-3.8-27b": 128_000,
        "gpt-oss-120b": 131_000,
    }

    # Max output tokens per model (paid tier).  Reasoning tokens count
    # against this budget, so we leave headroom below the model ceiling.
    MODEL_MAX_COMPLETION: dict[str, int] = {
        "qwen-3.8-27b": 40_000,
        "gpt-oss-120b": 40_000,
    }

    # Reasoning effort per model.  ``qwen-3.8-27b`` defaults to ``high``
    # on the server; we keep that.  Set a model to ``None`` to let the
    # server use its own default, or ``"none"`` to disable reasoning.
    MODEL_REASONING_EFFORT: dict[str, str | None] = {
        "qwen-3.8-27b": "high",
        "gpt-oss-120b": "medium",
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        cache_step: int = 4,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
        **_kwargs,
    ):
        # Per-model max output (falls back to the class default).
        self.MAX_COMPLETION_TOKENS = (
            self.MODEL_MAX_COMPLETION.get(model, self.MAX_COMPLETION_TOKENS)
        )
        super().__init__(
            model=model,
            base_url=base_url,
            cache_step=cache_step,
            stream_handler=stream_handler,
            temperature=temperature,
            **_kwargs,
        )

    # ── Subclass hooks ───────────────────────────────────────────────

    def _rate_limit_error_class(self) -> type:
        from cerebras.cloud.sdk import RateLimitError
        return RateLimitError

    def _resolve_credentials(self, base_url: str | None) -> tuple[str, str | None]:
        """Read ``CEREBRAS_API_KEY``; fall back to a placeholder for a
        custom (e.g. proxy) base URL with no key set."""
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            if base_url:
                # A custom endpoint (proxy / on-prem) may not require a key.
                api_key = "local"
            else:
                raise Exception("CEREBRAS_API_KEY Environment Variable Unset")
        return api_key, base_url

    def _build_client(self, api_key: str, base_url: str | None):
        """Build the Cerebras SDK client (default base URL if not given)."""
        from cerebras.cloud.sdk import Cerebras
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return Cerebras(**kwargs)

    def _extra_create_kwargs(self) -> dict:
        """Request the model's configured reasoning effort.

        ``reasoning_effort`` is a Cerebras extension; the shared base only
        sends OpenAI-standard parameters, so it is injected here.  A value
        of ``None`` means "let the server use its default" and is omitted.
        """
        effort = self.MODEL_REASONING_EFFORT.get(self.model)
        if effort:
            return {"reasoning_effort": effort}
        return {}
