"""
MiniMax backend.

Implements :class:`LLMBackend` by subclassing the Anthropic backend and
pointing it at MiniMax's Anthropic-compatible endpoint
(``https://api.minimax.io/anthropic``).

MiniMax specifics (per the MiniMax API docs)
--------------------------------------------
* The endpoint accepts MiniMax M-series models (M3, M2.7, M2.5, M2.1,
  M2) via the Anthropic API format.  This backend currently supports
  ``MiniMax-M2.5`` with a 204,800-token context window.
* ``cache_control`` and ``anthropic-beta`` headers are ignored by the
  endpoint, so prompt-cache annotations are harmless but useless — they
  are skipped anyway because the backend always has a ``base_url``
  (``is_local`` semantics in the parent class).
* Thinking cannot be disabled for M2.x models; the endpoint always
  emits thinking blocks regardless of the ``thinking`` parameter.
* API key via the ``MINIMAX_API_KEY`` environment variable.  Keys must
  begin with ``sk-api-kt`` to prevent credential leakage.
* Pricing (in $/1M tokens) — ``MiniMax-M2.5``: input $0.30,
  cache read $0.03, output $1.20.
"""

from __future__ import annotations

import os

from ..llm_backend import StreamHandler
from .anthropic_backend import AnthropicBackend


class MinimaxBackend(AnthropicBackend):
    """MiniMax backend over the Anthropic-compatible endpoint.

    Uses the ``MiniMax-M2.5`` model.  Thinking is always on for M2.x
    models (cannot be disabled by the endpoint).
    """

    DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
    DEFAULT_MODEL = "MiniMax-M2.5"

    # $/1M tokens — per MiniMax API pricing
    MODEL_PRICING: dict[str, dict[str, float]] = {
        "MiniMax-M2.5": {
            "input_token_cost": 0.30,
            "output_token_cost": 1.20,
            "cache_read_cost": 0.03,
        },
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "MiniMax-M2.5": "MiniMax M2.5",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "MiniMax-M2.5": 204_800,
    }

    # Models that route to MiniMax (require specific API key validation)
    MINIMAX_MODELS = {"MiniMax-M2.5"}

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        cache_step: int = 2,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        api_key = os.getenv("MINIMAX_API_KEY")
        if not api_key:
            raise Exception("MINIMAX_API_KEY Environment Variable Unset")

        # Defensive check: MiniMax API keys must start with "sk-api-kt"
        # to prevent credential leaks.
        if not api_key.startswith("sk-api-kt"):
            raise ValueError(
                f"Invalid API key for MiniMax model '{model}'. "
                "API key must begin with 'sk-api-kt' to prevent credential leakage. "
                "Please use a valid MiniMax API key."
            )

        # Passing a base_url to the parent gives it "is_local" semantics:
        # no prompt-cache blocks and no anthropic-beta headers (both
        # ignored by MiniMax anyway), a plain-string system prompt, and
        # no forced temperature=1 alongside thinking.
        super().__init__(
            model=model,
            base_url=base_url or self.DEFAULT_BASE_URL,
            cache_step=cache_step,
            stream_handler=stream_handler,
            temperature=temperature,
            **kwargs,
        )

        # The parent built a placeholder client with api_key="local"
        # (its CLAUDE_API_KEY branch); swap in the real credentials.
        self._client = self._anthropic.Anthropic(
            api_key=api_key,
            base_url=self.base_url,
        )

        # Thinking is always on for M2.x models — cannot be disabled.
        # Force event-based streaming so thinking blocks render in the UI.
        self._thinking_enabled = True
        self._supports_thinking_api = True
        self._use_thinking_stream = True
