"""
Kimi K3 backend.

Implements :class:`LLMBackend` by deriving from
:class:`.openai_compat_backend.OpenAICompatBackend` against the Kimi API
(OpenAI-compatible at ``api.moonshot.ai``).

Kimi K3 specifics
-----------------
* 1 M-token context window
* ``reasoning_effort="max"`` on every call (thinking always enabled)
* Fixed parameters: temperature=1.0, top_p=0.95, n=1, penalties=0
  (omitted from requests per Kimi docs — see ``SEND_TEMPERATURE``)
* Pricing: $3 / M input, $15 / M output, $0.30 / M cache-read input
* API key via ``MOONSHOT_API_KEY`` environment variable
"""

from __future__ import annotations

import os

from ..llm_backend import StreamHandler
from .openai_compat_backend import OpenAICompatBackend


class KimiBackend(OpenAICompatBackend):
    """Kimi K3 chat-completions backend (OpenAI-compatible)."""

    MODEL_PRICING: dict[str, dict[str, float]] = {
        "kimi-k3": {
            "input_token_cost": 3.00,
            "output_token_cost": 15.00,
            "cache_read_cost": 0.30,
        },
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "kimi-k3": "Kimi K3",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "kimi-k3": 1_000_000,
    }

    #: Kimi's max output (reasoning tokens count against this).
    MAX_COMPLETION_TOKENS = 131_072

    #: Kimi docs: temperature/top_p/n/penalties are fixed server-side and
    #: should be omitted from requests.
    SEND_TEMPERATURE = False

    def __init__(
        self,
        model: str = "kimi-k3",
        base_url: str | None = None,
        cache_step: int = 4,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
        **_kwargs,
    ):
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
        import openai
        return openai.RateLimitError

    def _resolve_credentials(self, base_url: str | None) -> tuple[str, str | None]:
        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise Exception("MOONSHOT_API_KEY Environment Variable Unset")
        return api_key, base_url

    def _build_client(self, api_key: str, base_url: str | None):
        import openai
        kwargs: dict = {"api_key": api_key}
        # Kimi's hosted endpoint; a custom base_url (proxy) overrides it.
        kwargs["base_url"] = base_url or "https://api.moonshot.ai/v1"
        return openai.OpenAI(**kwargs)

    def _extra_create_kwargs(self) -> dict:
        """Always request maximum reasoning effort."""
        return {"reasoning_effort": "max"}
