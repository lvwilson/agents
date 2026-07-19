"""
DeepSeek backend.

Implements :class:`LLMBackend` by subclassing the Anthropic backend and
pointing it at DeepSeek's Anthropic-compatible endpoint
(``https://api.deepseek.com/anthropic``).

DeepSeek specifics (per the DeepSeek Anthropic-API docs)
--------------------------------------------------------
* ``deepseek-v4-pro-max`` — max-reasoning variant; thinking is always
  enabled and ``output_config={"effort": "max"}`` is sent on every call
  (the ``thinking`` field is supported but ``budget_tokens`` is ignored,
  so reasoning depth is controlled via ``output_config.effort``).
* ``cache_control`` and ``anthropic-beta`` headers are ignored by the
  endpoint, so prompt-cache annotations are harmless but useless — they
  are skipped anyway because the backend always has a ``base_url``
  (``is_local`` semantics in the parent class).
* Temperature is supported in the range [0.0, 2.0]; unlike the Anthropic
  API there is no requirement to force temperature=1 when thinking is
  enabled (the parent only enforces that for non-local clients).
* API key via the ``DEEPSEEK_API_KEY`` environment variable.
* Pricing is not published in the provided docs, so the model is
  deliberately absent from ``MODEL_PRICING`` and costs report as $0
  rather than inventing numbers.
"""

from __future__ import annotations

import os

from ..llm_backend import StreamHandler
from .anthropic_backend import AnthropicBackend


class DeepSeekBackend(AnthropicBackend):
    """DeepSeek backend over the Anthropic-compatible endpoint.

    Thinking (reasoning) is always enabled at maximum effort — that is
    the entire point of selecting a ``-max`` model.
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"
    DEFAULT_MODEL = "deepseek-v4-pro-max"

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "deepseek-v4-pro-max": "DeepSeek V4 Pro Max",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "deepseek-v4-pro-max": 256_000,
    }

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        cache_step: int = 2,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise Exception("DEEPSEEK_API_KEY Environment Variable Unset")

        # The endpoint URL is fixed unless explicitly overridden (e.g. a
        # proxy).  Passing a base_url to the parent gives it "is_local"
        # semantics: no prompt-cache blocks and no anthropic-beta headers
        # (both ignored by DeepSeek anyway), a plain-string system prompt,
        # and no forced temperature=1 alongside thinking.
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

        # Max reasoning: thinking is always on for this backend.  The
        # parent will send thinking={"type": "enabled", "budget_tokens": …}
        # — DeepSeek ignores budget_tokens per the docs — and will use
        # event-based streaming so thinking blocks render dimmed in the
        # UI instead of leaking into the conversation.
        self._thinking_enabled = True
        self._supports_thinking_api = True
        self._use_thinking_stream = True

    # ── Request customisation ────────────────────────────────────────

    def _extra_stream_kwargs(self) -> dict:
        """Request maximum reasoning effort on every call.

        DeepSeek's Anthropic-compatible API ignores ``budget_tokens`` in
        the ``thinking`` field; reasoning depth is instead controlled via
        ``output_config.effort`` (the only ``output_config`` field the
        endpoint supports).
        """
        return {"output_config": {"effort": "max"}}
