"""
DeepSeek backend.

Implements :class:`LLMBackend` by subclassing the Anthropic backend and
pointing it at DeepSeek's Anthropic-compatible endpoint
(``https://api.deepseek.com/anthropic``).

DeepSeek specifics (per the DeepSeek Anthropic-API docs)
--------------------------------------------------------
* The endpoint accepts exactly two model names — ``deepseek-v4-pro``
  and ``deepseek-v4-flash``.  This backend defaults to ``deepseek-v4-pro``
  and requests maximum reasoning on every call: thinking is always
  enabled and ``output_config={"effort": "max"}`` is sent (the
  ``thinking`` field is requested bare — ``{"type": "enabled"}`` without
  ``budget_tokens``: the endpoint accepts that shape (verified live) and
  ignores the budget field anyway, so the harness never sends it, and
  reasoning depth is controlled via ``output_config.effort``).
  (An earlier revision used the invented name ``deepseek-v4-pro-max``,
  which the endpoint rejects with a 400 — the "max" lives in the effort
  parameter, not the model name.)
* ``cache_control`` and ``anthropic-beta`` headers are ignored by the
  endpoint, so prompt-cache annotations are harmless but useless — they
  are skipped anyway because the backend always has a ``base_url``
  (``is_local`` semantics in the parent class).
* Temperature is supported in the range [0.0, 2.0]; unlike the Anthropic
  API there is no requirement to force temperature=1 when thinking is
  enabled (the parent only enforces that for non-local clients).
* API key via the ``DEEPSEEK_API_KEY`` environment variable.
* Pricing (per https://api-docs.deepseek.com/quick_start/pricing/, in
  $/1M tokens) — ``deepseek-v4-pro``: input $0.435 (cache miss),
  $0.003625 (cache hit), output $0.87.  ``deepseek-v4-flash``: input
  $0.14 (miss), $0.0028 (hit), output $0.28.  Both models have a 1M
  context length.  DeepSeek's server-side context caching is automatic
  and carries no creation charge, so ``calculate_cost`` is overridden
  to bill cache reads at the published hit price rather than the
  parent's Anthropic heuristic (10% of the input price — ~12x the
  real hit rate).
"""

from __future__ import annotations

import os

from ..llm_backend import StreamHandler, map_effort
from .anthropic_backend import AnthropicBackend


class DeepSeekBackend(AnthropicBackend):
    """DeepSeek backend over the Anthropic-compatible endpoint.

    Uses the ``deepseek-v4-pro`` model with thinking (reasoning) always
    enabled at maximum effort (``output_config.effort=max``).
    """

    DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"
    DEFAULT_MODEL = "deepseek-v4-pro"

    #: Effort levels DeepSeek's ``output_config.effort`` accepts (per the
    #: DeepSeek API docs — "low/high/max"; no medium, no xhigh) and the
    #: default.  Thinking always stays on; ``low`` is the shallowest.
    EFFORT_LEVELS: tuple[str, ...] = ("low", "high", "max")
    DEFAULT_EFFORT = "max"

    # $/1M tokens — https://api-docs.deepseek.com/quick_start/pricing/
    MODEL_PRICING: dict[str, dict[str, float]] = {
        "deepseek-v4-pro": {
            "input_token_cost": 0.435,
            "output_token_cost": 0.87,
            "cache_read_cost": 0.003625,
        },
        "deepseek-v4-flash": {
            "input_token_cost": 0.14,
            "output_token_cost": 0.28,
            "cache_read_cost": 0.0028,
        },
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "deepseek-v4-pro": "DeepSeek V4 Pro Max",
        "deepseek-v4-flash": "DeepSeek V4 Flash",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "deepseek-v4-pro": 1_000_000,
        "deepseek-v4-flash": 1_000_000,
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
        # parent sends a thinking field per _thinking_config() (the bare
        # {"type": "enabled"} — DeepSeek's endpoint accepts it without
        # budget_tokens and ignores the budget per the docs) and uses
        # event-based streaming so thinking blocks render dimmed in the
        # UI instead of leaking into the conversation.
        self._thinking_enabled = True
        self._supports_thinking_api = True
        self._use_thinking_stream = True

    # ── Request customisation ────────────────────────────────────────

    def _thinking_config(self) -> dict | None:
        """Bare ``{"type": "enabled"}`` — no ``budget_tokens``.

        DeepSeek's Anthropic-compatible endpoint accepts the field
        without a budget (verified live) and reasoning depth is set
        via ``output_config.effort`` instead.  Sending the parent's
        8192 budget would be cargo-cult: parsed, ignored, and a
        source of confusion.
        """
        return {"type": "enabled"}

    def _extra_stream_kwargs(self) -> dict:
        """Request the resolved reasoning effort on every call.

        DeepSeek's Anthropic-compatible API ignores ``budget_tokens`` in
        the ``thinking`` field; reasoning depth is instead controlled via
        ``output_config.effort`` (the only ``output_config`` field the
        endpoint supports).  The effort defaults to ``max`` (current
        behaviour); a user ``--effort`` overrides it, clamped to the
        levels DeepSeek accepts (``low``/``high``/``max``).

        ``output_config`` is a DeepSeek extension unknown to the
        ``anthropic`` SDK, whose ``Messages.stream()`` accepts only its
        documented parameters — passing it directly raises
        ``TypeError: Messages.stream() got an unexpected keyword
        argument 'output_config'``.  It is therefore sent via
        ``extra_body``, which the SDK deep-merges into the JSON request
        body so the field reaches the endpoint as a top-level key.
        """
        # getattr (not attribute access) so a ``__new__``-constructed test
        # fixture that bypasses ``__init__`` still resolves to the model
        # default (max) instead of AttributeError-ing on a missing attr.
        effort = map_effort(getattr(self, "reasoning_effort", None),
                            self.EFFORT_LEVELS, self.DEFAULT_EFFORT)
        return {"extra_body": {"output_config": {"effort": effort}}}

    # ── Cost calculation ─────────────────────────────────────────────

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Bill usage at DeepSeek's published prices.

        Unlike Anthropic (cache reads at 10% of input, writes at 125%),
        DeepSeek's context caching is automatic with no creation
        charge, and cache hits are billed at an explicit published rate
        (~0.8% of the miss price).  ``cache_creation_tokens`` are
        therefore ignored.
        """
        pricing = self.MODEL_PRICING.get(self.model)
        if pricing is None:
            return 0.0
        cost = (
            input_tokens * pricing["input_token_cost"]
            + cache_read_tokens * pricing["cache_read_cost"]
            + output_tokens * pricing["output_token_cost"]
        ) / 1_000_000
        return cost
