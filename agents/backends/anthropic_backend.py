"""
Anthropic (Claude) backend.

Implements :class:`LLMBackend` using the ``anthropic`` Python SDK.
"""

from __future__ import annotations

import os

from ..llm_backend import LLMBackend, StreamHandler, RATE_LIMIT, TRANSIENT


class AnthropicBackend(LLMBackend):
    """Claude backend with streaming, prompt caching, and retry logic."""

    MODEL_PRICING = {
        "claude-3-5-sonnet-20240620":  {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-3-5-sonnet-20241022":  {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-3-7-sonnet-20250219":  {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-sonnet-4-20250514":    {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-sonnet-4-5-20250929":  {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-sonnet-4-6":           {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-opus-4-6":             {"input_token_cost": 5.00, "output_token_cost": 25.00},
        "MiniMax-M2.5" :               {"input_token_cost": 0.3,  "output_token_cost": 1.2},
    }

    MODEL_DISPLAY_NAMES = {
        "claude-3-5-sonnet-20240620":  "Claude 3.5 Sonnet",
        "claude-3-5-sonnet-20241022":  "Claude 3.5 Sonnet v2",
        "claude-3-7-sonnet-20250219":  "Claude 3.7 Sonnet",
        "claude-sonnet-4-20250514":    "Claude Sonnet 4",
        "claude-sonnet-4-5-20250929":  "Claude Sonnet 4.5",
        "claude-sonnet-4-6":           "Claude Sonnet 4.6",
        "claude-opus-4-6":             "Claude Opus 4.6",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "claude-3-5-sonnet-20240620":  200_000,
        "claude-3-5-sonnet-20241022":  200_000,
        "claude-3-7-sonnet-20250219":  200_000,
        "claude-sonnet-4-20250514":    200_000,
        "claude-sonnet-4-5-20250929":  200_000,
        "claude-sonnet-4-6":           200_000,
        "claude-opus-4-6":             200_000,
        "MiniMax-M2.5":                200_000,
    }

    # Models that route to MiniMax (require special API key validation)
    MINIMAX_MODELS = {"MiniMax-M2.5"}

    # Models that support the thinking extension (reasoning tokens)
    THINKING_MODELS = {
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-5-20250929",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    }

    # Models that require adaptive thinking type (vs "enabled")
    ADAPTIVE_THINKING_MODELS = {
        "claude-sonnet-4-6",
        "claude-opus-4-6",
    }

    # Default thinking configuration
    DEFAULT_THINKING_BUDGET = 8192
    MAX_OUTPUT_TOKENS = 64000

    @staticmethod
    def _get_thinking_enabled() -> bool:
        """Check if thinking is enabled via environment variable.

        Set CLAUDE_THINKING_ENABLED=true to enable extended thinking.
        Defaults to disabled to avoid unexpected cost increases (thinking
        tokens are billed at output token rates).
        """
        env_enabled = os.getenv("CLAUDE_THINKING_ENABLED", "false").lower().strip()
        return env_enabled in ("true", "1", "yes")

    @staticmethod
    def _get_thinking_budget() -> int:
        """Get thinking token budget from environment variable."""
        try:
            budget = int(os.getenv("CLAUDE_THINKING_BUDGET", str(AnthropicBackend.DEFAULT_THINKING_BUDGET)))
            return max(1024, budget)  # API requires minimum 1024 tokens for enabled mode
        except (ValueError, TypeError):
            return AnthropicBackend.DEFAULT_THINKING_BUDGET

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        base_url: str | None = None,
        cache_step: int = 2,
        stream_handler: StreamHandler | None = None,
        temperature: float = 0.6,
        **_kwargs,
    ):
        super().__init__(model=model, base_url=base_url, stream_handler=stream_handler, temperature=temperature)

        # Lazy import — only pull in anthropic when this backend is used
        import anthropic as _anthropic
        self._anthropic = _anthropic

        api_key = os.getenv("CLAUDE_API_KEY")

        if base_url:
            # Defensive check: MiniMax models require specific API key prefix
            # to prevent credential leaks. Only allow keys starting with "sk-api-kt"
            if model in self.MINIMAX_MODELS:
                if not api_key or not api_key.startswith("sk-api-kt"):
                    raise ValueError(
                        f"Invalid API key for MiniMax model '{model}'. "
                        "API key must begin with 'sk-api-kt' to prevent credential leakage. "
                        "Please use a valid MiniMax API key."
                    )
            if not api_key:
                api_key = "local"
            self._client = _anthropic.Anthropic(api_key=api_key, base_url=base_url)
        else:
            if not api_key:
                raise Exception("CLAUDE_API_KEY Environment Variable Unset")
            self._client = _anthropic.Anthropic(api_key=api_key)

        self.cache_step = cache_step

        # Thinking extension configuration
        self._thinking_enabled = self._get_thinking_enabled()
        self._thinking_budget = self._get_thinking_budget()

        # Whether the model is known to support the thinking API parameter.
        # For known Anthropic models we can send the thinking config to the
        # server.  For local/unknown models we still use event-based
        # streaming (to surface any thinking blocks the server may send)
        # but we don't inject the thinking API parameter — the server may
        # not understand it.
        self._supports_thinking_api = model in self.THINKING_MODELS

        # Use event-based streaming whenever the user has opted in to
        # thinking, regardless of model name.  This lets local models
        # that emit thinking blocks surface them in the UI without
        # requiring the model name to be in THINKING_MODELS.
        self._use_thinking_stream = self._thinking_enabled

        # Validate thinking budget against context window and max_tokens
        if self._supports_thinking_api and self._thinking_enabled:
            context_window = self.MODEL_CONTEXT_WINDOWS.get(model, 200_000)
            # budget_tokens must be < max_tokens and reasonable vs context window
            self._thinking_budget = max(1024, min(
                self._thinking_budget,
                context_window // 4,
                self.MAX_OUTPUT_TOKENS - 1,
            ))

    # ── Prompt-cache helpers ─────────────────────────────────────────

    @staticmethod
    def _has_cache_block(message: dict) -> bool:
        return any("cache_control" in item for item in message.get("content", []))

    @staticmethod
    def _add_cache_block(message: dict) -> None:
        for content_item in message.get("content", []):
            if content_item["type"] == "text":
                content_item["cache_control"] = {"type": "ephemeral"}
                break

    @staticmethod
    def _remove_cache_block(message: dict) -> None:
        for content_item in message.get("content", []):
            content_item.pop("cache_control", None)

    def mark_for_caching(self, message: dict) -> None:
        """Add an Anthropic ``cache_control`` annotation to *message*."""
        self._add_cache_block(message)

    def trim_cache_blocks(self, context: list[dict], max_blocks: int = 2) -> None:
        cached = [m for m in context if m["role"] == "user" and self._has_cache_block(m)]
        while len(cached) > max_blocks:
            self._remove_cache_block(cached.pop(0))

    # ── Cost calculation ─────────────────────────────────────────────

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        pricing = self.MODEL_PRICING.get(self.model)
        if pricing is None:
            return 0.0
        input_cost = pricing["input_token_cost"]
        output_cost = pricing["output_token_cost"]
        cost = (
            input_tokens * input_cost
            + cache_creation_tokens * input_cost * 1.25
            + cache_read_tokens * input_cost * 0.10
            + output_tokens * output_cost
        ) / 1_000_000
        return cost

    # ── Error classification ─────────────────────────────────────────

    def _classify_error(self, error: Exception) -> str:
        if isinstance(error, self._anthropic.RateLimitError):
            return RATE_LIMIT
        return TRANSIENT

    # ── Message format translation ───────────────────────────────────

    @staticmethod
    def _format_messages(context: list[dict]) -> list[dict]:
        """Convert internal flat image format to Anthropic's nested format.

        The internal format stores images as::

            {"type": "image", "media_type": "image/png", "data": "…"}

        Anthropic expects::

            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "…"}}

        Text blocks and cache_control annotations are passed through unchanged.
        """
        messages = []
        for msg in context:
            parts = msg.get("content", [])
            translated = []
            for part in parts:
                if part.get("type") == "image" and "source" not in part:
                    translated.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.get("media_type", "image/png"),
                            "data": part.get("data", ""),
                        },
                    })
                else:
                    translated.append(part)
            messages.append({"role": msg["role"], "content": translated})
        return messages

    # ── Core: get raw API response with retries ──────────────────────

    def _get_response(self, system_prompt: str, context: list[dict]):
        self.call_count += 1
        sh = self.stream_handler

        # Place a new cache block periodically.
        # Always cache on the first call so the initial task message is
        # written to the prompt cache immediately — without this, call 1
        # sends all message tokens as uncached input and the cache block
        # is only created on call 2 (at 1.25× cost), with reads not
        # benefiting until call 3.
        should_cache = (not self.is_local) and (
            self.call_count == 1 or self.call_count % self.cache_step == 0
        )
        if should_cache:
            for message in reversed(context):
                if message["role"] == "user" and not self._has_cache_block(message):
                    self._add_cache_block(message)
                    break
            self.trim_cache_blocks(context)

        # Pass system prompt as a cacheable content block so it is
        # written to the prompt cache on the first call and read from
        # cache on every subsequent call.
        if self.is_local:
            system_value = system_prompt
        else:
            system_value = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Translate flat image format → Anthropic nested source format.
        api_messages = self._format_messages(context)

        # TODO: max_tokens varies by backend (64K here, 16K for OpenAI/Gemini).
        # Consider making this configurable via the backend or constructor.
        stream_kwargs = dict(
            model=self.model,
            max_tokens=self.MAX_OUTPUT_TOKENS,
            temperature=self.temperature,
            system=system_value,
            messages=api_messages,
        )

        # Send the thinking API parameter only to servers known to
        # support it (Anthropic API with a recognised model name).
        # For local servers we skip this — the server may reject the
        # unknown parameter — but we still use event-based streaming
        # below so that any thinking blocks it sends are surfaced.
        if self._supports_thinking_api and self._thinking_enabled:
            thinking_type = "adaptive" if self.model in self.ADAPTIVE_THINKING_MODELS else "enabled"
            thinking_config = {"type": thinking_type}
            # Only include budget_tokens for "enabled" mode, not "adaptive" mode
            if thinking_type == "enabled":
                thinking_config["budget_tokens"] = self._thinking_budget
            stream_kwargs["thinking"] = thinking_config
            # Anthropic API requires temperature=1 when thinking is enabled.
            # Only enforce this for the Anthropic API; local servers may
            # not have this constraint.
            if not self.is_local:
                stream_kwargs["temperature"] = 1

        # Set beta headers for non-local models
        if not self.is_local:
            beta_features = ["output-128k-2025-02-19", "prompt-caching-2024-07-31"]
            if self._supports_thinking_api and self._thinking_enabled:
                beta_features.append("interleaved-thinking-2025-05-14")
            stream_kwargs["extra_headers"] = {
                "anthropic-beta": ", ".join(beta_features)
            }

        def attempt():
            with self._client.messages.stream(**stream_kwargs) as stream:
                if self._use_thinking_stream:
                    # Event-based streaming to handle interleaved thinking
                    # + text blocks correctly.  The SDK's MessageStream
                    # yields RawMessageStreamEvent objects when iterated.
                    # This path is used whenever the user opts in to
                    # thinking (CLAUDE_THINKING_ENABLED=true), even for
                    # local models whose names aren't in THINKING_MODELS,
                    # so that any thinking blocks the server emits are
                    # surfaced in the UI.
                    current_block_type = None
                    for event in stream:
                        if event.type == "content_block_start":
                            new_type = getattr(event.content_block, "type", None)
                            # Handle transitions to/from thinking blocks
                            if new_type == "thinking" and current_block_type != "thinking":
                                sh.on_stream_reasoning_start()
                            elif new_type != "thinking" and current_block_type == "thinking":
                                sh.on_stream_reasoning_end()
                            current_block_type = new_type
                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if current_block_type == "thinking":
                                chunk = getattr(delta, "thinking", "")
                                if chunk:
                                    sh.on_stream_reasoning_token(chunk)
                            elif current_block_type == "text":
                                chunk = getattr(delta, "text", "")
                                if chunk:
                                    sh.on_stream_token(chunk)
                    # Clean up if response ended during a thinking block
                    if current_block_type == "thinking":
                        sh.on_stream_reasoning_end()
                else:
                    # Simple text-only streaming for non-thinking models
                    for text in stream.text_stream:
                        sh.on_stream_token(text)
                response = stream.get_final_message()
            if response:
                return response
            raise Exception("No response received from Anthropic API")

        return self._run_with_retries(attempt)

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        response = self._get_response(system_prompt, context)

        self.last_input_tokens = response.usage.input_tokens
        self.last_output_tokens = response.usage.output_tokens

        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        total_input = self.last_input_tokens + cache_creation + cache_read

        # Note: output_tokens already includes thinking tokens when
        # extended thinking is enabled — no separate addition needed.
        self.last_total_context_tokens = total_input + self.last_output_tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self.last_total_context_tokens)

        self.cost += self.calculate_cost(
            self.last_input_tokens,
            self.last_output_tokens,
            cache_creation,
            cache_read,
        )

        # Track what this call would have cost without caching
        self.cost_without_cache += self.calculate_cost(
            self.last_input_tokens + cache_creation + cache_read,
            self.last_output_tokens,
        )

        # Collect text from all TextBlock objects in the response.
        # ThinkingBlock content is deliberately excluded — reasoning was
        # already streamed to the UI via the stream handler and must not
        # enter the conversation history (it would inflate input token
        # costs on every subsequent API call).
        text_parts = []
        for block in response.content:
            if hasattr(block, "text") and block.text:
                text_parts.append(block.text)

        text_content = "".join(text_parts)

        if not text_content:
            # Model returned no text content — provide feedback so the
            # agent knows it must complete its response block.
            content_types = [type(block).__name__ for block in response.content]
            return ("You must include a text response with your response block. "
                    "After calling any tools, you must provide a text response explaining "
                    "what you did and the results. Include your Completion: and Success: "
                    "fields at the end when the task is complete. "
                    f"(Debug: response contained blocks: {content_types})")

        return text_content
