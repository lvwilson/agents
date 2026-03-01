"""
Anthropic (Claude) backend.

Implements :class:`LLMBackend` using the ``anthropic`` Python SDK.
"""

from __future__ import annotations

import os
import random
import time

from llm_backend import LLMBackend, StreamHandler


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

    # Models that route to MiniMax (require special API key validation)
    MINIMAX_MODELS = {"MiniMax-M2.5"}

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        base_url: str | None = None,
        cache_step: int = 2,
        stream_handler: StreamHandler | None = None,
        **_kwargs,
    ):
        super().__init__(model=model, base_url=base_url, stream_handler=stream_handler)

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

    # ── Display name ─────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        if self.is_local:
            return f"{self.model} (local)"
        return self.MODEL_DISPLAY_NAMES.get(self.model, self.model)

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

    # ── Core: get raw API response with retries ──────────────────────

    def _get_response(self, system_prompt: str, context: list[dict]):
        self.call_count += 1
        sh = self.stream_handler

        # Place a new cache block periodically
        should_cache = (not self.is_local) and (self.call_count % self.cache_step == 0)
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

        start_time = time.monotonic()
        error_retries = 0
        current_delay = self.RETRY_BASE_DELAY

        while True:
            try:
                sh.on_stream_start()
                # TODO: max_tokens varies by backend (64K here, 16K for OpenAI/Gemini).
                # Consider making this configurable via the backend or constructor.
                stream_kwargs = dict(
                    model=self.model,
                    max_tokens=64000,
                    temperature=0.6,
                    system=system_value,
                    messages=context,
                )
                if not self.is_local:
                    stream_kwargs["extra_headers"] = {
                        "anthropic-beta": "output-128k-2025-02-19, prompt-caching-2024-07-31"
                    }
                with self._client.messages.stream(**stream_kwargs) as stream:
                    for text in stream.text_stream:
                        sh.on_stream_token(text)
                    sh.on_stream_end()
                response = stream.get_final_message()
                if response:
                    return response

            except self._anthropic.RateLimitError as e:
                sh.on_stream_end()
                sleep_time = current_delay
                if hasattr(e, "response") and e.response is not None:
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after is not None:
                        try:
                            sleep_time = max(int(retry_after), sleep_time)
                        except (ValueError, TypeError):
                            pass

                jitter = sleep_time * 0.25 * (2 * random.random() - 1)
                sleep_time = max(0, sleep_time + jitter)

                remaining = self.RETRY_TIMEOUT - (time.monotonic() - start_time)
                if remaining <= 0:
                    raise Exception(
                        f"Rate-limit retry timeout exceeded ({self.RETRY_TIMEOUT}s)"
                    )
                sleep_time = min(sleep_time, remaining)

                sh.on_retry(
                    f"Rate limited — retrying in {sleep_time:.1f}s "
                    f"({remaining:.0f}s remaining)"
                )
                time.sleep(sleep_time)
                current_delay = min(current_delay * self.RETRY_BACKOFF_FACTOR, self.RETRY_MAX_DELAY)

            except Exception as e:
                sh.on_stream_end()
                error_retries += 1
                if error_retries >= self.MAX_ERROR_RETRIES:
                    raise Exception(
                        f"Maximum retries exceeded ({self.MAX_ERROR_RETRIES}) "
                        f"on response request: {e}"
                    )
                sh.on_error(
                    f"Attempt {error_retries}/{self.MAX_ERROR_RETRIES} failed: {e}"
                )

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        response = self._get_response(system_prompt, context)

        self.last_input_tokens = response.usage.input_tokens
        self.last_output_tokens = response.usage.output_tokens

        cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        total_input = self.last_input_tokens + cache_creation + cache_read

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

        # Find the first TextBlock, skipping ThinkingBlock objects
        for block in response.content:
            if hasattr(block, "text") and block.text:
                return block.text

        # Model returned no text content - this happens when the model calls
        # tools but doesn't provide a text response. We must provide feedback
        # to the agent that it must complete its response block.
        content_types = [type(block).__name__ for block in response.content]
        return ("You must include a text response with your response block. "
                "After calling any tools, you must provide a text response explaining "
                "what you did and the results. Include your Completion: and Success: "
                "fields at the end when the task is complete. "
                f"(Debug: response contained blocks: {content_types})")
