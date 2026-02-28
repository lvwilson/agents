"""
Google Gemini backend.

Implements :class:`LLMBackend` using the ``google-genai`` unified SDK.
"""

from __future__ import annotations

import base64
import os
import random
import time

from llm_backend import LLMBackend
from ui import create_spinner, safe_console_print


class GeminiBackend(LLMBackend):
    """Gemini backend with streaming, context caching, and retry logic."""

    MODEL_PRICING: dict[str, dict[str, float]] = {
        # Pricing per 1M tokens (as of mid-2025, prompts <= 200k tokens)
        "gemini-3.1-pro-preview": {
            "input_token_cost": 2.00,
            "output_token_cost": 12.00,
            "cache_read_cost": 0.20,
            "cache_storage_cost_per_hour": 4.50,  # $/1M tokens/hour
        },
        "gemini-3.1-pro-preview-customtools": {
            "input_token_cost": 2.00,
            "output_token_cost": 12.00,
            "cache_read_cost": 0.20,
            "cache_storage_cost_per_hour": 4.50,  # $/1M tokens/hour
        },
        "gemini-3-flash-preview": {
            "input_token_cost": 0.50,
            "output_token_cost": 3.00,
            "cache_read_cost": 0.05,
            "cache_storage_cost_per_hour": 1.00,  # $/1M tokens/hour
        },
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "gemini-3.1-pro-preview":  "Gemini 3.1 Pro Preview",
        "gemini-3-flash-preview":  "Gemini 3 Flash Preview",
    }

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        base_url: str | None = None,
        cache_step: int = 2,
        **_kwargs,
    ):
        super().__init__(model=model, base_url=base_url)

        # Lazy import
        from google import genai as _genai
        from google.genai import types as _types
        self._genai = _genai
        self._types = _types

        api_key = os.getenv("GEMINI_API_KEY")
        if base_url:
            if not api_key:
                api_key = "local"
            self._client = _genai.Client(
                api_key=api_key,
                http_options={"base_url": base_url},
            )
        else:
            if not api_key:
                raise Exception("GEMINI_API_KEY Environment Variable Unset")
            self._client = _genai.Client(api_key=api_key)

        self.cache_step = cache_step

        # Context caching state
        self._cache_name: str | None = None       # Server-side cache resource name
        self._cached_msg_count: int = 0            # Number of context messages in the cache
        self._cached_system_prompt: str | None = None  # System prompt stored in cache

    # ── Display name ─────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        if self.is_local:
            return f"{self.model} (local)"
        return self.MODEL_DISPLAY_NAMES.get(self.model, self.model)

    # ── Message format translation ───────────────────────────────────

    def _translate_messages(self, context: list[dict]) -> list:
        """Convert internal message format to Gemini Content objects.

        Internal format::

            [{"role": "user"|"assistant",
              "content": [{"type": "text", "text": "…"}, …]}, …]

        Gemini format::

            [Content(role="user"|"model", parts=[Part(text="…"), …]), …]
        """
        types = self._types
        contents = []
        for msg in context:
            role = "model" if msg["role"] == "assistant" else "user"
            content_parts = msg.get("content", [])

            parts = []
            for part in content_parts:
                if part.get("type") == "text" and part.get("text"):
                    parts.append(types.Part(text=part["text"]))
                elif part.get("type") == "image":
                    source = part.get("source", {})
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    parts.append(types.Part(
                        inline_data=types.Blob(
                            mime_type=media_type,
                            data=base64.b64decode(data),
                        )
                    ))

            if parts:
                contents.append(types.Content(role=role, parts=parts))
        return contents

    # ── Context caching helpers ──────────────────────────────────────

    # Cache TTL in seconds — each cache is charged for this full duration
    # even if deleted early (over-estimate to keep budget tracking safe).
    CACHE_TTL = 300

    def _delete_cache(self) -> None:
        """Delete the current server-side cache if it exists."""
        if self._cache_name:
            try:
                self._client.caches.delete(name=self._cache_name)
            except Exception:
                pass  # Cache may have already expired
            self._cache_name = None
            self._cached_msg_count = 0
            self._cached_system_prompt = None

    def _create_cache(self, system_prompt: str, context: list[dict]) -> bool:
        """Create a server-side cache with the system prompt and context.

        Caches all messages except the last user message so that the
        next call can send only the final message as new content.
        Storage cost for the full TTL is charged immediately.

        Returns True if cache was created successfully, False otherwise.
        """
        if len(context) < 2:
            return False

        messages_to_cache = context[:-1]
        contents = self._translate_messages(messages_to_cache)
        if not contents:
            return False

        # Delete any existing cache before creating a new one
        self._delete_cache()

        try:
            cached_content = self._client.caches.create(
                model=self.model,
                config={
                    "contents": contents,
                    "system_instruction": system_prompt,
                    "ttl": f"{self.CACHE_TTL}s",
                    "display_name": "agent-context-cache",
                },
            )
            self._cache_name = cached_content.name
            self._cached_msg_count = len(messages_to_cache)
            self._cached_system_prompt = system_prompt

            # Charge storage for the full TTL up-front.  Use the
            # API-reported token count when available, otherwise
            # fall back to the total prompt tokens as an upper bound.
            usage = getattr(cached_content, "usage_metadata", None)
            token_count = (
                getattr(usage, "total_token_count", 0) if usage else 0
            ) or self.last_input_tokens
            self.cost += self.calculate_cost(
                0, 0,
                cache_storage_tokens=token_count,
                cache_storage_seconds=self.CACHE_TTL,
            )

            return True
        except Exception as e:
            safe_console_print(
                f"\n  ⚠ Cache creation failed (falling back to uncached): {e}",
                style="warning",
            )
            self._cache_name = None
            self._cached_msg_count = 0
            self._cached_system_prompt = None
            return False

    def _is_cache_valid(self, system_prompt: str, context: list[dict]) -> bool:
        """Check if the current cache is still usable for this request."""
        if not self._cache_name:
            return False
        if self._cached_system_prompt != system_prompt:
            return False
        # The cache covers messages [0:_cached_msg_count].  It's valid
        # as long as the context still starts with those same messages
        # and has grown (new messages appended after the cached portion).
        if len(context) <= self._cached_msg_count:
            return False
        return True

    # ── Cost calculation ─────────────────────────────────────────────

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_storage_tokens: int = 0,
        cache_storage_seconds: int = 0,
    ) -> float:
        pricing = self.MODEL_PRICING.get(self.model)
        if pricing is None:
            return 0.0
        input_cost = pricing["input_token_cost"]
        output_cost = pricing["output_token_cost"]
        # Gemini has an explicit per-model cache read price rather than
        # a fixed fraction of the input price.  Cache creation is charged
        # at the normal input rate (no surcharge).
        cache_cost = pricing.get("cache_read_cost", input_cost * 0.10)
        cost = (
            input_tokens * input_cost
            + cache_creation_tokens * input_cost
            + cache_read_tokens * cache_cost
            + output_tokens * output_cost
        ) / 1_000_000

        # Storage cost: charged per 1M tokens per hour, prorated by TTL
        if cache_storage_tokens > 0 and cache_storage_seconds > 0:
            storage_rate = pricing.get("cache_storage_cost_per_hour", 0.0)
            storage_hours = cache_storage_seconds / 3600.0
            cost += (cache_storage_tokens / 1_000_000) * storage_rate * storage_hours

        return cost

    # ── Core: streaming API call with retries ────────────────────────

    def _get_response(self, system_prompt: str, context: list[dict]):
        """Call the Gemini API with streaming and retries.

        Returns the collected response text and usage metadata.
        """
        self.call_count += 1
        types = self._types

        # Determine whether to use or create a cache
        use_cache = False
        should_create_cache = (
            not self.is_local
            and self.call_count % self.cache_step == 0
        )

        if not self.is_local and self._is_cache_valid(system_prompt, context):
            # Use existing cache — send only the uncached messages
            use_cache = True
        elif should_create_cache:
            # Create a new cache, then use it
            if self._create_cache(system_prompt, context):
                use_cache = True

        if use_cache:
            # Only send messages not covered by the cache
            uncached_messages = context[self._cached_msg_count:]
            contents = self._translate_messages(uncached_messages)
            config = types.GenerateContentConfig(
                cached_content=self._cache_name,
                temperature=0.6,
                max_output_tokens=16384,
                # system_instruction is in the cache — do not pass it again
            )
        else:
            contents = self._translate_messages(context)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.6,
                max_output_tokens=16384,
            )

        start_time = time.monotonic()
        error_retries = 0
        current_delay = self.RETRY_BASE_DELAY

        while True:
            try:
                spinner = create_spinner()
                spinner.start()

                stream = self._client.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                collected_text = ""
                usage_metadata = None
                first_chunk = True

                for chunk in stream:
                    # Capture usage metadata from any chunk that has it
                    if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                        usage_metadata = chunk.usage_metadata

                    if chunk.text:
                        if first_chunk:
                            spinner.stop()
                            first_chunk = False
                        safe_console_print(chunk.text, style="stream", end="")
                        collected_text += chunk.text

                if first_chunk:
                    spinner.stop()

                return collected_text, usage_metadata

            except Exception as e:
                spinner.stop()

                # If cache-related error, invalidate cache and retry without it
                if use_cache and ("cache" in str(e).lower() or "NOT_FOUND" in str(e)):
                    safe_console_print(
                        f"\n  ⚠ Cache expired or invalid, retrying without cache",
                        style="warning",
                    )
                    self._delete_cache()
                    use_cache = False
                    contents = self._translate_messages(context)
                    config = types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.6,
                        max_output_tokens=16384,
                    )
                    continue

                # Check for rate limiting (Google API uses 429 status)
                is_rate_limit = (
                    "429" in str(e)
                    or "RESOURCE_EXHAUSTED" in str(e)
                    or "rate limit" in str(e).lower()
                )

                if is_rate_limit:
                    sleep_time = current_delay
                    jitter = sleep_time * 0.25 * (2 * random.random() - 1)
                    sleep_time = max(0, sleep_time + jitter)

                    remaining = self.RETRY_TIMEOUT - (time.monotonic() - start_time)
                    if remaining <= 0:
                        raise Exception(
                            f"Rate-limit retry timeout exceeded ({self.RETRY_TIMEOUT}s)"
                        )
                    sleep_time = min(sleep_time, remaining)

                    safe_console_print(
                        f"\n  ⏳ Rate limited — retrying in {sleep_time:.1f}s "
                        f"({remaining:.0f}s remaining)",
                        style="warning",
                    )
                    time.sleep(sleep_time)
                    current_delay = min(
                        current_delay * self.RETRY_BACKOFF_FACTOR,
                        self.RETRY_MAX_DELAY,
                    )
                else:
                    error_retries += 1
                    if error_retries >= self.MAX_ERROR_RETRIES:
                        raise Exception(
                            f"Maximum retries exceeded ({self.MAX_ERROR_RETRIES}) "
                            f"on response request: {e}"
                        )
                    safe_console_print(
                        f"\n  ✗ Attempt {error_retries}/{self.MAX_ERROR_RETRIES} "
                        f"failed: {e}",
                        style="error",
                    )

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        text, usage_metadata = self._get_response(system_prompt, context)

        if usage_metadata is not None:
            self.last_input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
            self.last_output_tokens = (
                getattr(usage_metadata, "candidates_token_count", 0)
                or getattr(usage_metadata, "response_token_count", 0)
                or 0
            )
            cache_read = getattr(usage_metadata, "cached_content_token_count", 0) or 0
        else:
            self.last_input_tokens = 0
            self.last_output_tokens = 0
            cache_read = 0

        # prompt_token_count from Gemini already includes cached tokens,
        # so uncached input = total prompt tokens - cached tokens.
        uncached_input = max(0, self.last_input_tokens - cache_read)

        self.last_total_context_tokens = self.last_input_tokens + self.last_output_tokens
        self.peak_context_tokens = max(
            self.peak_context_tokens, self.last_total_context_tokens
        )

        self.cost += self.calculate_cost(
            uncached_input,
            self.last_output_tokens,
            cache_read_tokens=cache_read,
        )

        # Track what this call would have cost without caching
        self.cost_without_cache += self.calculate_cost(
            self.last_input_tokens,
            self.last_output_tokens,
        )

        if not text:
            raise Exception("No text content found in model response")

        return text
