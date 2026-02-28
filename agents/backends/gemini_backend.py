"""
Google Gemini backend.

Implements :class:`LLMBackend` using the ``google-genai`` unified SDK.
"""

from __future__ import annotations

import os
import random
import time

from llm_backend import LLMBackend
from ui import create_spinner, safe_console_print


class GeminiBackend(LLMBackend):
    """Gemini backend with streaming and retry logic."""

    MODEL_PRICING: dict[str, dict[str, float]] = {
        # Pricing per 1M tokens (as of mid-2025)
        "gemini-2.5-pro":          {"input_token_cost": 1.25,  "output_token_cost": 10.00},
        "gemini-2.5-flash":        {"input_token_cost": 0.15,  "output_token_cost": 0.60},
        "gemini-2.0-flash":        {"input_token_cost": 0.10,  "output_token_cost": 0.40},
        "gemini-2.0-flash-lite":   {"input_token_cost": 0.075, "output_token_cost": 0.30},
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "gemini-2.5-pro":          "Gemini 2.5 Pro",
        "gemini-2.5-flash":        "Gemini 2.5 Flash",
        "gemini-2.0-flash":        "Gemini 2.0 Flash",
        "gemini-2.0-flash-lite":   "Gemini 2.0 Flash Lite",
    }

    # Retry configuration
    RETRY_TIMEOUT = 300
    RETRY_BASE_DELAY = 1
    RETRY_MAX_DELAY = 60
    RETRY_BACKOFF_FACTOR = 2
    MAX_ERROR_RETRIES = 3

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        base_url: str | None = None,
        cache_step: int = 4,
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
                    # Gemini accepts inline_data with base64
                    import base64
                    parts.append(types.Part(
                        inline_data=types.Blob(
                            mime_type=media_type,
                            data=base64.b64decode(data),
                        )
                    ))

            if parts:
                contents.append(types.Content(role=role, parts=parts))
        return contents

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
        return (
            input_tokens * pricing["input_token_cost"]
            + output_tokens * pricing["output_token_cost"]
        ) / 1_000_000

    # ── Core: streaming API call with retries ────────────────────────

    def _get_response(self, system_prompt: str, context: list[dict]):
        """Call the Gemini API with streaming and retries.

        Returns the collected response text and usage metadata.
        """
        self.call_count += 1
        types = self._types
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
                # Check for rate limiting (Google API uses 429 status)
                is_rate_limit = (
                    "429" in str(e)
                    or "RESOURCE_EXHAUSTED" in str(e)
                    or "rate" in str(e).lower()
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
            self.last_output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
        else:
            self.last_input_tokens = 0
            self.last_output_tokens = 0

        self.last_total_context_tokens = self.last_input_tokens + self.last_output_tokens
        self.peak_context_tokens = max(
            self.peak_context_tokens, self.last_total_context_tokens
        )

        self.cost += self.calculate_cost(
            self.last_input_tokens, self.last_output_tokens
        )

        if not text:
            raise Exception("No text content found in model response")

        return text
