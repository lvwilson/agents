"""
OpenAI backend.

Implements :class:`LLMBackend` using the ``openai`` Python SDK.
Supports both the hosted OpenAI API and any OpenAI-compatible local
server (e.g. vLLM, llama.cpp, Ollama with OpenAI compat layer).
"""

from __future__ import annotations

import os
import random
import time

from llm_backend import LLMBackend
from ui import create_spinner, safe_console_print


class OpenAIBackend(LLMBackend):
    """OpenAI chat-completions backend with streaming and retry logic."""

    MODEL_PRICING: dict[str, dict[str, float]] = {
        "gpt-4o":                {"input_token_cost": 2.50,  "output_token_cost": 10.00},
        "gpt-4o-2024-11-20":    {"input_token_cost": 2.50,  "output_token_cost": 10.00},
        "gpt-4o-mini":          {"input_token_cost": 0.15,  "output_token_cost": 0.60},
        "gpt-4.1":              {"input_token_cost": 2.00,  "output_token_cost": 8.00},
        "gpt-4.1-mini":         {"input_token_cost": 0.40,  "output_token_cost": 1.60},
        "gpt-4.1-nano":         {"input_token_cost": 0.10,  "output_token_cost": 0.40},
        "o1":                   {"input_token_cost": 15.00, "output_token_cost": 60.00},
        "o1-mini":              {"input_token_cost": 1.10,  "output_token_cost": 4.40},
        "o3":                   {"input_token_cost": 2.00,  "output_token_cost": 8.00},
        "o3-mini":              {"input_token_cost": 1.10,  "output_token_cost": 4.40},
        "o4-mini":              {"input_token_cost": 1.10,  "output_token_cost": 4.40},
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "gpt-4o":               "GPT-4o",
        "gpt-4o-2024-11-20":   "GPT-4o (Nov 2024)",
        "gpt-4o-mini":         "GPT-4o Mini",
        "gpt-4.1":             "GPT-4.1",
        "gpt-4.1-mini":        "GPT-4.1 Mini",
        "gpt-4.1-nano":        "GPT-4.1 Nano",
        "o1":                  "o1",
        "o1-mini":             "o1 Mini",
        "o3":                  "o3",
        "o3-mini":             "o3 Mini",
        "o4-mini":             "o4 Mini",
    }

    # Retry configuration
    RETRY_TIMEOUT = 300
    RETRY_BASE_DELAY = 1
    RETRY_MAX_DELAY = 60
    RETRY_BACKOFF_FACTOR = 2
    MAX_ERROR_RETRIES = 3

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: str | None = None,
        cache_step: int = 4,
        **_kwargs,
    ):
        super().__init__(model=model, base_url=base_url)

        # Lazy import
        import openai as _openai
        self._openai = _openai

        api_key = os.getenv("OPENAI_API_KEY")
        client_kwargs: dict = {}
        if base_url:
            if not api_key:
                api_key = "local"
            client_kwargs["base_url"] = base_url
        else:
            if not api_key:
                raise Exception("OPENAI_API_KEY Environment Variable Unset")
        client_kwargs["api_key"] = api_key

        self._client = _openai.OpenAI(**client_kwargs)

    # ── Display name ─────────────────────────────────────────────────

    @property
    def display_name(self) -> str:
        if self.is_local:
            return f"{self.model} (local)"
        return self.MODEL_DISPLAY_NAMES.get(self.model, self.model)

    # ── Message format translation ───────────────────────────────────

    @staticmethod
    def _translate_messages(
        system_prompt: str, context: list[dict]
    ) -> list[dict]:
        """Convert internal message format to OpenAI chat-completions format.

        Internal format::

            [{"role": "user"|"assistant",
              "content": [{"type": "text", "text": "…"}, …]}, …]

        OpenAI format::

            [{"role": "system", "content": "…"},
             {"role": "user"|"assistant", "content": "…"}, …]
        """
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        for msg in context:
            role = msg["role"]
            content_parts = msg.get("content", [])

            # Handle image + text messages
            has_images = any(
                p.get("type") == "image" for p in content_parts
            )
            if has_images:
                oai_parts: list[dict] = []
                for part in content_parts:
                    if part.get("type") == "text":
                        oai_parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image":
                        source = part.get("source", {})
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        oai_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data}"
                            },
                        })
                messages.append({"role": role, "content": oai_parts})
            else:
                # Plain text — collapse content parts into a single string
                text = "\n".join(
                    p["text"] for p in content_parts
                    if p.get("type") == "text" and p.get("text")
                )
                messages.append({"role": role, "content": text})
        return messages

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
        """Call the OpenAI chat-completions API with streaming and retries.

        Returns the full collected response text and usage dict.
        """
        self.call_count += 1
        messages = self._translate_messages(system_prompt, context)

        start_time = time.monotonic()
        error_retries = 0
        current_delay = self.RETRY_BASE_DELAY

        while True:
            try:
                spinner = create_spinner()
                spinner.start()

                stream = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=16384,
                    temperature=0.6,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                collected_text = ""
                usage = None
                first_chunk = True

                for chunk in stream:
                    # Extract usage from the final chunk
                    if chunk.usage is not None:
                        usage = chunk.usage

                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            if first_chunk:
                                spinner.stop()
                                first_chunk = False
                            safe_console_print(delta.content, style="stream", end="")
                            collected_text += delta.content

                if first_chunk:
                    spinner.stop()

                return collected_text, usage

            except self._openai.RateLimitError as e:
                spinner.stop()
                sleep_time = current_delay

                # Try to read retry-after header
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

                safe_console_print(
                    f"\n  ⏳ Rate limited — retrying in {sleep_time:.1f}s "
                    f"({remaining:.0f}s remaining)",
                    style="warning",
                )
                time.sleep(sleep_time)
                current_delay = min(
                    current_delay * self.RETRY_BACKOFF_FACTOR, self.RETRY_MAX_DELAY
                )

            except Exception as e:
                spinner.stop()
                error_retries += 1
                if error_retries >= self.MAX_ERROR_RETRIES:
                    raise Exception(
                        f"Maximum retries exceeded ({self.MAX_ERROR_RETRIES}) "
                        f"on response request: {e}"
                    )
                safe_console_print(
                    f"\n  ✗ Attempt {error_retries}/{self.MAX_ERROR_RETRIES} failed: {e}",
                    style="error",
                )

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        text, usage = self._get_response(system_prompt, context)

        if usage is not None:
            self.last_input_tokens = usage.prompt_tokens or 0
            self.last_output_tokens = usage.completion_tokens or 0
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
