"""
Kimi K3 backend.

Implements :class:`LLMBackend` using the ``openai`` Python SDK against
the Kimi API (OpenAI-compatible at ``api.moonshot.ai``).

Kimi K3 specifics
-----------------
* 1 M-token context window
* ``reasoning_effort="max"`` on every call (thinking always enabled)
* Fixed parameters: temperature=1.0, top_p=0.95, n=1, penalties=0
  (omitted from requests per Kimi docs)
* Pricing: $3 / M input, $15 / M output, $0.30 / M cache-hit input
* API key via ``MOONSHOT_API_KEY`` environment variable
"""

from __future__ import annotations

import os

from ..llm_backend import (
    LLMBackend,
    StreamHandler,
    EmptyResponseError,
    RATE_LIMIT,
    TRANSIENT,
)


class KimiBackend(LLMBackend):
    """Kimi K3 completions backend with streaming and retry logic."""

    MODEL_PRICING: dict[str, dict[str, float]] = {
        "kimi-k3": {
            "input_token_cost": 3.00,
            "output_token_cost": 15.00,
            "cache_hit_token_cost": 0.30,
        },
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "kimi-k3": "Kimi K3",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "kimi-k3": 1_000_000,
    }

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
            stream_handler=stream_handler,
            temperature=temperature,
        )

        # Lazy import
        import openai as _openai

        self._openai = _openai

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise Exception("MOONSHOT_API_KEY Environment Variable Unset")

        self._client = _openai.OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.moonshot.ai/v1",
        )

    # ── Message format translation ───────────────────────────────────

    @staticmethod
    def _format_messages(system_prompt: str, context: list[dict]) -> list[dict]:
        """Convert internal message format to OpenAI chat-completions input."""

        def _to_content(parts: list[dict]) -> str | list[dict]:
            """Handle both text-only and multimodal content."""
            text_parts = [p for p in parts if p.get("type") == "text"]
            image_parts = [p for p in parts if p.get("type") == "image"]

            if not image_parts:
                # Plain text — return as string for simplicity
                texts = [p.get("text", "") for p in text_parts]
                combined = "\n".join(t for t in texts if t)
                return combined if combined else ""

            # Multimodal — return as list of content objects
            items: list[dict] = []
            for img in image_parts:
                media_type = img.get("media_type", "image/png")
                data = img.get("data", "")
                if data:
                    items.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{data}"
                            },
                        }
                    )
            for tp in text_parts:
                t = tp.get("text", "")
                if t:
                    items.append({"type": "text", "text": t})
            return items

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in context:
            role = msg.get("role")
            parts = msg.get("content", []) or []
            if role in ("user", "assistant", "system"):
                content = _to_content(parts)
                if content or role == "system":
                    messages.append({"role": role, "content": content})
            elif role == "tool":
                # Tool result messages
                text_parts = [p.get("text", "") for p in parts if p.get("type") == "text"]
                content = "\n".join(t for t in text_parts if t)
                if content:
                    messages.append(
                        {
                            "role": "tool",
                            "content": content,
                            "tool_call_id": msg.get("tool_call_id", ""),
                        }
                    )
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

        input_cost = pricing["input_token_cost"]
        output_cost = pricing["output_token_cost"]
        cache_cost = pricing.get("cache_hit_token_cost", input_cost * 0.1)

        uncached_input = max(0, input_tokens - cache_read_tokens)

        return (
            uncached_input * input_cost
            + cache_read_tokens * cache_cost
            + output_tokens * output_cost
        ) / 1_000_000

    # ── Error classification ─────────────────────────────────────────

    def _classify_error(self, error: Exception) -> str:
        if isinstance(error, self._openai.RateLimitError):
            return RATE_LIMIT
        return TRANSIENT

    # ── Core: streaming API call with retries ────────────────────────

    def _get_response(self, system_prompt: str, context: list[dict]):
        """Call the Kimi chat completions API with streaming and retries.

        Returns the full collected response text and usage dict.
        """
        self.call_count += 1
        sh = self.stream_handler
        messages = self._format_messages(system_prompt, context)

        def attempt():
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                reasoning_effort="max",
                max_completion_tokens=131_072,
                stream=True,
                stream_options={"include_usage": True},
            )

            collected_text = ""
            usage = None
            reasoning_started = False

            for event in stream:
                # Usage arrives on a dedicated final chunk whose choices
                # list is empty, so capture it before the guard below.
                chunk_usage = getattr(event, "usage", None)
                if chunk_usage:
                    usage = chunk_usage

                choices = getattr(event, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue

                # ``reasoning_content`` (thinking tokens) is streamed to
                # the UI via the reasoning hooks so it renders dimmed,
                # but it is never collected — thinking cannot leak into
                # the content stream or the conversation context.
                reasoning = getattr(delta, "reasoning_content", None)
                if reasoning:
                    if not reasoning_started:
                        sh.on_stream_reasoning_start()
                        reasoning_started = True
                    sh.on_stream_reasoning_token(reasoning)

                text = getattr(delta, "content", None)
                if text:
                    if reasoning_started:
                        sh.on_stream_reasoning_end()
                        reasoning_started = False
                    sh.on_stream_token(text)
                    collected_text += text

            # Clean up if the stream ended during a reasoning block.
            if reasoning_started:
                sh.on_stream_reasoning_end()

            return collected_text, usage

        return self._run_with_retries(attempt)

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        text, usage = self._get_response(system_prompt, context)

        if usage is not None:
            self.last_input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            self.last_output_tokens = getattr(usage, "completion_tokens", 0) or 0
            # Kimi may report cache hits in prompt_tokens_details
            details = getattr(usage, "prompt_tokens_details", None)
            cache_read = (
                getattr(details, "cached_tokens", 0) or 0
                if details
                else 0
            )
        else:
            self.last_input_tokens = 0
            self.last_output_tokens = 0
            cache_read = 0

        self.last_total_context_tokens = (
            self.last_input_tokens + self.last_output_tokens
        )
        self.peak_context_tokens = max(
            self.peak_context_tokens,
            self.last_total_context_tokens,
        )

        self.cost += self.calculate_cost(
            self.last_input_tokens,
            self.last_output_tokens,
            cache_read_tokens=cache_read,
        )

        self.cost_without_cache += self.calculate_cost(
            self.last_input_tokens,
            self.last_output_tokens,
            cache_read_tokens=0,
        )

        if not text:
            raise EmptyResponseError("No text content found in model response")

        return text
