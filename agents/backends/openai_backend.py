"""
OpenAI backend.

Implements :class:`LLMBackend` using the ``openai`` Python SDK.
Supports both the hosted OpenAI API and any OpenAI-compatible local
server (e.g. vLLM, llama.cpp, Ollama with OpenAI compat layer).
"""

from __future__ import annotations

import os

from ..llm_backend import LLMBackend, StreamHandler, RATE_LIMIT, TRANSIENT


class OpenAIBackend(LLMBackend):
    """OpenAI completions backend with streaming and retry logic."""

    MODEL_PRICING: dict[str, dict[str, float]] = {
        "gpt-5.2": {"input_token_cost": 2.50, "output_token_cost": 10.00},
        "gpt-5.2-mini": {"input_token_cost": 0.15, "output_token_cost": 0.60},
        "gpt-5.3": {"input_token_cost": 2.50, "output_token_cost": 10.00},
        "gpt-5.3-mini": {"input_token_cost": 0.15, "output_token_cost": 0.60},
        "gpt-5.3-codex": {"input_token_cost": 3.00, "output_token_cost": 12.00},
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "gpt-5.2": "GPT-5.2",
        "gpt-5.2-mini": "GPT-5.2 Mini",
        "gpt-5.3": "GPT-5.3",
        "gpt-5.3-mini": "GPT-5.3 Mini",
        "gpt-5.3-codex": "GPT-5.3 Codex",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "gpt-5.2":       128_000,
        "gpt-5.2-mini":  128_000,
        "gpt-5.3":       128_000,
        "gpt-5.3-mini":  128_000,
        "gpt-5.3-codex": 128_000,
    }

    def __init__(
        self,
        model: str = "gpt-5.3-codex",
        base_url: str | None = None,
        cache_step: int = 4,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
        **_kwargs,
    ):
        super().__init__(model=model, base_url=base_url, stream_handler=stream_handler, temperature=temperature)

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

    # ── Message format translation ───────────────────────────────────

    @staticmethod
    def _format_messages(system_prompt: str, context: list[dict]) -> list[dict]:
        """Convert internal message format to OpenAI Responses API input.

        Role/content constraints enforced:
        - system/user: input_text (and user may include input_image)
        - assistant: output_text (or refusal; we only emit output_text)
        """

        def _to_user_items(parts: list[dict]) -> list[dict]:
            items: list[dict] = []
            for part in parts:
                ptype = part.get("type")
                if ptype == "text":
                    text = part.get("text")
                    if text:
                        items.append({"type": "input_text", "text": text})
                elif ptype == "image":
                    media_type = part.get("media_type", "image/png")
                    data = part.get("data", "")
                    if data:
                        items.append(
                            {
                                "type": "input_image",
                                "image_url": f"data:{media_type};base64,{data}",
                            }
                        )
            return items

        def _to_text_items(parts: list[dict], item_type: str) -> list[dict]:
            text = "\n".join(
                p["text"] for p in parts if p.get("type") == "text" and p.get("text")
            )
            if not text:
                return []
            return [{"type": item_type, "text": text}]

        messages: list[dict] = []
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                }
            )

        for msg in context:
            role = msg.get("role")
            parts = msg.get("content", []) or []

            if role == "user":
                items = _to_user_items(parts)
            elif role == "assistant":
                # Assistant history must be output_* content types.
                # Ignore non-text parts (e.g. images) as they are not valid replay items here.
                items = _to_text_items(parts, "output_text")
            elif role == "system":
                items = _to_text_items(parts, "input_text")
            else:
                continue

            if items:
                messages.append({"role": role, "content": items})

        return messages

    @staticmethod
    def _validate_responses_input(messages: list[dict]) -> None:
        """Fail fast on role/content-type mismatches before API call."""
        for i, msg in enumerate(messages):
            role = msg.get("role")
            content = msg.get("content", []) or []
            for j, item in enumerate(content):
                ctype = item.get("type")
                if role == "assistant" and ctype not in {"output_text", "refusal"}:
                    raise ValueError(
                        f"Invalid assistant content type at input[{i}].content[{j}]: {ctype}"
                    )
                if role in {"user", "system"} and ctype == "output_text":
                    raise ValueError(
                        f"Invalid {role} content type at input[{i}].content[{j}]: {ctype}"
                    )

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

        uncached_input_tokens = max(0, input_tokens - cache_read_tokens)

        return (
            uncached_input_tokens * pricing["input_token_cost"]
            + cache_read_tokens * pricing["input_token_cost"] * 0.5
            + output_tokens * pricing["output_token_cost"]
        ) / 1_000_000

    # ── Error classification ─────────────────────────────────────────

    def _classify_error(self, error: Exception) -> str:
        if isinstance(error, self._openai.RateLimitError):
            return RATE_LIMIT
        return TRANSIENT

    # ── Core: streaming API call with retries ────────────────────────

    def _get_response(self, system_prompt: str, context: list[dict]):
        """Call the OpenAI responses API with streaming and retries.

        Returns the full collected response text and usage dict.
        """
        self.call_count += 1
        sh = self.stream_handler
        messages = self._format_messages(system_prompt, context)
        self._validate_responses_input(messages)

        def attempt():
            stream = self._client.responses.create(
                model=self.model,
                input=messages,
                max_output_tokens=16384,
                temperature=self.temperature,
                stream=True,
            )

            collected_text = ""
            usage = None

            for event in stream:
                if event.type == "response.output_text.delta":
                    text = event.delta
                    if text:
                        sh.on_stream_token(text)
                        collected_text += text
                elif event.type == "response.completed":
                    if hasattr(event, "response") and event.response:
                        usage = event.response.usage

            return collected_text, usage

        return self._run_with_retries(attempt)

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        text, usage = self._get_response(system_prompt, context)

        cache_read = 0
        if usage is not None:
            self.last_input_tokens = getattr(usage, "input_tokens", 0) or 0
            self.last_output_tokens = getattr(usage, "output_tokens", 0) or 0
            if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
                cache_read = (
                    getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
                )
        else:
            self.last_input_tokens = 0
            self.last_output_tokens = 0

        self.last_total_context_tokens = self.last_input_tokens + self.last_output_tokens
        self.peak_context_tokens = max(
            self.peak_context_tokens, self.last_total_context_tokens
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
            raise Exception("No text content found in model response")

        return text
