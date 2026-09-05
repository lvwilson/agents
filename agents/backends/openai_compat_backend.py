"""
OpenAI-compatible chat-completions backend (shared base).

This is the common implementation for any provider that exposes an
OpenAI-compatible **chat completions** endpoint (``POST /v1/chat/completions``
with ``stream`` + ``stream_options={"include_usage": true}``).  It is the
base class for:

* the ``openai`` provider **when a custom ``base_url`` is set** (local /
  self-hosted servers such as vLLM, llama.cpp, Ollama, LM Studio, … —
  these implement chat completions, not the Responses API), and
* provider-specific subclasses that swap in their own SDK client and
  credentials (e.g. :class:`.cerebras_backend.CerebrasBackend`,
  :class:`.kimi_backend.KimiBackend`).

The class is deliberately **client-agnostic**: it only relies on the
client exposing ``client.chat.completions.create(...)`` returning an
iterable of chunks with ``choices[0].delta`` (``content`` / ``reasoning`` /
``tool_calls``) and an optional trailing ``usage``.  That interface is
shared by the OpenAI and Cerebras SDKs (both Stainless-generated).

Reasoning (thinking) tokens are streamed to the UI via the reasoning
hooks and are never collected into the response text or conversation
context.  Both ``delta.reasoning`` (Cerebras) and ``delta.reasoning_content``
(vLLM / DeepSeek-style servers) are recognised.
"""

from __future__ import annotations

import os

from ..llm_backend import (
    LLMBackend,
    StreamHandler,
    EmptyResponseError,
    RATE_LIMIT,
    TRANSIENT,
    merge_consecutive_messages,
)


class OpenAICompatBackend(LLMBackend):
    """Chat-completions backend with streaming, reasoning, and retry logic.

    Subclasses should set ``MODEL_PRICING`` / ``MODEL_DISPLAY_NAMES`` /
    ``MODEL_CONTEXT_WINDOWS`` and, when they use a non-OpenAI SDK, override
    :meth:`_build_client`, :meth:`_resolve_credentials`, and
    :meth:`_rate_limit_error_class`.
    """

    MODEL_PRICING: dict[str, dict[str, float]] = {}
    MODEL_DISPLAY_NAMES: dict[str, str] = {}
    MODEL_CONTEXT_WINDOWS: dict[str, int] = {}

    #: Upper bound on generated tokens per call (includes reasoning tokens).
    MAX_COMPLETION_TOKENS: int = 16_384

    #: When False, ``temperature`` is omitted from the request (for models
    #: that reject it).  Default True — both OpenAI and Cerebras accept it.
    SEND_TEMPERATURE: bool = True

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        cache_step: int = 4,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
        **_kwargs,
    ):
        # super().__init__ sets self.base_url / self.is_local from the
        # *user-provided* base_url (None → not local → display_name shows
        # the model name).  We keep that semantics: a provider's default
        # URL (e.g. Cerebras' hosted API) is not "local".
        super().__init__(
            model=model,
            base_url=base_url,
            stream_handler=stream_handler,
            temperature=temperature,
        )
        self._rate_limit_exceptions = (self._rate_limit_error_class(),)
        api_key, resolved_base_url = self._resolve_credentials(base_url)
        self._client = self._build_client(api_key, resolved_base_url)

    # ── Subclass hooks ───────────────────────────────────────────────

    def _rate_limit_error_class(self) -> type:
        """Return the SDK's rate-limit exception class."""
        import openai
        return openai.RateLimitError

    def _resolve_credentials(self, base_url: str | None) -> tuple[str, str | None]:
        """Return ``(api_key, base_url)`` for the client.

        Default behaviour (OpenAI / generic OpenAI-compatible server):
        read ``OPENAI_API_KEY``; when a ``base_url`` is given and no key is
        set, fall back to a placeholder key (local servers ignore it).
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if base_url:
            if not api_key:
                api_key = "local"
        else:
            if not api_key:
                raise Exception("OPENAI_API_KEY Environment Variable Unset")
        return api_key, base_url

    def _build_client(self, api_key: str, base_url: str | None):
        """Build the API client.  Default: the OpenAI SDK."""
        import openai
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    def _extra_create_kwargs(self) -> dict:
        """Extra keyword arguments merged into the ``create()`` call.

        The base implementation returns an empty dict; subclasses override
        it to inject provider-specific parameters (e.g. Cerebras'
        ``reasoning_effort``) without duplicating the request builder.
        """
        return {}

    # ── Message format translation ───────────────────────────────────

    @staticmethod
    def _format_messages(system_prompt: str, context: list[dict]) -> list[dict]:
        """Convert internal message format to OpenAI chat-completions input.

        Consecutive same-role messages (produced by harness feedback
        injections) are merged first so strict servers that enforce role
        alternation don't reject the payload.
        """
        context = merge_consecutive_messages(context)

        def _to_content(parts: list[dict]) -> str | list[dict]:
            """Handle both text-only and multimodal content."""
            text_parts = [p for p in parts if p.get("type") == "text"]
            image_parts = [p for p in parts if p.get("type") == "image"]

            if not image_parts:
                texts = [p.get("text", "") for p in text_parts]
                combined = "\n".join(t for t in texts if t)
                return combined

            items: list[dict] = []
            for img in image_parts:
                media_type = img.get("media_type", "image/png")
                data = img.get("data", "")
                if data:
                    items.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{media_type};base64,{data}"},
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
        cache_cost = pricing.get("cache_read_cost", input_cost * 0.1)
        uncached_input = max(0, input_tokens - cache_read_tokens)
        return (
            uncached_input * input_cost
            + cache_read_tokens * cache_cost
            + output_tokens * output_cost
        ) / 1_000_000

    # ── Error classification ─────────────────────────────────────────

    def _classify_error(self, error: Exception) -> str:
        if isinstance(error, self._rate_limit_exceptions):
            return RATE_LIMIT
        return TRANSIENT

    # ── Core: streaming API call with retries ────────────────────────

    def _get_response(self, system_prompt: str, context: list[dict]):
        """Call the chat-completions API with streaming and retries.

        Returns the full collected response text and the usage object.
        """
        self.call_count += 1
        sh = self.stream_handler
        messages = self._format_messages(system_prompt, context)

        create_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.MAX_COMPLETION_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.SEND_TEMPERATURE:
            create_kwargs["temperature"] = self.temperature
        # Provider-specific extras (e.g. Cerebras' reasoning_effort).
        create_kwargs.update(self._extra_create_kwargs())

        def attempt():
            stream = self._client.chat.completions.create(**create_kwargs)

            collected_text = ""
            usage = None
            reasoning_started = False
            tool_calls: dict[int, dict] = {}

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

                # Native tool calls stream as per-index fragments.  This
                # harness executes textual Command: lines, not API tool
                # calls, so these are accumulated for post-response logging.
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = tool_calls.setdefault(
                        getattr(tc, "index", 0) or 0,
                        {"name": "", "arguments": []},
                    )
                    func = getattr(tc, "function", None)
                    if func is not None:
                        name = getattr(func, "name", None)
                        if name:
                            slot["name"] = name
                        args = getattr(func, "arguments", None)
                        if args:
                            slot["arguments"].append(args)

                # Reasoning tokens: Cerebras uses ``delta.reasoning``,
                # vLLM / DeepSeek-style servers use ``delta.reasoning_content``.
                reasoning = (
                    getattr(delta, "reasoning", None)
                    or getattr(delta, "reasoning_content", None)
                )
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

            if reasoning_started:
                sh.on_stream_reasoning_end()

            for slot in tool_calls.values():
                self._pending_tool_calls.append((
                    slot["name"] or "?",
                    "".join(slot["arguments"]),
                ))

            return collected_text, usage

        return self._run_with_retries(attempt)

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        text, usage = self._get_response(system_prompt, context)

        if usage is not None:
            self.last_input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            self.last_output_tokens = getattr(usage, "completion_tokens", 0) or 0
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

        self._emit_tool_calls()

        if not text:
            raise EmptyResponseError("No text content found in model response")
        return text
