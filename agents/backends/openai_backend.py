"""
OpenAI backend.

Implements :class:`LLMBackend` using the ``openai`` Python SDK.
Supports both the hosted OpenAI API and any OpenAI-compatible local
server (e.g. vLLM, llama.cpp, Ollama with OpenAI compat layer).
"""

from __future__ import annotations

import os

from ..llm_backend import (
    LLMBackend,
    StreamHandler,
    EmptyResponseError,
    RATE_LIMIT,
    TRANSIENT,
    map_effort,
    merge_consecutive_messages,
)


class OpenAIBackend(LLMBackend):
    """OpenAI completions backend with streaming and retry logic."""

    MODEL_PRICING: dict[str, dict[str, float]] = {
        "gpt-5.2": {"input_token_cost": 2.50, "output_token_cost": 10.00, "cache_read_cost": 1.25},
        "gpt-5.2-mini": {"input_token_cost": 0.15, "output_token_cost": 0.60, "cache_read_cost": 0.075},
        "gpt-5.3": {"input_token_cost": 2.50, "output_token_cost": 10.00, "cache_read_cost": 1.25},
        "gpt-5.3-mini": {"input_token_cost": 0.15, "output_token_cost": 0.60, "cache_read_cost": 0.075},
        "gpt-5.3-codex": {"input_token_cost": 3.00, "output_token_cost": 12.00, "cache_read_cost": 1.50},
        "gpt-6-astra": {"input_token_cost": 10.00, "output_token_cost": 50.00, "cache_read_cost": 1.00},
    }

    MODEL_DISPLAY_NAMES: dict[str, str] = {
        "gpt-5.2": "GPT-5.2",
        "gpt-5.2-mini": "GPT-5.2 Mini",
        "gpt-5.3": "GPT-5.3",
        "gpt-5.3-mini": "GPT-5.3 Mini",
        "gpt-5.3-codex": "GPT-5.3 Codex",
        "gpt-6-astra": "GPT-6 Astra",
    }

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "gpt-5.2":       128_000,
        "gpt-5.2-mini":  128_000,
        "gpt-5.3":       128_000,
        "gpt-5.3-mini":  128_000,
        "gpt-5.3-codex": 128_000,
        "gpt-6-astra":   1_100_000,
    }

    #: Max output tokens per call (reasoning tokens count against this
    #: budget).  All GPT-5.x models keep the class default (16,384);
    #: gpt-6-astra raises it to its 128K output ceiling.
    MODEL_MAX_COMPLETION: dict[str, int] = {
        "gpt-6-astra": 128_000,
    }

    #: Effort levels the GPT models accept via the Responses API
    #: ``reasoning={"effort": ...}`` parameter (installed openai SDK's
    #: ``Reasoning.effort``).  ``none``/``minimal`` are not supported by
    #: the flagship reasoning models (gpt-6-astra refuses them), so the
    #: shallowest sendable level is ``low`` and a user ``--effort none``
    #: clamps to it (reasoning stays on, as shallow as the model allows).
    #: The default is the model's own — the parameter is omitted entirely
    #: when the user did not request an effort.
    EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "xhigh")
    DEFAULT_EFFORT: str | None = None

    def __init__(
        self,
        model: str = "gpt-5.3-codex",
        base_url: str | None = None,
        cache_step: int = 4,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
        reasoning_effort: str | None = None,
        **_kwargs,
    ):
        super().__init__(model=model, base_url=base_url, stream_handler=stream_handler, temperature=temperature,
                         reasoning_effort=reasoning_effort)

        # Per-model max output (falls back to the class default).
        self.MAX_COMPLETION_TOKENS = (
            self.MODEL_MAX_COMPLETION.get(model, 16_384)
        )

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
    def _format_messages(system_prompt: str, context: list[dict], model: str | None = None) -> list[dict]:
        """Convert internal message format to OpenAI Responses API input.

        Role/content constraints enforced:
        - system/user: input_text (and user may include input_image)
        - assistant: output_text (or refusal; we only emit output_text)

        Consecutive same-role messages (produced by harness feedback
        injections) are merged first so strict servers that enforce
        role alternation don't reject the payload.
        """
        context = merge_consecutive_messages(context)

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
                native = msg.get("reasoning_context") or {}
                if (model is not None and native.get("format") == "openai_responses"
                        and native.get("model") == model):
                    messages.extend(
                        dict(item) for item in native.get("data", [])
                        if isinstance(item, dict) and item.get("type") == "reasoning"
                    )
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
        # Cached input is billed at the model's declared cache_read_cost
        # (declared rates match half the input price for the GPT-5.x
        # family; GPT-6 Astra's cache is 90% off, so the table must
        # win over any flat discount factor).
        cache_read_cost = pricing.get(
            "cache_read_cost", pricing["input_token_cost"] * 0.5
        )

        return (
            uncached_input_tokens * pricing["input_token_cost"]
            + cache_read_tokens * cache_read_cost
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
        messages = self._format_messages(system_prompt, context, self.model)
        self._validate_responses_input(messages)

        create_kwargs: dict = {
            "model": self.model,
            "input": messages,
            "max_output_tokens": self.MAX_COMPLETION_TOKENS,
            "temperature": self.temperature,
            "stream": True,
            # Required for native reasoning replay without previous_response_id.
            "include": ["reasoning.encrypted_content"],
        }
        # Reasoning effort: only sent when the user requested one
        # (``--effort`` / config ``effort``, forwarded by the Agent as a
        # canonical rank).  map_effort() clamps to the levels the GPT
        # models accept (max→xhigh, none→low — they cannot disable
        # reasoning), so one flag never 400s.  Omitted entirely when no
        # effort is requested: the model's own default applies, and the
        # wire format for existing deployments is unchanged.
        effort = getattr(self, "reasoning_effort", None)
        if effort is not None:
            create_kwargs["reasoning"] = {
                "effort": map_effort(
                    effort, self.EFFORT_LEVELS, self.DEFAULT_EFFORT
                )
            }

        def attempt():
            stream = self._client.responses.create(**create_kwargs)

            collected_text = ""
            usage = None
            completed = None
            reasoning_items = []
            in_reasoning = False

            for event in stream:
                if event.type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                }:
                    if not in_reasoning:
                        sh.on_stream_reasoning_start()
                        in_reasoning = True
                    if event.delta:
                        sh.on_stream_reasoning_token(event.delta)
                elif event.type in {
                    "response.reasoning_summary_text.done",
                    "response.reasoning_text.done",
                }:
                    if in_reasoning:
                        sh.on_stream_reasoning_end()
                        in_reasoning = False
                elif event.type == "response.output_text.delta":
                    if in_reasoning:
                        sh.on_stream_reasoning_end()
                        in_reasoning = False
                    text = event.delta
                    if text:
                        sh.on_stream_token(text)
                        collected_text += text
                elif event.type == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if getattr(item, "type", None) == "reasoning":
                        reasoning_items.append(item)
                    elif getattr(item, "type", None) == "function_call":
                        self._pending_tool_calls.append((
                            getattr(item, "name", "?") or "?",
                            getattr(item, "arguments", "") or "",
                        ))
                elif event.type == "response.completed":
                    if hasattr(event, "response") and event.response:
                        completed = event.response
                        usage = getattr(completed, "usage", None)
                elif event.type in {"response.failed", "response.incomplete", "error"}:
                    raise RuntimeError(f"OpenAI stream did not complete: {event.type}")

            if in_reasoning:
                sh.on_stream_reasoning_end()
            # Do not publish intermediate output_item.done data until the
            # entire stream has ended successfully and response.completed
            # confirms it. A dropped/failed stream cannot supply signatures.
            if completed is not None and collected_text:
                output = getattr(completed, "output", None)
                items = output if output is not None else reasoning_items
                native = [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in items
                    if getattr(item, "type", None) == "reasoning"
                    and hasattr(item, "model_dump")
                ]
                if native:
                    self.last_reasoning_context = {
                        "format": "openai_responses", "model": self.model,
                        "data": native,
                    }
            return collected_text, usage

        return self._run_with_retries(attempt)

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        self.last_reasoning_context = {}
        text, usage = self._get_response(system_prompt, context)

        if usage is not None:
            cache_read = 0
            self.last_input_tokens = getattr(usage, "input_tokens", 0) or 0
            self.last_output_tokens = getattr(usage, "output_tokens", 0) or 0
            if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
                cache_read = (
                    getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
                )

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
        # else: no usage on this call — keep the last known tracking values
        # untouched (zeroing them makes the session context appear to
        # collapse to ~0 mid-run and silently disables the context-usage
        # guard for the rest of the session).  Cost can't be computed
        # without usage, so it isn't accumulated.

        self._emit_tool_calls()

        if not text:
            raise EmptyResponseError("No text content found in model response")

        return text
