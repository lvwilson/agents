"""
Google Gemini backend.

Implements :class:`LLMBackend` using the ``google-genai`` unified SDK.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os

from ..llm_backend import (
    LLMBackend,
    StreamHandler,
    EmptyResponseError,
    RATE_LIMIT,
    TRANSIENT,
    merge_consecutive_messages,
    map_effort,
)


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

    MODEL_CONTEXT_WINDOWS: dict[str, int] = {
        "gemini-3.1-pro-preview":             2_000_000,
        "gemini-3.1-pro-preview-customtools": 2_000_000,
        "gemini-3-flash-preview":             1_000_000,
    }

    DEFAULT_CONTEXT_WINDOW: int = 1_000_000

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        base_url: str | None = None,
        cache_step: int = 2,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
        reasoning_effort: str | int | None = None,
        **_kwargs,
    ):
        super().__init__(
            model=model, base_url=base_url, stream_handler=stream_handler,
            temperature=temperature, reasoning_effort=reasoning_effort,
        )

        # Lazy import
        from google import genai as _genai
        from google.genai import types as _types
        self._genai = _genai
        self._types = _types

        # Argon hard prompts may take several minutes (official migration guide).
        http_options = {"timeout": 900_000} if model == "gemini-4-argon" else {}
        api_key = os.getenv("GEMINI_API_KEY")
        if base_url:
            if not api_key:
                api_key = "local"
            self._client = _genai.Client(
                api_key=api_key,
                http_options={**http_options, "base_url": base_url},
            )
        else:
            if not api_key:
                raise Exception("GEMINI_API_KEY Environment Variable Unset")
            self._client = _genai.Client(
                api_key=api_key,
                **({"http_options": http_options} if http_options else {}),
            )

        self.cache_step = cache_step

        # Context caching state
        self._cached_prefix_digest: str | None = None
        self._cache_name: str | None = None       # Server-side cache resource name
        self._cached_msg_count: int = 0            # Number of context messages in the cache
        self._cached_system_prompt: str | None = None  # System prompt stored in cache

    # ── Message format translation ───────────────────────────────────

    def _translate_messages(self, context: list[dict]) -> list:
        """Convert internal message format to Gemini Content objects.

        Internal format::

            [{"role": "user"|"assistant",
              "content": [{"type": "text", "text": "…"}, …]}, …]

        Gemini format::

            [Content(role="user"|"model", parts=[Part(text="…"), …]), …]

        Consecutive same-role messages (produced by harness feedback
        injections) are merged first so the strict role-alternation
        enforced by the Gemini API doesn't reject the payload.
        """
        context = merge_consecutive_messages(context)
        types = self._types
        contents = []
        for msg in context:
            role = "model" if msg["role"] == "assistant" else "user"
            content_parts = msg.get("content", [])

            parts = []
            native = msg.get("reasoning_context") or {}
            if (role == "model" and native.get("format") == "gemini"
                    and native.get("model") == self.model
                    and isinstance(native.get("data"), list)
                    and any(isinstance(p, dict) and p.get("thought_signature")
                            for p in native["data"])):
                # model_validate(dict) treats a base64 signature as literal
                # UTF-8 bytes. JSON validation uses the SDK's bytes codec.
                parts = [
                    types.Part.model_validate_json(json.dumps(part))
                    for part in native["data"]
                ]
                # These are the complete ordered text/thought parts, not
                # an extra prefix: appending visible text would duplicate it.
                contents.append(types.Content(role=role, parts=parts))
                continue
            for part in content_parts:
                if part.get("type") == "text" and part.get("text"):
                    parts.append(types.Part(text=part["text"]))
                elif part.get("type") == "image":
                    media_type = part.get("media_type", "image/png")
                    data = part.get("data", "")
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
            self._cached_prefix_digest = None

    def _prefix_digest(self, context: list[dict]) -> str:
        """Fingerprint the wire prefix, including native thought signatures."""
        payload = [
            content.model_dump(mode="json", exclude_none=True)
            for content in self._translate_messages(context)
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def supports_explicit_caching(self) -> bool:
        return self.model != "gemini-4-argon"

    def _create_cache(self, system_prompt: str, context: list[dict]) -> bool:
        """Create a server-side cache with the system prompt and context.

        Caches all messages except the last user message so that the
        next call can send only the final message as new content.
        Storage cost for the full TTL is charged immediately.

        Returns True if cache was created successfully, False otherwise.
        """
        if not self.supports_explicit_caching or len(context) < 2:
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
            self._cached_prefix_digest = self._prefix_digest(messages_to_cache)

            # Charge storage for the full TTL up-front.  Use the
            # API-reported token count when available, otherwise
            # fall back to the total prompt tokens as an upper bound.
            usage = getattr(cached_content, "usage_metadata", None)
            token_count = (
                getattr(usage, "total_token_count", 0) if usage else 0
            ) or self.last_input_tokens
            self.cost += self._calculate_storage_cost(token_count, self.CACHE_TTL)

            return True
        except Exception as e:
            self.stream_handler.on_retry(
                f"Cache creation failed (falling back to uncached): {e}"
            )
            self._cache_name = None
            self._cached_msg_count = 0
            self._cached_system_prompt = None
            self._cached_prefix_digest = None
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
        return getattr(self, "_cached_prefix_digest", None) == self._prefix_digest(
            context[:self._cached_msg_count]
        )

    @staticmethod
    def _is_cache_error(error: Exception) -> bool:
        """Return True if *error* looks like a stale/invalid cache error."""
        error_str = str(error)
        return "cache" in error_str.lower() or "NOT_FOUND" in error_str

    # ── Cost calculation ─────────────────────────────────────────────

    def _calculate_storage_cost(self, token_count: int, seconds: int) -> float:
        """Return the dollar cost for caching *token_count* tokens for *seconds*.

        Storage is charged per 1 M tokens per hour, prorated by the
        actual duration.
        """
        pricing = self.MODEL_PRICING.get(self.model)
        if not pricing:
            return 0.0
        storage_rate = pricing.get("cache_storage_cost_per_hour", 0.0)
        storage_hours = seconds / 3600.0
        return (token_count / 1_000_000) * storage_rate * storage_hours

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
        # Gemini has an explicit per-model cache read price rather than
        # a fixed fraction of the input price.  Cache creation is charged
        # at the normal input rate (no surcharge).
        cache_cost = pricing.get("cache_read_cost", input_cost * 0.10)
        return (
            input_tokens * input_cost
            + cache_creation_tokens * input_cost
            + cache_read_tokens * cache_cost
            + output_tokens * output_cost
        ) / 1_000_000

    # ── Error classification ─────────────────────────────────────────

    def _classify_error(self, error: Exception) -> str:
        # Prefer typed exception check when google.api_core is available
        try:
            from google.api_core.exceptions import ResourceExhausted, TooManyRequests
            if isinstance(error, (ResourceExhausted, TooManyRequests)):
                return RATE_LIMIT
        except ImportError:
            pass

        # Fall back to string matching for wrapped or non-typed errors
        error_str = str(error)
        if (
            "429" in error_str
            or "RESOURCE_EXHAUSTED" in error_str
            or "rate limit" in error_str.lower()
        ):
            return RATE_LIMIT

        return TRANSIENT

    # ── Core: streaming API call with retries ────────────────────────

    def _get_response(self, system_prompt: str, context: list[dict]):
        """Call the Gemini API with streaming and retries.

        Returns the collected response text and usage metadata.
        """
        self.call_count += 1
        sh = self.stream_handler
        types = self._types

        # Determine whether to use or create a cache
        should_create_cache = (
            self.supports_explicit_caching
            and not self.is_local
            and self.call_count % self.cache_step == 0
        )

        use_cache = False
        if (self.supports_explicit_caching and not self.is_local
                and self._is_cache_valid(system_prompt, context)):
            use_cache = True
        elif should_create_cache:
            if self._create_cache(system_prompt, context):
                use_cache = True

        def _do_stream(cached):
            """Run one streaming call, optionally using the server cache."""
            options = {
                "max_output_tokens": getattr(self, "MODEL_MAX_COMPLETION", {}).get(
                    self.model, 16384),
            }
            if getattr(self, "SEND_TEMPERATURE", True):
                options["temperature"] = self.temperature
            levels = getattr(self, "EFFORT_LEVELS", ())
            if levels and getattr(self, "reasoning_effort", None) is not None:
                level = map_effort(self.reasoning_effort, levels, levels[0])
                # Canonical "none" means minimal, not disabled, on Gemini 3.
                options["thinking_config"] = types.ThinkingConfig(
                    thinking_level="minimal" if level == "none" else level)
            if cached:
                contents = self._translate_messages(context[self._cached_msg_count:])
                options["cached_content"] = self._cache_name
            else:
                contents = self._translate_messages(context)
                options["system_instruction"] = system_prompt
            config = types.GenerateContentConfig(**options)

            stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

            collected_text = ""
            usage_metadata = None
            native_parts = []
            has_signature = False
            finished = False
            in_reasoning = False

            for chunk in stream:
                if getattr(chunk, "usage_metadata", None):
                    usage_metadata = chunk.usage_metadata
                candidates = getattr(chunk, "candidates", None)
                if candidates:
                    # We request one candidate; never mix alternate answers
                    # or their signatures into the same assistant turn.
                    candidate = candidates[0]
                    finish_reason = getattr(candidate, "finish_reason", None)
                    if finish_reason:
                        reason = getattr(finish_reason, "value", finish_reason)
                        finished = reason in {"STOP", "MAX_TOKENS"}
                    content = getattr(candidate, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        text = getattr(part, "text", None)
                        thought = bool(getattr(part, "thought", False))
                        signature = getattr(part, "thought_signature", None)
                        # Preserve boundaries and signature association, even
                        # for signature-only parts. Never concatenate these.
                        if text is not None or thought or signature:
                            native_parts.append(
                                part.model_dump(mode="json", exclude_none=True)
                            )
                            has_signature = has_signature or bool(signature)
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            self._pending_tool_calls.append((
                                getattr(fc, "name", "?") or "?",
                                str(getattr(fc, "args", "") or ""),
                            ))
                        if text:
                            if thought:
                                if not in_reasoning:
                                    sh.on_stream_reasoning_start()
                                    in_reasoning = True
                                sh.on_stream_reasoning_token(text)
                            else:
                                if in_reasoning:
                                    sh.on_stream_reasoning_end()
                                    in_reasoning = False
                                sh.on_stream_token(text)
                                collected_text += text
                elif not hasattr(chunk, "candidates"):
                    # Legacy lightweight client wrappers may expose only text.
                    # Real SDK chunks always have candidates; never use their
                    # convenience .text accessor (it can obscure thoughts).
                    for fc in getattr(chunk, "function_calls", None) or []:
                        self._pending_tool_calls.append((
                            getattr(fc, "name", "?") or "?",
                            str(getattr(fc, "args", "") or ""),
                        ))
                    text = getattr(chunk, "text", None)
                    if text:
                        sh.on_stream_token(text)
                        collected_text += text

            if in_reasoning:
                sh.on_stream_reasoning_end()
            if finished and collected_text and has_signature:
                self.last_reasoning_context = {
                    "format": "gemini", "model": self.model, "data": native_parts,
                }
            return collected_text, usage_metadata

        def attempt():
            nonlocal use_cache
            try:
                return _do_stream(use_cache)
            except Exception as e:
                # If the error is cache-related, invalidate and retry
                # once without the cache before letting _run_with_retries
                # handle rate-limit / transient classification.
                if use_cache and self._is_cache_error(e):
                    sh.on_retry("Cache expired or invalid, retrying without cache")
                    self._delete_cache()
                    use_cache = False
                    return _do_stream(False)
                raise

        return self._run_with_retries(attempt)

    # ── Public interface ─────────────────────────────────────────────

    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        self.last_reasoning_context = {}
        text, usage_metadata = self._get_response(system_prompt, context)

        if usage_metadata is not None:
            self.last_input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
            self.last_output_tokens = (
                getattr(usage_metadata, "candidates_token_count", 0)
                or getattr(usage_metadata, "response_token_count", 0)
                or 0
            )
            self.last_output_tokens += (
                getattr(usage_metadata, "thoughts_token_count", 0) or 0
            )
            cache_read = getattr(usage_metadata, "cached_content_token_count", 0) or 0

            # prompt_token_count from Gemini already includes cached
            # tokens, so uncached input = total prompt tokens - cached
            # tokens.
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
        # else: no usage on this call — keep the last known tracking
        # values untouched (zeroing them makes the session context appear
        # to collapse to ~0 mid-run and silently disables the
        # context-usage guard for the rest of the session).  Cost can't
        # be computed without usage, so it isn't accumulated.

        self._emit_tool_calls()

        if not text:
            raise EmptyResponseError("No text content found in model response")

        return text
