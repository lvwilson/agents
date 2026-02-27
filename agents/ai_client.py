#!/usr/bin/env python3
import anthropic
import os
import random
import time

from ui import console, safe_console_print, create_spinner


def convert_string_to_dict(string, cache=False):
    result = []
    if cache:
        result.append({
            "type": "text",
            "text": string,
            "cache_control": {"type": "ephemeral"}
        })
    else:
        result.append({
            "type": "text",
            "text": string,
        })

    return result

class ClaudeClient():
    MODEL_PRICING = {
        "claude-3-5-sonnet-20240620": {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-3-5-sonnet-20241022": {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-3-7-sonnet-20250219": {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-sonnet-4-20250514" : {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-sonnet-4-5-20250929" : {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-sonnet-4-6" : {"input_token_cost": 3.00, "output_token_cost": 15.00},
        "claude-opus-4-6" : {"input_token_cost": 5.00, "output_token_cost": 25.00}
    }

    # Friendly display names for models
    MODEL_DISPLAY_NAMES = {
        "claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
        "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet v2",
        "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
        "claude-sonnet-4-20250514": "Claude Sonnet 4",
        "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
        "claude-sonnet-4-6": "Claude Sonnet 4.6",
        "claude-opus-4-6": "Claude Opus 4.6",
    }

    def __init__(self, model="claude-sonnet-4-5-20250929", cache_step=4, base_url=None):
        api_key = os.getenv("CLAUDE_API_KEY")
        if base_url:
            # Local backend: use dummy key if none is set
            if not api_key:
                api_key = "local"
            self.client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
            self.is_local = True
        else:
            if not api_key:
                raise Exception("CLAUDE_API_KEY Environment Variable Unset")
            self.client = anthropic.Anthropic(api_key=api_key)
            self.is_local = False
        self.model = model
        self.cost = 0.0
        self.call_count = 0
        self.cache_step = cache_step
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.peak_context_tokens = 0

    @property
    def display_name(self):
        if self.is_local:
            return f"{self.model} (local)"
        return self.MODEL_DISPLAY_NAMES.get(self.model, self.model)

    def _has_cache_block(self, message):
        """Return True if a user message already has a cache_control block."""
        return any("cache_control" in item for item in message.get("content", []))

    def _add_cache_block(self, message):
        """Add a cache_control block to the first text item of a user message."""
        for content_item in message.get("content", []):
            if content_item["type"] == "text":
                content_item["cache_control"] = {"type": "ephemeral"}
                break

    def _remove_cache_block(self, message):
        """Remove cache_control from all content items in a user message."""
        for content_item in message.get("content", []):
            content_item.pop("cache_control", None)

    # Retry configuration — exponential backoff for rate limits only
    RETRY_TIMEOUT = 300        # 5 minutes overall timeout for rate-limit retries
    RETRY_BASE_DELAY = 1       # Initial backoff delay in seconds
    RETRY_MAX_DELAY = 60       # Maximum backoff delay in seconds
    RETRY_BACKOFF_FACTOR = 2   # Exponential backoff multiplier
    MAX_ERROR_RETRIES = 3      # Fixed retry limit for non-rate-limit errors

    def _get_response(self, system_prompt, context):
        # Increment call counter
        self.call_count += 1

        # Determine if we should place a new cache block this turn
        should_cache = (not self.is_local) and (self.call_count % self.cache_step == 0)

        if should_cache:
            # Collect all user messages that currently carry a cache block, in context order
            cached_messages = [m for m in context if m["role"] == "user" and self._has_cache_block(m)]

            # Trim oldest cache blocks until only one remains, so the new one makes two
            while len(cached_messages) >= 2:
                self._remove_cache_block(cached_messages.pop(0))

            # Add a new cache block at the latest user message that doesn't already have one
            for message in reversed(context):
                if message["role"] == "user" and not self._has_cache_block(message):
                    self._add_cache_block(message)
                    break
        
        start_time = time.monotonic()
        error_retries = 0
        current_delay = self.RETRY_BASE_DELAY
        
        while True:
            try:
                spinner = create_spinner()
                spinner.start()
                stream_kwargs = dict(
                    model=self.model,
                    max_tokens=64000,
                    temperature=0.6,
                    system=system_prompt,
                    messages=context,
                )
                if not self.is_local:
                    stream_kwargs["extra_headers"] = {
                        "anthropic-beta": "output-128k-2025-02-19, prompt-caching-2024-07-31"
                    }
                with self.client.messages.stream(**stream_kwargs) as stream:
                    first_chunk = True
                    for text in stream.text_stream:
                        if first_chunk:
                            spinner.stop()
                            first_chunk = False
                        safe_console_print(text, style="stream", end="")
                    if first_chunk:
                        spinner.stop()
                response = stream.get_final_message()
                if response:  # If a valid response is received, return it
                    return response
            except anthropic.RateLimitError as e:
                spinner.stop()

                # Exponential backoff with 5-minute timeout for rate limits
                sleep_time = current_delay
                if hasattr(e, 'response') and e.response is not None:
                    headers = e.response.headers
                    retry_after = headers.get('retry-after')
                    if retry_after is not None:
                        try:
                            sleep_time = max(int(retry_after), sleep_time)
                        except (ValueError, TypeError):
                            pass  # fall back to exponential backoff delay

                # Add jitter: ±25% randomisation to avoid thundering herd
                jitter = sleep_time * 0.25 * (2 * random.random() - 1)
                sleep_time = max(0, sleep_time + jitter)

                # Ensure we don't exceed the overall timeout
                remaining = self.RETRY_TIMEOUT - (time.monotonic() - start_time)
                if remaining <= 0:
                    raise Exception(
                        f"Rate-limit retry timeout exceeded ({self.RETRY_TIMEOUT}s)"
                    )
                sleep_time = min(sleep_time, remaining)

                safe_console_print(
                    f"\n  ⏳ Rate limited — retrying in {sleep_time:.1f}s "
                    f"({remaining:.0f}s remaining)",
                    style="warning"
                )
                time.sleep(sleep_time)

                # Escalate delay for next rate-limit retry
                current_delay = min(current_delay * self.RETRY_BACKOFF_FACTOR, self.RETRY_MAX_DELAY)

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
                    style="error"
                )
                    
    def generate_response(self, system_prompt, context):
        response = self._get_response(system_prompt, context)
        self.last_input_tokens = response.usage.input_tokens
        self.last_output_tokens = response.usage.output_tokens
        self.peak_context_tokens = max(self.peak_context_tokens, self.last_input_tokens)
        self.cost += self.calculate_cost(self.last_input_tokens, self.last_output_tokens)
        # Find the first TextBlock, skipping ThinkingBlock objects from reasoning models
        for block in response.content:
            if hasattr(block, 'text'):
                if block.text:
                    return block.text
        raise Exception("No text content found in model response")

    def calculate_cost(self, input_tokens, output_tokens):
        pricing = self.MODEL_PRICING.get(self.model)
        if pricing is None:
            return 0.0
        cost = (input_tokens * pricing['input_token_cost'] + output_tokens * pricing['output_token_cost']) / 1_000_000
        return cost
    
def form_message(role, content):
    message = {
        "role": role,
        "content": convert_string_to_dict(content)
    }
    return message
