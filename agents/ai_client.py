#!/usr/bin/env python3
import anthropic
import os
import time

from ui import console, safe_console_print


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

    def __init__(self, model="claude-sonnet-4-5-20250929", cache_step=4):
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise Exception("CLAUDE_API_KEY Environment Variable Unset")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cost = 0.0
        self.call_count = 0
        self.cache_step = cache_step
        self.last_input_tokens = 0
        self.last_output_tokens = 0

    @property
    def display_name(self):
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

    def _get_response(self, system_prompt, context, max_retries=3):
        # Increment call counter
        self.call_count += 1

        # Determine if we should place a new cache block this turn
        should_cache = (self.call_count % self.cache_step == 0)

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
        
        response = None
        retries = 0
        
        while retries < max_retries:
            try:
                with self.client.messages.stream(
                    model=self.model,
                    max_tokens=64000,
                    temperature=0.6,
                    system=system_prompt,
                    messages=context,
                    extra_headers={"anthropic-beta": "output-128k-2025-02-19, prompt-caching-2024-07-31"}
                ) as stream:
                    for text in stream.text_stream:
                        safe_console_print(text, style="stream", end="")
                response = stream.get_final_message()
                if response:  # If a valid response is received, return it
                    return response
            except anthropic.RateLimitError as e:
                retries += 1
                if hasattr(e, 'response') and e.response is not None:
                    headers = e.response.headers
                    retry_after = int(headers.get('retry-after', 1))  # Default to 1 second if header is missing
                    safe_console_print(f"\n  ⏳ Rate limited — retrying in {retry_after}s", style="warning")
                    time.sleep(retry_after + 1)
            except Exception as e:
                retries += 1
                # Log or handle the exception as needed
                safe_console_print(f"\n  ✗ Attempt {retries}/{max_retries} failed: {e}", style="error")
        
        raise Exception("Maximum retries exceeded on response request")
                    
    def generate_response(self, system_prompt, context):
        response = self._get_response(system_prompt, context)
        self.last_input_tokens = response.usage.input_tokens
        self.last_output_tokens = response.usage.output_tokens
        self.cost += self.calculate_cost(self.last_input_tokens, self.last_output_tokens)
        return response.content[0].text

    def calculate_cost(self, input_tokens, output_tokens):
        pricing = self.MODEL_PRICING[self.model]
        cost = (input_tokens * pricing['input_token_cost'] + output_tokens * pricing['output_token_cost']) / 1_000_000
        return cost
    
def form_message(role, content):
    message = {
        "role": role,
        "content": convert_string_to_dict(content)
    }
    return message
