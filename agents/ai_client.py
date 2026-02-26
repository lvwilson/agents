#!/usr/bin/env python3
import anthropic
import os
import time
from rich.console import Console

console = Console()

def safe_console_print(text, style="default", end="\n"):
    try:
        console.print(text, style=style, end=end)
    except Exception:
        print(text)

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
        "claude-sonnet-4-6" : {"input_token_cost": 3.00, "output_token_cost": 15.00}
    }

    def __init__(self, model="claude-sonnet-4-5-20250929", cache_step=2):
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise Exception("CLAUDE_API_KEY Environment Variable Unset")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cost = 0.0
        self.call_count = 0
        self.cache_step = cache_step

    def _get_response(self, system_prompt, context, max_retries=3):
        # Increment call counter
        self.call_count += 1
        
        # Determine if we should cache this request
        should_cache = (self.call_count % self.cache_step == 0)
        
        # Remove any existing cache_control blocks from context
        for message in context:
            if message["role"] == "user" and "content" in message:
                for content_item in message["content"]:
                    if "cache_control" in content_item:
                        del content_item["cache_control"]
        
        # Add cache_control to the latest user message if needed
        if should_cache and context:
            for i in range(len(context) - 1, -1, -1):
                if context[i]["role"] == "user":
                    for content_item in context[i]["content"]:
                        if content_item["type"] == "text":
                            content_item["cache_control"] = {"type": "ephemeral"}
                            break
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
                        safe_console_print(text, style="cyan", end="")
                response = stream.get_final_message()
                if response:  # If a valid response is received, return it
                    return response
            except anthropic.RateLimitError as e:
                retries += 1
                if hasattr(e, 'response') and e.response is not None:
                    headers = e.response.headers
                    retry_after = int(headers.get('retry-after', 1))  # Default to 1 second if header is missing
                    safe_console_print(f"Rate limit exceeded, retrying in: {retry_after}s", style="yellow")
                    time.sleep(retry_after + 1)
            except Exception as e:
                retries += 1
                # Log or handle the exception as needed
                safe_console_print(f"Attempt {retries} failed: {e}", style="red")
        
        raise Exception("Maximum retries exceeded on response request")
                    
    def generate_response(self, system_prompt, context):
        response = self._get_response(system_prompt, context)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self.cost += self.calculate_cost(input_tokens, output_tokens)
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
