#!/usr/bin/env python3
"""
Claude Agent - An autonomous AI agent using Anthropic's Claude API.
"""
# Standard library imports
import argparse
import logging
import os
import pickle
import platform
import signal
import sys
import traceback
import time
import re

# Third-party imports
import yaml

# llmide
from llmide.llmide import process_content, filter_content, terminate_process
from llmide.llmide_functions import get_default_shell

# Local imports
from backends import create_backend
from ai_client import convert_string_to_dict
from ui import (
    RichStreamHandler,
    print_banner,
    print_iteration_header,
    print_summary,
    print_completion_result,
    print_budget_warning,
    print_budget_exceeded,
    print_error,
    print_interrupted,
    print_sigterm,
    print_clipped,
    safe_console_print,
)

# Global state
script_dir = os.path.dirname(os.path.realpath(__file__))
pickle_path = os.path.join(script_dir, 'context.pkl')
iterations = 0


def extract_completion(text, backticks=5):
    """
    Extract the completion section from the given text.
    
    Args:
        text (str): The text to extract the completion from.
        backticks (int): The number of backticks used to wrap the YAML section (default: 5).
    
    Returns:
        dict: A dictionary containing the completion information, or None if no completion was found.
    """
    # Create the pattern for matching the backtick-wrapped YAML section
    backtick_pattern = '`' * backticks
    pattern = rf"{backtick_pattern}(Completion:[\s\S]*?){backtick_pattern}"
    
    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    # Extract the YAML content
    yaml_content = match.group(1).strip()
    
    try:
        # Parse the YAML content using PyYAML
        completion_data = yaml.safe_load(yaml_content)
        completion_text = completion_data['Completion']
        if isinstance(completion_text, str):
            completion_text = completion_text.strip()
        return completion_text, completion_data['Success']
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}", file=sys.stderr)
        return "Task could not be verified.", False


def sigterm_handler(_signo, _stack_frame):
    """Handle SIGTERM signal by terminating subprocess."""
    print_sigterm()
    terminate_process()

# Register signal handler
signal.signal(signal.SIGTERM, sigterm_handler)


def read_yaml_file(file_path):
    """Read and parse a YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        dict: Parsed YAML content
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def read_configuration(configuration_name):
    """Read agent configuration from a YAML file.
    
    Args:
        configuration_name: Name of the configuration file
        
    Returns:
        dict: Configuration data
    """
    script_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, configuration_name)
    return read_yaml_file(config_path)


class ClaudeAgent:
    """An autonomous agent powered by Claude AI.
    
    This agent can execute tasks, maintain context, and manage compute budget.
    """

    def __init__(self, configuration_name, task, compute_budget=1.0, context=None,
                 local_model=None, local_port=8000):
        """Initialize the Claude Agent.
        
        Args:
            configuration_name: Name of the YAML configuration file
            task: The task to be performed
            compute_budget: Maximum allowed cost in dollars
            context: Optional list of previous conversation messages
            local_model: If set, use a local Anthropic-compatible API with this model name
            local_port: Port for the local API server (default 8000)
        """
        if context is None:
            context = []
            
        # Load configuration
        configuration = read_configuration(configuration_name)
        
        # Determine provider from environment variable or config
        provider = os.environ.get("AGENT_MODEL_PROVIDER", configuration.get("provider", "anthropic"))
        
        # Determine model from environment variable with provider-specific defaults
        model_env = os.environ.get("AGENT_MODEL")
        if local_model:
            self.model_name = local_model
            base_url = f"http://localhost:{local_port}"
        elif model_env:
            self.model_name = model_env
            base_url = configuration.get("base_url", None)
        else:
            # Provider-specific defaults
            provider_defaults = {
                "anthropic": "claude-opus-4-6",
                "openai": "gpt-5.3-codex",
                "gemini": "gemini-3.1-pro-preview",
            }
            self.model_name = provider_defaults.get(provider, "claude-opus-4-6")
            base_url = configuration.get("base_url", None)

        self.client = create_backend(
            provider,
            model=self.model_name,
            base_url=base_url,
            stream_handler=RichStreamHandler(),
        )
        
        # Set up system prompt with environment information.
        # NOTE: This prompt is persisted in save_context() and restored on
        # resume so that the prompt-cache prefix stays identical.  Any
        # dynamic content (e.g. the timestamp below) would otherwise
        # invalidate the entire Anthropic prompt cache on resumption.
        self.system_prompt = configuration["system_prompt"]
        os_info = platform.platform()
        self.system_prompt += f"\nOperating System: {os_info}"
        self.system_prompt += f"\nShell: {get_default_shell()}"
        self.system_prompt += f"\nSystem Date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        self.system_prompt += f"\nWorking Directory: {os.getcwd()}"
        self.system_prompt += f"\nUser: {os.environ.get('USER', 'unknown')}"
        
        # Set remaining attributes
        self.overbudget_prompt = configuration["overbudget"]
        self.context = context
        self.task = task
        self.context.append(ClaudeAgent._form_message("user", self.task))
        self.compute_budget = compute_budget
        self.start_time = None

        # Display startup banner
        print_banner(self.client.display_name, self.compute_budget, platform.platform())

    @staticmethod
    def _form_message(role, content, cache=False):
        """Create a message dictionary for Claude API.
        
        Args:
            role: Either "user" or "assistant"
            content: The message content
            
        Returns:
            dict: Formatted message
        """
        message = {
            "role": role,
            "content": convert_string_to_dict(content, cache)
        }
        return message

    @staticmethod
    def _form_message_with_images(role, content, image_media_type_tuple_array):
        """Create a message dictionary that includes images for Claude API.
        
        Args:
            role: Either "user" or "assistant"
            content: The text content
            image_media_type_tuple_array: List of (image_base64, media_type) tuples
            
        Returns:
            dict: Formatted message with images
        """
        images = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64,
                },
            }
            for image_base64, media_type in image_media_type_tuple_array
        ]

        text_content = {
            "type": "text",
            "text": content
        }

        combined_content = images + [text_content]

        return {
            "role": role,
            "content": combined_content
        }
    
    def _iterate(self):
        """Perform one iteration of the conversation with Claude.
        
        Returns:
            bool: True if the agent should continue running, False otherwise
        """
        global iterations
        print_iteration_header(
            iterations, self.client.cost, self.compute_budget,
            self.client.last_input_tokens, self.client.last_output_tokens,
            self.client.last_total_context_tokens,
            cost_without_cache=self.client.cost_without_cache,
        )
        iterations += 1
        
        # Generate response from Claude
        response = self.client.generate_response(self.system_prompt, self.context)
        
        if not response:
            return False

        # Filter response content
        response_length = len(response)
        response = filter_content(response)
        filtered_length = len(response)
        
        if response_length > filtered_length:
            clipped = response_length - filtered_length
            print_clipped(clipped, response)
        
        # Add response to context and process it
        self.context.append(ClaudeAgent._form_message("assistant", response))
        command_response, image_media_tuple_array = process_content(response)

        # Check compute budget
        if self.client.cost > 0.75 * self.compute_budget:
            command_response += "\n" + self.overbudget_prompt
            print_budget_warning(self.client.cost, self.compute_budget)

        # Add user message to context (with or without images)
        if len(image_media_tuple_array) == 0:
            self.context.append(ClaudeAgent._form_message("user", command_response))
        else:
            message = ClaudeAgent._form_message_with_images("user", command_response, image_media_tuple_array)
            self.context.append(message)
            
        # Determine if we should continue running
        command_called = not (command_response == "End.")
        return command_called
    
    def run(self):
        """Run the agent until completion or interruption."""
        self.start_time = time.time()
        try:
            running = self._iterate()
            while running:
                running = self._iterate()
                if self.client.cost > self.compute_budget:
                    print_budget_exceeded(self.client.cost, self.compute_budget)
                    break
        except KeyboardInterrupt:
            print_interrupted()
        except Exception as e:
            print_error(e, traceback.format_exc())
        
        # Print final summary
        elapsed = time.time() - self.start_time
        print_summary(self.client.cost, iterations, elapsed, self.compute_budget,
                      self.client.peak_context_tokens,
                      cost_without_cache=self.client.cost_without_cache)

    def save_context(self, filename='context.pkl'):
        """Save conversation context and token state to a pickle file.
        
        Args:
            filename: Path to save the pickle file
        """
        state = {
            'context': self.context,
            'system_prompt': self.system_prompt,
            'total_context_tokens': self.client.last_total_context_tokens,
            'peak_context_tokens': self.client.peak_context_tokens,
            'last_input_tokens': self.client.last_input_tokens,
            'last_output_tokens': self.client.last_output_tokens,
            'cost': self.client.cost,
            'cost_without_cache': self.client.cost_without_cache,
            'call_count': self.client.call_count,
        }
        with open(filename, 'wb') as file:
            pickle.dump(state, file)

    def load_context(self, filename='context.pkl'):
        """Load conversation context and token state from a pickle file.
        
        Args:
            filename: Path to the pickle file
        """
        with open(filename, 'rb') as file:
            data = pickle.load(file)

        # Support both old format (bare list) and new format (dict with state)
        if isinstance(data, dict):
            self.context = data['context']
            # Restore the original system prompt so the prompt cache
            # remains valid across resumed sessions.
            if 'system_prompt' in data:
                self.system_prompt = data['system_prompt']
            self.client.last_total_context_tokens = data.get('total_context_tokens', 0)
            self.client.peak_context_tokens = data.get('peak_context_tokens', 0)
            self.client.last_input_tokens = data.get('last_input_tokens', 0)
            self.client.last_output_tokens = data.get('last_output_tokens', 0)
            self.client.cost = data.get('cost', 0.0)
            self.client.cost_without_cache = data.get('cost_without_cache', 0.0)
            self.client.call_count = data.get('call_count', 0)
        else:
            # Legacy format: data is just the context list
            self.context = data

        # Remove the last user message and replace with current task
        self.context.pop()  # TODO: Check that the last message is from the user
        self.context.append(ClaudeAgent._form_message("user", self.task, True))
        # Trim so at most 2 cache blocks remain after adding the new one
        self.client.trim_cache_blocks(self.context)


def run_agent(agent_definition, command, budget, save=True, restore=False,
              local_model=None, local_port=8000):
    agent = ClaudeAgent(agent_definition, command, budget,
                        local_model=local_model, local_port=local_port)
    if restore:
        agent.load_context()
    agent.run()
    completion = "Error"
    success = False
    if len(agent.context) > 2:
        final_content = agent.context[-2]['content'][0]['text']
        result = extract_completion(final_content)
        if result is not None:
            completion, success = result
        else:
            # Give the agent one more chance to provide a completion block,
            # but only if there is budget remaining.
            if agent.client.cost <= agent.compute_budget:
                feedback = ("Feedback: No completion block was found in your response. "
                            "Please provide a completion block with "
                            "'Completion: <description>' and 'Success: True/False' "
                            "at the end of your response.")
                agent.context.append(ClaudeAgent._form_message("user", feedback))
                try:
                    agent._iterate()
                except Exception as e:
                    logging.warning("Completion-retry iteration failed: %s", e)
                # Check the new response for a completion block
                if len(agent.context) > 2:
                    final_content = agent.context[-2]['content'][0]['text']
                    result = extract_completion(final_content)
                    if result is not None:
                        completion, success = result

    if save:
        agent.save_context()
    
    return completion, success


def main():
    """Parse arguments and run the Claude Agent."""
    parser = argparse.ArgumentParser(description="Autonomous AI agent")
    parser.add_argument('command', type=str, help='A command string like "update my system"')
    parser.add_argument('-b', '--compute-budget', type=float, default=1.0, help='Compute budget in dollars')
    parser.add_argument('-r', '--restore', action='store_true', help='Restore previous context')
    parser.add_argument('-l', '--local', action='store_true',
                        help='Use a local Anthropic-compatible API (requires LOCAL_MODEL environment variable)')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port for the local API server (default: 8000)')

    args = parser.parse_args()

    command = args.command
    if not sys.stdin.isatty():
        piped_content = sys.stdin.read()
        if piped_content:
            backticks = '`' * 5
            command = command + "\n" + backticks + "\n" + piped_content + "\n" + backticks

    # Resolve local model: only use local mode when -l is explicitly passed
    local_model = None
    if args.local:
        local_model = os.environ.get('LOCAL_MODEL')
        if not local_model:
            parser.error('--local requires the LOCAL_MODEL environment variable to be set')

    completion, success = run_agent('basic_agent.yaml', command, args.compute_budget,
                                    restore=args.restore, local_model=local_model,
                                    local_port=args.port)
    print_completion_result(completion, success)


if __name__ == "__main__":
    main()
