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

# Third-party imports
import yaml
from rich.console import Console

# Add llmide to path
sys.path.insert(2, '/home/loki/dev/llmide')
from llmide.llmide import process_content, filter_content, terminate_process
from llmide.llmide_functions import get_default_shell

# Local imports
from ai_client import ClaudeClient, safe_console_print, convert_string_to_dict

# Initialize console and global variables
console = Console()
script_dir = os.path.dirname(os.path.realpath(__file__))
pickle_path = os.path.join(script_dir, 'context.pkl')
iterations = 0

def sigterm_handler(_signo, _stack_frame):
    """Handle SIGTERM signal by terminating subprocess."""
    console.print("Sigterm caught, terminating subprocess...", style="red")
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

    def __init__(self, configuration_name, task, compute_budget=1.0, context=None):
        """Initialize the Claude Agent.
        
        Args:
            configuration_name: Name of the YAML configuration file
            task: The task to be performed
            compute_budget: Maximum allowed cost in dollars
            context: Optional list of previous conversation messages
        """
        if context is None:
            context = []
            
        # Load configuration
        configuration = read_configuration(configuration_name)
        self.model_name = configuration["model"]
        self.client = ClaudeClient(model=self.model_name)
        
        # Set up system prompt with environment information
        self.system_prompt = configuration["system_prompt"]
        os_info = platform.platform()
        self.system_prompt += f"\nOperating System: {os_info}"
        self.system_prompt += f"\nShell: {get_default_shell()}"
        
        # Set remaining attributes
        self.overbudget_prompt = configuration["overbudget"]
        self.context = context
        self.task = task
        self.context.append(ClaudeAgent._form_message("user", self.task))
        self.compute_budget = compute_budget

    @staticmethod
    def _form_message(role, content):
        """Create a message dictionary for Claude API.
        
        Args:
            role: Either "user" or "assistant"
            content: The message content
            
        Returns:
            dict: Formatted message
        """
        message = {
            "role": role,
            "content": convert_string_to_dict(content)
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
        console.print(f"[{iterations}] - ${self.client.cost}")
        iterations += 1
        
        # Generate response from Claude
        response = self.client.generate_response(self.system_prompt, self.context)
        
        # Filter response content
        response_length = len(response)
        response = filter_content(response)
        filtered_length = len(response)
        
        if response_length > filtered_length:
            clipped = response_length - filtered_length
            safe_console_print(f"\nClipped {clipped} characters from response", style="yellow")
            safe_console_print(response, style="cyan")
        
        # Add response to context and process it
        self.context.append(ClaudeAgent._form_message("assistant", response))
        command_response, image_media_tuple_array = process_content(response)

        # Check compute budget
        if self.client.cost > self.compute_budget:
            command_response += "\n" + self.overbudget_prompt
            console.print("Compute budget warning", style="yellow")

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
        try:
            running = self._iterate()
            while running:
                running = self._iterate()
                if self.client.cost > self.compute_budget:
                    console.print("Compute budget exceeded", style="red")
                    break
        except KeyboardInterrupt:
            console.print("Interrupted by user", style="yellow")
        except Exception as e:
            console.print(e, style="red")
            trace_info = traceback.format_exc()
            console.print(trace_info, style="white")
        
        console.print(f'Total Cost: ${self.client.cost:.4f}')

    def save_context(self, filename='context.pkl'):
        """Save conversation context to a pickle file.
        
        Args:
            filename: Path to save the pickle file
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.context, file)

    def load_context(self, filename='context.pkl'):
        """Load conversation context from a pickle file.
        
        Args:
            filename: Path to the pickle file
        """
        with open(filename, 'rb') as file:
            self.context = pickle.load(file)
        # Remove the last user message and replace with current task
        self.context.pop()  # TODO: Check that the last message is from the user
        self.context.append(ClaudeAgent._form_message("user", self.task))

def main():
    """Parse arguments and run the Claude Agent."""
    parser = argparse.ArgumentParser(description="Autonomous AI agent")
    parser.add_argument('command', type=str, help='A command string like "update my system"')
    parser.add_argument('-b', '--compute-budget', type=float, default=1.0, help='Compute budget in dollars')
    parser.add_argument('-r', '--restore', action='store_true', help='Restore previous context')

    args = parser.parse_args()
    
    # Create agent with basic agent configuration
    agent = ClaudeAgent('basic_agent.yaml', args.command, args.compute_budget)
    
    if args.restore:
        agent.load_context()
    
    agent.run()
    agent.save_context()

if __name__ == "__main__":
    main()
