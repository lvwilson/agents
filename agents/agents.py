#!/usr/bin/env python3
import sys
sys.path.insert(2, '/home/loki/dev/llmide')
from llmide.llmide import process_content
from llmide.llmide_functions import get_default_shell

import anthropic
import argparse
import os
from rich.console import Console
import yaml
import pickle
import time
import platform

console = Console()
script_dir = os.path.dirname(os.path.realpath(__file__))
pickle_path = os.path.join(script_dir, 'context.pkl')

def convert_string_to_dict(string):
    result = []
    result.append({
        "type": "text",
        "text": string
    })
    return result

class ClaudeClient():
    MODEL_PRICING = {
        "claude-3-5-sonnet-20240620": {"input_token_cost": 3.00, "output_token_cost": 15.00}
    }

    def __init__(self, model="claude-3-5-sonnet-20240620"):
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise Exception("CLAUDE_API_KEY Environment Variable Unset")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.cost = 0.0

    def _get_response(self, system_prompt, context, max_retries=3):
        response = None
        retries = 0
        
        while retries < max_retries:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    temperature=0.5,
                    system=system_prompt,
                    messages=context,
                    extra_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"}
                )
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
                print(f"Attempt {retries} failed: {e}")
        
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
            "role":role,
            "content":convert_string_to_dict(content)
        }
        return message

def safe_console_print(text, style="default"):
    try:
        console.print(text, style=style)
    except Exception:
        print(text)

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def read_configuration(configuration_name):
    script_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, configuration_name)

    configuration = read_yaml_file(config_path)
    return configuration

class ClaudeAgent:

    def __init__(self, configuration_name, task, compute_budget=1.0, context=[]):
        configuration = read_configuration(configuration_name)
        self.model_name = configuration["model"]
        self.client = ClaudeClient(model=self.model_name)
        self.system_prompt = configuration["system_prompt"]
        os_info = platform.platform()
        self.system_prompt += f"\nOperating System: {os_info}"
        self.system_prompt += f"\nShell: {get_default_shell()}"
        self.overbudget_prompt = configuration["overbudget"]
        self.context = context
        self.task = task
        self.context.append(ClaudeAgent._form_message("user", self.task))
        self.compute_budget = compute_budget

    @staticmethod
    def _form_message(role, content):
        message = {
            "role":role,
            "content":convert_string_to_dict(content)
        }
        return message
    
    def _iterate(self):
        response = self.client.generate_response(self.system_prompt, self.context)
        self.context.append(ClaudeAgent._form_message("assistant", response))
        safe_console_print(response, style="cyan")
        command_response = process_content(response)

        self.context.append(ClaudeAgent._form_message("user", command_response))
        command_called = not (command_response == "End.")
        return command_called
    
    def run(self):
        try:
            running = self._iterate()
            cost = 0.0
            while (running):
                running = self._iterate()
                if self.client.cost > self.compute_budget:
                    console.print("Compute budget exceeded", style="red")
                    break
        except Exception as e:
            console.print(e, style="red")
        console.print(f'Total Cost: {self.client.cost}')

    def save_context(self, filename='context.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump(self.context, file)

    def load_context(self, filename='context.pkl'):
        with open(filename, 'rb') as file:
            self.context = pickle.load(file)
        self.context.pop() #todo: this should check that the last message is from the user but for now whatever
        self.context.append(ClaudeAgent._form_message("user", self.task))

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI agent")
    parser.add_argument('command', type=str, help='A command string like "update my system"')
    parser.add_argument('-b', '--compute-budget', type=float, default=1.0, help='Compute budget in dollars')
    parser.add_argument('-r', '--restore', action='store_true', help='Restore previous context')

    args = parser.parse_args()
    #agent = ClaudeAgent('manipulator_agent.yaml', args.command, args.compute_budget)
    agent = ClaudeAgent('basic_agent.yaml', args.command, args.compute_budget)
    if args.restore:
        agent.load_context()
    
    agent.run()
    agent.save_context()


if __name__ == "__main__":
    main()