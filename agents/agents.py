#!/usr/bin/env python3
import signal
import logging
import traceback
import sys
sys.path.insert(2, '/home/loki/dev/llmide')
from llmide.llmide import process_content, filter_content, terminate_process
from llmide.llmide_functions import get_default_shell

# Import from the new ai_client module
from ai_client import ClaudeClient, safe_console_print, convert_string_to_dict

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
iterations = 0

def sigterm_handler(_signo, _stack_frame):
    console.print("Sigterm caught, terminating subprocess...", style="red")
    terminate_process()

signal.signal(signal.SIGTERM, sigterm_handler)

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

    def _form_message_with_images(role, content, image_media_type_tuple_array):
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
        global iterations
        console.print(f"[{iterations}] - ${self.client.cost}")
        iterations+=1
        response = self.client.generate_response(self.system_prompt, self.context)
        response_length = len(response)
        response = filter_content(response)
        filtered_length = len(response)
        if (response_length > filtered_length):
            clipped = response_length - filtered_length
            safe_console_print(f"\nClipped {clipped} characters from response", style="yellow")
            safe_console_print(response, style="cyan")
        
        self.context.append(ClaudeAgent._form_message("assistant", response))
        command_response, image_media_tuple_array = process_content(response)

        if self.client.cost > self.compute_budget:
            command_response +=  "/n" + self.overbudget_prompt
            console.print("Compute budget warning", style="yellow")

        if len(image_media_tuple_array) == 0:
            self.context.append(ClaudeAgent._form_message("user", command_response))
        else:
            message = ClaudeAgent._form_message_with_images("user", command_response, image_media_tuple_array)
            #print(message)
            self.context.append(message)
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
        except KeyboardInterrupt:
            pass
        except Exception as e:
            console.print(e, style="red")
            trace_info = traceback.format_exc()
            console.print(trace_info, style="white")
        
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
