#!/usr/bin/env python3
import sys
#sys.path.insert(2, '/home/loki/dev/llmide')
from llmide.llmide import process_content

import os
import anthropic
import argparse
from rich.console import Console

console = Console()

system_prompt = """
You are an AI software engineer. Respond to requests with detailed thoughts and plans, followed by the appropriate command and code if needed. Use the following format:

Response Format:
Task: What you are working on as given by the user
End conditions: When you know you are finished, explcitly when the user has asked you to finish
Worklog: What you have done
Detailed thoughts and Plans:
Provide a detailed explanation of your thought process and plans for addressing the request.

Command: command_name arg1 arg2 ... argn
```Python
 #Code goes here when the command requires code.
```

The system automatically parses your response for the above structure, when no command is given it is assumed that you have finished.

Address Definition:
Addresses are a dot-separated path indicating the location of the target node. This can be a top-level function or class ("FunctionName"), a method within a class ("ClassName.method_name"), or elements within nested classes ("OuterClass.InnerClass.method_name").

Commands:
read_code_signatures_and_docstrings file_path
write_code_to_file file_path ```code```
read_code_from_file file_path
insert_code_before_matching_line file_path line ```code```
insert_code_after_matching_line file_path line ```code```
replace_code_before_matching_line file_path line ```code```
replace_code_after_matching_line file_path line ```code```
replace_code_between_matching_lines file_path line1 line2 ```code```

Note regarding backticks: to simplify syntax understanding they are presented on the same line but they must be on a new line under the command as in the examples.
run_console_command "arguments"

Example output: 
Detailed thoughts and Plans: This is how to read a file
Command: read_code_from_file file.py

Example output:
Detailed thoughts and Plans: This is how to write code to a file
Command: write_code_to_file file.py
```Python
print("Hello World!")
```

Example output:
Detailed thoughts and Plans: This is how to execute a console command. Avoid running interactive commands as they are not supported.
Command: run_console_command "ls -l"

Example output:
Detailed thoughts and Plans: This is how to manipulate code using insert_code_after_matching_line.
Command: insert_code_after_matching_line "import foo" 
```Python
import bar
```

Only one command at a time can be extracted so only issue one command per step.
Indentation is crucial, ensure that indentation matches the indentation of the target document when manipulating code.
When writing code using write_code_to_file always include all the code. If there is a significant ammount of code to leave intact please use the other manipulation functions.
"""


def form_message(role, content):
    message = {
        "role":role,
        "content":convert_string_to_dict(content)
    }
    return message

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

    def generate_response(self, system_prompt, context):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4095,
            temperature=0.5,
            system=system_prompt,
            messages=context
        )
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        self.cost += self.calculate_cost(input_tokens, output_tokens)
        return response.content[0].text

    def calculate_cost(self, input_tokens, output_tokens):
        pricing = self.MODEL_PRICING[self.model]
        cost = (input_tokens * pricing['input_token_cost'] + output_tokens * pricing['output_token_cost']) / 1_000_000
        return cost


def simple_agent (task_prompt):
    global system_prompt
    user_prompt = task_prompt + "\nPlease execute the given task to the best of your ability. The user role will be used for tool responses going forward."
    context = []
    context.append(form_message("user", user_prompt))
    client = ClaudeClient()
    response = client.generate_response(system_prompt, context)
    context.append(form_message("assistant", response))
    console.print (response, style="cyan")
    command_response = process_content(response)
    console.print(command_response)
    context.append(form_message("user", command_response))
    try:
        while command_response != "End.":
            response = client.generate_response(system_prompt, context)
            context.append(form_message("assistant", response))
            console.print (response, style="cyan")
            command_response = process_content(response)
            console.print(command_response)
            context.append(form_message("user", command_response))
    except KeyboardInterrupt:
        pass
    print(f'Total Cost: {client.cost}')

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI agent")
    parser.add_argument('command', type=str, help='A command string like "update my system"')
    args = parser.parse_args()
    simple_agent(args.command)

if __name__ == "__main__":
    main()