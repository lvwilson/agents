"""
Command parser — extracts commands from LLM output, dispatches to tool functions.

This module owns all parsing and dispatch logic:
- ``process_content()`` — parse and execute commands
- ``filter_content()`` — trim output when multiple read commands are queued
- ``terminate_process()`` — kill any running subprocess
"""

import re
import os
import io
import base64
from collections import namedtuple

from PIL import Image, UnidentifiedImageError

from . import functions


def split_preserving_quotes(s):
    """Split by spaces but preserve quoted segments."""
    pattern = r'(?:"[^"]*"|\'[^\']*\'|\S)+'
    matches = re.findall(pattern, s)
    result = [match[1:-1] if match[0] in ('"', "'") else match for match in matches]
    return result


def process_slice(content):
    """Extract the first command, its arguments, backtick payload, and remaining content."""
    command_pattern = r"^Command: (\S+)\s*(.*)$"
    backtick_pattern = r"`````(?:[\w#\+\-]+)?\s*(.*?)`````"

    command_match = re.search(command_pattern, content, re.MULTILINE)
    if command_match:
        command = command_match.group(1)
        arguments = command_match.group(2)
        command_end_pos = command_match.end()
    else:
        command = None
        arguments = None
        command_end_pos = -1

    backtick_match = re.search(backtick_pattern, content, re.DOTALL)
    if backtick_match:
        backtick_content = backtick_match.group(1)
        backtick_start_pos = backtick_match.start()
        backtick_end_pos = backtick_match.end()
    else:
        backtick_content = None
        backtick_end_pos = -1
        backtick_start_pos = -1

    # Ignore backticks if not directly attached to command
    if (backtick_start_pos - command_end_pos > 1):
        backtick_content = None
        backtick_end_pos = -1

    split_position = max(command_end_pos, backtick_end_pos)
    remaining_content = content[split_position:].strip()
    if command:
        return command, arguments, backtick_content, remaining_content
    else:
        return None, None, None, None


CommandInfo = namedtuple('CommandInfo', ['command', 'arguments', 'backtick_content'])


def concise_representation(input_string, max_chars):
    """Truncate a string to max_chars, showing start and end with ellipsis."""
    if len(input_string) <= max_chars:
        return input_string
    part_length = (max_chars - 3) // 2
    first_part = input_string[:part_length]
    last_part = input_string[-part_length:] if (max_chars % 2 == 0) else input_string[-(part_length + 1):]
    return f"{first_part}...{last_part}"


# Commands that can be stacked (queued together like read_file)
STACKABLE_READ_COMMANDS = {'read_file', 'read_page', 'read_page_html', 'page_links', 'view_page', 'web_search'}


def filter_content(content):
    """Cut output at the final read command or first non-read command after a read command."""
    read_command_encountered = False
    command, arguments, backtick_content, remaining_content = process_slice(content)
    if command:
        command = CommandInfo(command, arguments, backtick_content)
        if command.command in STACKABLE_READ_COMMANDS:  # FIX: was `command == 'read_file'`
            read_command_encountered = True
    previous_remaining_content = remaining_content
    while command:
        command, arguments, backtick_content, remaining_content = process_slice(remaining_content)
        if command:
            if command in STACKABLE_READ_COMMANDS:  # `command` is a raw string here
                read_command_encountered = True
            elif read_command_encountered:
                n_to_copy = len(content) - len(previous_remaining_content)
                return content[:n_to_copy]
            previous_remaining_content = remaining_content
    return content


def process_content(content):
    """Parse and execute all commands from LLM output.

    Returns
    -------
    tuple[str, list]
        ``(text_result, image_data_tuples)``
    """
    commands = []
    command, arguments, backtick_content, remaining_content = process_slice(content)
    if command:
        commands.append(CommandInfo(command, arguments, backtick_content))
    while command:
        command, arguments, backtick_content, remaining_content = process_slice(remaining_content)
        if command:
            commands.append(CommandInfo(command, arguments, backtick_content))

    response = ""
    image_data_tuple_array = []
    if len(commands) == 0:
        return "End.", []

    for command in commands:
        if command.command == "view_image":
            command_response, image_array = _view_images(command.arguments)
            for image_mediatype_tuple in image_array:
                image_data_tuple_array.append(image_mediatype_tuple)
        elif command.command == "create_image":
            args = split_preserving_quotes(command.arguments)
            command_response, image_array = _create_image(*args)
        elif command.command == "view_page":
            result = _execute_command(command.command, command.arguments, command.backtick_content)
            if isinstance(result, tuple):
                command_response, screenshot_path = result
                if screenshot_path and os.path.exists(screenshot_path):
                    image_base64, media_type = _load_and_resize_image(screenshot_path)
                    if media_type:
                        image_data_tuple_array.append((image_base64, media_type))
                command_response = (command_response or "ok") + "\n"
            else:
                command_response = (result or "ok") + "\n"
        else:
            command_response = (_execute_command(command.command, command.arguments, command.backtick_content) or "ok") + "\n"
            if command.command == "run_console_command":
                limit = 10000
                if len(command_response) >= limit:
                    concise_command_response = concise_representation(command_response, limit)
                    command_response = f"Truncating command response to {limit} characters...\n" + concise_command_response
        response += command_response
    return response, image_data_tuple_array


# ── Image handling ──────────────────────────────────────────────────

def _load_and_resize_image(image_path):
    """Load an image, resize for LLM vision, return (base64, media_type)."""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file at {image_path} does not exist.")
        image = Image.open(image_path)
    except FileNotFoundError as e:
        return str(e), None
    except UnidentifiedImageError:
        return "The file is not a valid image.", None
    except Exception as e:
        return f"An error occurred: {e}", None

    original_width, original_height = image.size
    max_pixels = 1_150_000
    max_dimension = 1568

    scaling_factor = min(
        1,
        max_dimension / max(original_width, original_height),
        (max_pixels / (original_width * original_height)) ** 0.5
    )

    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    buffer = io.BytesIO()
    resized_image.save(buffer, format=image.format)
    buffer.seek(0)

    resized_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    media_type = Image.MIME.get(image.format, "")

    if media_type not in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
        return f"{media_type} is an unsupported media type.", None

    return resized_image_base64, media_type


def _view_images(arguments):
    """Load and encode one or more images for the LLM."""
    image_data_tuple_array = []
    command_response = "Image(s) loaded successfully"
    args = split_preserving_quotes(arguments)
    try:
        for argument in args:
            image_base64, media_type = _load_and_resize_image(argument)
            image_data_tuple_array.append((image_base64, media_type))
    except Exception as e:
        command_response = f"An error occured loading image(s): {e}"
        image_data_tuple_array = []
    return command_response, image_data_tuple_array


def _create_image(prompt, output_file, width=1024, height=1024):
    """Generate an image using the getimg.ai API."""
    try:
        import requests
    except ImportError:
        return "Image generation error: 'requests' package is not installed.", []

    auth_key = os.getenv("GETIMG_API_KEY")
    if not auth_key:
        return "Image generation error: getimg API key not found in environment variables.", []

    url = "https://api.getimg.ai/v1/flux-schnell/text-to-image"
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {auth_key}",
        "content-type": "application/json",
    }
    data = {
        "prompt": prompt,
        "output_format": "png",
        "response_format": "b64",
        "width": width,
        "height": height,
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()

        if "image" in response_data:
            image_b64 = response_data["image"]
            image_data = base64.b64decode(image_b64)
            with open(output_file, "wb") as image_file:
                image_file.write(image_data)
            return "Image generation successful.", [(image_b64, "image/png")]
        else:
            return "Image generation error: Image not found in the response.", []

    except requests.exceptions.RequestException as e:
        return f"Image generation request failed: {e}", []


# ── Command dispatch ────────────────────────────────────────────────

def _execute_command(command, arguments, backticks):
    """Dispatch a parsed command to the appropriate tool function."""
    if command is None:
        return "Error: Command name must be specified correctly."
    if command != "run_console_command":
        args = split_preserving_quotes(arguments)
    else:
        args = arguments
    if not isinstance(args, list):
        args = [args]
    if backticks is not None:
        args.append(backticks)
    try:
        function = getattr(functions, command.lower())
    except AttributeError:
        return "Error: Command not found"
    try:
        result = function(*args) if args else function()
        return result if result is not None else "ok"
    except Exception as e:
        return f"Error executing command: {e}\n {command}, {arguments}, {backticks}"


def terminate_process():
    """Terminate any running subprocess."""
    functions.terminate_process()

