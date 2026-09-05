"""
Command parser — extracts commands from LLM output, dispatches to tool functions.

This module owns all parsing and dispatch logic:
- ``process_content()`` — parse and execute commands (the end_session
  sentinel is resolved here, never dispatched)
- ``strip_end_session()`` — remove the explicit end_session command(s)
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
    command_pattern = r"^Command: (\S+)[ \t]*(.*)$"
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


# Commands that can be stacked (queued together like read_file).
# request_approval is a no-op side-effect-wise, so it is safe to queue
# alongside reads — without this, filter_content would clip the
# approval request off the end of a read-heavy turn.
STACKABLE_READ_COMMANDS = {'read_file', 'deep_read', 'read_page', 'read_page_html', 'page_links', 'view_page', 'web_search', 'request_approval'}


#: Explicit end-of-session command.  It never dispatches to a tool
#: function: on its own it ends the session, and when it is queued
#: alongside other commands it is REJECTED — the other commands run
#: first (so the model reflects on their output) and the model must
#: then re-issue end_session by itself.
END_SESSION_COMMAND = "end_session"

#: Matches an end_session command line anywhere in a response,
#: forgiving about leading whitespace and the case of the name.
END_SESSION_RE = re.compile(
    r"^[ \t]*Command:\s*end_session(?:[ \t].*)?$",
    re.MULTILINE | re.IGNORECASE,
)

_FENCE_5 = "`" * 5

#: Matches an end_session command line plus any directly-attached
#: 5-backtick payload block (the completion note); used by
#: :func:`strip_end_session` to remove the whole span.
_END_SESSION_SPAN_RE = re.compile(
    r"^[ \t]*Command:\s*end_session(?:[ \t].*)?[ \t]*\r?\n?"
    r"(?:" + _FENCE_5 + r"(?:[\w#\+\-]+)?\s*[\s\S]*?" + _FENCE_5 + r")?"
    r"[ \t]*\r?\n?",
    re.MULTILINE | re.IGNORECASE,
)

#: Prepended to the tool-result message when end_session was issued
#: alongside other commands (a rejected end attempt).
END_SESSION_REJECTED_NOTICE = (
    "end_session REJECTED: you issued it in the same response as other "
    "commands, but end_session is only accepted when it is the sole "
    "command. The other commands were executed — reflect on their "
    "output below before finishing. When you are done reviewing it, "
    "respond with 'Command: end_session' (with your completion note in "
    "the backtick block) and nothing else to end the session."
)


def strip_end_session(content):
    """Remove every end_session command line from *content*.

    A directly-attached 5-backtick payload (the completion note) is
    removed together with its command line.

    Returns:
        tuple: ``(cleaned_content, found)`` where *found* is True if at
        least one end_session command line was present.
    """
    cleaned, count = re.subn(_END_SESSION_SPAN_RE, "", content)
    return cleaned, count > 0


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


def process_content(content, blocked_commands=None):
    """Parse and execute all commands from LLM output.

    An explicit ``end_session`` command is never dispatched: on its own
    it yields the ``"End."`` sentinel (the session ends); alongside
    other commands it is rejected (a notice is prepended to their
    output) and the remaining commands run as usual.

    Args:
        content: The raw LLM output text.
        blocked_commands: Optional iterable of command names to reject.
            Blocked commands — including one wrapped in ``deep_read`` —
            return a BLOCKED notice instead of executing.  Used by
            planning mode to forbid state-changing tools.

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

    # Explicit end_session handling (self-consistency for direct
    # callers — Agent._iterate strips it before calling): alone it is
    # an intentional stop; alongside other commands it is rejected and
    # the rest run anyway.
    reject_prefix = ""
    if any(c.command.lower() == END_SESSION_COMMAND for c in commands):
        commands = [
            c for c in commands
            if c.command.lower() != END_SESSION_COMMAND
        ]
        if not commands:
            return "End.", []
        reject_prefix = END_SESSION_REJECTED_NOTICE + "\n"

    if len(commands) == 0:
        return "End.", []

    for command in commands:
        # Planning-mode guard: reject blocked commands, looking through
        # the deep_read wrapper (otherwise deep_read write_file … would
        # bypass the guard).
        blocked_name = command.command
        if command.command == "deep_read":
            inner_args = split_preserving_quotes(command.arguments)
            if inner_args:
                blocked_name = inner_args[0]
        if blocked_commands and blocked_name.lower() in blocked_commands:
            command_response = (
                f"[BLOCKED: planning mode] {blocked_name.lower()} changes "
                "state and is not allowed in planning mode. Produce your "
                "plan as visible text and, when it is ready, ask to "
                "continue with 'Command: request_approval'.\n"
            )
            response += command_response
            continue
        if command.command == "view_image":
            command_response, image_array = _view_images(command.arguments)
            for image_mediatype_tuple in image_array:
                image_data_tuple_array.append(image_mediatype_tuple)
        elif command.command == "create_image":
            args = split_preserving_quotes(command.arguments)
            command_response, image_array = _create_image(*args)
            for image_mediatype_tuple in image_array:
                image_data_tuple_array.append(image_mediatype_tuple)
        elif command.command == "view_page":
            result = _execute_command(command.command, command.arguments, command.backtick_content)
            if isinstance(result, tuple):
                command_response, screenshot_path = result
                if screenshot_path and os.path.exists(screenshot_path):
                    image_base64, media_type = _load_and_resize_image(screenshot_path)
                    if media_type:
                        image_data_tuple_array.append((image_base64, media_type))
                    else:
                        # image_base64 contains the error message when media_type is None
                        command_response = (command_response or "") + f"\n[Screenshot load failed: {image_base64}]"
                command_response = (command_response or "ok") + "\n"
            else:
                command_response = (result or "ok") + "\n"
        elif command.command == "mcp_call":
            result = _execute_command(command.command, command.arguments, command.backtick_content)
            if isinstance(result, tuple):
                command_response, images = result
                for image_b64, media_type in (images or []):
                    if media_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
                        command_response = (command_response or "") + f"\n[Unsupported image type: {media_type}]"
                        continue
                    try:
                        raw = base64.b64decode(image_b64)
                    except Exception as e:
                        command_response = (command_response or "") + f"\n[Image decode failed: {e}]"
                        continue
                    resized_b64, ok_type = _resize_image_bytes(raw)
                    if ok_type:
                        image_data_tuple_array.append((resized_b64, ok_type))
                    else:
                        command_response = (command_response or "") + f"\n[Image resize failed: {resized_b64}]"
                command_response = (command_response or "ok") + "\n"
            else:
                command_response = (result or "ok") + "\n"
        elif command.command == "deep_read":
            # deep_read wraps another command, bypassing truncation
            inner_result = _execute_command(
                command.arguments, None, command.backtick_content, truncate=False
            )
            if isinstance(inner_result, tuple):
                # e.g. view_page returns (text, screenshot_path) — keep
                # only the text part for the deep_read response.
                inner_result = inner_result[0] if inner_result else None
            if not isinstance(inner_result, str):
                inner_result = str(inner_result) if inner_result is not None else "ok"
            command_response = (inner_result or "ok") + "\n"
        else:
            result = _execute_command(command.command, command.arguments, command.backtick_content)
            # Defensive: a tool without a dedicated branch that returns a
            # tuple (e.g. (text, images)) would otherwise crash on
            # tuple + "\n".  Use the text part only.
            if isinstance(result, tuple):
                result = result[0] if result and isinstance(result[0], str) else None
            command_response = (result or "ok") + "\n"
            if command.command == "run_console_command":
                limit = 10000
                if len(command_response) >= limit:
                    concise_command_response = concise_representation(command_response, limit)
                    command_response = f"Truncating command response to {limit} characters...\n" + concise_command_response
        response += command_response
    return reject_prefix + response, image_data_tuple_array


# ── Image handling ──────────────────────────────────────────────────

def _resize_image_bytes(data):
    """Resize raw image bytes for LLM vision, return (base64, media_type).

    Returns ``(error_message, None)`` when the data is not a valid image
    or its format is unsupported.
    """
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        return "The image data is not a valid image.", None
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


def _load_and_resize_image(image_path):
    """Load an image file, resize for LLM vision, return (base64, media_type)."""
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The file at {image_path} does not exist.")
        with open(image_path, "rb") as f:
            data = f.read()
    except FileNotFoundError as e:
        return str(e), None
    except Exception as e:
        return f"An error occurred: {e}", None

    return _resize_image_bytes(data)


def _view_images(arguments):
    """Load and encode one or more images for the LLM."""
    image_data_tuple_array = []
    errors = []
    args = split_preserving_quotes(arguments)
    try:
        for argument in args:
            image_base64, media_type = _load_and_resize_image(argument)
            if media_type:
                image_data_tuple_array.append((image_base64, media_type))
            else:
                # image_base64 contains the error message when media_type is None
                errors.append(f"{argument}: {image_base64}")
    except Exception as e:
        errors.append(f"An error occurred loading image(s): {e}")
        image_data_tuple_array = []

    if errors:
        command_response = "Image loading error(s):\n" + "\n".join(errors)
        if image_data_tuple_array:
            command_response += "\nOther image(s) loaded successfully."
    else:
        command_response = "Image(s) loaded successfully"
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


# ── Output safety truncation ───────────────────────────────────────

_TRUNCATE_THRESHOLD = 60_000
_TRUNCATE_KEEP = 30_000


def truncate_output(text):
    """Truncate tool output that exceeds the safety threshold.

    If *text* is longer than 60,000 characters, return the first 30,000
    and last 30,000 characters with a notification in the middle and at
    the end indicating that content was clipped.

    Parameters
    ----------
    text : str
        The raw tool output.

    Returns
    -------
    str
        The original text if within limits, or a truncated version.
    """
    if not isinstance(text, str) or len(text) <= _TRUNCATE_THRESHOLD:
        return text

    total = len(text)
    clipped = total - (_TRUNCATE_KEEP * 2)

    head = text[:_TRUNCATE_KEEP]
    tail = text[-_TRUNCATE_KEEP:]

    middle_notice = (
        f"\n\n... [OUTPUT TRUNCATED — {clipped:,} characters clipped from middle "
        f"({total:,} total characters)] ...\n\n"
    )
    end_notice = (
        f"\n\n[END OF TRUNCATED OUTPUT — Showed first {_TRUNCATE_KEEP:,} and "
        f"last {_TRUNCATE_KEEP:,} of {total:,} total characters]"
    )

    return head + middle_notice + tail + end_notice


# ── Command dispatch ────────────────────────────────────────────────

def _execute_command(command, arguments, backticks, truncate=True):
    """Dispatch a parsed command to the appropriate tool function.

    When *command* is a string like ``"read_file /path/to/file"`` (as used
    by ``deep_read``), the first token is taken as the command name and the
    rest as arguments.

    When *truncate* is ``False``, the output is returned without the
    60 000-character safety truncation.
    """
    if command is None:
        return "Error: Command name must be specified correctly."

    # deep_read passes the inner command as a single string argument
    if isinstance(command, str):
        parts = split_preserving_quotes(command)
        if not parts:
            return "Error: deep_read requires an inner command (e.g. deep_read read_file path)."
        cmd_name = parts[0].lower()
        remaining_args = parts[1:] if len(parts) > 1 else []
    else:
        cmd_name = command.lower()
        remaining_args = []

    if cmd_name != "run_console_command":
        args = remaining_args + (split_preserving_quotes(arguments) if isinstance(arguments, str) else [])
    else:
        # Preserve any args split out of a deep_read inner-command string
        # (e.g. deep_read run_console_command "ls -la") — they precede
        # the explicitly-passed arguments.
        args = remaining_args + ([arguments] if isinstance(arguments, str) and arguments else [])

    if not isinstance(args, list):
        args = [args]
    if backticks is not None:
        args.append(backticks)
    try:
        function = getattr(functions, cmd_name)
    except AttributeError:
        return f"Error: Command not found: {cmd_name}"
    try:
        result = function(*args) if args else function()
        if result is None:
            return "ok"
        # Handle tuple results (e.g. view_page returns (text, path))
        if isinstance(result, tuple):
            first = truncate_output(result[0]) if truncate and isinstance(result[0], str) else result[0]
            return (first,) + result[1:]
        if truncate:
            return truncate_output(result)
        return result
    except Exception as e:
        return f"Error executing command: {e}\n {cmd_name}, {arguments}, {backticks}"


def terminate_process():
    """Terminate any running subprocess."""
    functions.terminate_process()

