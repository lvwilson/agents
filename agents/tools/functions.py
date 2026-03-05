"""
Tool functions — filesystem, shell, code manipulation, web browser, and misc tools.

This module provides the concrete implementations of every tool command
that the LLM can invoke.  The command dispatcher in ``parser.py`` resolves
command names to functions in this module via ``getattr(functions, name)``.
"""

import atexit as _atexit
import difflib
import io
import os
import pty
import pwd
import subprocess
import sys
import threading

from . import code_scissors
from . import codemanipulator
from . import findreplace


# ── TTY management ──────────────────────────────────────────────────
#
# All feedback output goes to /dev/tty so that stdout is reserved
# exclusively for the ``stdout`` tool.  The handle is opened lazily
# so that importing this module in a headless environment doesn't fail.

_tty = None


def _cleanup_tty():
    """Close the TTY file handle if we opened one."""
    global _tty
    if _tty is not None and _tty is not sys.stderr:
        try:
            _tty.close()
        except Exception:
            pass
        _tty = None


_atexit.register(_cleanup_tty)


def _get_tty():
    """Return the TTY file object, opening it lazily.  Falls back to stderr."""
    global _tty
    if _tty is None:
        try:
            _tty = open("/dev/tty", "w")
        except OSError:
            _tty = sys.stderr
    return _tty


# ── Subprocess management ──────────────────────────────────────────
#
# A module-level handle to the currently running subprocess so that
# ``terminate_process()`` can kill it from a signal handler.

_process = None


def terminate_process():
    """Terminate any running subprocess."""
    global _process
    if _process:
        print("Terminating process...", file=_get_tty())
        _process.terminate()


# ── Shell helpers ───────────────────────────────────────────────────

def get_default_shell():
    """Return the default shell for the current user."""
    if sys.platform == "win32":
        return os.getenv("COMSPEC", "cmd.exe")
    return pwd.getpwuid(os.getuid()).pw_shell


def run_console_command(command: str) -> str:
    """Execute a console command via the user's shell and return its output.

    :param command: The console command to execute.
    :return: Combined stdout/stderr output from the command.
    """
    global _process

    def _strip_quotes(s: str) -> str:
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    def _read_output(fd, output_list):
        try:
            while True:
                data = os.read(fd, io.DEFAULT_BUFFER_SIZE).decode()
                if not data:
                    break
                _get_tty().write(data)
                _get_tty().flush()
                output_list.append(data)
        except OSError:
            pass

    try:
        output = []
        try:
            master_fd, slave_fd = pty.openpty()
            stripped_command = _strip_quotes(command)
            stripped_command = stripped_command.replace('\\"', '"')

            shell_path = get_default_shell()
            popen_kwargs = dict(
                shell=True, stdin=slave_fd, stdout=slave_fd,
                stderr=slave_fd, text=True, close_fds=True,
            )
            if shell_path:
                popen_kwargs["executable"] = shell_path

            _process = subprocess.Popen(stripped_command, **popen_kwargs)
            os.close(slave_fd)

            output_thread = threading.Thread(target=_read_output, args=(master_fd, output))
            output_thread.start()
            _process.wait()
            output_thread.join()
            os.close(master_fd)
        except KeyboardInterrupt:
            pass

        _process = None
        combined_output = "".join(output)
        return combined_output if combined_output else "ok"
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e.stderr}"
    except Exception as e:
        return f"An error occurred: {str(e)}"


# ── File I/O tools ─────────────────────────────────────────────────

def read_file(file_path):
    """Read and return the entire contents of a file.

    :param file_path: Path to the file.
    :return: File contents as a string.
    """
    with open(file_path, "r") as f:
        return f.read()


def write_file(file_path, code):
    """Write *code* to *file_path*, creating directories as needed.

    Returns a status message including a unified diff of the changes.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    try:
        with open(file_path, "r") as f:
            original_content = f.read()
    except FileNotFoundError:
        original_content = ""

    original_length = len(original_content)
    new_length = len(code)

    diff = "\n".join(difflib.unified_diff(
        original_content.splitlines(), code.splitlines(),
        lineterm="", fromfile="original", tofile="new",
    ))

    try:
        with open(file_path, "w") as f:
            f.write(code)
        return (f"{file_path} successfully written. Original length: {original_length}, "
                f"New length: {new_length}.\n\nDiff:\n{diff}")
    except Exception as e:
        return f"{file_path} write error: {e}"


def append_to_file(file_path, log_message):
    """Append *log_message* to *file_path*.

    Returns a status message including a unified diff of the changes.
    """
    try:
        with open(file_path, "r") as f:
            original_content = f.read()
    except FileNotFoundError:
        original_content = ""

    original_length = len(original_content)

    try:
        with open(file_path, "a") as f:
            f.write(log_message + "\n")

        with open(file_path, "r") as f:
            updated_content = f.read()

        new_length = len(updated_content)
        diff = "\n".join(difflib.unified_diff(
            original_content.splitlines(), updated_content.splitlines(),
            lineterm="", fromfile="original", tofile="updated",
        ))
        return (f"{file_path} successfully appended. Original length: {original_length}, "
                f"New length: {new_length}.\n\nDiff:\n{diff}")
    except Exception as e:
        return f"{file_path} append error: {e}"


# ── Find-and-replace tool ──────────────────────────────────────────

def find_and_replace(file_path, command):
    """Perform a find-and-replace on *file_path* using the SEARCH/REPLACE block syntax.

    Returns a status message with a unified diff of the changes.
    """
    try:
        with open(file_path, "r") as f:
            original_content = f.read()
    except Exception as e:
        return f"{file_path} read error: {e}"

    modified_content = findreplace.find_replace(original_content, command)

    diff = "\n".join(difflib.unified_diff(
        original_content.splitlines(), modified_content.splitlines(),
        lineterm="", fromfile="original", tofile="modified",
    ))

    try:
        with open(file_path, "w") as f:
            f.write(modified_content)
        return f"{file_path} successfully written.\n\nDiff:\n{diff}"
    except Exception as e:
        return f"{file_path} write error: {e}"


# ── Code-scissors (line-based insert/replace) ──────────────────────

def _read_or_error(file_path):
    """Read a file, returning ``(content, None)`` or ``(None, error_msg)``."""
    try:
        with open(file_path, "r") as f:
            return f.read(), None
    except Exception as e:
        return None, f"{file_path} read error: {e}"


def _write_or_error(file_path, content):
    """Write *content* to *file_path*, returning a status message."""
    try:
        with open(file_path, "w") as f:
            f.write(content)
        return f"{file_path} successfully written."
    except Exception as e:
        return f"{file_path} write error: {e}"


def insert_text_after_matching_line(file_path, line, new_code):
    source, err = _read_or_error(file_path)
    if err:
        return err
    return _write_or_error(file_path, code_scissors.insert_after(source, line, new_code))


def insert_text_before_matching_line(file_path, line, new_code):
    source, err = _read_or_error(file_path)
    if err:
        return err
    return _write_or_error(file_path, code_scissors.insert_before(source, line, new_code))


def replace_text_before_matching_line(file_path, line, new_code):
    source, err = _read_or_error(file_path)
    if err:
        return err
    return _write_or_error(file_path, code_scissors.replace_before(source, line, new_code))


def replace_text_after_matching_line(file_path, line, new_code):
    source, err = _read_or_error(file_path)
    if err:
        return err
    return _write_or_error(file_path, code_scissors.replace_after(source, line, new_code))


def replace_text_between_matching_lines(file_path, line1, line2, new_code):
    source, err = _read_or_error(file_path)
    if err:
        return err
    return _write_or_error(file_path, code_scissors.replace_between(source, line1, line2, new_code))


# ── AST-based code manipulation (codemanipulator) ──────────────────

def read_code_signatures_and_docstrings(file_path):
    """Return function/class signatures and docstrings from *file_path*."""
    try:
        with open(file_path, "r") as f:
            return codemanipulator.get_signatures_and_docstrings(f.read())
    except Exception as e:
        return f"{file_path} read error: {e}"


def replace_docstring_at_address(file_path, address, new_docstring):
    """Replace the docstring at *address* inside *file_path*."""
    source, err = _read_or_error(file_path)
    if err:
        return err
    source = codemanipulator.change_docstring(source, address, new_docstring)
    try:
        return codemanipulator.write_code(file_path, source)
    except Exception as e:
        return f"{file_path} write error: {e}"


def read_code_at_address(file_path, address):
    """Return the source code at *address* inside *file_path*."""
    try:
        with open(file_path, "r") as f:
            return codemanipulator.read_code_at_address(f.read(), address)
    except Exception as e:
        return f"{file_path} read error: {e}"


def replace_code_at_address(file_path, address, new_code):
    """Replace the code at *address* inside *file_path* with *new_code*."""
    source, err = _read_or_error(file_path)
    if err:
        return err
    try:
        source = codemanipulator.replace_code(source, address, new_code)
    except Exception as e:
        return f"{file_path} replace error: {e}"
    try:
        return codemanipulator.write_code(file_path, source)
    except Exception as e:
        return f"{file_path} write error: {e}"


def add_code_after_address(file_path, address, new_code):
    """Add *new_code* after *address* inside *file_path*."""
    source, err = _read_or_error(file_path)
    if err:
        return err
    source = codemanipulator.insert_code_after(source, address, new_code)
    return _write_or_error(file_path, source)


def add_code_before_address(file_path, address, new_code):
    """Add *new_code* before *address* inside *file_path*."""
    source, err = _read_or_error(file_path)
    if err:
        return err
    source = codemanipulator.insert_code_before(source, address, new_code)
    return _write_or_error(file_path, source)


def remove_code_at_address(file_path, address):
    """Remove the code at *address* inside *file_path*."""
    source, err = _read_or_error(file_path)
    if err:
        return err
    source = codemanipulator.remove_code(source, address)
    return _write_or_error(file_path, source)


# ── stdout tool ─────────────────────────────────────────────────────

def stdout(*args):
    """Write content to stdout (the only way to produce stdout output).

    All other feedback is directed to the terminal (tty).
    """
    content = args[-1] if args else ""
    sys.stdout.write(content)
    sys.stdout.flush()
    return "Content written to stdout."


# ── Summarize tool ──────────────────────────────────────────────────

def summarize(*args):
    """Summarize a file or folder using an LLM.

    Usage::

        summarize path/to/file.py
        summarize path/to/folder "*.py"
        summarize path/to/folder "*.py" --recursive

    The optional backtick block provides additional instructions.
    """
    from . import summarize as _summarize_mod

    positional = list(args)
    instruction = ""

    recursive = False
    cleaned = []
    for a in positional:
        if a in ("--recursive", "-r"):
            recursive = True
        else:
            cleaned.append(a)
    positional = cleaned

    if not positional:
        return "Error: summarize requires at least a file or folder path."

    path = positional[0]

    if len(positional) >= 2:
        last = positional[-1]
        if "\n" in last or (len(last) > 60 and " " in last):
            instruction = last
            positional = positional[:-1]

    filter_pattern = positional[1] if len(positional) >= 2 else "*"

    try:
        if os.path.isfile(path):
            return _summarize_mod.summarize_file(path, instruction=instruction)
        elif os.path.isdir(path):
            return _summarize_mod.summarize_folder(
                path, filter_pattern=filter_pattern,
                recursive=recursive, instruction=instruction,
            )
        else:
            return f"Error: {path} is not a file or directory."
    except Exception as e:
        return f"Error during summarization: {e}"


# ── Web browser tools (Playwright) ─────────────────────────────────

from .web_browser import get_browser, close_browser


def web_navigate(url):
    """Navigate the browser to *url*."""
    return get_browser().navigate(url)


def web_back():
    """Go back one page in browser history."""
    return get_browser().back()


def web_forward():
    """Go forward one page in browser history."""
    return get_browser().forward()


def web_read(*args):
    """Read the visible text of the current page or a CSS-selected element."""
    selector = args[0] if args else None
    return get_browser().read_text(selector)


def web_read_html(*args):
    """Read the HTML of the current page or a CSS-selected element."""
    selector = args[0] if args else None
    return get_browser().read_html(selector)


def web_links():
    """List all links on the current page."""
    return get_browser().get_links()


def web_click(selector):
    """Click the element matching *selector*."""
    return get_browser().click(selector)


def web_type(selector, text):
    """Type *text* into the input matching *selector*."""
    return get_browser().type_text(selector, text)


def web_press_key(key):
    """Press a keyboard key (e.g. ``Enter``, ``Tab``, ``Escape``)."""
    return get_browser().press_key(key)


def web_select(selector, value):
    """Select *value* from the ``<select>`` matching *selector*."""
    return get_browser().select_option(selector, value)


def web_screenshot(*args):
    """Take a screenshot.  Args: ``file_path [css_selector]``."""
    if not args:
        return "Error: web_screenshot requires at least a file path."
    file_path = args[0]
    selector = args[1] if len(args) > 1 else None
    return get_browser().screenshot(file_path, selector=selector)


def web_execute_js(*args):
    """Execute JavaScript on the current page (code via backtick block)."""
    script = args[-1] if args else None
    if not script:
        return "Error: web_execute_js requires JavaScript code in a backtick block."
    return get_browser().execute_js(script)


def web_wait(selector, *args):
    """Wait for *selector* to appear.  Optional timeout in ms (default 10 000)."""
    timeout = int(args[0]) if args else 10000
    return get_browser().wait_for_selector(selector, timeout=timeout)


def web_page_info():
    """Return the current URL, title, and viewport size."""
    return get_browser().get_page_info()


def web_close():
    """Close the browser and free resources."""
    close_browser()
    return "Browser closed."
