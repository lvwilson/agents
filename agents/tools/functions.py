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
from .docs import docs


# ── Agent pool (sub-agents) ────────────────────────────────────────
#
# The pool is injected at runtime by Agent.__init__() via register_pool().

_agent_pool = None


def register_pool(pool):
    """Register the agent pool for sub-agent tools."""
    global _agent_pool
    _agent_pool = pool


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


def run_console_command(command: str, backtick_content: str = None) -> str:
    """Execute a console command via the user's shell and return its output.

    Supports multi-line commands via the optional backtick_content parameter.
    When backtick_content is provided, it is used as the command body,
    allowing multi-line shell scripts to be passed cleanly.

    :param command: The console command to execute (single-line).
    :param backtick_content: Optional multi-line command from backtick block.
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
                raw = os.read(fd, io.DEFAULT_BUFFER_SIZE)
                if not raw:
                    break
                try:
                    data = raw.decode("utf-8")
                except UnicodeDecodeError:
                    data = raw.decode("latin-1")
                _get_tty().write(data)
                _get_tty().flush()
                output_list.append(data)
        except OSError:
            pass

    try:
        output = []
        try:
            master_fd, slave_fd = pty.openpty()
            # Use backtick content as the command if provided (multi-line
            # support).  Backtick content is passed to the shell verbatim
            # (no quote stripping/unescaping) so scripts are not mangled.
            if backtick_content is not None:
                stripped_command = backtick_content.strip()
            elif "\n" in command:
                # Multi-line script arrived via the positional argument
                # (e.g. backtick block with no inline command) — verbatim.
                stripped_command = command.strip()
            else:
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

def _read_file_safe(file_path):
    """Read a file as UTF-8, falling back to latin-1 on decode errors.

    Returns ``(content, None)`` on success or ``(None, error_msg)`` on failure.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read(), None
    except UnicodeDecodeError:
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read(), None
        except Exception as e:
            return None, f"{file_path} read error (latin-1 fallback): {e}"
    except Exception as e:
        return None, f"{file_path} read error: {e}"


def read_file(file_path):
    """Read and return the entire contents of a file.

    :param file_path: Path to the file.
    :return: File contents as a string.
    """
    content, err = _read_file_safe(file_path)
    if err:
        return err
    return content


def write_file(file_path, code):
    """Write *code* to *file_path*, creating directories as needed.

    Returns a concise status message with line count, byte size, and
    (for existing files) lines added/removed from the diff.
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    existing = os.path.isfile(file_path)
    new_length = len(code)
    lines = len(code.splitlines())

    if existing:
        original_content, err = _read_file_safe(file_path)
        if err:
            return err
    else:
        original_content = ""

    try:
        with open(file_path, "w") as f:
            f.write(code)
    except Exception as e:
        return f"{file_path} write error: {e}"

    if existing and original_content:
        added = removed = 0
        for line in difflib.unified_diff(
            original_content.splitlines(), code.splitlines(),
            lineterm="", fromfile="original", tofile="new",
        ):
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return (f"{file_path} written: {lines} lines, {new_length} bytes "
                f"(+{added}/-{removed})")

    return f"{file_path} written: {lines} lines, {new_length} bytes"


def append_to_file(file_path, log_message):
    """Append *log_message* to *file_path*.

    Returns a status message including a unified diff of the changes.
    """
    original_content, err = _read_file_safe(file_path)
    if err:
        return err

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
    original_content, err = _read_file_safe(file_path)
    if err:
        return err

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
    return _read_file_safe(file_path)


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
    source, err = _read_file_safe(file_path)
    if err:
        return err
    try:
        return codemanipulator.get_signatures_and_docstrings(source)
    except Exception as e:
        return f"{file_path} parse error: {e}"


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
    source, err = _read_file_safe(file_path)
    if err:
        return err
    try:
        return codemanipulator.read_code_at_address(source, address)
    except Exception as e:
        return f"{file_path} parse error: {e}"


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


# ── Planning-mode tool ───────────────────────────────────────────────

def request_approval(*args):
    """Ask the user to approve the current plan (planning mode).

    In planning mode (started with -p / --plan) the harness intercepts
    this command and pauses at the terminal: the user approves (the
    agent continues into execution) or sends feedback (the agent
    revises the plan).  The decision is delivered as tool output.
    Outside planning mode the command is a harmless no-op.
    """
    return (
        "request_approval acknowledged. Planning mode is not active, so "
        "there is no plan to approve — continue working on the task."
    )


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


# ── Web search (DuckDuckGo) ────────────────────────────────────────

def web_search(query, max_results=None):
    """Search the web using DuckDuckGo and return formatted results.

    :param query: Search query string.
    :param max_results: Maximum number of results (default 5).
    :return: Formatted search results with title, URL, and snippet.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return "Error: ddgs package not installed. Run: pip install ddgs"

    n = int(max_results) if max_results else 5
    n = max(1, min(n, 20))

    try:
        results = list(DDGS().text(query, max_results=n))
    except Exception as e:
        return f"Search error: {e}"

    if not results:
        return f"No results found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        href = r.get("href", "")
        body = r.get("body", "")
        lines.append(f"  {i}. {title}")
        lines.append(f"     {href}")
        if body:
            lines.append(f"     {body}")
        lines.append("")
    return "\n".join(lines)


# ── Web browser tools (Playwright) ─────────────────────────────────

from .web_browser import get_browser, close_browser


# ── New core tools (stateless reading) ──────────────────────────────

def read_page(url, *args):
    """Navigate to *url* and return visible text. Optional CSS selector."""
    selector = args[0] if args else None
    return get_browser().read_page(url, selector)


def read_page_html(url, *args):
    """Navigate to *url* and return HTML. Optional CSS selector."""
    selector = args[0] if args else None
    return get_browser().read_page_html(url, selector)


def page_links(url):
    """Navigate to *url* and return all links."""
    return get_browser().page_links(url)


def view_page(url, *args):
    """Navigate to *url*, screenshot + text + interactive elements.

    Returns ``(rich_text_response, file_path)`` tuple for parser.
    """
    file_path = args[0] if args else None
    return get_browser().view_page(url, file_path)


# ── New interactive tools (stateful browsing) ───────────────────────

def browse_open(url):
    """Open URL in browser session, auto-reads page content."""
    return get_browser().browse_open(url)


def browse_read(*args):
    """Read current page text, optionally scoped by CSS selector."""
    selector = args[0] if args else None
    return get_browser().browse_read(selector)


def browse_click(selector):
    """Click element (auto-waits), returns resulting page content."""
    return get_browser().browse_click(selector)


def browse_type(selector, text):
    """Type text into element. Supports [Enter], [Tab], [Escape] inline."""
    return get_browser().browse_type(selector, text)


def browse_js(*args):
    """Execute JavaScript on the current page (code via backtick block)."""
    script = args[-1] if args else None
    if not script:
        return "Error: browse_js requires JavaScript code in a backtick block."
    return get_browser().execute_js(script)


# ── Sub-agent tools ────────────────────────────────────────────────

def create_agent(*args):
    """Create a sub-agent with a given name and role.

    Usage: create_agent name ["description"]
    Backtick block: The role/persona instructions for the sub-agent.
    """
    if _agent_pool is None:
        return "Error: agent pool not initialized."

    if not args:
        return "Error: create_agent requires a name argument."

    name = args[0]

    # The backtick block (role prompt) is always the last arg when present.
    # We need at least 2 args (name + backtick block) for a valid call.
    if len(args) < 2:
        return "Error: create_agent requires a role description in a backtick block."

    role_prompt = args[-1]

    # Optional description between name and backtick block.
    description = args[1] if len(args) >= 3 else ""

    return _agent_pool.create(name, role_prompt, description)


def list_agents(*args):
    """List all registered sub-agents."""
    if _agent_pool is None:
        return "Error: agent pool not initialized."
    return _agent_pool.list()


def run_agent(*args):
    """Run a sub-agent on a task.

    Usage: run_agent name [budget] [timeout]
    Backtick block: The task description for the sub-agent.
    """
    if _agent_pool is None:
        return "Error: agent pool not initialized."

    if not args:
        return "Error: run_agent requires a name argument."

    name = args[0]
    task = args[-1] if len(args) >= 2 else None

    if task is None or task == name:
        return "Error: run_agent requires a task in a backtick block."

    # Parse optional numeric arguments between name and backtick block.
    # First numeric arg is budget, second is timeout.
    budget = 1.00
    timeout = 300
    numeric_args = []
    for arg in args[1:-1]:
        try:
            numeric_args.append(float(arg))
        except ValueError:
            pass

    if len(numeric_args) >= 1:
        budget = numeric_args[0]
    if len(numeric_args) >= 2:
        timeout = int(numeric_args[1])

    return _agent_pool.run(name, task, budget, timeout)


# ── MCP (Model Context Protocol) tools ─────────────────────────────

def mcp_list_tools(*args):
    """List available tools from configured MCP servers.

    Usage: mcp_list_tools [server_name]

    With no arguments, lists tools from all configured servers.
    With a server name, lists tools from that server only.

    Servers are configured in ~/.config/agents/mcp_servers.json
    """
    from .mcp_client import get_manager

    server_name = args[0] if args else None
    try:
        tools = get_manager().list_tools(server_name)
    except Exception as e:
        return f"Error: {e}"

    if not tools:
        return "No MCP tools found. Configure servers in ~/.config/agents/mcp_servers.json"

    lines = []
    current_server = None
    for t in tools:
        if t["server"] != current_server:
            current_server = t["server"]
            lines.append(f"\n[{current_server}]")
        desc = t["description"][:100] if t["description"] else "(no description)"
        lines.append(f"  {t['name']} -- {desc}")
        if t.get("schema", {}).get("properties"):
            props = t["schema"]["properties"]
            params = ", ".join(
                f"{k}: {v.get('type', '?')}" for k, v in props.items()
            )
            lines.append(f"    params: {params}")
    return "\n".join(lines)


def mcp_call(*args):
    """Call a tool on an MCP server.

    Usage: mcp_call server_name tool_name [JSON args in backtick block]

    The backtick block contains JSON arguments for the tool.
    If the tool takes no arguments, the backtick block can be omitted.

    If the tool returns images (e.g. cua-driver screenshots), they are
    displayed to you inline alongside the text response.
    """
    import json as _json
    from .mcp_client import get_manager

    if len(args) < 2:
        return "Error: mcp_call requires server_name and tool_name arguments."

    server_name = args[0]
    tool_name = args[1]

    # Parse JSON arguments from backtick block (last arg if present)
    arguments = {}
    if len(args) >= 3:
        json_str = args[-1]
        try:
            arguments = _json.loads(json_str)
        except _json.JSONDecodeError as e:
            return f"Error parsing JSON arguments: {e}\nReceived: {json_str[:200]}"

    try:
        return get_manager().call_tool(server_name, tool_name, arguments)
    except Exception as e:
        return f"Error calling {server_name}/{tool_name}: {e}"

# ── Memory / Notes tools ─────────────────────────────────────────────

def note(*args):
    from .. import memory as _mem
    if not args:
        return "Error: note requires a subcommand (append, replace, rewrite)."
    subcmd = args[0].lower()
    if subcmd == "append":
        text = args[-1] if len(args) >= 2 else None
        if not text:
            return "Error: note append requires text in a backtick block."
        updated = _mem.note_append(text)
        return f"Notes appended. Current notes ({len(updated)} chars):\n{updated}"
    elif subcmd == "replace":
        block = args[-1] if len(args) >= 2 else None
        if not block:
            return "Error: note replace requires a SEARCH/REPLACE block."
        current = _mem.get_notes()
        try:
            updated = findreplace.find_replace(current, block)
        except ValueError as e:
            return f"Error: {e}"
        _mem.note_rewrite(updated)
        return f"Notes updated. Current notes ({len(updated)} chars):\n{updated}"
    elif subcmd == "rewrite":
        new_content = args[-1] if len(args) >= 2 else None
        if not new_content:
            return "Error: note rewrite requires new notes in a backtick block."
        updated = _mem.note_rewrite(new_content)
        return f"Notes rewritten. Current notes ({len(updated)} chars):\n{updated}"
    else:
        return f"Error: unknown note subcommand '{subcmd}'. Use: append, replace, rewrite."
