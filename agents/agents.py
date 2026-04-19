#!/usr/bin/env python3
"""
Agent - An autonomous AI agent with pluggable LLM backends.
"""
# Standard library imports
import argparse
from dataclasses import dataclass
import logging
import os
import platform
import re
import signal
import sys
import time
import traceback
from typing import Optional

# Third-party imports
import yaml

# Tools (command parsing and execution)
from .tools import process_content, filter_content, terminate_process
from .tools import get_default_shell
from .tools import register_llm as _register_summarize_llm
from .tools import register_pool as _register_pool

# Local imports
from .backends import create_backend
from .session import (
    generate_session_id,
    validate_session_id,
    get_latest_session_for_dir,
    save_session,
    load_session,
)
from .llm_backend import InterruptedResponse
from .ui import (
    RichStreamHandler,
    print_banner,
    print_iteration_header,
    print_summary,
    print_completion_result,
    print_budget_warning,
    print_budget_exceeded,
    print_error,
    print_interrupted,
    print_interrupt_feedback,
    get_user_feedback,
    print_sigterm,
    print_clipped,
    safe_console_print,
)

# ── Global state ─────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.realpath(__file__))


# ── Message helpers ──────────────────────────────────────────────────

def _text_block(text):
    """Wrap *text* in the internal content-block format."""
    return [{"type": "text", "text": text}]


def _form_message(role, content):
    """Create a message dict in the internal format.

    Args:
        role: ``"user"`` or ``"assistant"``
        content: Plain text string

    Returns:
        dict with ``role`` and ``content`` keys.
    """
    return {"role": role, "content": _text_block(content)}


def _form_message_with_images(role, content, image_media_type_tuple_array):
    """Create a message dict that includes images.

    Args:
        role: ``"user"`` or ``"assistant"``
        content: Plain text string
        image_media_type_tuple_array: List of ``(image_base64, media_type)`` tuples

    Returns:
        dict with ``role`` and ``content`` keys.
    """
    images = [
        {
            "type": "image",
            "media_type": media_type,
            "data": image_base64,
        }
        for image_base64, media_type in image_media_type_tuple_array
    ]
    text_content = {"type": "text", "text": content}
    return {"role": role, "content": images + [text_content]}


@dataclass
class CompletionResult:
    """Represents the result of an agent's task execution."""
    text: str
    success: bool

def extract_completion(text, backticks=5) -> Optional[CompletionResult]:
    """Extract the completion section from the given text.

    Args:
        text (str): The text to extract the completion from.
        backticks (int): The number of backticks used to wrap the section (default: 5).

    Returns:
        CompletionResult: The completion result, or None if no completion was found.
    """
    # Create the pattern for matching the backtick-wrapped section
    backtick_pattern = '`' * backticks
    pattern = rf"{backtick_pattern}(Completion:[\s\S]*?Success:\s*(True|False)[\s\S]*?){backtick_pattern}"

    # Search for the pattern in the text
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    # Extract the content
    content = match.group(1).strip()

    # Parse using simple regex (no YAML)
    completion_match = re.search(r'Completion:\s*(.+)', content)
    success_match = re.search(r'Success:\s*(True|False)', content)

    if completion_match and success_match:
        completion_text = completion_match.group(1).strip()
        success = success_match.group(1) == 'True'
        return CompletionResult(text=completion_text, success=success)

    return CompletionResult(text="Task could not be verified.", success=False)


def sigterm_handler(_signo, _stack_frame):
    """Handle SIGTERM signal by terminating subprocess."""
    print_sigterm()
    terminate_process()


def _format_host_for_url(host):
    """Wrap an IPv6 address in brackets for use in URLs.

    IPv6 addresses contain colons which conflict with the host:port
    separator, so they must be enclosed in square brackets per RFC 2732.
    IPv4 addresses and hostnames are returned unchanged.

    Examples:
        >>> _format_host_for_url("localhost")
        'localhost'
        >>> _format_host_for_url("::1")
        '[::1]'
        >>> _format_host_for_url("[::1]")
        '[::1]'
        >>> _format_host_for_url("192.168.1.50")
        '192.168.1.50'
    """
    # Already bracketed
    if host.startswith('[') and host.endswith(']'):
        return host
    # Contains a colon → IPv6 address, needs brackets
    if ':' in host:
        return f'[{host}]'
    return host


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


class Agent:
    """An autonomous agent powered by an LLM backend.

    This agent can execute tasks, maintain context, and manage compute budget.
    """

    # When a user message (e.g. file-read output) exceeds this many
    # characters, the harness asks the backend to cache it so that
    # subsequent API calls don't re-process the same large payload.
    LARGE_MESSAGE_CACHE_THRESHOLD = 10_000

    def __init__(self, configuration_name, task, compute_budget=1.0, context=None,
                 local_model=None, local_port=8000, local_host="localhost",
                 session_id=None):
        """Initialize the Agent.

        Args:
            configuration_name: Name of the YAML configuration file
            task: The task to be performed
            compute_budget: Maximum allowed cost in dollars
            context: Optional list of previous conversation messages
            local_model: If set, use a local Anthropic-compatible API with this model name
            local_port: Port for the local API server (default 8000)
            local_host: Hostname for the local API server (default "localhost")
            session_id: Optional session ID for saving/restoring context
        """
        if context is None:
            context = []

        # Session management
        self.session_id = session_id or generate_session_id()
        self.working_dir = os.getcwd()

        # Load configuration
        configuration = read_configuration(configuration_name)

        # Determine provider from environment variable or config
        provider = os.environ.get("AGENT_MODEL_PROVIDER", configuration.get("provider", "anthropic"))

        # Determine model from environment variable with provider-specific defaults
        model_env = os.environ.get("AGENT_MODEL")
        if local_model:
            self.model_name = local_model
            base_url = f"http://{_format_host_for_url(local_host)}:{local_port}"
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

        # Temperature can be configured per-agent in the YAML config.
        # If not specified, each backend uses its own default (1.0 for
        # most providers, 0.6 for Anthropic).
        backend_kwargs = {}
        if "temperature" in configuration:
            backend_kwargs["temperature"] = configuration["temperature"]

        self.client = create_backend(
            provider,
            model=self.model_name,
            base_url=base_url,
            stream_handler=RichStreamHandler(),
            **backend_kwargs,
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
        self.context.append(_form_message("user", self.task))
        self.compute_budget = compute_budget
        self.iterations = 0
        self.start_time = None
        self._last_assistant_response = None
        self._loop_count = 0

        # Register the LLM backend for the summarize tool so that
        # the tools layer can make one-shot LLM calls without a circular import.
        self._register_summarize_backend()

        # Create and register the sub-agent pool.
        self._init_agent_pool()

        # Display startup banner
        print_banner(self.client.display_name, self.compute_budget, platform.platform(),
                     self.client.context_window_size)

    def _register_summarize_backend(self):
        """Wire the agent's LLM backend into the tools summarize module.

        Creates a thin wrapper that converts the ``(system_prompt, user_message)``
        signature expected by :func:`agents.tools.summarize.register_llm` into a
        single-turn conversation call through the agent's backend.
        """
        client = self.client  # capture for the closure

        def _generate(system_prompt: str, user_message: str) -> str:
            context = [_form_message("user", user_message)]
            return client.generate_response(system_prompt, context)

        _register_summarize_llm(_generate)

    def _init_agent_pool(self):
        """Create the sub-agent pool and register it with the tools layer."""
        from .agent_pool import AgentPool

        self._agent_pool = AgentPool()
        _register_pool(self._agent_pool)

    def _iterate(self):
        """Perform one iteration of the conversation with Claude.

        Returns:
            bool: True if the agent should continue running, False otherwise
        """
        print_iteration_header(
            self.iterations, self.client.cost, self.compute_budget,
            self.client.last_input_tokens, self.client.last_output_tokens,
            self.client.last_total_context_tokens,
            cost_without_cache=self.client.cost_without_cache,
            context_window_tokens=self.client.context_window_size,
        )
        self.iterations += 1

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

        # Anti-looping check: detect if the LLM produced the exact same output twice in a row
        if self._last_assistant_response is not None and response == self._last_assistant_response:
            self._loop_count += 1
            if self._loop_count >= 3:
                raise RuntimeError("Looping error: LLM produced identical response 3 times in a row.")

            print_error(f"Loop detected (attempt {self._loop_count}/3): LLM produced identical response. Injecting feedback.", None)
            # Remove the previous identical assistant message (the new
            # duplicate has not been appended to context yet).
            if self.context and self.context[-1]["role"] == "assistant":
                self.context.pop()
            # Remove the command-result user message that preceded it,
            # so the injected feedback replaces the stale exchange.
            if self.context and self.context[-1]["role"] == "user":
                self.context.pop()
            # Inject feedback to prevent looping
            loop_feedback = "Feedback: avoid looping and work towards finishing your task."
            self.context.append(_form_message("user", loop_feedback))
            # Reset the assistant response tracker to allow recovery
            self._last_assistant_response = None
            return True

        # Add response to context and process it
        self.context.append(_form_message("assistant", response))
        self._last_assistant_response = response
        self._loop_count = 0
        command_response, image_media_tuple_array = process_content(response)

        # Determine if we should continue running.  This must be checked
        # *before* the overbudget prompt is appended — otherwise the
        # "End." sentinel is mutated and the agent fails to terminate
        # even when no commands were found.
        command_called = command_response != "End."

        # Check compute budget
        if self.client.cost > 0.80 * self.compute_budget:
            command_response += "\n" + self.overbudget_prompt
            print_budget_warning(self.client.cost, self.compute_budget)

        # Add user message to context (with or without images)
        if len(image_media_tuple_array) == 0:
            message = _form_message("user", command_response)
        else:
            message = _form_message_with_images("user", command_response, image_media_tuple_array)
        self.context.append(message)

        # Large command outputs (e.g. file reads) are expensive to
        # re-process on every subsequent call.  Ask the backend to
        # cache them so the prefix stays warm.
        if len(command_response) >= self.LARGE_MESSAGE_CACHE_THRESHOLD:
            self.client.mark_for_caching(message)
            self.client.trim_cache_blocks(self.context)

        return command_called

    def _enter_feedback_mode(self, partial_response=None):
        """Pause the agent and wait for user feedback.

        If *partial_response* is provided it is added to the context as
        an assistant message so the conversation remains coherent.

        Returns
        -------
        str | None
            The user's feedback text, or ``None`` if the user chose to
            exit (Ctrl+C in feedback mode).
        """
        if partial_response:
            self.context.append(_form_message("assistant", partial_response))
        print_interrupt_feedback()
        return get_user_feedback()

    def run(self):
        """Run the agent until completion or interruption.

        Ctrl+C behaviour
        -----------------
        * **First Ctrl+C** — the current iteration finishes normally,
          then the agent pauses and waits for user feedback.
        * **Second Ctrl+C** — any running subprocess is terminated,
          output is suspended immediately, partial output is captured,
          and the agent waits for user feedback.
        * **Third Ctrl+C** (or Ctrl+C in feedback mode) — the agent
          exits immediately.
        """
        self.start_time = time.time()
        self._interrupt_requested = False
        original_sigint = signal.getsignal(signal.SIGINT)

        def _tty_msg(text):
            """Write a bright-yellow message directly to the terminal."""
            try:
                with open("/dev/tty", "w") as tty:
                    tty.write(f"\n\033[93m  ⚠  {text}\033[0m\n")
            except OSError:
                pass

        def _hard_interrupt(signum, frame):
            """Second Ctrl+C: stop current work immediately."""
            terminate_process()
            _tty_msg("Stopping current work…")
            # Restore original handler so a third Ctrl+C kills the agent.
            signal.signal(signal.SIGINT, original_sigint)
            raise KeyboardInterrupt

        def _soft_interrupt(signum, frame):
            """First Ctrl+C: set flag so the loop pauses after the current iteration."""
            self._interrupt_requested = True
            _tty_msg("Interrupt received — will pause after current step. Press Ctrl+C again to stop immediately.")
            # Install hard-interrupt so the next Ctrl+C escalates.
            signal.signal(signal.SIGINT, _hard_interrupt)

        try:
            running = True
            while running:
                # Arm the soft-interrupt handler before each iteration
                self._interrupt_requested = False
                signal.signal(signal.SIGINT, _soft_interrupt)

                try:
                    running = self._iterate()
                except InterruptedResponse as ir:
                    # Hard interrupt during streaming — partial output captured
                    signal.signal(signal.SIGINT, original_sigint)
                    feedback = self._enter_feedback_mode(ir.partial_text)
                    if feedback is None:
                        print_interrupted()
                        break
                    self.context.append(_form_message("user", feedback))
                    running = True
                    continue
                except KeyboardInterrupt:
                    # Hard interrupt outside of streaming (e.g. during
                    # command execution).  The response was already fully
                    # streamed and added to context so we do not pass
                    # partial text to avoid duplication.
                    signal.signal(signal.SIGINT, original_sigint)
                    feedback = self._enter_feedback_mode()
                    if feedback is None:
                        print_interrupted()
                        break
                    self.context.append(_form_message("user", feedback))
                    running = True
                    continue

                # Restore default handler while checking flags / feedback
                signal.signal(signal.SIGINT, original_sigint)

                if self.client.cost > self.compute_budget:
                    print_budget_exceeded(self.client.cost, self.compute_budget)
                    break

                # First Ctrl+C was pressed — iteration finished normally,
                # now pause for user feedback.
                if self._interrupt_requested:
                    self._interrupt_requested = False
                    feedback = self._enter_feedback_mode()
                    if feedback is None:
                        print_interrupted()
                        break
                    self.context.append(_form_message("user", feedback))
                    running = True

        except Exception as e:
            print_error(e, traceback.format_exc())
        finally:
            signal.signal(signal.SIGINT, original_sigint)

        # Print final summary
        elapsed = time.time() - self.start_time
        print_summary(self.client.cost, self.iterations, elapsed, self.compute_budget,
                      self.client.peak_context_tokens,
                      cost_without_cache=self.client.cost_without_cache,
                      context_window_tokens=self.client.context_window_size)

    def request_completion(self) -> bool:
        """Ask the LLM for a completion block if none was found.

        Appends a feedback message requesting a completion block and
        runs one more iteration.

        Returns True if an iteration was performed, False if budget
        was already exhausted.
        """
        if self.client.cost > self.compute_budget:
            return False
        feedback = (
            "Feedback: No completion block was found in your response. "
            "Please provide a completion block with "
            "'Completion: <description>' and 'Success: True/False' "
            "at the end of your response."
        )
        self.context.append(_form_message("user", feedback))
        self._iterate()
        return True

    def save_context(self):
        """Save conversation context and token state to a JSON session file.

        Delegates to :func:`.session.save_session` which handles atomic
        writes, file permissions, index updates, and stale-session pruning.
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
        save_session(self.session_id, self.working_dir, state)

    def load_context(self, session_id=None):
        """Load conversation context and token state from a JSON session file.

        Args:
            session_id: Session ID to load.  If None, uses self.session_id.
        """
        sid = session_id or self.session_id
        data = load_session(sid)

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

        # Adopt the loaded session's ID so subsequent saves go to the
        # same file.
        self.session_id = sid

        # Remove the last user message and replace with current task
        if self.context and self.context[-1]["role"] == "user":
            self.context.pop()
        else:
            logging.warning(
                "load_context: expected last message role='user', got '%s'. "
                "Appending new task without removing last message.",
                self.context[-1]["role"] if self.context else "<empty>",
            )
        new_message = _form_message("user", self.task)
        self.context.append(new_message)
        # Let the backend annotate the new message for caching (e.g.
        # Anthropic adds cache_control blocks) and trim stale markers.
        self.client.mark_for_caching(new_message)
        self.client.trim_cache_blocks(self.context)


class SessionNotFoundError(Exception):
    """Raised when no restorable session can be found."""


def run_agent(agent_definition, command, budget, save=True, restore=False,
              session_id=None, local_model=None, local_port=8000,
              local_host="localhost"):
    """Create and run an agent, optionally restoring a previous session.

    Args:
        agent_definition: YAML config filename
        command: The task string
        budget: Compute budget in dollars
        save: Whether to save context after running
        restore: Whether to restore a previous session
        session_id: Explicit session ID to use/restore.  When restoring
                    without a session ID, the latest session for the
                    current working directory is used.
        local_model: Local model name (if using local API)
        local_port: Port for local API server
        local_host: Hostname for local API server (default "localhost")

    Returns:
        tuple: (completion_text, success_bool, session_id)

    Raises:
        SessionNotFoundError: When restoring and no session can be found.
    """
    # Resolve which session to restore
    restore_sid = None
    if restore:
        if session_id:
            restore_sid = session_id
        else:
            restore_sid = get_latest_session_for_dir(os.getcwd())
            if restore_sid is None:
                raise SessionNotFoundError(
                    "No previous session found for this directory."
                )

    # Use the restore session ID if we have one, otherwise use provided
    effective_sid = restore_sid or session_id

    agent = Agent(agent_definition, command, budget,
                  local_model=local_model, local_port=local_port,
                  local_host=local_host, session_id=effective_sid)

    if restore and restore_sid:
        agent.load_context(restore_sid)

    agent.run()
    completion_result = None
    if len(agent.context) > 2:
        final_content = agent.context[-2]['content'][0]['text']
        completion_result = extract_completion(final_content)
        if completion_result is None:
            # Give the agent one more chance to provide a completion block.
            try:
                if agent.request_completion() and len(agent.context) > 2:
                    final_content = agent.context[-2]['content'][0]['text']
                    completion_result = extract_completion(final_content)
            except Exception as e:
                logging.warning("Completion-retry iteration failed: %s", e)

    completion = completion_result.text if completion_result else "Error"
    success = completion_result.success if completion_result else False

    if save:
        agent.save_context()

    return completion, success, agent.session_id


def main():
    """Parse arguments and run the agent."""
    # Register signal handler here rather than at import time so that
    # importing this module as a library doesn't install a handler as a
    # side-effect.
    signal.signal(signal.SIGTERM, sigterm_handler)
    parser = argparse.ArgumentParser(description="Autonomous AI agent")
    parser.add_argument('command', type=str, help='A command string like "update my system"')
    parser.add_argument('-b', '--compute-budget', type=float, default=1.0, help='Compute budget in dollars')
    parser.add_argument('-r', '--restore', action='store_true',
                        help='Restore the latest session for the current directory')
    parser.add_argument('-s', '--session', type=str, default=None,
                        help='Session ID to use or resume (max 10 alphanumeric chars)')
    parser.add_argument('-l', '--local', action='store_true',
                        help='Use a local Anthropic-compatible API (requires LOCAL_MODEL environment variable)')
    parser.add_argument('-p', '--port', type=int, default=None,
                        help='Port for the local API server (default: LOCAL_LLM_PORT or 8000)')
    parser.add_argument('-H', '--host', type=str, default=None,
                        help='Hostname for the LLM API server (default: LOCAL_LLM_HOST or localhost)')
    parser.add_argument('-a', '--agent', type=str, default='basic_agent.yaml',
                        help='Agent definition YAML file (default: basic_agent.yaml)')

    args = parser.parse_args()

    # Resolve --port default: CLI flag > LOCAL_LLM_PORT env var > 8000
    if args.port is None:
        port_env = os.environ.get('LOCAL_LLM_PORT')
        if port_env is not None:
            try:
                args.port = int(port_env)
            except ValueError:
                parser.error(f'LOCAL_LLM_PORT must be a valid integer, got: {port_env!r}')
        else:
            args.port = 8000

    # Resolve --host default: CLI flag > LOCAL_LLM_HOST env var > localhost
    if args.host is None:
        args.host = os.environ.get('LOCAL_LLM_HOST', 'localhost')

    # Validate session ID if provided
    if args.session:
        try:
            validate_session_id(args.session)
        except ValueError as e:
            parser.error(str(e))

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

    try:
        completion, success, sid = run_agent(
            args.agent, command, args.compute_budget,
            restore=args.restore, session_id=args.session,
            local_model=local_model, local_port=args.port,
            local_host=args.host)
    except SessionNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # Display session info
    if args.restore:
        safe_console_print(f"  ↻  Resumed session [bright_cyan]{sid}[/]", style="info")
    else:
        safe_console_print(f"  ◈  Session [bright_cyan]{sid}[/]", style="info")

    print_completion_result(completion, success)


if __name__ == "__main__":
    main()
