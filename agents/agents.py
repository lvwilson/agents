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
from .git_utils import is_git_repo, check_git_clean, git_add_and_commit
from .memory import (
    format_memory_view,
    add_episode,
    get_episode_count,
    squash_episodes,
    notes_need_compact,
    get_notes,
    MAX_NOTES_CHARS,
    EPISODES_BEFORE_SQUASH,
)
from .session import (
    generate_session_id,
    validate_session_id,
    get_latest_session_for_dir,
    save_session,
    load_session,
)
from .llm_backend import InterruptedResponse, EmptyResponseError
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


# Known online models mapped to their provider.  When -m specifies one
# of these, the provider is auto-detected and the -o flag is not required.
_ONLINE_MODELS: dict[str, str] = {
    # Anthropic
    "claude-3-5-sonnet-20240620": "anthropic",
    "claude-3-5-sonnet-20241022": "anthropic",
    "claude-3-7-sonnet-20250219": "anthropic",
    "claude-sonnet-4-20250514": "anthropic",
    "claude-sonnet-4-5-20250929": "anthropic",
    "claude-sonnet-4-6": "anthropic",
    "claude-opus-4-6": "anthropic",
    "claude-fable-5": "anthropic",
    "MiniMax-M2.5": "anthropic",
    # OpenAI
    "gpt-5.2": "openai",
    "gpt-5.2-mini": "openai",
    "gpt-5.3": "openai",
    "gpt-5.3-mini": "openai",
    "gpt-5.3-codex": "openai",
    # Gemini
    "gemini-3.1-pro-preview": "gemini",
    "gemini-3.1-pro-preview-customtools": "gemini",
    "gemini-3-flash-preview": "gemini",
    # Kimi
    "kimi-k3": "kimi",
}


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


def _find_latest_completion(context, scan_limit=5):
    """Find the most recent completion block in the conversation.

    Scans assistant messages from newest to oldest and returns the
    first successfully parsed :class:`CompletionResult`.  Only
    assistant messages are inspected, so completion-looking text inside
    user messages (e.g. tool output echoing these instructions) is
    never mistaken for the agent's own completion.

    Args:
        context: Conversation in the internal message format.
        scan_limit: Maximum number of assistant messages to inspect.

    Returns:
        CompletionResult, or None if no recent assistant message
        contains a valid completion block.
    """
    checked = 0
    for message in reversed(context):
        if message.get("role") != "assistant":
            continue
        checked += 1
        if checked > scan_limit:
            break
        try:
            text = message["content"][0]["text"]
        except (KeyError, IndexError, TypeError):
            continue
        result = extract_completion(text)
        if result is not None:
            return result
    return None


#: Reminder injected (at most once per incident) when a model response
#: contains neither a command nor a completion block — an almost
#: certainly unintended session end.  It explains both output
#: mechanisms and that this is the only warning before the session ends.
NO_OUTPUT_REMINDER = (
    "Feedback: Your previous response contained no commands and no "
    "completion block, so the session is about to end — which you "
    "almost certainly did not intend. You have two ways to produce "
    "output:\n"
    "1. To keep working, issue one or more commands as visible lines "
    "of the form 'Command: name args', each optionally followed by a "
    "5-backtick payload block. Command output is returned to you in a "
    "'=== Tool Results ===' message.\n"
    "2. To intentionally finish the task, end your response with a "
    "completion block wrapped in 5 backticks:\n"
    f"`````\n"
    "Completion: <description of what you accomplished>\n"
    "Success: True or False\n"
    f"`````\n"
    "This is your only warning: if your next response again contains "
    "neither commands nor a completion block, the session will end. "
    "Please respond to your task now."
)


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


_AGENTS_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".agents")


def read_configuration(configuration_name):
    """Read agent configuration from a YAML file.

    Looks for the config in ``~/.agents/`` first, then falls back to
    the package directory (for backward compatibility).

    Args:
        configuration_name: Name of the configuration file

    Returns:
        dict: Configuration data
    """
    # Prefer user-level config in ~/.agents/
    user_config = os.path.join(_AGENTS_CONFIG_DIR, configuration_name)
    if os.path.isfile(user_config):
        return read_yaml_file(user_config)

    # Fall back to package-level config
    script_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
    config_path = os.path.join(script_dir, configuration_name)
    return read_yaml_file(config_path)


# ── Context guard ─────────────────────────────────────────────────────

#: Header line identifying the context-guard block inside the first
#: user message of a session.
CONTEXT_GUARD_HEADER = "=== Context Guard (session start) ==="

#: Header line identifying the task block of the first user message.
TASK_HEADER = "=== Task ==="

#: Header line identifying a tool-results user message.
TOOL_RESULTS_HEADER = "=== Tool Results ==="


def build_task_message(task):
    """Wrap *task* in an explicit task block.

    The first user message of a session is the only user-authored text
    in the conversation; labelling it makes it unambiguous against the
    tool-result user messages that follow.
    """
    return f"{TASK_HEADER}\n{task}\n=== End Task ==="


def build_tool_results_message(command_response):
    """Wrap *command_response* in an explicit tool-results block.

    Tool output is delivered to the model as a ``user`` message (the
    chat APIs have no tool role), so without a label a bare ``ok`` reads
    as if the human typed it.  The guard makes the origin explicit.
    """
    return f"{TOOL_RESULTS_HEADER}\n{command_response}\n=== End Tool Results ==="


def build_context_guard():
    """Build the context guard prepended to the first user message.

    The guard carries everything that used to live in the system prompt
    — timestamp, working directory, platform, shell, user — plus the
    folder-memory snapshot (episodes + notes) and the notes-compact hint.
    It is captured exactly once per session: on resume the original
    guard is kept untouched (see :meth:`Agent.load_context`) so the
    Anthropic prompt-cache prefix is never invalidated.
    """
    lines = [
        CONTEXT_GUARD_HEADER,
        f"System Date: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Working Directory: {os.getcwd()}",
        f"Operating System: {platform.platform()}",
        f"Shell: {get_default_shell()}",
        f"User: {os.environ.get('USER', 'unknown')}",
        "Note: these values were captured when this session started and "
        "may be stale after a resume; re-check them with a console "
        "command if they matter.",
    ]

    memory_view = format_memory_view()
    if memory_view:
        lines.append("")
        lines.append(memory_view)

    if notes_need_compact():
        lines.append("")
        lines.append(
            f"NOTE: Your project notes have exceeded {MAX_NOTES_CHARS} characters. "
            "Please use the `note rewrite` command to make them more compact "
            "(approximately half their current size)."
        )

    return "\n".join(lines)


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
                 session_id=None, model=None):
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
            model: Explicit model name.  Online models are auto-detected
                   (no -o needed).  Unknown names are treated as local.
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

        # ── Resolve model and provider ──────────────────────────────
        # Priority: explicit -m flag > local_model (env/flag) > AGENT_MODEL > default
        if model is not None:
            # Explicit model via -m: auto-detect online vs local
            detected_provider = _ONLINE_MODELS.get(model)
            if detected_provider is not None:
                # Known online model — auto-select provider
                self.model_name = model
                provider = detected_provider
                base_url = configuration.get("base_url", None)
            else:
                # Unknown model name — treat as local
                self.model_name = model
                base_url = f"http://{_format_host_for_url(local_host)}:{local_port}"
        elif local_model:
            self.model_name = local_model
            base_url = f"http://{_format_host_for_url(local_host)}:{local_port}"
        elif os.environ.get("AGENT_MODEL"):
            self.model_name = os.environ["AGENT_MODEL"]
            base_url = configuration.get("base_url", None)
        else:
            # Provider-specific defaults
            provider_defaults = {
                "anthropic": "claude-opus-4-6",
                "openai": "gpt-5.3-codex",
                "gemini": "gemini-3.1-pro-preview",
                "kimi": "kimi-k3",
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

        # Set up the system prompt.  It is intentionally IMMUTABLE: no
        # timestamps, no working directory, no memory snapshot — nothing
        # that varies between runs.  Its Anthropic prompt-cache entry
        # therefore stays valid across every session in every project,
        # and resume never needs to choose between freshness and cache
        # validity.  All per-run environment context lives in the
        # context guard (below) instead.
        self.system_prompt = configuration["system_prompt"]

        # Build the context guard — a snapshot of the environment and
        # folder memory captured once, at session start — and prepend it
        # to the first user message.  Memory is thus loaded on every new
        # session, while the volatile text sits in the cheap front of
        # the conversation rather than in the cached system prefix.
        context_guard = build_context_guard()

        # Set remaining attributes
        self.overbudget_prompt = configuration["overbudget"]
        self.context = context
        self.task = task
        task_block = build_task_message(task)
        first_message = f"{context_guard}\n\n{task_block}" if context_guard else task_block
        self.context.append(_form_message("user", first_message))
        self.compute_budget = compute_budget
        self.iterations = 0
        self.start_time = None
        self._last_assistant_response = None
        self._loop_count = 0
        self._empty_response_count = 0
        self._no_output_reminded = False

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
        self._agent_pool.model = self.model_name
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

        # Generate response from the LLM.  A blank turn (only thinking
        # tokens, no visible text) must NOT end the session — the model
        # simply failed to emit content, usually because it wrote its
        # commands into a reasoning block.  Feed the failure back so it
        # can retry; only give up after several consecutive blanks.
        try:
            response = self.client.generate_response(self.system_prompt, self.context)
        except EmptyResponseError:
            self._empty_response_count += 1
            if self._empty_response_count >= 3:
                raise
            print_error(
                f"Model returned no text content "
                f"(attempt {self._empty_response_count}/3). Injecting feedback.",
                None,
            )
            empty_feedback = (
                "Feedback: Your previous response contained no text content — "
                "it was completely blank. A blank response is interpreted as a "
                "request to end the session, which is almost certainly not what "
                "you intended. This usually happens when commands or replies "
                "are written into reasoning/thinking instead of visible output. "
                "You must always produce visible text, and commands must be "
                "issued as 'Command: name args' lines in your visible response — "
                "never inside thinking. Please respond to your task now."
            )
            self.context.append(_form_message("user", empty_feedback))
            return True
        self._empty_response_count = 0

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
        # A response with no completion block also ended the turn: the
        # model stopped without either mechanism the harness understands.
        completion_found = extract_completion(response) is not None

        # Check compute budget
        if self.client.cost > 0.80 * self.compute_budget:
            command_response += "\n" + self.overbudget_prompt
            print_budget_warning(self.client.cost, self.compute_budget)

        # Label the tool output so it is unmistakably tool-generated
        # rather than user-authored.  The loop-control sentinel above
        # uses the raw command_response, so wrapping here cannot break
        # termination.
        framed_response = build_tool_results_message(command_response)

        # Add user message to context (with or without images)
        if len(image_media_tuple_array) == 0:
            message = _form_message("user", framed_response)
        else:
            message = _form_message_with_images("user", framed_response, image_media_tuple_array)
        self.context.append(message)

        # Large command outputs (e.g. file reads) are expensive to
        # re-process on every subsequent call.  Ask the backend to
        # cache them so the prefix stays warm.
        if len(command_response) >= self.LARGE_MESSAGE_CACHE_THRESHOLD:
            self.client.mark_for_caching(message)
            self.client.trim_cache_blocks(self.context)

        # Accidental-stop guard: a response with neither commands nor a
        # completion block almost certainly was not meant to end the
        # session — the model simply forgot both output mechanisms.
        # Give it ONE reminder explaining both mechanisms and one chance
        # to self-correct; if the next response is again content-free,
        # let the session end as it normally would.  A response that
        # does contain a command or a completion block counts as
        # intentional and ends (or continues) normally, and also re-arms
        # the reminder so each incident gets its own single warning.
        if command_called or completion_found:
            self._no_output_reminded = False
        elif self._no_output_reminded:
            print_error(
                "Model produced neither commands nor a completion block "
                "again after one reminder. Ending the session.",
                None,
            )
        else:
            self._no_output_reminded = True
            print_error(
                "Model response contained neither commands nor a "
                "completion block. Injecting reminder (only warning).",
                None,
            )
            self.context.append(_form_message("user", NO_OUTPUT_REMINDER))
            return True

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

    def _request_episode_summary(self) -> str | None:
        """Ask the LLM for a short episode summary of this session.

        Returns the summary string, or None if budget was exhausted.
        """
        if self.client.cost > self.compute_budget:
            return None
        feedback = (
            "Feedback: Session complete. Please provide a brief summary "
            "(2-4 sentences) of what you accomplished in this session. "
            "Focus on key decisions, changes made, and any outstanding work. "
            "Do not include any commands — just the summary text."
        )
        self.context.append(_form_message("user", feedback))
        try:
            self._iterate()
        except Exception as e:
            logging.warning("Episode-summary iteration failed: %s", e)
            return None
        for msg in reversed(self.context):
            if msg["role"] == "assistant":
                return msg["content"][0]["text"].strip()
        return None

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

    def request_commit_message(self) -> str | None:
        """Ask the LLM for a git commit message.

        Appends a feedback message requesting a commit message and
        runs one more iteration.

        Returns the commit message string, or None if budget was
        exhausted or no message was produced.
        """
        if self.client.cost > self.compute_budget:
            return None
        feedback = (
            "Feedback: Your work has changed files in the repository. "
            "Please provide a concise git commit message on a single line. "
            "Do not include any other content or commands — just the commit message."
        )
        self.context.append(_form_message("user", feedback))
        try:
            self._iterate()
        except Exception as e:
            # A failure here (loop detection, retry exhaustion) must not
            # crash the CLI after the session completed successfully.
            logging.warning("Commit-message iteration failed: %s", e)
            return None
        # Extract the last assistant response as the commit message
        for msg in reversed(self.context):
            if msg["role"] == "assistant":
                raw = msg["content"][0]["text"].strip()
                # Strip markdown fences the model may wrap the message in
                raw = re.sub(r"`{3,5}", "", raw).strip()
                first_line = raw.split("\n")[0].strip()
                return first_line or None
        return None

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

        # Resume is deliberately cache-pure: the restored context is
        # only ever appended to, never mutated.  In particular the
        # original context guard in the first user message (timestamp,
        # working directory, memory snapshot) is left in place so the
        # cached prefix stays byte-identical.  Memory is NOT re-loaded
        # here — only the replaced task message at the tail differs
        # from the previous run.
        first = self.context[0] if self.context else None
        if (
            first is not None
            and first["role"] == "user"
            and first.get("content")
            and CONTEXT_GUARD_HEADER not in first["content"][0].get("text", "")
        ):
            logging.info(
                "load_context: resumed session predates the context-guard "
                "format; leaving context untouched (cache-pure resume)."
            )

        # Remove the last user message and replace with current task
        if self.context and self.context[-1]["role"] == "user":
            self.context.pop()
        else:
            logging.warning(
                "load_context: expected last message role='user', got '%s'. "
                "Appending new task without removing last message.",
                self.context[-1]["role"] if self.context else "<empty>",
            )
        new_message = _form_message("user", build_task_message(self.task))
        self.context.append(new_message)
        # Let the backend annotate the new message for caching (e.g.
        # Anthropic adds cache_control blocks) and trim stale markers.
        self.client.mark_for_caching(new_message)
        self.client.trim_cache_blocks(self.context)


class SessionNotFoundError(Exception):
    """Raised when no restorable session can be found."""


def run_agent(agent_definition, command, budget, save=True, restore=False,
              session_id=None, local_model=None, local_port=8000,
              local_host="localhost", nogit=False, model=None):
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
        nogit: If True, skip git status check and auto-commit
        model: Explicit model name.  Online models are auto-detected
               (no -o needed).  Unknown names are treated as local.

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
                  local_host=local_host, session_id=effective_sid,
                  model=model)

    if restore and restore_sid:
        agent.load_context(restore_sid)

    agent.run()
    # The completion block is not guaranteed to sit at context[-2]:
    # if the agent's final reply ends with a Worklog line or a trailing
    # Command, the loop runs one more iteration and the completion
    # slides one message earlier.  Scan recent assistant messages
    # newest-to-oldest so a correctly written completion is never
    # reported as a failure merely because of its position.
    completion_result = _find_latest_completion(agent.context)
    if completion_result is None and len(agent.context) > 2:
        # Give the agent one more chance to provide a completion block.
        try:
            if agent.request_completion():
                completion_result = _find_latest_completion(agent.context)
        except Exception as e:
            logging.warning("Completion-retry iteration failed: %s", e)

    completion = completion_result.text if completion_result else "Error"
    success = completion_result.success if completion_result else False

    if save:
        agent.save_context()

    # ── Episode memory ──────────────────────────────────────────────
    # Prompt the agent for a short episode summary, store it, then
    # squash old episodes if the threshold has been reached.
    try:
        episode_summary = agent._request_episode_summary()
        if episode_summary:
            add_episode(episode_summary, session_id=agent.session_id)

            # After storing, check if we need to squash old episodes.
            if get_episode_count() >= EPISODES_BEFORE_SQUASH:
                squash_prompt = read_configuration("memory.yaml").get(
                    "squash_prompt",
                    "Compress these episode summaries into one concise paragraph.",
                )

                def _squash(input_text: str) -> str:
                    ctx = [_form_message("user", squash_prompt + "\n\n" + input_text)]
                    return agent.client.generate_response(
                        "You are a concise summarizer.", ctx
                    )

                squash_episodes(_squash)
    except Exception as e:
        logging.warning("Episode memory update failed: %s", e)

    # Auto-commit if there are uncommitted changes and git is enabled
    if not nogit and is_git_repo():
        clean, _ = check_git_clean()
        if not clean:
            commit_msg = agent.request_commit_message()
            if commit_msg:
                # First line only; fences already stripped above
                first_line = commit_msg.split("\n")[0].strip()
                author_name = agent.model_name
                author_email = f"agent@{platform.node()}"
                ok, err = git_add_and_commit(first_line, author_name=author_name, author_email=author_email)
                if ok:
                    safe_console_print(f"  ⚡  Auto-committed: [green]{first_line}[/] [dim]by {author_name} <{author_email}>[/]", style="info")
                else:
                    safe_console_print(f"  ⚠  Auto-commit failed: {err}", style="warning")

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
    local_group = parser.add_mutually_exclusive_group()
    local_group.add_argument('-l', '--local', action='store_true',
                             help='Use a local Anthropic-compatible API (also enabled automatically when LOCAL_MODEL env var is set)')
    local_group.add_argument('-o', '--online', action='store_true',
                             help='Use the online model provider (ignores LOCAL_MODEL env var)')
    parser.add_argument('-m', '--model', type=str, default=None,
                        help='Model to use. Online models are auto-detected '
                             '(no -o needed). Unknown model names are treated '
                             'as local.')
    parser.add_argument('-p', '--port', type=int, default=None,
                        help='Port for the local API server (default: LOCAL_LLM_PORT or 8000)')
    parser.add_argument('-H', '--host', type=str, default=None,
                        help='Hostname for the LLM API server (default: LOCAL_LLM_HOST or localhost)')
    parser.add_argument('-a', '--agent', type=str, default='basic_agent.yaml',
                        help='Agent definition YAML file (default: basic_agent.yaml)')
    parser.add_argument('--nogit', action='store_true',
                        help='Disable git status check and auto-commit')

    args = parser.parse_args()

    # Pre-flight: ensure git working tree is clean (unless --nogit)
    if not args.nogit and is_git_repo():
        clean, msg = check_git_clean()
        if not clean:
            print_error(msg, None)
            sys.exit(1)

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

    # Resolve model selection:
    # -m/--model takes highest priority — auto-detects online vs local
    # -o/--online forces online (ignores LOCAL_MODEL)
    # -l/--local forces local (requires LOCAL_MODEL)
    # If neither -m nor -o nor -l, LOCAL_MODEL env var enables local mode.
    model_arg = args.model
    local_model = None
    if model_arg is None and not args.online:
        if args.local:
            local_model = os.environ.get('LOCAL_MODEL')
            if not local_model:
                parser.error('--local requires the LOCAL_MODEL environment variable to be set')
        else:
            local_model = os.environ.get('LOCAL_MODEL')

    try:
        completion, success, sid = run_agent(
            args.agent, command, args.compute_budget,
            restore=args.restore, session_id=args.session,
            local_model=local_model, local_port=args.port,
            local_host=args.host, nogit=args.nogit,
            model=model_arg)
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
