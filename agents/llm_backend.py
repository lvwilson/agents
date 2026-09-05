"""
Abstract base class for LLM backends.

Every backend (Anthropic, OpenAI, Gemini, …) implements this interface.
Backends are lazily loaded via the factory in ``backends/__init__.py``.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from .loop_detector import LoopDetectedError

_T = TypeVar("_T")


class StreamHandler:
    """Callback interface for streaming events from backends.

    Backends call these methods to report streaming progress.  The default
    implementation is a silent no-op so that backends work headlessly
    without any UI dependency.  Pass a ``RichStreamHandler`` (from ``ui``)
    for interactive terminal output.

    The handler also accumulates streamed tokens in an internal buffer so
    that partial output can be recovered after a ``KeyboardInterrupt``.
    """

    def __init__(self):
        self._buffer: list[str] = []
        self._reasoning_buffer: list[str] = []

    def on_stream_start(self) -> None:
        """Called once before the first token of a new API call."""
        self._buffer = []
        self._reasoning_buffer = []

    def on_stream_token(self, token: str) -> None:
        """Called for each streamed token/chunk of text."""
        self._buffer.append(token)

    def on_stream_end(self) -> None:
        """Called after the last token (or if no tokens were received)."""

    # ── Reasoning token streaming ────────────────────────────────────

    def on_stream_reasoning_start(self) -> None:
        """Called once before the first reasoning token of a new API call."""

    def on_stream_reasoning_token(self, token: str) -> None:
        """Called for each streamed reasoning/thinking token chunk."""
        self._reasoning_buffer.append(token)

    def on_stream_reasoning_end(self) -> None:
        """Called after the last reasoning token (or if no reasoning tokens)."""

    def on_tool_call(self, name: str, arguments: str = "") -> None:
        """Called when the model emits a native API tool/function call.

        This harness does not execute native tool calls — it parses
        textual ``Command:`` lines instead — so any native tool call the
        model emits would otherwise be silently dropped.  Backends call
        this hook so the UI can log the call (in yellow) and the user
        can see that the model is emitting commands in the wrong place.
        """

    def on_retry(self, message: str) -> None:
        """Called when a retryable error occurs (rate-limit or transient)."""

    def on_error(self, message: str) -> None:
        """Called when a non-retryable attempt fails."""

    def get_buffered_text(self) -> str:
        """Return all tokens accumulated since the last ``on_stream_start``."""
        return "".join(self._buffer)

    def get_buffered_reasoning(self) -> str:
        """Return all reasoning tokens accumulated since the last start."""
        return "".join(self._reasoning_buffer)


# Convenience alias — a handler that does nothing.
NullStreamHandler = StreamHandler


# ── Error classification constants ───────────────────────────────────
RATE_LIMIT = "rate_limit"
TRANSIENT = "transient"


def merge_consecutive_messages(context: list[dict]) -> list[dict]:
    """Merge consecutive messages that share the same role.

    Several harness feedback paths (empty-response feedback, the
    no-output reminder, episode-summary and commit-message requests,
    user feedback mode) append a ``user`` message immediately after a
    previous ``user`` (tool-results) message.  Strict chat APIs
    (OpenAI, some Anthropic-compatible local servers) reject payloads
    with consecutive same-role messages with a 400 error.

    This normalization pass merges such runs into a single message by
    concatenating their content part lists, preserving order.  Image
    and text parts are kept as separate content blocks, which every
    supported backend accepts.  The input is not mutated; a new list
    of (shallow-copied) messages is returned.

    Args:
        context: Conversation in the internal message format.

    Returns:
        A new list of messages with no consecutive same-role pairs.
    """
    merged: list[dict] = []
    for msg in context:
        role = msg.get("role")
        parts = list(msg.get("content", []) or [])
        if merged and merged[-1].get("role") == role:
            merged[-1]["content"] = merged[-1]["content"] + parts
        else:
            merged.append({"role": role, "content": parts})
    return merged


class InterruptedResponse(Exception):
    """Raised when a streaming response is interrupted by the user.

    Carries the partial text that was streamed before the interruption
    so the caller can still make use of it.
    """

    def __init__(self, partial_text: str):
        self.partial_text = partial_text
        super().__init__(f"Response interrupted ({len(partial_text)} chars received)")


class EmptyResponseError(Exception):
    """Raised when the model returns no text content at all.

    This is distinct from a network/transient failure: the API call
    succeeded but the assistant turn contained only reasoning/thinking
    tokens (or nothing).  The agent loop catches this and feeds it back
    to the model as an instruction to produce visible output, rather
    than treating the blank turn as a request to end the session.
    """


class LLMBackend(ABC):
    """Unified interface for large-language-model providers.

    Subclasses must implement ``generate_response``.  They should also
    populate the class-level ``MODEL_DISPLAY_NAMES`` and
    ``MODEL_CONTEXT_WINDOWS`` dicts; the base class provides default
    ``display_name`` and ``context_window_size`` implementations that
    look up the current model in those dicts.

    Token-tracking and cost attributes have sensible defaults so that
    backends which don't support them still satisfy the interface.

    The ``_run_with_retries`` template method provides shared retry /
    back-off logic.  Backends customise behaviour by overriding
    ``_classify_error`` and optionally ``_extract_retry_after``.
    """

    # Subclasses should populate these — the base class uses them for
    # display_name and context_window_size lookups.
    MODEL_DISPLAY_NAMES: dict[str, str] = {}
    MODEL_CONTEXT_WINDOWS: dict[str, int] = {}

    # Default context window when the model isn't in MODEL_CONTEXT_WINDOWS.
    DEFAULT_CONTEXT_WINDOW: int = 256_000

    # Retry configuration — shared defaults for all backends
    RETRY_TIMEOUT = 300        # 5 minutes overall timeout for rate-limit retries
    RETRY_BASE_DELAY = 1       # Initial backoff delay in seconds
    RETRY_MAX_DELAY = 60       # Maximum backoff delay in seconds
    RETRY_BACKOFF_FACTOR = 2   # Exponential backoff multiplier
    MAX_ERROR_RETRIES = 3      # Fixed retry limit for non-rate-limit errors
    TRANSIENT_RETRY_DELAY = 2  # Seconds to wait between transient-error retries

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        stream_handler: StreamHandler | None = None,
        temperature: float = 1.0,
    ):
        self.model: str = model
        self.base_url: str | None = base_url
        self.is_local: bool = base_url is not None
        self.stream_handler: StreamHandler = stream_handler or NullStreamHandler()
        self.temperature: float = temperature

        # Running totals
        self.cost: float = 0.0
        self.cost_without_cache: float = 0.0
        self.call_count: int = 0

        # Session throughput / cost-rate metrics (driven by the per-call
        # duration recorded in _run_with_retries and folded in here via
        # record_step_metrics()).  See the final-metrics panel in ui.py.
        self.last_call_duration: float = 0.0
        self.total_output_tokens: int = 0
        self.total_call_duration: float = 0.0
        self.output_rate_tokens_per_sec: float | None = None
        self.cost_per_hour: float | None = None
        self.step_rate_tokens_per_sec: float | None = None

        # Per-call token bookkeeping
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_total_context_tokens: int = 0
        self.peak_context_tokens: int = 0

        # Native tool/function calls detected during the current call.
        # Backends append (name, arguments) tuples here while parsing
        # the response; generate_response drains the list via
        # _emit_tool_calls() after the stream has ended.
        self._pending_tool_calls: list[tuple[str, str]] = []

    # ── Retry template method ────────────────────────────────────────

    def _run_with_retries(self, attempt_fn: Callable[[], _T]) -> _T:
        """Execute *attempt_fn* in a retry loop with exponential back-off.

        ``attempt_fn`` is a zero-argument callable that performs a single
        streaming API call.  It may call ``self.stream_handler.on_stream_token``
        to deliver tokens but must **not** call ``on_stream_start`` or
        ``on_stream_end`` — those are managed by this method.

        On success, ``attempt_fn`` returns a result which is passed through.
        On failure it should let exceptions propagate.

        Error classification is delegated to ``_classify_error``:

        * ``RATE_LIMIT`` — exponential back-off with jitter; honours
          ``_extract_retry_after`` if available.
        * ``TRANSIENT`` — fixed retry count (``MAX_ERROR_RETRIES``) with
          a short delay (``TRANSIENT_RETRY_DELAY``) between attempts.

        Exhausted retries are re-raised with exception chaining.
        """
        sh = self.stream_handler
        start_time = time.monotonic()
        error_retries = 0
        current_delay = self.RETRY_BASE_DELAY
        # Fresh-generation timestamp: taken just before each attempt
        # streams, so sleep time spent between retries never inflates
        # the measured call duration.  Callers must only read
        # self.last_call_duration on success.
        fresh_start_time = time.monotonic()

        while True:
            try:
                fresh_start_time = time.monotonic()
                sh.on_stream_start()
                result = attempt_fn()
                sh.on_stream_end()
                # Record the wall-clock duration of the *final* (successful)
                # generation so the agent can show live tokens/second.  Only
                # success should update metrics — a failed/retried attempt is
                # not a completed step.
                self.last_call_duration = time.monotonic() - fresh_start_time
                return result

            except KeyboardInterrupt:
                sh.on_stream_end()
                partial = sh.get_buffered_text()
                raise InterruptedResponse(partial)

            except LoopDetectedError:
                # A loop was detected mid-stream.  This is not a
                # transient failure — retrying the same request would
                # just stream the same loop again.  Propagate it to the
                # agent, which discards the partial response and redoes
                # the generation.
                sh.on_stream_end()
                raise

            except Exception as e:
                sh.on_stream_end()
                # Authentication failures are never retryable — fail fast
                # with a clear message instead of burning retries on bad
                # credentials.
                if self._is_auth_error(e):
                    raise Exception(
                        f"Authentication failed (HTTP 401): {e}. "
                        "Check that the correct API key is set in the "
                        "environment."
                    ) from e

                classification = self._classify_error(e)

                if classification == RATE_LIMIT:
                    sleep_time = current_delay

                    retry_after = self._extract_retry_after(e)
                    if retry_after is not None:
                        sleep_time = max(retry_after, sleep_time)

                    jitter = sleep_time * 0.25 * (2 * random.random() - 1)
                    sleep_time = max(0, sleep_time + jitter)

                    remaining = self.RETRY_TIMEOUT - (time.monotonic() - start_time)
                    if remaining <= 0:
                        raise Exception(
                            f"Rate-limit retry timeout exceeded ({self.RETRY_TIMEOUT}s)"
                        ) from e
                    sleep_time = min(sleep_time, remaining)

                    sh.on_retry(
                        f"Rate limited — retrying in {sleep_time:.1f}s "
                        f"({remaining:.0f}s remaining)"
                    )
                    time.sleep(sleep_time)
                    current_delay = min(
                        current_delay * self.RETRY_BACKOFF_FACTOR,
                        self.RETRY_MAX_DELAY,
                    )

                else:  # TRANSIENT (or unknown — fail after retries)
                    error_retries += 1
                    if error_retries >= self.MAX_ERROR_RETRIES:
                        raise Exception(
                            f"Maximum retries exceeded ({self.MAX_ERROR_RETRIES}) "
                            f"on response request: {e}"
                        ) from e
                    sh.on_error(
                        f"Attempt {error_retries}/{self.MAX_ERROR_RETRIES} "
                        f"failed: {e}"
                    )
                    time.sleep(self.TRANSIENT_RETRY_DELAY)

    def _is_auth_error(self, error: Exception) -> bool:
        """Return True if *error* is an authentication failure (HTTP 401).

        These are never retryable — retrying the same bad credentials only
        wastes time.  Works across SDKs by inspecting the HTTP status code
        carried on the exception (both the OpenAI and Cerebras SDKs expose
        ``error.status_code`` / ``error.response.status_code``).
        """
        status = getattr(error, "status_code", None)
        if status is None:
            response = getattr(error, "response", None)
            if response is not None:
                status = getattr(response, "status_code", None)
        return status == 401

    def _classify_error(self, error: Exception) -> str:
        """Classify *error* for the retry loop.

        Returns ``RATE_LIMIT`` or ``TRANSIENT``.  The default treats
        every error as transient.  Subclasses should override to detect
        provider-specific rate-limit exceptions.
        """
        return TRANSIENT

    def _extract_retry_after(self, error: Exception) -> float | None:
        """Extract a ``Retry-After`` hint (in seconds) from *error*.

        Returns ``None`` when the error carries no such hint.  The
        default implementation inspects ``error.response.headers``
        which works for both the Anthropic and OpenAI SDKs.
        """
        if hasattr(error, "response") and error.response is not None:
            retry_after = error.response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass
        return None

    # ── Native tool-call logging ─────────────────────────────────────

    def _emit_tool_calls(self) -> None:
        """Drain ``_pending_tool_calls`` through the stream handler.

        Called by ``generate_response`` after the response has been
        fully processed (stream ended, usage recorded).  This harness
        executes textual ``Command:`` lines, not native API tool calls,
        so any tool calls the model emitted are logged (yellow in the
        UI) to alert the user that commands are landing in the wrong
        place, rather than being silently dropped.
        """
        # getattr for subclasses that bypass LLMBackend.__init__ (e.g.
        # test fixtures that construct backends with __new__).
        pending = getattr(self, "_pending_tool_calls", None) or []
        self._pending_tool_calls = []
        for name, arguments in pending:
            self.stream_handler.on_tool_call(name, arguments)

    # ── Abstract methods ─────────────────────────────────────────────

    @abstractmethod
    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        """Send *context* to the model and return the assistant's text reply.

        The method is responsible for:
        * notifying ``self.stream_handler`` of streaming progress,
        * updating ``cost`` and all token-tracking attributes,
        * retry / back-off on transient errors.

        Parameters
        ----------
        system_prompt : str
            The system-level instruction for the model.
        context : list[dict]
            Conversation history in the standard format::

                [{"role": "user"|"assistant", "content": [{"type": "text", "text": "…"}, …]}, …]

        Returns
        -------
        str
            The text content of the model's response.
        """

    @property
    def display_name(self) -> str:
        """Human-readable model name shown in the UI banner.

        Looks up ``self.model`` in the subclass's ``MODEL_DISPLAY_NAMES``
        dict, falling back to the raw model string.  Local/remote models
        get a suffix showing the server location.
        """
        if self.is_local:
            # Show the host:port for remote servers, just "(local)" for localhost
            if self.base_url:
                # Strip protocol prefix for display
                host_display = self.base_url
                for prefix in ("http://", "https://"):
                    if host_display.startswith(prefix):
                        host_display = host_display[len(prefix):]
                        break
                # Recognise all common localhost representations
                _local_prefixes = (
                    "localhost:", "127.0.0.1:", "[::1]:", "[0:0:0:0:0:0:0:1]:",
                )
                if any(host_display.startswith(p) for p in _local_prefixes):
                    return f"{self.model} (local)"
                return f"{self.model} ({host_display})"
            return f"{self.model} (local)"
        return self.MODEL_DISPLAY_NAMES.get(self.model, self.model)

    @property
    def context_window_size(self) -> int:
        """Maximum context window size in tokens for the current model.

        Looks up ``self.model`` in the subclass's ``MODEL_CONTEXT_WINDOWS``
        dict, falling back to ``DEFAULT_CONTEXT_WINDOW``.
        """
        return self.MODEL_CONTEXT_WINDOWS.get(self.model, self.DEFAULT_CONTEXT_WINDOW)

    # ── Optional overrides ───────────────────────────────────────────

    def mark_for_caching(self, message: dict) -> None:
        """Annotate *message* so the backend will cache it on the next call.

        The default implementation is a no-op.  Backends that support
        prompt caching (e.g. Anthropic) should override this to add
        provider-specific cache annotations.
        """

    def trim_cache_blocks(self, context: list[dict], max_blocks: int = 2) -> None:
        """Remove stale prompt-cache markers from *context*.

        The default implementation is a no-op.  Backends that support
        prompt caching (e.g. Anthropic) should override this.
        """

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> float:
        """Return the dollar cost for a single API call.

        The default implementation returns ``0.0``.  Subclasses should
        override with their own pricing table.
        """
        return 0.0

    def record_step_metrics(self) -> None:
        """Fold one completed generation into the session metrics.

        Called by the agent after each successful :meth:`generate_response`
        (including free-form internal turns).  Accumulates the
        tokens/second and cost-per-hour values the UI displays:

        * per-step rate — this step's output tokens over the wall-clock
          duration of this generation (shown in the step header);
        * session rate — all output tokens over all generation time
          (shown in the final metrics panel);
        * session cost rate — dollars spent so far over elapsed
          generation time, i.e. the estimated cost per hour if the task
          kept going at this pace.
        """
        duration = self.last_call_duration
        if duration <= 0:
            return
        self.total_output_tokens += self.last_output_tokens
        self.total_call_duration += duration
        if self.last_output_tokens > 0:
            self.step_rate_tokens_per_sec = self.last_output_tokens / duration
        if self.total_call_duration > 0:
            self.output_rate_tokens_per_sec = (
                self.total_output_tokens / self.total_call_duration
            )
        self.cost_per_hour = self.cost / (self.total_call_duration / 3600)
