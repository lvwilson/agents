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

_T = TypeVar("_T")


class StreamHandler:
    """Callback interface for streaming events from backends.

    Backends call these methods to report streaming progress.  The default
    implementation is a silent no-op so that backends work headlessly
    without any UI dependency.  Pass a ``RichStreamHandler`` (from ``ui``)
    for interactive terminal output.
    """

    def on_stream_start(self) -> None:
        """Called once before the first token of a new API call."""

    def on_stream_token(self, token: str) -> None:
        """Called for each streamed token/chunk of text."""

    def on_stream_end(self) -> None:
        """Called after the last token (or if no tokens were received)."""

    def on_retry(self, message: str) -> None:
        """Called when a retryable error occurs (rate-limit or transient)."""

    def on_error(self, message: str) -> None:
        """Called when a non-retryable attempt fails."""


# Convenience alias — a handler that does nothing.
NullStreamHandler = StreamHandler


# ── Error classification constants ───────────────────────────────────
RATE_LIMIT = "rate_limit"
TRANSIENT = "transient"


class LLMBackend(ABC):
    """Unified interface for large-language-model providers.

    Subclasses must implement ``generate_response`` and ``display_name``.
    Token-tracking and cost attributes have sensible defaults so that
    backends which don't support them still satisfy the interface.

    The ``_run_with_retries`` template method provides shared retry /
    back-off logic.  Backends customise behaviour by overriding
    ``_classify_error`` and optionally ``_extract_retry_after``.
    """

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
    ):
        self.model: str = model
        self.base_url: str | None = base_url
        self.is_local: bool = base_url is not None
        self.stream_handler: StreamHandler = stream_handler or NullStreamHandler()

        # Running totals
        self.cost: float = 0.0
        self.cost_without_cache: float = 0.0
        self.call_count: int = 0

        # Per-call token bookkeeping
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_total_context_tokens: int = 0
        self.peak_context_tokens: int = 0

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

        while True:
            try:
                sh.on_stream_start()
                result = attempt_fn()
                sh.on_stream_end()
                return result

            except Exception as e:
                sh.on_stream_end()
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
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable model name shown in the UI banner."""

    # ── Optional overrides ───────────────────────────────────────────

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
