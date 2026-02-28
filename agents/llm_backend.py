"""
Abstract base class for LLM backends.

Every backend (Anthropic, OpenAI, Gemini, …) implements this interface.
Backends are lazily loaded via the factory in ``backends/__init__.py``.
"""

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """Unified interface for large-language-model providers.

    Subclasses must implement ``generate_response`` and ``display_name``.
    Token-tracking and cost attributes have sensible defaults so that
    backends which don't support them still satisfy the interface.
    """

    # Retry configuration — shared defaults for all backends
    RETRY_TIMEOUT = 300        # 5 minutes overall timeout for rate-limit retries
    RETRY_BASE_DELAY = 1       # Initial backoff delay in seconds
    RETRY_MAX_DELAY = 60       # Maximum backoff delay in seconds
    RETRY_BACKOFF_FACTOR = 2   # Exponential backoff multiplier
    MAX_ERROR_RETRIES = 3      # Fixed retry limit for non-rate-limit errors

    def __init__(self, model: str, base_url: str | None = None):
        self.model: str = model
        self.base_url: str | None = base_url
        self.is_local: bool = base_url is not None

        # Running totals
        self.cost: float = 0.0
        self.cost_without_cache: float = 0.0
        self.call_count: int = 0

        # Per-call token bookkeeping
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.last_total_context_tokens: int = 0
        self.peak_context_tokens: int = 0

    # ── Abstract methods ─────────────────────────────────────────────

    @abstractmethod
    def generate_response(self, system_prompt: str, context: list[dict]) -> str:
        """Send *context* to the model and return the assistant's text reply.

        The method is responsible for:
        * streaming output to the console (via ``ui`` helpers),
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
