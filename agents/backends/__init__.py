"""
Lazy-loading backend registry.

Each provider module is imported only when first requested, keeping
startup fast and avoiding hard dependencies on SDKs the user hasn't
installed.

Usage
-----
::

    from backends import create_backend

    client = create_backend("anthropic", model="claude-sonnet-4-5-20250929")
    client = create_backend("openai",    model="gpt-4o", base_url="http://localhost:8000")
    client = create_backend("gemini",    model="gemini-3.1-pro-preview")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_backend import LLMBackend, StreamHandler

# Maps provider name → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "anthropic": ("backends.anthropic_backend", "AnthropicBackend"),
    "openai":    ("backends.openai_backend",    "OpenAIBackend"),
    "gemini":    ("backends.gemini_backend",    "GeminiBackend"),
}

# Cache of already-imported classes so we import each module at most once.
_CLASS_CACHE: dict[str, type] = {}


def _load_class(provider: str) -> type:
    """Import and return the backend class for *provider* (lazy)."""
    if provider in _CLASS_CACHE:
        return _CLASS_CACHE[provider]

    if provider not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown LLM provider {provider!r}. "
            f"Available providers: {available}"
        )

    module_path, class_name = _REGISTRY[provider]
    import importlib
    module = importlib.import_module(module_path, package=__name__)
    cls = getattr(module, class_name)
    _CLASS_CACHE[provider] = cls
    return cls


def create_backend(
    provider: str,
    *,
    model: str,
    base_url: str | None = None,
    cache_step: int = 2,
    stream_handler: "StreamHandler | None" = None,
    **kwargs,
) -> "LLMBackend":
    """Instantiate an LLM backend by provider name.

    Parameters
    ----------
    provider : str
        One of the registered provider names (``"anthropic"``, …).
    model : str
        Model identifier to pass to the backend.
    base_url : str | None
        Optional override URL (e.g. for local / self-hosted inference).
    cache_step : int
        How often to place prompt-cache markers (Anthropic-specific;
        ignored by backends that don't support it).  Default is ``2``
        meaning a cache block is placed every 2 API calls.
    stream_handler : StreamHandler | None
        Optional callback handler for streaming events.  When ``None``
        a silent no-op handler is used (headless mode).  Pass a
        ``RichStreamHandler`` for interactive terminal output.
    **kwargs
        Forwarded to the backend constructor.
    """
    cls = _load_class(provider)
    return cls(
        model=model,
        base_url=base_url,
        cache_step=cache_step,
        stream_handler=stream_handler,
        **kwargs,
    )
