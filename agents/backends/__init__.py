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
    from ..llm_backend import LLMBackend, StreamHandler

# Maps provider name → (module_path, class_name)
_REGISTRY: dict[str, tuple[str, str]] = {
    "anthropic": (".anthropic_backend", "AnthropicBackend"),
    "cerebras":  (".cerebras_backend",  "CerebrasBackend"),
    "deepseek":  (".deepseek_backend",  "DeepSeekBackend"),
    "gemini":    (".gemini_backend",    "GeminiBackend"),
    "kimi":      (".kimi_backend",      "KimiBackend"),
    "minimax":   (".minimax_backend",   "MinimaxBackend"),
    "openai":    (".openai_backend",    "OpenAIBackend"),
}

# When a provider is given a custom ``base_url`` (local / self-hosted /
# proxy), some providers should use an OpenAI-compatible *chat-completions*
# backend instead of their hosted one.  The hosted ``openai`` backend uses
# the Responses API, which most local servers (vLLM, llama.cpp, Ollama, …)
# do not implement — they implement chat completions.  So ``openai`` + a
# base_url routes to the shared chat-completions backend.
_BASE_URL_OVERRIDES: dict[str, tuple[str, str]] = {
    "openai": (".openai_compat_backend", "OpenAICompatBackend"),
}

# Cache of already-imported classes so we import each module at most once.
# Keyed by (provider, base_url_override) so the hosted and custom-URL
# variants of a provider are cached independently.
_CLASS_CACHE: dict[tuple[str, bool], type] = {}


def _load_class(provider: str, base_url: str | None = None) -> type:
    """Import and return the backend class for *provider* (lazy).

    When *base_url* is set and the provider has a chat-completions
    override (see ``_BASE_URL_OVERRIDES``), that class is returned
    instead of the hosted one.
    """
    use_override = bool(base_url) and provider in _BASE_URL_OVERRIDES
    cache_key = (provider, use_override)
    if cache_key in _CLASS_CACHE:
        return _CLASS_CACHE[cache_key]

    if provider not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown LLM provider {provider!r}. "
            f"Available providers: {available}"
        )

    if use_override:
        module_path, class_name = _BASE_URL_OVERRIDES[provider]
    else:
        module_path, class_name = _REGISTRY[provider]
    import importlib
    module = importlib.import_module(module_path, package=__name__)
    cls = getattr(module, class_name)
    _CLASS_CACHE[cache_key] = cls
    return cls


def list_available_models(provider_filter: str | None = None) -> list[dict]:
    """Return model information from all registered backends.

    Parameters
    ----------
    provider_filter : str or None
        If given, only return models from this provider.

    Returns
    -------
    list[dict]
        Each dict has keys: provider, model, display, input_cost,
        output_cost, cache_read_cost, context.  Costs are in dollars
        per million tokens; context is the context-window size in
        tokens.  ``None`` values indicate unavailable information.
    """
    results: list[dict] = []
    for provider_name in sorted(_REGISTRY):
        if provider_filter and provider_name != provider_filter:
            continue
        try:
            cls = _load_class(provider_name)
        except Exception:
            continue

        # Use __dict__ rather than getattr so subclassed backends are
        # not credited with models they merely inherit — DeepSeekBackend
        # subclasses AnthropicBackend, so a plain getattr would list all
        # nine Claude models under the "deepseek" provider too.
        pricing: dict = cls.__dict__.get("MODEL_PRICING", {}) or {}
        contexts: dict = cls.__dict__.get("MODEL_CONTEXT_WINDOWS", {}) or {}
        displays: dict = cls.__dict__.get("MODEL_DISPLAY_NAMES", {}) or {}

        all_models = set(pricing.keys()) | set(contexts.keys()) | set(displays.keys())

        for model_name in sorted(all_models):
            price = pricing.get(model_name, {})
            results.append(
                {
                    "provider": provider_name,
                    "model": model_name,
                    "display": displays.get(model_name, model_name),
                    "input_cost": price.get("input_token_cost"),
                    "output_cost": price.get("output_token_cost"),
                    "cache_read_cost": price.get("cache_read_cost"),
                    "context": contexts.get(model_name),
                }
            )
    return results


def create_backend(
    provider: str,
    *,
    model: str,
    base_url: str | None = None,
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
    stream_handler : StreamHandler | None
        Optional callback handler for streaming events.  When ``None``
        a silent no-op handler is used (headless mode).  Pass a
        ``RichStreamHandler`` for interactive terminal output.
    **kwargs
        Forwarded to the backend constructor.  Backends that support
        prompt caching (Anthropic, Gemini) define their own
        ``cache_step`` default in their constructor.
    """
    cls = _load_class(provider, base_url=base_url)
    return cls(
        model=model,
        base_url=base_url,
        stream_handler=stream_handler,
        **kwargs,
    )
