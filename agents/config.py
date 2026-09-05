"""
``.agent`` configuration file support.

A small YAML file (``.agent``) lets you pin the LLM backend for a
project — or globally — without touching your shell environment or the
agent YAML configs.  This is the primary mechanism for switching
backends: pick a provider, a model, and (optionally) a custom base URL
in a few lines.

Resolution order (highest precedence wins, key by key):

1. **Project** ``.agent`` — searched upward from the current directory,
   so a file in the repo root governs every subdirectory.
2. **Home** ``~/.agent`` — a global default for all projects.
3. **Environment variables** — ``AGENT_MODEL_PROVIDER``, ``AGENT_MODEL``,
   ``AGENT_BASE_URL``, ``AGENT_TEMPERATURE`` (kept for backward
   compatibility; the file overrides them).

CLI flags (``--provider`` / ``--model``) sit above all of these and are
handled in :mod:`agents.agents`.

Only the recognised keys are kept: ``provider``, ``model``, ``base_url``,
``temperature``.  API keys are deliberately *not* read from this file —
they stay in the environment (see each backend's ``*_API_KEY``).

Example ``.agent``::

    # Use Cerebras' Qwen 3.8 27B
    provider: cerebras
    model: qwen-3.8-27b

Example with a custom OpenAI-compatible endpoint::

    provider: openai
    model: qwen3.8-27b
    base_url: http://localhost:8000/v1
"""

from __future__ import annotations

import os

import yaml

#: File name used at both the project and home level.
AGENT_FILENAME = ".agent"

#: Recognised config keys and the environment variable that maps to each.
_ENV_MAP: dict[str, str] = {
    "provider": "AGENT_MODEL_PROVIDER",
    "model": "AGENT_MODEL",
    "base_url": "AGENT_BASE_URL",
    "temperature": "AGENT_TEMPERATURE",
}

_VALID_KEYS: tuple[str, ...] = ("provider", "model", "base_url", "temperature")


def _find_project_agent(start: str | None = None) -> str | None:
    """Search upward from *start* (default: cwd) for a ``.agent`` file.

    Returns the path of the nearest ``.agent`` found, or ``None``.
    """
    d = os.path.abspath(start or os.getcwd())
    while True:
        candidate = os.path.join(d, AGENT_FILENAME)
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:  # reached filesystem root
            return None
        d = parent


def _home_agent() -> str | None:
    """Return the path of ``~/.agent`` if it exists, else ``None``."""
    candidate = os.path.join(os.path.expanduser("~"), AGENT_FILENAME)
    return candidate if os.path.isfile(candidate) else None


def _read_yaml(path: str) -> dict:
    """Read a YAML file, returning ``{}`` on any error or non-dict."""
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _coerce(key: str, value):
    """Coerce a config value to its expected type."""
    if key == "temperature":
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if isinstance(value, str):
        return value.strip()
    return value


def _merge(target: dict, source: dict) -> None:
    """Merge recognised keys from *source* into *target* (source wins)."""
    for key in _VALID_KEYS:
        if key in source and source[key] is not None:
            target[key] = _coerce(key, source[key])


def load_agent_config(start: str | None = None) -> dict:
    """Return the resolved ``.agent`` configuration.

    Merges, from lowest to highest precedence:

    1. Environment variables
    2. Home ``~/.agent``
    3. Project ``.agent`` (nearest ancestor of *start* that has one)

    Returns a dict containing any of ``provider``, ``model``, ``base_url``,
    ``temperature``.  Missing keys are simply absent, so callers can layer
    their own defaults underneath.
    """
    config: dict = {}

    # Lowest precedence: environment variables.
    for key, env_name in _ENV_MAP.items():
        val = os.environ.get(env_name)
        if val is not None and val != "":
            config[key] = _coerce(key, val)

    # Home config overrides env vars.
    home = _home_agent()
    if home:
        _merge(config, _read_yaml(home))

    # Project config (highest file precedence) overrides home.
    project = _find_project_agent(start)
    if project:
        _merge(config, _read_yaml(project))

    return config


def agent_config_path(start: str | None = None) -> str | None:
    """Return the path of the ``.agent`` file that would be used, or ``None``.

    Useful for diagnostics (``agents --show-config`` style tooling).
    """
    return _find_project_agent(start) or _home_agent()
