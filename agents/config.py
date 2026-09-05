"""
``.agent`` configuration file support.

A small YAML file (``.agent``) lets you pin the LLM backend for a
project — or globally — without touching your shell environment or the
agent YAML configs.  This is the primary mechanism for switching
backends: pick a provider, a model, and (optionally) a custom base URL
in a few lines.

Configuration locations and their roles:

* ``.agent`` — **project** config.  Searched upward from the current
  directory, so a file in the repo root governs every subdirectory.
* ``~/.agents/agent_config.yaml`` — **global** config, the single
  canonical home for a cross-project backend pin, living inside the
  agent state directory next to ``sessions/``, ``memory/`` and
  ``browser_profile/``.

The old global location (a bare ``~/.agent`` file in the home
directory) has been removed from the code and is no longer read:
stray files at that path are simply ignored, so a stale or squatted
``~/.agent`` can neither pin a backend nor silence one.  Move it to
``~/.agents/agent_config.yaml`` if it still holds a pin you want.

Resolution order (highest precedence wins, key by key):

1. **Project** ``.agent`` (nearest ancestor of the working directory).
2. **Global** ``~/.agents/agent_config.yaml``.
3. **Environment variables** — ``AGENT_MODEL_PROVIDER``, ``AGENT_MODEL``,
   ``AGENT_BASE_URL``, ``AGENT_TEMPERATURE`` (kept for backward
   compatibility; the config files override them).

CLI flags (``--provider`` / ``--model``) sit above all of these and are
handled in :mod:`agents.agents`.

Only the recognised keys are kept: ``provider``, ``model``, ``base_url``,
``temperature``.  API keys are deliberately *not* read from these files —
they stay in the environment (see each backend's ``*_API_KEY``).

Example project ``.agent``::

    # Use Cerebras' Qwen 3.8 27B
    provider: cerebras
    model: qwen-3.8-27b

Example global ``~/.agents/agent_config.yaml``::

    provider: openai
    model: qwen3.8-27b
    base_url: http://localhost:8000/v1

Note on ``LOCAL_MODEL``: the auto-enable of local mode by the
``LOCAL_MODEL`` env var (see :func:`agents.agents.main`) yields to any
``provider`` or ``model`` pinned in config — an explicit ``--local``
flag still forces local mode.
"""

from __future__ import annotations

import os

import yaml

#: File name used at the project level (searched upward from the cwd).
AGENT_FILENAME = ".agent"

#: File name of the **global** config, stored inside the agent state
#: directory (``~/.agents/``) next to sessions, memory and the browser
#: profile.  This is the single canonical home for the cross-project
#: backend pin.
HOME_AGENT_FILENAME = "agent_config.yaml"

#: Recognised config keys and the environment variable that maps to each.
_ENV_MAP: dict[str, str] = {
    "provider": "AGENT_MODEL_PROVIDER",
    "model": "AGENT_MODEL",
    "base_url": "AGENT_BASE_URL",
    "temperature": "AGENT_TEMPERATURE",
}

_VALID_KEYS: tuple[str, ...] = ("provider", "model", "base_url", "temperature")

#: Fallback model per provider, used when no model is pinned anywhere
#: (``-m`` flag, ``AGENT_MODEL`` env var, or config file).  One
#: shared constant so the CLI and ``agent-commit`` never drift.
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-5.3-codex",
    "gemini": "gemini-3.1-pro-preview",
    "kimi": "kimi-k3",
    "deepseek": "deepseek-v4-pro",
    "cerebras": "qwen-3.8-27b",
}


def _find_project_agent(start: str | None = None) -> str | None:
    """Search upward from *start* (default: cwd) for a ``.agent`` file.

    Returns the path of the nearest ``.agent`` found, or ``None``.
    Directories named ``.agent`` are skipped (e.g. ``openclaw`` keeps a
    tracked ``.agent/`` directory of workflow files in-tree).
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


def _global_new() -> str:
    """Path of the canonical global config:
    ``~/.agents/agent_config.yaml``."""
    return os.path.join(
        os.path.expanduser("~"), ".agents", HOME_AGENT_FILENAME
    )


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


def home_agent_status() -> str:
    """Report the state of the global backend config.

    Returns:
        * ``"ok"`` — ``~/.agents/agent_config.yaml`` is a regular file
          and will be loaded.
        * ``"missing"`` — no global config (nothing to load; env vars
          and provider defaults apply).
        * ``"squat"`` — the path exists but is not a regular file
          (e.g. a stale directory squatting the name); nothing usable
          can be loaded, and :data:`HOME_AGENT_SQUAT_WARNING` explains
          the fix at startup.
    """
    path = _global_new()
    if os.path.isfile(path):
        return "ok"
    if os.path.exists(path):
        return "squat"
    return "missing"


#: Warning shown at startup when the global backend config path is
#: squatted by a non-file, so the reason "the global pin does nothing"
#: is visible instead of a silent fallback to the default backend.
HOME_AGENT_SQUAT_WARNING = (
    "The global backend config path ~/.agents/agent_config.yaml exists "
    "but is not a regular file, so the global .agent config is being "
    "IGNORED - every folder without its own .agent file falls back to "
    "the default backend. Rename whatever squats this path and create a "
    "real config file (e.g. 'provider: cerebras / "
    "model: qwen-3.8-27b')."
)


def report_home_config(verbose: bool = True) -> None:
    """Print startup messages for the global config.

    Reports the ``"squat"`` state so a squatted global config path
    cannot silently drop the backend pin for every project.
    ``verbose=False`` silences output (for tests / library use).
    """
    from . import ui as _ui  # local import: keeps config import-light

    if (
        home_agent_status() == "squat"
        and verbose
    ):
        _ui.safe_console_print(HOME_AGENT_SQUAT_WARNING, style="warning")


def _home_agent() -> str | None:
    """Return the canonical global config path to load, or ``None``.

    ``~/.agents/agent_config.yaml`` is the only global location the
    loader reads — a squatted (non-file) path is skipped, and the
    retired ``~/.agent`` home-dir file is not consulted at all.
    """
    path = _global_new()
    return path if os.path.isfile(path) else None


def load_agent_config(start: str | None = None) -> dict:
    """Return the resolved ``.agent`` configuration.

    Merges, from lowest to highest precedence:

    1. Environment variables
    2. Global config: ``~/.agents/agent_config.yaml``
    3. Project ``.agent`` (nearest ancestor of *start* that has one)

    Returns a dict containing any of ``provider``, ``model``,
    ``base_url``, ``temperature``.  Missing keys are simply absent, so
    callers can layer their own defaults underneath.
    """
    config: dict = {}

    # Lowest precedence: environment variables.
    for key, env_name in _ENV_MAP.items():
        val = os.environ.get(env_name)
        if val is not None and val != "":
            config[key] = _coerce(key, val)

    # Global config (single canonical location).
    home = _home_agent()
    if home:
        _merge(config, _read_yaml(home))

    # Project config (highest file precedence) overrides global.
    project = _find_project_agent(start)
    if project:
        _merge(config, _read_yaml(project))

    return config


def agent_config_path(start: str | None = None) -> str | None:
    """Return the path of the config file that would be used, or ``None``.

    The nearest project ``.agent`` file if one exists, otherwise the
    canonical global config path.  Useful for diagnostics (``agents
    --show-config`` style tooling).
    """
    return _find_project_agent(start) or _home_agent()
