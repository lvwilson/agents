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
* ``~/.agents/agent_config.yaml`` — **global** config, living inside the
  agent state directory (next to ``sessions/``, ``memory/`` and
  ``browser_profile/``).
* ``~/.agent`` — **legacy** global location (a bare file in the home
  directory).  Still read for compatibility, but deprecated: the first
  CLI run after this convention changes moves it to the new location
  (one-time, never deleted, never overwritten — see
  :func:`migrate_legacy_home_agent` and the ``HOME_AGENT_*_NOTICE``
  constants).  If both locations carry content, a warning is shown and
  the new location wins.

Resolution order (highest precedence wins, key by key):

1. **Project** ``.agent`` (nearest ancestor of the working directory).
2. **Global** ``~/.agents/agent_config.yaml``, falling back to the
   legacy ``~/.agent`` file.
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
import shutil

import yaml

#: File name used at the project level (searched upward from the cwd).
AGENT_FILENAME = ".agent"

#: File name of the **global** config, stored inside the agent state
#: directory (``~/.agents/``) next to sessions, memory and the browser
#: profile.  Keeps the home directory free of lone agent dotfiles.
HOME_AGENT_FILENAME = "agent_config.yaml"

#: Legacy global location: a bare ``.agent`` file in the home directory.
#: Still read (lowest preference of the two global locations) and
#: automatically moved to the new location on the first CLI run.
LEGACY_HOME_AGENT_FILENAME = ".agent"

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
    """Path of the new global config: ``~/.agents/agent_config.yaml``."""
    return os.path.join(
        os.path.expanduser("~"), ".agents", HOME_AGENT_FILENAME
    )


def _global_legacy() -> str:
    """Path of the legacy global config: ``~/.agent`` in the home dir."""
    return os.path.join(
        os.path.expanduser("~"), LEGACY_HOME_AGENT_FILENAME
    )


def _global_candidates() -> tuple[str, str]:
    """``(new, legacy)`` global config paths, highest precedence first."""
    return _global_new(), _global_legacy()


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
        * ``"ok"`` — a global config file (new or legacy location)
          exists and will be loaded.
        * ``"missing"`` — no global config anywhere.
        * ``"squat"`` — a global config path exists but is not a regular
          file (e.g. a stale directory squatting the name), so nothing
          usable can be loaded from it.
        * ``"multi"`` — both the new ``~/.agents/agent_config.yaml`` and
          the legacy ``~/.agent`` carry content; the new one wins and a
          warning (``HOME_AGENT_MULTI_WARNING``) explains the fix.

    The one-time ``~/.agent`` → ``~/.agents/agent_config.yaml`` move is
    triggered by :func:`migrate_legacy_home_agent` (called from the CLI
    entry points), not by this status check, so checking state has no
    filesystem side effects.
    """
    new_path, legacy_path = _global_candidates()
    new_is_file = os.path.isfile(new_path)
    legacy_is_file = os.path.isfile(legacy_path)
    if new_is_file and legacy_is_file:
        return "multi"
    for path in (new_path, legacy_path):
        if not os.path.exists(path):
            continue
        if os.path.isfile(path):
            return "ok"
        return "squat"
    return "missing"


#: Visible notice (startup) when a legacy ``~/.agent`` file is moved to
#: the new ``~/.agents/agent_config.yaml`` location.  The file is moved,
#: never deleted or copied twice, so the notice fires at most once.
HOME_AGENT_MOVED_NOTICE = (
    "Moved legacy global config from ~/.agent to "
    "~/.agents/agent_config.yaml"
)

#: Warning (startup) when both the new and the legacy global config
#: carry content: the legacy file cannot be auto-migrated without
#: risking an overwrite, so the new one wins and the leftover legacy
#: file is left for the user to inspect and remove.
HOME_AGENT_MULTI_WARNING = (
    "Both ~/.agents/agent_config.yaml (new) and ~/.agent (legacy) exist — "
    "the new location is in use and ~/.agent was left untouched; remove "
    "~/.agent or merge its provider/model/base_url/temperature values "
    "into ~/.agents/agent_config.yaml."
)

#: Warning shown at startup when the global backend config path is
#: squatted by a non-file, most commonly the 2024 stale venv directory
#: that once squatted ``~/.agent``: without the notice the global pin
#: silently disappears and every project without its own ``.agent``
#: file falls back to the default backend.
HOME_AGENT_SQUAT_WARNING = (
    "A global backend config path exists but is not a regular file, so "
    "the global .agent config is being IGNORED - every folder without "
    "its own .agent file falls back to the default backend. The global "
    "config lives at ~/.agents/agent_config.yaml (the legacy ~/.agent "
    "location is deprecated): rename whatever directory squats the name "
    "and create a real config file (e.g. 'provider: cerebras / "
    "model: qwen-3.8-27b')."
)


def migrate_legacy_home_agent() -> str | None:
    """Perform the one-time ``~/.agent`` → ``~/.agents/agent_config.yaml`` move.

    Rules (safety first):

    * The file is **moved**, never deleted or duplicated — the legacy
      path disappears and the new path holds the same bytes.
    * An existing ``~/.agents/agent_config.yaml`` is **never
      overwritten**: a leftover legacy file (or a hand-crafted legacy
      file after the switch) only yields the ``HOME_AGENT_MULTI_WARNING``
      notice.
    * Migration runs only when the legacy path is a regular file; a
      squatted ``~/.agent`` (e.g. a directory) is left alone — the
      squat warning covers it.

    Returns ``HOME_AGENT_MOVED_NOTICE`` when a move happened,
    ``HOME_AGENT_MULTI_WARNING`` when the legacy leftover is kept, and
    ``None`` when there is nothing to do (typical on machines that
    never used the legacy location).
    """
    legacy_path = _global_legacy()
    if not os.path.isfile(legacy_path):
        return None
    new_path = _global_new()
    if os.path.isfile(new_path):
        return HOME_AGENT_MULTI_WARNING
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    shutil.move(legacy_path, new_path)
    return HOME_AGENT_MOVED_NOTICE


def report_home_config(verbose: bool = True) -> None:
    """Print startup messages for the global config, in the right order.

    Runs :func:`migrate_legacy_home_agent` first (so the status check
    inspects the post-migration state), then reports ``"squat"`` and
    ``"multi"`` states.  ``"ok"`` after an automatic move also reports
    the move, so the user always sees where their config went.
    ``verbose=False`` silences everything (for tests / library use).
    """
    from . import ui as _ui  # local import: keeps config import-light

    moved = migrate_legacy_home_agent()
    status = home_agent_status()
    message: str | None = None
    style = "warning"
    if status == "squat":
        message = HOME_AGENT_SQUAT_WARNING
    elif status == "multi":
        message = HOME_AGENT_MULTI_WARNING
    elif moved == HOME_AGENT_MOVED_NOTICE:
        # Status is "ok" now; the "moved" notice explains why.
        message = moved
        style = "info"
    if message is not None and verbose:
        _ui.safe_console_print(message, style=style)


def _home_agent() -> str | None:
    """Return the global config path to load, or ``None``.

    New location first (``~/.agents/agent_config.yaml``), then the
    legacy ``~/.agent`` file.  A squatted (non-file) path is skipped —
    nothing usable can be loaded from it either way.
    """
    for path in _global_candidates():
        if os.path.isfile(path):
            return path
    return None


def load_agent_config(start: str | None = None) -> dict:
    """Return the resolved ``.agent`` configuration.

    Merges, from lowest to highest precedence:

    1. Environment variables
    2. Global config: ``~/.agents/agent_config.yaml``, falling back to
       the legacy ``~/.agent`` file
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

    # Global config (new location wins over the legacy file).
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
    global config path (new location, then legacy).  Useful for
    diagnostics (``agents --show-config`` style tooling).
    """
    return _find_project_agent(start) or _home_agent()
