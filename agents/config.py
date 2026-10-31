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
   ``AGENT_BASE_URL``, ``AGENT_TEMPERATURE``, ``AGENT_EFFORT`` (kept for
   backward compatibility; the config files override them).

CLI flags (``--provider`` / ``--model``) sit above all of these and are
handled in :mod:`agents.agents`.

Only the recognised keys are kept: ``provider``, ``model``, ``base_url``,
``temperature``, ``effort``.  The project-local ``folder_hash`` metadata
is managed separately by the memory system and is not a backend option.
API keys are deliberately *not* read from these files — they stay in the
environment (see each backend's ``*_API_KEY``).

Example project ``.agent``::

    # Use Cerebras' Qwen 3.8 27B
    provider: cerebras
    model: qwen-3.8-27b

Example global ``~/.agents/agent_config.yaml``::

    provider: openai
    model: qwen3.8-27b
    base_url: http://localhost:8000/v1

Note on ``LOCAL_MODEL``: the ``LOCAL_MODEL`` env var auto-enables
local mode unless ``-m`` names a model or ``-o``/``--online`` is
passed; an explicit ``--local`` flag forces local mode (and errors
cleanly when no model is named).  All routing rules — hosted-model
auto-detection, local traffic always going to the OpenAI-compatible
chat-completions client, a pinned ``base_url`` being the destination —
live in :func:`agents.agents.resolve_backend_target`.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile

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
    "effort": "AGENT_EFFORT",
}

_VALID_KEYS: tuple[str, ...] = ("provider", "model", "base_url",
                                "temperature", "effort")

#: Editable metadata is not inherited or passed to backend configuration.
_WRITABLE_KEYS = _VALID_KEYS + ("folder_hash",)

#: Fallback model per provider, used when no model is pinned anywhere
#: (``-m`` flag, ``AGENT_MODEL`` env var, or config file).  One
#: shared constant so the CLI, ``agent-commit`` and the backend
#: resolution rules (see :func:`agents.agents.resolve_backend_target`)
#: never drift.
PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-5.3-codex",
    "gemini": "gemini-4-argon",
    "kimi": "kimi-k3",
    "deepseek": "deepseek-v4-pro",
    "cerebras": "qwen-3.8-27b",
    "minimax": "MiniMax-M2.5",
}


def _find_project_agent(
    start: str | None = None, *, backend_only: bool = False,
) -> str | None:
    """Search upward from *start* (default: cwd) for a ``.agent`` file.

    Returns the path of the nearest ``.agent`` found, or ``None``.
    Directories named ``.agent`` are skipped (e.g. ``openclaw`` keeps a
    tracked ``.agent/`` directory of workflow files in-tree). Backend lookups
    skip memory-only files so initializing identity does not hide an inherited pin.
    """
    d = os.path.abspath(start or os.getcwd())
    while True:
        candidate = os.path.join(d, AGENT_FILENAME)
        if os.path.isfile(candidate):
            data = _read_yaml(candidate) if backend_only else {}
            memory_only = "folder_hash" in data and not any(
                key in data for key in _VALID_KEYS
            )
            if not memory_only:
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
    3. Project ``.agent`` (nearest ancestor, skipping memory-only files)

    Returns a dict containing any of ``provider``, ``model``,
    ``base_url``, ``temperature``, ``effort``.  Missing keys are simply
    absent, so callers can layer their own defaults underneath.
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
    project = _find_project_agent(start, backend_only=True)
    if project:
        _merge(config, _read_yaml(project))

    return config


def agent_config_path(start: str | None = None) -> str | None:
    """Return the path of the config file that would be used, or ``None``.

    The nearest non-memory-only project ``.agent`` if one exists, otherwise the
    canonical global config path.  Useful for diagnostics (``agents
    --show-config`` style tooling).
    """
    return _find_project_agent(start, backend_only=True) or _home_agent()


DEFAULT_CONFIG_COMMENT = "# Agents backend configuration"


def global_config_path() -> str:
    """Return the canonical global config path: ``~/.agents/agent_config.yaml``."""
    return _global_new()


def project_config_path(start: str | None = None) -> str | None:
    """Return the nearest existing project config path, if any."""
    return _find_project_agent(start)


def _format_yaml_scalar(value) -> str:
    """Produce exactly one YAML line, including for strings containing newlines."""
    if isinstance(value, str):
        return yaml.safe_dump(
            value, default_style='"', allow_unicode=False, width=2**31 - 1
        ).strip()
    return yaml.safe_dump(value).splitlines()[0]


def _config_mapping(text: str) -> dict[str, tuple]:
    """Validate editable YAML without changing the permissive config loader."""
    if any(char in text for char in ("\x85", "\u2028", "\u2029", "\ufeff")):
        raise ValueError("unsupported YAML line separator or byte-order mark")
    if "\r" in text.replace("\r\n", ""):
        raise ValueError("bare carriage-return line endings are unsupported")
    unsupported = (
        yaml.tokens.AnchorToken, yaml.tokens.AliasToken, yaml.tokens.TagToken,
        yaml.tokens.DirectiveToken, yaml.tokens.DocumentStartToken,
        yaml.tokens.DocumentEndToken,
    )
    if any(isinstance(token, unsupported) for token in yaml.scan(text)):
        raise ValueError("anchors, aliases, tags and document markers are unsupported")
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if root is None:
        return {}
    if not isinstance(root, yaml.MappingNode) or root.flow_style:
        raise ValueError("backend config must be a block YAML mapping")

    def check(node):
        if isinstance(node, yaml.MappingNode):
            seen = set()
            for key, value in node.value:
                if not isinstance(key, yaml.ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                    raise ValueError("YAML mapping keys must be strings")
                if key.value in seen:
                    raise ValueError(f"duplicate YAML key {key.value!r}")
                seen.add(key.value)
                check(value)
        elif isinstance(node, yaml.SequenceNode):
            for value in node.value:
                check(value)

    check(root)
    yaml.safe_load(text)  # Reject unsupported values/tags before touching the file.
    result = {}
    for key, value in root.value:
        if key.start_mark.column != 0 or key.start_mark.line != key.end_mark.line:
            raise ValueError("top-level keys must be single-line, unindented keys")
        # Refuse explicit-key syntax and unusual separators, even for other keys.
        separator = text[key.end_mark.index:value.start_mark.index]
        if not separator.startswith(":") and not re.match(r"^[ \t]+:", separator):
            raise ValueError("explicit YAML mapping keys are unsupported")
        result[key.value] = (key, value)
    return result


def _check_config_path(path: str) -> os.stat_result | None:
    """Refuse symlinks (including parent directories) and nonregular targets."""
    parent = os.path.dirname(path)
    while True:
        try:
            info = os.lstat(parent)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"config parent is not a real directory: {parent}")
        ancestor = os.path.dirname(parent)
        if ancestor == parent:
            break
        parent = ancestor
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"config target is not a regular file: {path}")
    return info


def write_agent_config_keys(
    path: str,
    *,
    set_keys: dict[str, object] | None = None,
    remove_keys: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Atomically update backend keys or folder_hash, preserving other text.

    Key names in requests are case-insensitive. None removes a key; an explicit
    non-None assignment wins over remove_keys. Inline comments on removed keys
    become standalone comments. Returns (path, written_keys, removed_keys).

    This conservative editor refuses duplicate/non-string mapping keys, root
    flow mappings, anchors/aliases, explicit tags/document markers, indented or
    explicit root keys, BOMs/unusual line endings, and multiline/collection values
    being changed. LF and CRLF are supported. Unrelated
    nested mappings and multiline values are preserved verbatim. Existing
    permissions are retained; new files are private (0600). Symlinks, including
    symlinked parents, are refused. Failures raise ValueError without replacing
    the original file. This is not a locking API for concurrent writers.
    """
    def normalize(key):
        if not isinstance(key, str) or key.lower() not in _WRITABLE_KEYS:
            raise ValueError(f"unknown backend config key {key!r}")
        return key.lower()

    if set_keys is not None and not isinstance(set_keys, dict):
        raise ValueError("set_keys must be a mapping")
    if remove_keys is not None and not isinstance(remove_keys, (tuple, list)):
        raise ValueError("remove_keys must be a tuple or list")

    def fingerprint(info):
        if info is None:
            return None
        return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns)

    assignments = {}
    removals = []
    seen = set()
    for key, value in (set_keys or {}).items():
        key = normalize(key)
        if key in seen:
            raise ValueError(f"duplicate backend config key {key!r}")
        seen.add(key)
        if value is None:
            removals.append(key)
        else:
            if type(value) not in (str, bool, int, float):
                raise ValueError(f"backend config value for {key!r} must be scalar")
            assignments[key] = _coerce(key, value)
    for key in remove_keys or ():
        key = normalize(key)
        if key not in removals:
            removals.append(key)

    target = os.path.abspath(path)
    temp_path = None
    try:
        original = _check_config_path(target)
        if not assignments and not removals:
            return path, (), ()
        text = ""
        if original is not None:
            # Do not follow a target swapped for a symlink after lstat.
            fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "r", encoding="utf-8", newline="") as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or fingerprint(opened) != fingerprint(original):
                    raise ValueError("backend config changed while opening")
                text = stream.read()

        nodes = _config_mapping(text)
        edits = []
        newline = "\r\n" if "\r\n" in text else "\n"
        for name in assignments.keys() | set(removals):
            if name not in nodes:
                continue
            key, value = nodes[name]
            if (not isinstance(value, yaml.ScalarNode)
                    or value.style in ("|", ">")
                    or key.start_mark.line != value.start_mark.line
                    or value.start_mark.line != value.end_mark.line):
                raise ValueError(f"cannot edit multiline or collection value for {name!r}")
            start, end = value.start_mark.index, value.end_mark.index
            if name in assignments:
                replacement = _format_yaml_scalar(assignments[name])
                if text[start - 1:start] == ":":
                    replacement = " " + replacement
                edits.append((start, end, replacement))
            else:
                line_end = text.find("\n", end)
                line_end = len(text) if line_end < 0 else line_end + 1
                suffix = text[end:line_end].lstrip(" \t")
                comment = suffix if suffix.startswith("#") else ""
                edits.append((key.start_mark.index, line_end, comment))

        content = text
        for start, end, replacement in sorted(edits, reverse=True):
            content = content[:start] + replacement + content[end:]
        additions = [
            f"{name}: {_format_yaml_scalar(assignments[name])}"
            for name in _WRITABLE_KEYS if name in assignments and name not in nodes
        ]
        if additions:
            if not content:
                content = DEFAULT_CONFIG_COMMENT + newline
            elif not content.endswith("\n"):
                content += newline
            content += newline.join(additions) + newline

        written = tuple(name for name in _WRITABLE_KEYS if name in assignments)
        removed = tuple(name for name in removals if name in nodes)
        if content == text:
            return path, written, removed
        # Verify that editing did not produce unexpected YAML before any write.
        _config_mapping(content)
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        _check_config_path(target)
        fd, temp_path = tempfile.mkstemp(prefix=f".{os.path.basename(target)}.",
                                         suffix=".tmp", dir=parent)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            if original is not None:
                os.fchmod(stream.fileno(), stat.S_IMODE(original.st_mode))
            os.fsync(stream.fileno())
        # Catch ordinary concurrent modification and target/symlink swaps.
        current = _check_config_path(target)
        if fingerprint(current) != fingerprint(original):
            raise ValueError("backend config changed while writing")
        os.replace(temp_path, target)
        temp_path = None
        return path, written, removed
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"cannot write backend config {path!r}: {exc}") from exc
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
