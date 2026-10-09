"""Persist a backend default without changing the running agent."""

import argparse
import os
from urllib.parse import urlsplit

from ..config import (
    AGENT_FILENAME,
    global_config_path,
    project_config_path,
    write_agent_config_keys,
)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _parser():
    parser = _ArgumentParser(prog="set_default_model", add_help=False, allow_abbrev=False)
    parser.add_argument("model")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--scope", choices=("global", "project"), default="global")
    scope.add_argument("--global", dest="scope", action="store_const", const="global")
    scope.add_argument("--project", dest="scope", action="store_const", const="project")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--online", action="store_true")
    parser.add_argument("--provider")
    parser.add_argument("--base-url")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    return parser


def _validate_url(url):
    try:
        parsed = urlsplit(url)
        valid = (
            parsed.scheme in ("http", "https") and parsed.hostname
            and not parsed.username and not parsed.password
            and not parsed.query and not parsed.fragment
            and (parsed.port is None or 1 <= parsed.port <= 65535)
            and not any(char.isspace() or ord(char) < 32 for char in url)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "--base-url must be an absolute HTTP(S) API URL without "
            "credentials, query parameters, or a fragment"
        )


def set_default_model(*args):
    """Tool entry point; accepts model and CLI-style selection options."""
    parser = _parser()
    if args in (("--help",), ("-h",)):
        return parser.format_help()
    try:
        options = parser.parse_args(args)
        return _save(options)
    except (ValueError, OSError, argparse.ArgumentError) as exc:
        return f"Error: {exc}\nUsage: {parser.format_usage().strip()}"


def _save(options):
    # Import lazily: agents.py imports the tool dispatcher during startup.
    from ..agents import (
        _local_base_url,
        _looks_local_base_url,
        _online_models,
        resolve_backend_target,
    )
    from ..backends import _REGISTRY

    model = options.model.strip()
    if not model or any(ord(char) < 32 for char in model):
        raise ValueError("model must be a nonempty, single-line name")
    provider = options.provider
    if provider is not None:
        provider = provider.strip().lower()
        if provider not in _REGISTRY:
            raise ValueError(f"unknown provider; choose from {', '.join(sorted(_REGISTRY))}")

    url = options.base_url
    if url is not None:
        _validate_url(url)
    if url is not None and (options.host is not None or options.port is not None):
        raise ValueError("--base-url cannot be combined with --host or --port")

    hosting = _online_models().get(model)
    local = options.local or (
        not options.online and provider is None and hosting is None and url is None
    )
    if options.online and url and _looks_local_base_url(url):
        raise ValueError("--online conflicts with a loopback --base-url")
    if options.online and not hosting and provider is None:
        raise ValueError("unknown online model; specify --provider")
    if (options.host is not None or options.port is not None) and not local:
        raise ValueError("--host and --port require a local selection (use --local)")
    if local:
        if provider not in (None, "openai", "anthropic"):
            raise ValueError("--local supports --provider openai or anthropic")
        if url is None:
            host = options.host if options.host is not None else os.getenv("LOCAL_LLM_HOST", "localhost")
            try:
                port = options.port if options.port is not None else int(os.getenv("LOCAL_LLM_PORT", "8000"))
            except ValueError as exc:
                raise ValueError("LOCAL_LLM_PORT must be an integer") from exc
            if not 1 <= port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            if not host or any(char in host for char in "/?#@"):
                raise ValueError("host must be a hostname or IP address")
            url = _local_base_url(host, port)
            _validate_url(url)

    target = resolve_backend_target(
        model_name=model,
        cli_provider=provider,
        cli_local=local,
        cli_base_url=url or "",
    )
    resolved_provider, resolved_model, resolved_url = target
    # A file provider pin is weaker than an explicit CLI provider. Never write
    # a configuration that would silently use a different protocol on restart.
    hosting_conflict = (
        hosting and resolved_provider != hosting
        and not (resolved_url and _looks_local_base_url(resolved_url))
    )
    if hosting_conflict:
        raise ValueError(
            f"cannot persist provider {resolved_provider!r} for catalogued model "
            f"{model!r}: startup selects {hosting!r} for that name. "
            "Use a distinct server model alias or pass --provider on the CLI"
        )
    saved_target = resolve_backend_target(
        config_model=resolved_model,
        config_provider=resolved_provider,
        config_base_url=resolved_url or "",
    )
    if saved_target != target:
        raise ValueError("this backend selection cannot be represented by a default config")

    project = project_config_path()
    path = (
        global_config_path() if options.scope == "global"
        else project or os.path.join(os.getcwd(), AGENT_FILENAME)
    )
    write_agent_config_keys(path, set_keys={
        "model": resolved_model,
        "provider": resolved_provider,
        # Null/removal would reveal a stale lower-priority URL.
        "base_url": resolved_url or "",
    })
    result = (
        f"Saved {options.scope} default in {path}\n"
        f"Model: {resolved_model}\nProvider: {resolved_provider}\n"
        f"Endpoint: {resolved_url or '(provider default; inherited base_url cleared)'}\n"
        "Applies to future sessions. The running session and its sub-agents "
        "keep their current backend."
    )
    if options.scope == "global" and project:
        result += f"\nWarning: project config {project} takes precedence over global defaults."
    if os.getenv("LOCAL_MODEL"):
        result += (
            "\nWarning: LOCAL_MODEL still enables local mode at startup; "
            "unset it or use --online (-o) to use the saved default."
        )
    return result
