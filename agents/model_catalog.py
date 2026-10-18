"""Offline, versioned model metadata layered over backend defaults.

The bundled catalog is reviewed data; the user overlay is updated explicitly.
No network requests, SDK imports, credentials or endpoint overrides live here.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

SCHEMA_VERSION = 1
BUNDLED_CATALOG = Path(__file__).with_name("model_catalog.json")
_FIELDS = {
    "provider", "model", "display", "context", "pricing", "source",
    "discovered_at", "discovery_source", "verified_at", "max_output", "effort_levels", "send_temperature",
}
_PRICE_FIELDS = {
    "input_token_cost", "output_token_cost", "cache_read_cost",
    "cache_write_cost", "cache_storage_cost_per_hour",
    "long_context_threshold", "long_input_multiplier", "long_output_multiplier",
}
_OPENAI_FIELDS = {"max_output", "effort_levels", "send_temperature"}
_OPENAI_PRICES = {
    "cache_write_cost", "long_context_threshold",
    "long_input_multiplier", "long_output_multiplier",
}


def catalog_path() -> Path:
    """An environment override is useful for isolated testing/deployments."""
    return Path(os.environ.get("AGENT_MODEL_CATALOG",
                               "~/.agents/model_catalog.json")).expanduser()


def validate_catalog(data: object) -> list[dict]:
    """Reject malformed/unsupported records rather than silently ignoring typos."""
    from .backends import _REGISTRY

    if (not isinstance(data, dict) or set(data) != {"version", "models"}
            or type(data["version"]) is not int or data["version"] != SCHEMA_VERSION
            or not isinstance(data["models"], list)):
        raise ValueError("catalog must contain version: 1 and a models list")
    seen = set()
    for record in data["models"]:
        if not isinstance(record, dict) or set(record) - _FIELDS:
            raise ValueError("invalid model record or unknown metadata field")
        provider, model = record.get("provider"), record.get("model")
        if not isinstance(provider, str) or provider not in _REGISTRY:
            raise ValueError(f"unknown provider: {provider!r}")
        if (not isinstance(model, str) or not model or len(model) > 512
                or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in model)):
            raise ValueError("model must be a nonempty ID without whitespace/control characters")
        key = (provider, model)
        if key in seen:
            raise ValueError(f"duplicate model: {provider}/{model}")
        seen.add(key)
        for field in ("display", "source", "discovered_at", "discovery_source", "verified_at"):
            if field in record and (not isinstance(record[field], str)
                                    or not record[field].strip()
                                    or any(ord(c) < 32 for c in record[field])):
                raise ValueError(f"{field} must be nonempty single-line text")
        for field in ("context", "max_output"):
            if field in record and (type(record[field]) is not int or record[field] <= 0):
                raise ValueError(f"{field} must be a positive integer")
        if provider != "openai" and set(record) & _OPENAI_FIELDS:
            raise ValueError("request-option metadata currently supports openai only")
        if "send_temperature" in record and type(record["send_temperature"]) is not bool:
            raise ValueError("send_temperature must be boolean")
        if "effort_levels" in record:
            levels = record["effort_levels"]
            allowed = {"none", "low", "medium", "high", "xhigh", "max"}
            if (not isinstance(levels, list) or
                    any(not isinstance(v, str) or v not in allowed for v in levels)
                    or len(set(levels)) != len(levels)):
                raise ValueError("effort_levels must be a list of unique canonical efforts")
        if "pricing" in record:
            prices = record["pricing"]
            if (not isinstance(prices, dict) or set(prices) - _PRICE_FIELDS
                    or not {"input_token_cost", "output_token_cost", "cache_read_cost"} <= set(prices)):
                raise ValueError("pricing requires input_token_cost, output_token_cost and cache_read_cost")
            if provider != "openai" and set(prices) & _OPENAI_PRICES:
                raise ValueError("tier/cache-write pricing metadata supports openai only")
            if provider != "gemini" and "cache_storage_cost_per_hour" in prices:
                raise ValueError("cache storage pricing supports gemini only")
            for name, value in prices.items():
                if (type(value) not in (int, float) or value < 0
                        or value > 10**15 or not math.isfinite(value)):
                    raise ValueError(f"invalid price: {name}")
            if "long_context_threshold" in prices:
                if (type(prices["long_context_threshold"]) is not int
                        or prices["long_context_threshold"] <= 0
                        or not {"long_input_multiplier", "long_output_multiplier"} <= set(prices)):
                    raise ValueError("long-context pricing needs positive threshold and both multipliers")
            elif {"long_input_multiplier", "long_output_multiplier"} & set(prices):
                raise ValueError("long-context multipliers require a threshold")
    return copy.deepcopy(data["models"])


def read_catalog(path: Path) -> list[dict]:
    try:
        return validate_catalog(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read model catalog {path}: {exc}") from exc


def model_overrides() -> dict[tuple[str, str], dict]:
    """Merge reviewed package data with the user overlay, field by field.

    Pricing is replaced as a complete object to avoid retaining obsolete tiers.
    """
    result = {}
    paths = [BUNDLED_CATALOG, catalog_path()]
    for index, path in enumerate(paths):
        if index == 1 and not path.exists():
            continue
        for record in read_catalog(path):
            key = (record["provider"], record["model"])
            result[key] = {**result.get(key, {}), **record}
    _validate_ownership(list(result.values()))
    return result


def _validate_ownership(records: list[dict]) -> None:
    """Do not let an import silently redirect an already catalogued model ID."""
    from .backends import _REGISTRY, _load_class
    owners = {}
    for provider in _REGISTRY:
        cls = _load_class(provider)
        for table in ("MODEL_PRICING", "MODEL_CONTEXT_WINDOWS", "MODEL_DISPLAY_NAMES"):
            for model in cls.__dict__.get(table, {}):
                owners.setdefault(model, set()).add(provider)
    for record in records:
        model, provider = record["model"], record["provider"]
        previous = owners.setdefault(model, set())
        if previous - {provider}:
            raise ValueError(
                f"ambiguous model ID {model!r}: already registered for "
                f"{', '.join(sorted(previous))}; cannot register for {provider}. "
                "Use an explicit --provider without registering this duplicate ID."
            )
        previous.add(provider)


def apply_catalog(records: list[dict], path: Path | None = None) -> Path:
    """Validate then atomically merge selected records; never prune absent IDs."""
    records = validate_catalog({"version": SCHEMA_VERSION, "models": records})
    target = Path(path) if path is not None else catalog_path()
    # Reuse the config writer's conservative symlink/regular-file checks.
    from .config import _check_config_path
    target = target.absolute()
    original = _check_config_path(str(target))
    existing = read_catalog(target) if original is not None else []
    merged = {(r["provider"], r["model"]): r for r in existing}
    for record in records:
        key = (record["provider"], record["model"])
        merged[key] = {**merged.get(key, {}), **record}
    data = {"version": SCHEMA_VERSION, "models": [merged[k] for k in sorted(merged)]}
    validate_catalog(data)
    _validate_ownership(read_catalog(BUNDLED_CATALOG) + data["models"])
    target.parent.mkdir(parents=True, exist_ok=True)
    _check_config_path(str(target))
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        def fingerprint(info):
            if info is None:
                return None
            return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
                    info.st_mtime_ns, info.st_ctime_ns)
        if fingerprint(_check_config_path(str(target))) != fingerprint(original):
            raise ValueError("model catalog changed while writing; retry")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)
    return target


def apply_backend_metadata(backend) -> None:
    """Apply metadata before subclass initialization, without class mutation.

    Generic/local OpenAICompatBackend has no hosted provider match, deliberately.
    """
    from .backends import _REGISTRY
    module = type(backend).__module__.rsplit(".", 1)[-1]
    provider = next((p for p, (m, _) in _REGISTRY.items()
                     if m.lstrip(".") == module), None)
    if provider is None:
        return
    record = model_overrides().get((provider, backend.model))
    if record is None:
        return
    for field, table in (("display", "MODEL_DISPLAY_NAMES"),
                         ("context", "MODEL_CONTEXT_WINDOWS"),
                         ("pricing", "MODEL_PRICING"),
                         ("max_output", "MODEL_MAX_COMPLETION")):
        values = copy.deepcopy(getattr(backend, table, {}))
        if field in record:
            values[backend.model] = record[field]
        elif field == "context" and backend.model not in values:
            values[backend.model] = 0  # Unknown, not the generic guessed window.
        setattr(backend, table, values)
    if "effort_levels" in record:
        backend.EFFORT_LEVELS = tuple(record["effort_levels"])
    if "send_temperature" in record:
        backend.SEND_TEMPERATURE = record["send_temperature"]
    if backend.model not in backend.MODEL_PRICING:
        warnings.warn(
            f"Pricing unknown for {provider}/{backend.model}; costs are not tracked "
            "and the dollar budget cannot limit spending. Import verified pricing.",
            RuntimeWarning, stacklevel=3,
        )
