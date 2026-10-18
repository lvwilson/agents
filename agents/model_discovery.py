"""Explicit provider model discovery. Never called by normal agent startup."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, HTTPRedirectHandler, build_opener

# Do not use AGENT_BASE_URL or SDK endpoint environment overrides: registering
# IDs from an arbitrary proxy as hosted models would change their routing.
PROVIDERS = {
    "openai": ("https://api.openai.com/v1/models", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/models", "CLAUDE_API_KEY"),
    "cerebras": ("https://api.cerebras.ai/v1/models", "CEREBRAS_API_KEY"),
    "deepseek": ("https://api.deepseek.com/models", "DEEPSEEK_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/models", "GEMINI_API_KEY"),
    "kimi": ("https://api.moonshot.ai/v1/models", "MOONSHOT_API_KEY"),
}


class DiscoveryError(ValueError):
    """A failed discovery must never produce a partial applied catalog."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _get_json(url: str, headers: dict) -> dict:
    request = Request(url, headers={**headers, "Accept": "application/json"})
    try:
        with build_opener(_NoRedirect).open(request, timeout=30) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise DiscoveryError("model listing exceeds 8 MiB")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise DiscoveryError("provider returned a non-object model listing")
        return data
    except HTTPError as exc:
        # Do not echo headers, credentials, response bodies or arbitrary URLs.
        raise DiscoveryError(f"model listing failed (HTTP {exc.code})") from None
    except (URLError, TimeoutError, OSError, UnicodeError, ValueError) as exc:
        if isinstance(exc, DiscoveryError):
            raise
        raise DiscoveryError(f"model listing failed ({type(exc).__name__})") from None


def discover_models(provider: str) -> list[dict]:
    """Fetch model identities, plus metadata explicitly supplied by the API.

    Listings do not guarantee inference access or compatibility. No prices or
    reasoning options are inferred from a model ID or a neighbouring model.
    """
    if provider not in PROVIDERS:
        raise DiscoveryError(
            f"No model-list adapter for {provider!r}; import reviewed metadata instead."
        )
    endpoint, key_name = PROVIDERS[provider]
    key = os.getenv(key_name)
    if not key:
        raise DiscoveryError(f"{key_name} is required for {provider} discovery")
    if provider == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    elif provider == "gemini":
        headers = {"x-goog-api-key": key}
    else:
        headers = {"Authorization": f"Bearer {key}"}
    stamp = datetime.now(timezone.utc).isoformat()
    records = {}
    cursor = None
    seen_cursors = set()
    for _ in range(100):
        query = {}
        if provider == "anthropic":
            query["limit"] = 100
            if cursor:
                query["after_id"] = cursor
        elif provider == "gemini":
            query["pageSize"] = 1000
            if cursor:
                query["pageToken"] = cursor
        url = endpoint + ("?" + urlencode(query) if query else "")
        data = _get_json(url, headers)
        items = data.get("models" if provider == "gemini" else "data")
        if not isinstance(items, list):
            raise DiscoveryError("provider listing is missing its model array")
        for item in items:
            if not isinstance(item, dict):
                raise DiscoveryError("invalid model entry")
            if provider == "gemini":
                methods = item.get("supportedGenerationMethods", [])
                if not isinstance(methods, list):
                    raise DiscoveryError("invalid supportedGenerationMethods")
                if "generateContent" not in methods:
                    continue
                model = item.get("name")
                if isinstance(model, str):
                    model = model.removeprefix("models/")
            else:
                model = item.get("id")
            record = {"provider": provider, "model": model,
                      "discovery_source": endpoint, "discovered_at": stamp}
            # Do not overwrite reviewed display names with a raw ID.
            display = item.get("displayName" if provider == "gemini" else "display_name")
            if display:
                record["display"] = display
            # Gemini supplies its input limit, not necessarily total context;
            # do not incorrectly label it a total context window.
            from .model_catalog import validate_catalog
            try:
                validate_catalog({"version": 1, "models": [record]})
            except ValueError as exc:
                raise DiscoveryError(f"invalid provider model metadata: {exc}") from None
            if model in records and records[model] != record:
                raise DiscoveryError(f"conflicting duplicate model ID: {model}")
            records[model] = record
        if provider == "anthropic" and data.get("has_more"):
            cursor = data.get("last_id")
            if not cursor:
                raise DiscoveryError("paginated listing omitted last_id")
        elif provider == "gemini":
            cursor = data.get("nextPageToken")
        else:
            cursor = None
            if data.get("has_more"):
                raise DiscoveryError("unexpected pagination; refusing a partial listing")
        if not cursor:
            return [records[k] for k in sorted(records)]
        if not isinstance(cursor, str) or cursor in seen_cursors:
            raise DiscoveryError("invalid or repeated pagination cursor")
        seen_cursors.add(cursor)
    raise DiscoveryError("model listing exceeded 100 pages")
