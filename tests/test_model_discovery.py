"""Offline discovery, safe catalog updates, and GPT-6.1 integration."""
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import HTTPError

import pytest

from agents import model_catalog as catalog
from agents import model_discovery as discovery
from agents.backends import create_backend, list_available_models
from agents.backends.openai_backend import OpenAIBackend
from agents.cli.models_cli import main


@pytest.fixture(autouse=True)
def isolated_catalog(tmp_path, monkeypatch):
    path = tmp_path / "model_catalog.json"
    monkeypatch.setenv("AGENT_MODEL_CATALOG", str(path))
    return path


def record(model="future-chat", **extra):
    return {"provider": "openai", "model": model, **extra}


def document(*records):
    return {"version": 1, "models": list(records)}


def make_backend(model="gpt-6.1-sol", **kwargs):
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), mock.patch("openai.OpenAI"):
        return OpenAIBackend(model=model, **kwargs)


@pytest.fixture(params=[
    ("gpt-6.1-sol", "GPT-6.1 Sol", 2, 10, .1, 2.5),
    ("gpt-6.1-terra", "GPT-6.1 Terra", 1, 5, .05, 1.25),
], ids=["sol", "terra"])
def gpt61(request):
    return request.param


def test_gpt61_offline_metadata_and_routing(gpt61):
    from agents.agents import resolve_backend_target
    model, display, input_price, output_price, cache_read, _ = gpt61
    entry = next(r for r in list_available_models("openai") if r["model"] == model)
    assert entry["display"] == display
    assert (entry["context"], entry["input_cost"], entry["cache_read_cost"],
            entry["output_cost"]) == (1050000, input_price, cache_read, output_price)
    assert resolve_backend_target(model_name=model) == ("openai", model, None)
    assert resolve_backend_target(model_name=model, cli_online=True) == (
        "openai", model, None)
    assert resolve_backend_target(config_model=model, config_provider="cerebras") == (
        "openai", model, None)
    b = make_backend(model)
    assert b.display_name == display
    assert b.context_window_size == 1050000
    assert b.MAX_COMPLETION_TOKENS == 128000


@pytest.mark.parametrize("rank", [None, 0, 1, 2, 3, 4, 5])
def test_gpt61_request_and_completed_usage(gpt61, rank):
    model, _, input_price, output_price, cache_read, cache_write = gpt61
    b = make_backend(model, reasoning_effort=rank)
    b._client.responses.create.return_value = iter([
        SimpleNamespace(type="response.output_text.delta", delta="OK"),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1000, output_tokens=100,
                                  input_tokens_details=SimpleNamespace(
                                      cached_tokens=500, cache_creation_tokens=100)))),
    ])
    assert b.generate_response("sys", []) == "OK"
    kwargs = b._client.responses.create.call_args.kwargs
    assert kwargs["model"] == model
    assert kwargs["max_output_tokens"] == 128000
    if rank is None:
        assert "reasoning" not in kwargs
    else:
        expected = ("none", "low", "medium", "high", "xhigh", "max")[rank]
        if model == "gpt-6.1-sol" and rank == 0:
            expected = "low"
        assert kwargs["reasoning"] == {"effort": expected}
    assert kwargs["stream"] is True
    assert b.cost == pytest.approx(
        (400*input_price + 500*cache_read + 100*cache_write + 100*output_price)/1e6)


@pytest.mark.parametrize("tokens,mult_in,mult_out", [(272000, 1, 1), (272001, 2, 1.5)])
def test_gpt61_long_context_threshold(gpt61, tokens, mult_in, mult_out):
    model, _, input_price, output_price, cache_read, cache_write = gpt61
    b = make_backend(model)
    assert b.calculate_cost(tokens, 100, cache_read_tokens=1000,
                            cache_creation_tokens=500) == pytest.approx(
        (((tokens-1500)*input_price + 1000*cache_read + 500*cache_write)*mult_in
         + 100*output_price*mult_out)/1e6)


def test_overlay_refresh_routing_and_no_shared_mutation(isolated_catalog):
    from agents.agents import _online_models, resolve_backend_target
    assert "future-chat" not in _online_models()
    catalog.apply_catalog([record(context=9999, display="Future", max_output=500,
                                  effort_levels=["low", "max"], send_temperature=False,
                                  pricing={"input_token_cost": 3,
                                           "output_token_cost": 8, "cache_read_cost": .3})])
    assert _online_models()["future-chat"] == "openai"
    assert resolve_backend_target(model_name="future-chat") == ("openai", "future-chat", None)
    b = make_backend("future-chat")
    assert (b.display_name, b.context_window_size, b.MAX_COMPLETION_TOKENS) == ("Future", 9999, 500)
    assert b.SEND_TEMPERATURE is False
    assert "future-chat" not in OpenAIBackend.MODEL_PRICING
    isolated_catalog.unlink()
    assert "future-chat" not in _online_models()
    assert make_backend().SEND_TEMPERATURE is True


def test_discovery_unknown_metadata_and_local_isolation(gpt61):
    catalog.apply_catalog([record()])
    with pytest.warns(RuntimeWarning, match="dollar budget cannot limit spending"):
        b = make_backend("future-chat")
    assert b.context_window_size == 0
    assert b.calculate_cost(1000, 1000) == 0
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}), mock.patch("openai.OpenAI"):
        local = create_backend("openai", model=gpt61[0], base_url="http://localhost:8000/v1")
    assert local.context_window_size != 1050000
    assert local.MAX_COMPLETION_TOKENS == 16384


def test_merge_preserves_reviewed_fields_and_other_models(isolated_catalog):
    catalog.apply_catalog([record(display="Future", context=10000), record("another")])
    catalog.apply_catalog([record(source="https://api.openai.com/v1/models")])
    saved = catalog.read_catalog(isolated_catalog)
    future = next(r for r in saved if r["model"] == "future-chat")
    assert future["display"] == "Future" and future["context"] == 10000
    assert len(saved) == 2
    assert isolated_catalog.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("bad", [
    {"version": 2, "models": []},
    {"version": True, "models": []},
    document(record(unknown=1)),
    document(record(provider="invalid")),
    document(record(model="")),
    document(record(model="bad\nid")),
    document(record(context=True)),
    document(record(max_output=0)),
    document(record(effort_levels=["imaginary"])),
    document(record(effort_levels=["max", "max"])),
    document(record(pricing={"input_token_cost": 1, "output_token_cost": 2})),
    document(record(pricing={"input_token_cost": 10**400, "output_token_cost": 2, "cache_read_cost": 1})),
    document(record(pricing={"input_token_cost": float("nan"), "output_token_cost": 2, "cache_read_cost": 1})),
    document(record(provider="deepseek", send_temperature=True)),
    document(record(pricing={"input_token_cost": 1, "output_token_cost": 2, "cache_read_cost": 1,
                             "cache_storage_cost_per_hour": 1})),
    document(record(), record()),
])
def test_reject_invalid_catalog(bad):
    with pytest.raises(ValueError):
        catalog.validate_catalog(bad)


def test_failed_write_leaves_old_catalog_and_cleans_temporary(isolated_catalog):
    catalog.apply_catalog([record()])
    previous = isolated_catalog.read_bytes()
    with mock.patch("agents.model_catalog.os.replace", side_effect=OSError("disk error")):
        with pytest.raises(OSError):
            catalog.apply_catalog([record("another")])
    assert isolated_catalog.read_bytes() == previous
    assert list(isolated_catalog.parent.iterdir()) == [isolated_catalog]


def test_invalid_existing_catalog_not_overwritten(isolated_catalog):
    isolated_catalog.write_text('{"version":2,"models":[]}')
    with pytest.raises(ValueError):
        catalog.apply_catalog([record()])
    assert json.loads(isolated_catalog.read_text())["version"] == 2


def test_symlink_refused(tmp_path, isolated_catalog):
    original = tmp_path / "original.json"
    original.write_text(json.dumps(document()))
    isolated_catalog.symlink_to(original)
    with pytest.raises(ValueError, match="regular file"):
        catalog.apply_catalog([record()])
    assert json.loads(original.read_text()) == document()


def test_cross_provider_collision_rejected(isolated_catalog):
    from agents.agents import resolve_backend_target
    with pytest.raises(ValueError, match="ambiguous model ID"):
        catalog.apply_catalog([record("deepseek-v4-pro")])
    assert not isolated_catalog.exists()
    assert resolve_backend_target(config_model="deepseek-v4-pro",
                                  config_provider="deepseek")[0] == "deepseek"
    # Manual edits are validated too; no silently misrouted startup.
    isolated_catalog.write_text(json.dumps(document(record("deepseek-v4-pro"))))
    with pytest.raises(ValueError, match="ambiguous model ID"):
        list_available_models()


@pytest.mark.parametrize("provider,module,cls", [
    ("anthropic", "anthropic_backend", "AnthropicBackend"),
    ("minimax", "minimax_backend", "MinimaxBackend"),
    ("deepseek", "deepseek_backend", "DeepSeekBackend"),
])
def test_declared_cache_price_matches_billing(provider, module, cls):
    import importlib
    catalog.apply_catalog([record("cache-probe", provider=provider,
                                 pricing={"input_token_cost": 10, "output_token_cost": 20,
                                          "cache_read_cost": 7})])
    klass = getattr(importlib.import_module(f"agents.backends.{module}"), cls)
    b = klass.__new__(klass)
    b.model = "cache-probe"
    catalog.apply_backend_metadata(b)
    assert b.calculate_cost(0, 0, cache_read_tokens=1000000) == 7


@pytest.mark.parametrize("provider,key", [
    ("openai", "OPENAI_API_KEY"), ("deepseek", "DEEPSEEK_API_KEY"),
    ("cerebras", "CEREBRAS_API_KEY"), ("kimi", "MOONSHOT_API_KEY"),
])
def test_bearer_discovery(provider, key, monkeypatch):
    monkeypatch.setenv(key, "secret")
    with mock.patch.object(discovery, "_get_json", return_value={"data": [{"id": "future-chat"}]}) as get:
        rows = discovery.discover_models(provider)
    assert rows[0]["model"] == "future-chat"
    assert "pricing" not in rows[0] and "context" not in rows[0]
    assert get.call_args.args[1] == {"Authorization": "Bearer secret"}
    assert get.call_args.args[0].startswith("https://")


def test_anthropic_pagination(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "secret")
    pages = [
        {"data": [{"id": "a", "display_name": "A"}], "has_more": True, "last_id": "a"},
        {"data": [{"id": "b"}], "has_more": False},
    ]
    with mock.patch.object(discovery, "_get_json", side_effect=pages) as get:
        assert [r["model"] for r in discovery.discover_models("anthropic")] == ["a", "b"]
    assert "after_id=a" in get.call_args.args[0]
    assert get.call_args.args[1]["x-api-key"] == "secret"


def test_gemini_pagination_and_generation_filter(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    pages = [
        {"models": [{"name": "models/embed", "supportedGenerationMethods": ["embedContent"]}],
         "nextPageToken": "p/2"},
        {"models": [{"name": "models/chat", "displayName": "Chat",
                     "supportedGenerationMethods": ["generateContent"], "inputTokenLimit": 999}]},
    ]
    with mock.patch.object(discovery, "_get_json", side_effect=pages) as get:
        rows = discovery.discover_models("gemini")
    assert rows[0]["model"] == "chat"
    assert "context" not in rows[0]
    assert "pageToken=p%2F2" in get.call_args.args[0]
    assert get.call_args.args[1] == {"x-goog-api-key": "secret"}


@pytest.mark.parametrize("page", [
    {}, {"data": "wrong"}, {"data": [{"id": "bad id"}]},
    {"data": [], "has_more": True},
])
def test_malformed_listing_rejected(page, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    with mock.patch.object(discovery, "_get_json", return_value=page):
        with pytest.raises(discovery.DiscoveryError):
            discovery.discover_models("openai")


def test_pagination_loop_rejected(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "secret")
    with mock.patch.object(discovery, "_get_json", return_value={
        "data": [{"id": "a"}], "has_more": True, "last_id": "a",
    }):
        with pytest.raises(discovery.DiscoveryError, match="repeated"):
            discovery.discover_models("anthropic")


def test_missing_key_and_unsupported_adapter(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(discovery.DiscoveryError, match="OPENAI_API_KEY"):
        discovery.discover_models("openai")
    with pytest.raises(discovery.DiscoveryError, match="import reviewed"):
        discovery.discover_models("minimax")


def test_transport_errors_redacted_and_redirects_refused():
    opener = mock.Mock()
    opener.open.side_effect = HTTPError("https://secret.invalid", 401, "secret", {}, None)
    with mock.patch.object(discovery, "build_opener", return_value=opener):
        with pytest.raises(discovery.DiscoveryError, match=r"HTTP 401") as exc:
            discovery._get_json("https://example.com", {"Authorization": "secret"})
    assert "secret" not in str(exc.value)
    assert discovery._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil") is None


def test_cli_preview_apply_and_absent_selection(isolated_catalog, capsys):
    with mock.patch("agents.cli.models_cli.discover_models", return_value=[record()]):
        assert main(["discover", "openai", "--model", "future-chat"]) == 0
        assert not isolated_catalog.exists()
        assert json.loads(capsys.readouterr().out)["models"] == [record()]
        assert main(["discover", "openai", "--model", "absent", "--apply"]) == 1
        assert not isolated_catalog.exists()
        assert main(["discover", "openai", "--model", "future-chat", "--apply"]) == 0
    assert catalog.read_catalog(isolated_catalog) == [record()]


def test_apply_requires_exact_selection_without_network():
    with mock.patch("agents.cli.models_cli.discover_models") as get:
        with pytest.raises(SystemExit):
            main(["discover", "openai", "--apply"])
        get.assert_not_called()


def test_import_preview_and_apply(tmp_path, isolated_catalog, capsys):
    source = tmp_path / "import.json"
    source.write_text(json.dumps(document(record(context=2048))))
    assert main(["import", str(source)]) == 0
    assert not isolated_catalog.exists()
    assert main(["import", str(source), "--apply"]) == 0
    assert catalog.read_catalog(isolated_catalog)[0]["context"] == 2048
    capsys.readouterr()


def test_failure_does_not_apply_partial_results(isolated_catalog):
    with mock.patch("agents.cli.models_cli.discover_models",
                    side_effect=discovery.DiscoveryError("HTTP 503")):
        assert main(["discover", "openai", "--model", "future-chat", "--apply"]) == 1
    assert not isolated_catalog.exists()


def test_metadata_controls_wire_options_without_affecting_other_models():
    catalog.apply_catalog([record("gpt-6.1-sol", effort_levels=[], send_temperature=False)])
    b = make_backend(reasoning_effort=5)
    b._client.responses.create.return_value = iter([
        SimpleNamespace(type="response.output_text.delta", delta="OK"),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None)),
    ])
    b.generate_response("sys", [])
    kwargs = b._client.responses.create.call_args.kwargs
    assert "reasoning" not in kwargs and "temperature" not in kwargs
    assert make_backend("gpt-6-astra").EFFORT_LEVELS == ("low", "medium", "high", "xhigh")


def test_discovery_preserves_reviewed_provenance(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    with mock.patch.object(discovery, "_get_json",
                           return_value={"data": [{"id": "gpt-6.1-sol"}]}):
        catalog.apply_catalog(discovery.discover_models("openai"))
    metadata = catalog.model_overrides()[("openai", "gpt-6.1-sol")]
    assert metadata["source"].startswith("https://developers.openai.com/")
    assert metadata["discovery_source"] == "https://api.openai.com/v1/models"
    assert metadata["pricing"]["input_token_cost"] == 2
