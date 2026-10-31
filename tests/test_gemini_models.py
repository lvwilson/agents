"""Reviewed Gemini models: offline routing, wire options, and dated billing."""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest import mock

import pytest

from agents import model_catalog as catalog
from agents.agents import resolve_backend_target
from agents.backends import list_available_models
from agents.backends.gemini_backend import GeminiBackend

MODELS = {
    "gemini-4-argon": ("Gemini 4 Argon", 2, 10, .1, None),
    "gemini-3.8-flash": ("Gemini 3.8 Flash", .75, 3.75, .075, .5),
    "gemini-3.6-flash": ("Gemini 3.6 Flash", .75, 3.75, .075, .5),
    "gemini-3.5-flash-lite": ("Gemini 3.5 Flash-Lite", .3, 2.5, .03, 1),
}
CONTEXT = [
    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
    {"role": "user", "content": [{"type": "text", "text": "continue"}]},
]


def set_date(monkeypatch, day):
    clock = mock.Mock()
    clock.now.return_value = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    monkeypatch.setattr(catalog, "datetime", clock)


@pytest.fixture(autouse=True)
def isolated_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_MODEL_CATALOG", str(tmp_path / "catalog.json"))
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    set_date(monkeypatch, "2026-10-31")


def make_backend(model, **kwargs):
    with mock.patch("google.genai.Client"):
        return GeminiBackend(model=model, **kwargs)


@pytest.mark.parametrize("model", MODELS)
def test_metadata_and_routing(model):
    display, input_rate, output_rate, cache_rate, storage = MODELS[model]
    entry = next(r for r in list_available_models("gemini") if r["model"] == model)
    assert (entry["display"], entry["context"], entry["input_cost"],
            entry["output_cost"], entry["cache_read_cost"]) == (
                display, 1048576, input_rate, output_rate, cache_rate)
    for options in ({}, {"cli_online": True}):
        assert resolve_backend_target(model_name=model, **options) == ("gemini", model, None)
    assert resolve_backend_target(config_model=model, config_provider="cerebras") == (
        "gemini", model, None)
    b = make_backend(model)
    assert b.display_name == display
    assert b.context_window_size == 1048576
    assert b.MODEL_MAX_COMPLETION[model] == 65536
    assert b.calculate_cost(100, 20, cache_read_tokens=50) == pytest.approx(
        (100 * input_rate + 20 * output_rate + 50 * cache_rate) / 1e6)
    assert b._calculate_storage_cost(1_000_000, 3600) == (storage or 0)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", [None, 0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("cached", [False, True])
def test_request_options(model, effort, cached):
    b = make_backend(model, reasoning_effort=effort)
    b._client.models.generate_content_stream.return_value = []
    if cached:
        b._cache_name = "cachedContents/test"
        b._cached_msg_count = 2
        b._cached_system_prompt = "system"
        b._cached_prefix_digest = b._prefix_digest(CONTEXT[:2])
    b._get_response("system", CONTEXT)
    options = b._client.models.generate_content_stream.call_args.kwargs
    cfg = options["config"].model_dump(exclude_none=True)
    assert options["model"] == model
    assert cfg["max_output_tokens"] == 65536
    if model == "gemini-4-argon":
        assert "temperature" not in cfg
        assert "cached_content" not in cfg
        assert cfg["system_instruction"] == "system"
        assert len(options["contents"]) == 3
    else:
        assert cfg["temperature"] == 1
        assert ("cached_content" in cfg) == cached
    if effort is None:
        assert "thinking_config" not in cfg  # Leave the server default alone.
    else:
        supports_minimal = model not in {"gemini-4-argon", "gemini-3.8-flash"}
        expected = "minimal" if effort == 0 and supports_minimal else (
            "low" if effort <= 1 else "medium" if effort == 2 else "high")
        assert cfg["thinking_config"]["thinking_level"].lower() == expected


def test_argon_never_creates_explicit_cache():
    b = make_backend("gemini-4-argon", cache_step=1)
    assert not b._create_cache("system", CONTEXT)
    b._client.models.generate_content_stream.return_value = []
    b._get_response("system", CONTEXT)
    b._get_response("system", CONTEXT)
    b._client.caches.create.assert_not_called()


def test_flash_still_attempts_explicit_cache():
    b = make_backend("gemini-3.8-flash", cache_step=1)
    b._client.caches.create.return_value = SimpleNamespace(
        name="cachedContents/test", usage_metadata=SimpleNamespace(total_token_count=100))
    b._client.models.generate_content_stream.return_value = []
    b._get_response("system", CONTEXT)
    b._client.caches.create.assert_called_once()
    assert b._client.models.generate_content_stream.call_args.kwargs[
        "config"].cached_content == "cachedContents/test"


@pytest.mark.parametrize("day,argon,flash", [
    ("2026-11-30", (2, 10, .1), (.75, 3.75, .075)),
    ("2026-12-01", (4, 20, .2), (.75, 3.75, .075)),
    ("2026-12-31", (4, 20, .2), (.75, 3.75, .075)),
    ("2027-01-01", (4, 20, .2), (1.5, 7.5, .15)),
])
def test_pricing_transitions(monkeypatch, day, argon, flash):
    set_date(monkeypatch, day)
    for model, rates in [("gemini-4-argon", argon),
                         ("gemini-3.8-flash", flash), ("gemini-3.6-flash", flash)]:
        entry = next(r for r in list_available_models("gemini") if r["model"] == model)
        assert (entry["input_cost"], entry["output_cost"], entry["cache_read_cost"]) == rates
        b = make_backend(model)
        assert b.calculate_cost(1_000_000, 1_000_000, cache_read_tokens=1_000_000) == sum(rates)
        if model != "gemini-4-argon":
            assert b._calculate_storage_cost(1_000_000, 3600) == (
                1 if day >= "2027-01-01" else .5)


def test_overlay_pricing_replaces_bundled_schedule(monkeypatch):
    prices = dict(input_token_cost=9, output_token_cost=8, cache_read_cost=7)
    catalog.apply_catalog([dict(provider="gemini", model="gemini-4-argon", pricing=prices)])
    set_date(monkeypatch, "2027-01-01")
    assert make_backend("gemini-4-argon").MODEL_PRICING["gemini-4-argon"] == prices


def test_overlay_can_clear_its_own_schedule(monkeypatch):
    prices = dict(input_token_cost=9, output_token_cost=8, cache_read_cost=7)
    record = dict(provider="gemini", model="gemini-4-argon", pricing=prices)
    catalog.apply_catalog([{**record, "pricing_schedule": [
        dict(effective_from="2027-01-01", pricing={**prices, "input_token_cost": 20})]}])
    catalog.apply_catalog([record])
    set_date(monkeypatch, "2027-01-02")
    assert make_backend("gemini-4-argon").MODEL_PRICING["gemini-4-argon"] == prices


@pytest.mark.parametrize("schedule", [
    [], None, [{}], [{"effective_from": "2026-02-30", "pricing": {}}],
    [{"effective_from": "2027-01-01", "pricing": {"input_token_cost": 1}}],
    [{"effective_from": 20270101, "pricing": {}}],
    [{"effective_from": "2027-01-01", "pricing": None}],
])
def test_bad_pricing_schedules_rejected(schedule):
    with pytest.raises(ValueError):
        catalog.validate_catalog({"version": 1, "models": [{
            "provider": "gemini", "model": "test", "pricing_schedule": schedule,
            "pricing": dict(input_token_cost=1, output_token_cost=2, cache_read_cost=.1),
        }]})


def test_no_carbon_or_non_text_models_added():
    records = catalog.read_catalog(catalog.BUNDLED_CATALOG)
    assert {r["model"] for r in records if r["provider"] == "gemini"} == set(MODELS)


def test_old_model_and_unknown_model_request_unchanged():
    for model in ("gemini-3-flash-preview", "unknown-gemini"):
        b = make_backend(model, reasoning_effort="max")
        b._client.models.generate_content_stream.return_value = []
        b._get_response("system", CONTEXT)
        cfg = b._client.models.generate_content_stream.call_args.kwargs["config"]
        assert cfg.max_output_tokens == 16384
        assert cfg.thinking_config is None
        assert cfg.temperature == 1


def test_argon_timeout_and_proxy():
    with mock.patch("google.genai.Client") as client:
        GeminiBackend(model="gemini-4-argon", base_url="https://proxy.example")
    assert client.call_args.kwargs["http_options"] == {
        "timeout": 900000, "base_url": "https://proxy.example"}


@pytest.mark.parametrize("model", MODELS)
def test_streamed_usage_and_signed_reasoning_roundtrip(model):
    from google.genai import types
    b = make_backend(model)
    parts = [
        types.Part(text="Summary", thought=True),
        types.Part(text="Answer", thought_signature=b"signed-state"),
    ]
    usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=12,
                            thoughts_token_count=8, cached_content_token_count=40)
    b._client.models.generate_content_stream.return_value = [
        SimpleNamespace(candidates=[SimpleNamespace(
            content=types.Content(role="model", parts=parts), finish_reason="STOP")],
            usage_metadata=usage),
    ]
    assert b.generate_response("system", CONTEXT) == "Answer"
    assert b.last_input_tokens == 100
    assert b.last_output_tokens == 20
    assert b.last_total_context_tokens == 120
    _, input_rate, output_rate, cache_rate, _ = MODELS[model]
    assert b.cost == pytest.approx((60 * input_rate + 40 * cache_rate + 20 * output_rate) / 1e6)
    assert b.last_reasoning_context["model"] == model
    translated = b._translate_messages([{
        "role": "assistant",
        "content": [{"type": "text", "text": "Answer"}],
        "reasoning_context": b.last_reasoning_context,
    }])
    assert len(translated[0].parts) == 2
    assert translated[0].parts[1].thought_signature == b"signed-state"


@pytest.mark.parametrize("dates", [
    ["2027-01-01", "2027-01-01"],
    ["2027-02-01", "2027-01-01"],
])
def test_schedule_dates_must_increase(dates):
    prices = dict(input_token_cost=1, output_token_cost=2, cache_read_cost=.1)
    with pytest.raises(ValueError, match="ordered unique"):
        catalog.validate_catalog({"version": 1, "models": [{
            "provider": "gemini", "model": "test", "pricing": prices,
            "pricing_schedule": [
                dict(effective_from=day, pricing=prices) for day in dates],
        }]})


def test_schedule_requires_base_pricing():
    with pytest.raises(ValueError, match="base pricing"):
        catalog.validate_catalog({"version": 1, "models": [{
            "provider": "gemini", "model": "test",
            "pricing_schedule": [dict(effective_from="2027-01-01", pricing={
                "input_token_cost": 1, "output_token_cost": 2, "cache_read_cost": .1})],
        }]})


REMOVED_MODELS = (
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-flash-lite",
)


def test_gemini_31_models_not_registered():
    from agents.agents import _build_online_models_catalog

    listed = {r["model"] for r in list_available_models("gemini")}
    hosted = _build_online_models_catalog()
    for model in REMOVED_MODELS:
        assert model not in listed
        assert model not in hosted
        assert model not in GeminiBackend.MODEL_PRICING
        assert model not in GeminiBackend.MODEL_DISPLAY_NAMES
        assert model not in GeminiBackend.MODEL_CONTEXT_WINDOWS


def test_gemini_31_models_not_in_fallback_catalog():
    from agents.agents import _build_online_models_catalog

    with mock.patch("agents.backends.list_available_models", return_value=[]):
        hosted = _build_online_models_catalog()
    assert hosted["gemini-4-argon"] == "gemini"
    assert not set(REMOVED_MODELS) & hosted.keys()


def test_gemini_default_is_argon():
    from agents.config import PROVIDER_DEFAULT_MODELS

    assert PROVIDER_DEFAULT_MODELS["gemini"] == "gemini-4-argon"
    with mock.patch("google.genai.Client"):
        backend = GeminiBackend()
    assert backend.model == "gemini-4-argon"
    assert backend.context_window_size == 1048576
    assert backend.MODEL_MAX_COMPLETION[backend.model] == 65536
    assert not backend.supports_explicit_caching
