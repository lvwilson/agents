"""Tests for the Cerebras backend.

Cerebras exposes an OpenAI-compatible *chat completions* endpoint, so
``CerebrasBackend`` derives from ``OpenAICompatBackend`` and swaps in the
official ``cerebras_cloud_sdk`` client.  These tests mock the SDK client
so no real API calls are made.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.backends import create_backend, list_available_models  # noqa: E402
from agents.backends.cerebras_backend import CerebrasBackend  # noqa: E402
from agents.backends.openai_compat_backend import OpenAICompatBackend  # noqa: E402


def _make_backend(**kwargs):
    """Construct a CerebrasBackend with the SDK client mocked out."""
    with mock.patch.dict(os.environ, {"CEREBRAS_API_KEY": "sk-test-cerebras"}):
        with mock.patch("cerebras.cloud.sdk.Cerebras") as client_cls:
            backend = CerebrasBackend(**kwargs)
    return backend, client_cls


class TestRegistration(unittest.TestCase):
    """The provider is registered and the model auto-resolves."""

    def test_factory_creates_cerebras_backend(self):
        with mock.patch.dict(os.environ, {"CEREBRAS_API_KEY": "sk-test"}):
            with mock.patch("cerebras.cloud.sdk.Cerebras"):
                backend = create_backend("cerebras", model="qwen-3.8-27b")
        self.assertIsInstance(backend, CerebrasBackend)
        self.assertIsInstance(backend, OpenAICompatBackend)

    def test_model_names_map_to_provider(self):
        self.assertEqual(agents_module._ONLINE_MODELS["qwen-3.8-27b"], "cerebras")
        self.assertEqual(agents_module._ONLINE_MODELS["gpt-oss-120b"], "cerebras")

    def test_resolve_model_auto_detects_online(self):
        provider, base_url = agents_module.resolve_model("qwen-3.8-27b")
        self.assertEqual(provider, "cerebras")
        self.assertIsNone(base_url)  # online — no local URL

    def test_list_available_models_includes_cerebras(self):
        entries = [e for e in list_available_models("cerebras")]
        self.assertTrue(entries)
        models = {e["model"] for e in entries}
        self.assertIn("qwen-3.8-27b", models)
        self.assertIn("gpt-oss-120b", models)


class TestConstructor(unittest.TestCase):
    """Constructor wires credentials, endpoint and per-model settings."""

    def test_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "CEREBRAS_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(Exception) as ctx:
                CerebrasBackend()
        self.assertIn("CEREBRAS_API_KEY", str(ctx.exception))

    def test_default_model_and_hosted_endpoint(self):
        backend, client_cls = _make_backend()
        self.assertEqual(backend.model, "qwen-3.8-27b")
        # Hosted API: no base_url passed to the SDK (it defaults internally).
        self.assertIsNone(backend.base_url)
        client_cls.assert_called_with(api_key="sk-test-cerebras")
        # Not "local" — the hosted endpoint is a real remote provider.
        self.assertFalse(backend.is_local)

    def test_base_url_override_respected(self):
        backend, client_cls = _make_backend(base_url="http://proxy.local:9000/v1")
        self.assertEqual(backend.base_url, "http://proxy.local:9000/v1")
        client_cls.assert_called_with(
            api_key="sk-test-cerebras",
            base_url="http://proxy.local:9000/v1",
        )
        self.assertTrue(backend.is_local)

    def test_per_model_max_completion(self):
        backend, _ = _make_backend(model="qwen-3.8-27b")
        self.assertEqual(backend.MAX_COMPLETION_TOKENS, 40_000)

    def test_display_name_and_context_window(self):
        backend, _ = _make_backend()
        self.assertEqual(backend.display_name, "Cerebras Qwen 3.8 27B")
        self.assertEqual(backend.context_window_size, 128_000)


class TestPricing(unittest.TestCase):
    """Cost accounting follows MODEL_PRICING.

    Cerebras cache reads are billed at the full input price (no hit
    discount), so the expected cost for a fully-cached input equals the
    uncached cost.  Expectations are derived from the class dict so vendor
    price changes never break the suite.
    """

    def test_pricing_structure_and_cost_formula(self):
        for model, price in CerebrasBackend.MODEL_PRICING.items():
            for key in ("input_token_cost", "output_token_cost",
                        "cache_read_cost"):
                self.assertGreater(price.get(key, 0), 0, f"{model}.{key}")
            # No cache discount: a cache read costs the same as a miss.
            self.assertEqual(price["cache_read_cost"], price["input_token_cost"])

            backend, _ = _make_backend(model=model)
            # Fully-cached input: cost == uncached cost (no discount).
            cost_cached = backend.calculate_cost(
                1_000_000, 1_000_000, cache_read_tokens=1_000_000)
            cost_uncached = backend.calculate_cost(
                1_000_000, 1_000_000, cache_read_tokens=0)
            self.assertAlmostEqual(cost_cached, cost_uncached)
            self.assertAlmostEqual(
                cost_uncached,
                price["input_token_cost"] + price["output_token_cost"],
            )


class TestReasoningRequest(unittest.TestCase):
    """Every API call requests the model's configured reasoning effort."""

    def test_extra_create_kwargs(self):
        backend, _ = _make_backend(model="qwen-3.8-27b")
        self.assertEqual(backend._extra_create_kwargs(),
                         {"reasoning_effort": "high"})
        backend2, _ = _make_backend(model="gpt-oss-120b")
        self.assertEqual(backend2._extra_create_kwargs(),
                         {"reasoning_effort": "medium"})


class TestStreaming(unittest.TestCase):
    """Streaming parses content, reasoning, usage and tool calls."""

    def _run_stream(self, backend, chunks, handler=None):
        """Point the mocked client at *chunks* and run one generation."""
        if handler is None:
            handler = __import__(
                "agents.llm_backend", fromlist=["StreamHandler"]).StreamHandler()
        backend.stream_handler = handler
        create = backend._client.chat.completions.create
        create.return_value = iter(chunks)
        ctx = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        return backend.generate_response("system", ctx)

    def test_collects_text_and_ignores_reasoning(self):
        backend, _ = _make_backend()
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning="thinking...", content=None, tool_calls=None))],
                usage=None),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="Hel", tool_calls=None))],
                usage=None),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="lo", tool_calls=None))],
                usage=None),
            SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5,
                prompt_tokens_details=None)),
        ]
        result = self._run_stream(backend, chunks)
        self.assertEqual(result, "Hello")
        self.assertEqual(backend.last_input_tokens, 10)
        self.assertEqual(backend.last_output_tokens, 5)
        # Reasoning was streamed to the handler but NOT collected.
        self.assertEqual(
            backend.stream_handler.get_buffered_reasoning(), "thinking...")
        self.assertNotIn("thinking", result)

    def test_sends_reasoning_effort_and_stream_options(self):
        backend, _ = _make_backend(model="qwen-3.8-27b")
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="ok", tool_calls=None))],
                usage=None),
            SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1,
                prompt_tokens_details=None)),
        ]
        self._run_stream(backend, chunks)
        kwargs = backend._client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen-3.8-27b")
        self.assertEqual(kwargs["reasoning_effort"], "high")
        self.assertTrue(kwargs["stream"])
        self.assertEqual(kwargs["stream_options"], {"include_usage": True})
        self.assertEqual(kwargs["max_completion_tokens"], 40_000)

    def test_tool_calls_are_logged_not_collected(self):
        backend, _ = _make_backend()
        tc = SimpleNamespace(index=0, function=SimpleNamespace(
            name="mytool", arguments='{"a": 1}'))
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="done", tool_calls=[tc]))],
                usage=None),
            SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1,
                prompt_tokens_details=None)),
        ]
        handler = mock.MagicMock()
        result = self._run_stream(backend, chunks, handler=handler)
        self.assertEqual(result, "done")
        self.assertEqual(backend._pending_tool_calls, [])  # drained
        # The tool call was surfaced via the handler during the call.
        handler.on_tool_call.assert_called_once_with("mytool", '{"a": 1}')


if __name__ == "__main__":
    unittest.main()
