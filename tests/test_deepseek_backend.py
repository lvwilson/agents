"""Tests for the DeepSeek backend.

DeepSeek exposes an Anthropic-compatible endpoint
(``https://api.deepseek.com/anthropic``), so ``DeepSeekBackend``
subclasses ``AnthropicBackend`` with three customisations:

* authentication via ``DEEPSEEK_API_KEY`` against the fixed endpoint,
* thinking (reasoning) always enabled, streamed via the reasoning
  hooks, with ``output_config={"effort": "max"}`` on every call —
  selecting a ``-max`` model means maximum reasoning,
* prompt-cache annotations and beta headers skipped (the endpoint
  ignores them), which falls out of the parent's ``is_local`` path.

These tests mock the ``anthropic`` SDK client so no real API calls are
made.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.backends import create_backend  # noqa: E402
from agents.backends.anthropic_backend import AnthropicBackend  # noqa: E402
from agents.backends.deepseek_backend import DeepSeekBackend  # noqa: E402


def _make_backend(**kwargs):
    """Construct a DeepSeekBackend with the SDK client mocked out."""
    with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test-deepseek"}):
        with mock.patch("anthropic.Anthropic") as client_cls:
            backend = DeepSeekBackend(**kwargs)
    return backend, client_cls


class TestRegistration(unittest.TestCase):
    """The provider is registered and the model auto-resolves."""

    def test_factory_creates_deepseek_backend(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
            with mock.patch("anthropic.Anthropic"):
                backend = create_backend("deepseek", model="deepseek-v4-pro")
        self.assertIsInstance(backend, DeepSeekBackend)
        self.assertIsInstance(backend, AnthropicBackend)

    def test_model_name_maps_to_provider(self):
        self.assertEqual(
            agents_module._ONLINE_MODELS["deepseek-v4-pro"], "deepseek"
        )

    def test_resolve_model_auto_detects_online(self):
        provider, base_url = agents_module.resolve_model("deepseek-v4-pro")
        self.assertEqual(provider, "deepseek")
        self.assertIsNone(base_url)  # online — no local URL


class TestConstructor(unittest.TestCase):
    """Constructor wires credentials, endpoint and max reasoning."""

    def test_missing_api_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "DEEPSEEK_API_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(Exception) as ctx:
                DeepSeekBackend()
        self.assertIn("DEEPSEEK_API_KEY", str(ctx.exception))

    def test_default_model_and_endpoint(self):
        backend, client_cls = _make_backend()
        self.assertEqual(backend.model, "deepseek-v4-pro")
        self.assertEqual(backend.base_url, "https://api.deepseek.com/anthropic")
        # The real client is authenticated with the DeepSeek key, not the
        # parent's placeholder "local" key.
        client_cls.assert_called_with(
            api_key="sk-test-deepseek",
            base_url="https://api.deepseek.com/anthropic",
        )

    def test_base_url_override_respected(self):
        backend, client_cls = _make_backend(base_url="http://proxy.local:9000")
        self.assertEqual(backend.base_url, "http://proxy.local:9000")
        client_cls.assert_called_with(
            api_key="sk-test-deepseek",
            base_url="http://proxy.local:9000",
        )

    def test_thinking_forced_on(self):
        backend, _ = _make_backend()
        self.assertTrue(backend._thinking_enabled)
        self.assertTrue(backend._supports_thinking_api)
        self.assertTrue(backend._use_thinking_stream)

    def test_display_name_and_context_window(self):
        backend, _ = _make_backend()
        # base_url is set, so display_name shows the endpoint host.
        self.assertIn("deepseek-v4-pro", backend.display_name)
        self.assertEqual(backend.context_window_size, 1_000_000)


class TestPricing(unittest.TestCase):
    """Cost accounting follows MODEL_PRICING.

    All expectations are derived from the class dict, so vendor price
    changes never break the suite — the tests pin structure and the
    cost formula, not numbers.
    """

    def test_pricing_structure_and_cost_formula(self):
        for model, price in DeepSeekBackend.MODEL_PRICING.items():
            # The three rates the cost formula relies on.
            for key in ("input_token_cost", "output_token_cost",
                        "cache_read_cost"):
                self.assertGreater(price.get(key, 0), 0, f"{model}.{key}")
            # The invariant behind the calculate_cost override: hits
            # are far cheaper than misses, so the parent's 10%-of-input
            # heuristic would be badly wrong for this provider.
            self.assertLess(price["cache_read_cost"],
                            price["input_token_cost"] * 0.05)

            backend, _ = _make_backend(model=model)
            # Formula: miss-rate input + hit-rate reads + output, with
            # cache creation free (automatic server-side caching).
            cost = backend.calculate_cost(
                1_000_000, 1_000_000,
                cache_creation_tokens=1_000_000,
                cache_read_tokens=1_000_000,
            )
            expected = (price["input_token_cost"]
                        + price["cache_read_cost"]
                        + price["output_token_cost"])
            self.assertAlmostEqual(cost, expected)

class TestMaxReasoningRequest(unittest.TestCase):
    """Every API call requests thinking + max effort."""

    def test_extra_stream_kwargs(self):
        backend, _ = _make_backend()
        self.assertEqual(
            backend._extra_stream_kwargs(),
            {"extra_body": {"output_config": {"effort": "max"}}},
        )

    def test_get_response_sends_thinking_and_effort(self):
        backend, _ = _make_backend()
        backend.stream_handler = __import__(
            "agents.llm_backend", fromlist=["StreamHandler"]).StreamHandler()

        # Mock the streaming context manager.
        stream = mock.MagicMock()
        stream.__iter__ = lambda self: iter([])
        final = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
            content=[SimpleNamespace(type="text", text="done")],
        )
        stream.get_final_message.return_value = final
        ctx = backend._client.messages.stream.return_value
        ctx.__enter__.return_value = stream

        result = backend.generate_response(
            "system", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        )
        self.assertEqual(result, "done")

        kwargs = backend._client.messages.stream.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek-v4-pro")
        # output_config is a DeepSeek extension the anthropic SDK doesn't
        # model, so it must travel via extra_body (merged into the JSON
        # request body) rather than as a direct stream() kwarg.
        self.assertEqual(
            kwargs["extra_body"], {"output_config": {"effort": "max"}}
        )
        self.assertNotIn("output_config", kwargs)
        # Thinking is requested; budget_tokens is ignored by DeepSeek but
        # harmless (parent builds "enabled" mode config).
        self.assertEqual(kwargs["thinking"]["type"], "enabled")
        # is_local path: plain-string system prompt, no beta headers.
        self.assertEqual(kwargs["system"], "system")
        self.assertNotIn("extra_headers", kwargs)


if __name__ == "__main__":
    unittest.main()
