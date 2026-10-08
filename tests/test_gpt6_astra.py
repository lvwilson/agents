"""Tests for gpt-6-astra (OpenAI flagship, 2026-09) hosted-model support.

Pins:
* registration in the OpenAI backend's model tables (pricing, display
  name, 1.1M context) so --list-models, routing, the context-usage
  guard and banner all see it;
* hosted-model auto-detection: ``resolve_backend_target`` selects the
  openai provider for ``-m gpt-6-astra`` instead of treating the unknown
  name as a local server (the failure mode before registration);
* cost math: cached input billed at the declared $1/M hit rate (90% off),
  and unchanged results for the older GPT models (whose declared cache
  rates equal the legacy 50% heuristic exactly);
* reasoning-effort wiring: omitted by default (wire format unchanged),
  clamped to the GPT effort levels when requested (max→xhigh,
  none→low — the models cannot disable reasoning), always sent as
  ``reasoning={"effort": …}`` on the Responses API parameter the
  installed openai SDK documents;
* max output: gpt-6-astra raises the per-call ceiling to its 128K
  output limit; other models keep 16,384.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.backends import list_available_models  # noqa: E402
from agents.backends.openai_backend import OpenAIBackend  # noqa: E402
from agents.llm_backend import StreamHandler  # noqa: E402


def _make_openai(model="gpt-6-astra", **kwargs):
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
        with mock.patch("openai.OpenAI"):
            return OpenAIBackend(model=model, **kwargs)


def _run_one_turn(backend):
    """Drive one full generate_response() against a mocked client.

    The fake stream emits one text delta and a response.completed
    event with usage, then returns.  Returns the kwargs the backend
    passed to ``client.responses.create(…)``.
    """
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="PONG"),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=1_000,
                    output_tokens=100,
                    input_tokens_details=SimpleNamespace(cached_tokens=500),
                )
            ),
        ),
    ]
    backend._client.responses.create.return_value = iter(events)
    backend.generate_response("sys", [
        {"role": "user", "content": [{"type": "text", "text": "ping"}]}
    ])
    return backend._client.responses.create.call_args.kwargs


class TestRegistration(unittest.TestCase):
    """gpt-6-astra is a registered, hosted OpenAI model."""

    def test_in_model_table(self):
        entries = {e["model"]: e for e in list_available_models("openai")}
        self.assertIn("gpt-6-astra", entries)
        e = entries["gpt-6-astra"]
        self.assertEqual(e["display"], "GPT-6 Astra")
        self.assertEqual(e["input_cost"], 10.00)
        self.assertEqual(e["output_cost"], 50.00)
        self.assertEqual(e["cache_read_cost"], 1.00)
        self.assertEqual(e["context"], 1_100_000)

    def test_display_and_context(self):
        backend = _make_openai()
        self.assertEqual(backend.display_name, "GPT-6 Astra")
        self.assertEqual(backend.context_window_size, 1_100_000)

    def test_hosted_catalog_and_resolution(self):
        # The live catalog (same source --list-models renders) must map
        # the name to openai, and the resolver must route it to the
        # hosted provider with NO base_url (pre-registration this name
        # fell through to the local server — the "does not work" bug).
        catalog = agents_module._online_models()
        self.assertEqual(catalog.get("gpt-6-astra"), "openai")
        provider, model, base_url = agents_module.resolve_backend_target(
            model_name="gpt-6-astra"
        )
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-6-astra")
        self.assertIsNone(base_url)


class TestCost(unittest.TestCase):
    """Cache hits bill at the declared rate; older models unchanged."""

    def test_astra_cache_hit_rate(self):
        backend = _make_openai()
        # 1000 input (500 cached) + 100 output:
        # 500 * $10 + 500 * $1 + 100 * $50 all per M.
        self.assertAlmostEqual(
            backend.calculate_cost(1000, 100, cache_read_tokens=500),
            5_000 + 500 + 5_000,
            places=6,
        )
        self.assertAlmostEqual(
            backend.calculate_cost(1000, 100, cache_read_tokens=0),
            10_000 + 5_000,
            places=6,
        )

    def test_legacy_models_keep_legacy_rates(self):
        # cache_read_cost == input*0.5 for every pre-existing model, so
        # the declared-rate formula reproduces the old 50%-heuristic
        # results exactly.
        backend = _make_openai(model="gpt-5.3-codex")
        self.assertAlmostEqual(
            backend.calculate_cost(1_000_000, 0, cache_read_tokens=500_000),
            500_000 * 3.00 + 500_000 * 1.50,
            places=6,
        )
        mini = _make_openai(model="gpt-5.2-mini")
        self.assertAlmostEqual(
            mini.calculate_cost(1_000_000, 0, cache_read_tokens=500_000),
            500_000 * 0.15 + 500_000 * 0.075,
            places=6,
        )


class TestEffort(unittest.TestCase):
    """reasoning={"effort": …} is clamped, and omitted by default."""

    def test_default_sends_no_reasoning_param(self):
        kwargs = _run_one_turn(_make_openai())
        self.assertNotIn("reasoning", kwargs)
        # Wire format for existing deployments is otherwise unchanged.
        self.assertEqual(kwargs["model"], "gpt-6-astra")
        self.assertEqual(kwargs["temperature"], 1.0)
        self.assertTrue(kwargs["stream"])

    def test_effort_rank_clamps_to_gpt_levels(self):
        # Canonical ranks from parse_effort(): 0=none … 5=max.
        # max(5) → xhigh (SDK ceiling); none(0) → low (models cannot
        # disable reasoning).
        self.assertEqual(
            _run_one_turn(_make_openai(reasoning_effort=5))["reasoning"],
            {"effort": "xhigh"},
        )
        self.assertEqual(
            _run_one_turn(_make_openai(reasoning_effort=0))["reasoning"],
            {"effort": "low"},
        )
        self.assertEqual(
            _run_one_turn(_make_openai(reasoning_effort=3))["reasoning"],
            {"effort": "high"},
        )
        self.assertEqual(
            _run_one_turn(_make_openai(reasoning_effort=2))["reasoning"],
            {"effort": "medium"},
        )


class TestOutputCap(unittest.TestCase):
    """gpt-6-astra may use its 128K output budget per call."""

    def test_astra_gets_128k(self):
        backend = _make_openai()
        self.assertEqual(backend.MAX_COMPLETION_TOKENS, 128_000)
        kwargs = _run_one_turn(backend)
        self.assertEqual(kwargs["max_output_tokens"], 128_000)

    def test_other_models_keep_default(self):
        backend = _make_openai(model="gpt-5.3-codex")
        self.assertEqual(backend.MAX_COMPLETION_TOKENS, 16_384)
        kwargs = _run_one_turn(backend)
        self.assertEqual(kwargs["max_output_tokens"], 16_384)


if __name__ == "__main__":
    unittest.main()
