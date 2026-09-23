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
from agents.llm_backend import (  # noqa: E402
    EmptyResponseError,
    StreamHandler,
    merge_consecutive_messages,
)


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


class TestMultiTurnReasoningEcho(unittest.TestCase):
    """Cerebras is a stateless API: the model's own historical thinking
    is preserved only when each assistant turn is re-sent with its
    ``reasoning`` field (docs: "Multi-turn reasoning and
    clear_thinking").  ``clear_thinking`` is never sent — omission
    means the server preserves the reasoning, which is the desired
    behaviour.
    """

    def test_format_messages_echoes_assistant_reasoning(self):
        backend, _ = _make_backend()
        ctx = [
            {"role": "user",
             "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant",
             "content": [{"type": "text", "text": "turn 1 answer"}],
             "reasoning": "turn 1 thinking"},
            {"role": "user",
             "content": [{"type": "text", "text": "second"}]},
        ]
        msgs = backend._format_messages("sys", ctx)
        self.assertEqual(
            msgs,
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "turn 1 answer",
                 "reasoning": "turn 1 thinking"},
                {"role": "user", "content": "second"},
            ],
        )

    def test_no_reasoning_field_when_turn_had_none(self):
        backend, _ = _make_backend()
        msgs = backend._format_messages(
            "sys",
            [{"role": "assistant",
              "content": [{"type": "text", "text": "no thinking"}]}],
        )
        self.assertNotIn("reasoning", msgs[0])

    def test_clear_thinking_never_sent(self):
        backend, _ = _make_backend(model="qwen-3.8-27b")
        self.assertNotIn("clear_thinking", backend._extra_create_kwargs())
        # Same when a user effort is set.
        backend.reasoning_effort = 2  # canonical rank for "medium"
        self.assertNotIn("clear_thinking", backend._extra_create_kwargs())

    def test_base_class_does_not_echo(self):
        """The shared base (llama.cpp / vLLM / local OpenAI-compatible
        servers) must keep the plain {role, content} wire format even
        when the context carries a reasoning field — those servers
        either reject unknown fields or already handle thinking
        themselves.
        """
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with mock.patch("openai.OpenAI"):
                backend = OpenAICompatBackend(model="local-model")
        msgs = backend._format_messages(
            "sys",
            [{"role": "assistant",
              "content": [{"type": "text", "text": "x"}],
              "reasoning": "y"}],
        )
        self.assertEqual(msgs[1], {"role": "assistant", "content": "x"})
        self.assertEqual(msgs[0], {"role": "system", "content": "sys"})

    def test_captures_streamed_reasoning(self):
        """last_reasoning accumulates the streamed thinking tokens and
        is committed only after a completed stream."""
        backend, _ = _make_backend()
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning="thinking a", content=None, tool_calls=None))],
                usage=None),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="Hello", tool_calls=None))],
                usage=None),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning="more thinking", content=None, tool_calls=None))],
                usage=None),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="!", tool_calls=None))],
                usage=None),
            SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5,
                prompt_tokens_details=None)),
        ]
        handler = StreamHandler()
        backend.stream_handler = handler
        backend._client.chat.completions.create.return_value = iter(chunks)
        result = backend.generate_response(
            "system",
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        self.assertEqual(result, "Hello!")
        # Reasoning chunks are captured in arrival order (raw
        # concatenation, no separators) and kept out of the text.
        self.assertEqual(backend.last_reasoning, "thinking amore thinking")
        self.assertNotIn("thinking", result)


class TestMergePreservesExtraKeys(unittest.TestCase):
    """merge_consecutive_messages must not drop assistant 'reasoning'
    fields when it folds same-role runs together."""

    def test_merge_keeps_extra_keys(self):
        ctx = [
            {"role": "assistant",
             "content": [{"type": "text", "text": "a"}],
             "reasoning": "r1"},
            {"role": "assistant",
             "content": [{"type": "text", "text": "b"}]},
            {"role": "user",
             "content": [{"type": "text", "text": "c"}]},
            {"role": "user",
             "content": [{"type": "text", "text": "d"}],
             "extra": 1},
        ]
        merged = merge_consecutive_messages(ctx)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["reasoning"], "r1")
        # First occurrence wins on a key collision.
        self.assertEqual(merged[1]["extra"], 1)
        self.assertEqual(
            merged[0]["content"],
            [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])

    def test_merge_concatenates_reasoning(self):
        """When two consecutive assistant turns both carry reasoning,
        the merged message keeps BOTH (concatenated in order) — neither
        turn's thinking is silently dropped."""
        ctx = [
            {"role": "assistant",
             "content": [{"type": "text", "text": "a"}],
             "reasoning": "r1 "},
            {"role": "assistant",
             "content": [{"type": "text", "text": "b"}],
             "reasoning": "r2"},
        ]
        merged = merge_consecutive_messages(ctx)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["reasoning"], "r1 r2")


_FAKE_CONFIG = {
    "system_prompt": "TEST SYSTEM PROMPT",
    "overbudget": "over budget",
    "provider": "cerebras",
}


def _make_agent_with_client(client):
    """Build an Agent whose LLM client is the given backend instance."""
    with mock.patch.object(
            agents_module, "read_configuration", return_value=_FAKE_CONFIG), \
         mock.patch.object(agents_module, "format_memory_view",
                           return_value=""), \
         mock.patch.object(agents_module, "notes_need_compact",
                           return_value=False), \
         mock.patch.object(agents_module, "create_backend",
                           return_value=client), \
         mock.patch.object(agents_module, "print_banner"), \
         mock.patch.object(agents_module, "print_iteration_header"), \
         mock.patch.object(agents_module, "print_error"):
        return agents_module.Agent("fake.yaml", "do the thing",
                                   session_id="reason1")


class TestAgentReasoningCapture(unittest.TestCase):
    """Agent-level flow: per-turn reasoning is attached to the stored
    assistant message, consumed exactly once, and echoed back on the
    next Cerebras request."""

    def _run_two_turns(self):
        """Two real CerebrasBackend.generate_response() calls through a
        mocked SDK client, driving the full Agent._iterate loop."""
        handler = StreamHandler()

        def make_chunks(text, reasoning):
            chunks = []
            for r in reasoning:
                chunks.append(SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(
                        reasoning=r, content=None, tool_calls=None))],
                    usage=None))
            for t in text:
                chunks.append(SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(
                        reasoning=None, content=t, tool_calls=None))],
                    usage=None))
            chunks.append(SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=4,
                prompt_tokens_details=None)))
            return chunks

        with mock.patch.dict(os.environ,
                             {"CEREBRAS_API_KEY": "sk-test"}):
            with mock.patch("cerebras.cloud.sdk.Cerebras") as client_cls:
                backend = CerebrasBackend(stream_handler=handler)
        # The seed ends with a user message (as real harness flow does —
        # a tool-result always separates assistant turns), so no two
        # assistant messages are ever consecutive.
        seed = [{"role": "user",
                 "content": [{"type": "text", "text": "design it"}]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": "seed answer"}],
                 "reasoning": "seed thinking"},
                {"role": "user",
                 "content": [{"type": "text", "text": "ok, and add "
                                    "error handling"}]}]
        agent = _make_agent_with_client(backend)
        agent.context = list(seed)
        backend._client.chat.completions.create.side_effect = [
            iter(make_chunks(["ok", "."], ["think ", "one "])),
            iter(make_chunks(["ok", "2"], ["think ", "two "])),
        ]
        agent._iterate()
        agent._iterate()
        return agent, backend

    def test_reasoning_attached_and_echoed_next_turn(self):
        agent, backend = self._run_two_turns()

        # Every assistant turn's thinking was stored on its own message
        # (the seed's first, then each generated turn).
        assistants = [m for m in agent.context if m["role"] == "assistant"]
        self.assertEqual(assistants[0].get("reasoning"), "seed thinking")
        self.assertEqual(assistants[1].get("reasoning"), "think one ")
        self.assertEqual(assistants[2].get("reasoning"), "think two ")
        # ...and consumed: no stale value left on the client.
        self.assertEqual(backend.last_reasoning, "")

        # Turn 1's request carried only the seed's reasoning; turn 2's
        # request carried both.
        calls = backend._client.chat.completions.create.call_args_list
        self.assertEqual(
            [m.get("reasoning") for m in calls[0].kwargs["messages"]
             if m["role"] == "assistant"],
            ["seed thinking"])
        self.assertEqual(
            [m.get("reasoning") for m in calls[1].kwargs["messages"]
             if m["role"] == "assistant"],
            ["seed thinking", "think one "])

    def test_no_reasoning_field_when_turn_thinks_nothing(self):
        """A turn with no thinking stores no 'reasoning' key and sends
        a plain assistant message on the next request."""
        backend, _ = _make_backend()
        agent = _make_agent_with_client(backend)
        seed = [{"role": "user",
                 "content": [{"type": "text", "text": "design it"}]}]
        agent.context = list(seed)

        def chunks_no_reasoning():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="ok.", tool_calls=None))],
                usage=None)
            yield SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=1, completion_tokens=1,
                prompt_tokens_details=None))
        backend._client.chat.completions.create.return_value = (
            chunks_no_reasoning())
        agent._iterate()
        assistants = [m for m in agent.context
                      if m["role"] == "assistant"]
        self.assertNotIn("reasoning", assistants[0])
        sent = backend._client.chat.completions.create.call_args.kwargs
        self.assertEqual(
            [m.get("reasoning") for m in sent["messages"]
             if m["role"] == "assistant"],
            [])

    def test_blank_turn_discards_its_reasoning(self):
        """A blank turn (thinking only, no visible text) must NOT carry
        its reasoning onto the next turn — there is no assistant message
        to attach it to, so it is discarded on the spot."""
        backend, _ = _make_backend()
        agent = _make_agent_with_client(backend)
        # Simulate: the generation streamed thinking but no visible
        # text, so generate_response raised EmptyResponseError.
        backend.last_reasoning = "orphan thinking"
        backend.generate_response = mock.Mock(
            side_effect=EmptyResponseError("no text"))
        running = agent._iterate()
        self.assertTrue(running)
        # The orphan thinking was cleared from the client, so the next
        # real turn cannot inherit it.
        self.assertEqual(backend.last_reasoning, "")
        # No assistant message exists to carry it.
        self.assertFalse(
            [m for m in agent.context if m["role"] == "assistant"])
        # The blank-turn feedback was injected instead.
        user_texts = [m["content"][0]["text"]
                      for m in agent.context if m["role"] == "user"]
        self.assertTrue(any("no text content" in t for t in user_texts))


if __name__ == "__main__":
    unittest.main()
