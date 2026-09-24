"""Regression tests: context tracking must never decrease while in use.

Two real defects were found by auditing the context-calculation paths
and confirmed with live provider probes:

1. **Mid-turn one-shot calls on the shared client.**  The ``summarize``
   tool (and the episode-squash summarizer) issue one-shot
   ``generate_response`` calls on the *same* backend instance the agent
   uses for its main conversation, with a fresh, much smaller context.
   In ``_iterate`` those calls fire inside ``process_content``, and the
   context-usage guard reads ``last_total_context_tokens`` *afterwards*
   — so a summarize command made tracked context collapse from e.g.
   100K to ~400 mid-session, and a once-per-session wrap-up warning
   could be permanently lost.  The one-shot wrappers now snapshot and
   restore the main conversation's display token fields (cost and
   call_count still accumulate — real spend).

2. **Zero-out on missing usage.**  When a streamed response carries no
   usage object (local OpenAI-compatible servers that ignore
   ``stream_options``, a lost final chunk, …) the ``openai_compat``,
   ``openai`` (Responses) and ``gemini`` backends used to reset
   ``last_input_tokens`` / ``last_output_tokens`` to 0, so
   ``last_total_context_tokens`` — what the guard, the per-turn header
   and the session file all read — dropped to ~0 while the session was
   mid-flight.  They now keep the last known values (context only ever
   grows, so the stale value is a safe lower bound); cost can't be
   computed without usage and isn't accumulated for such calls.

Provider usage semantics were verified live and are NOT bugs
(documented here so the probes are not re-run needlessly):

* Cerebras chat-completions: ``prompt_tokens`` INCLUDES the cached
  prefix and grows monotonically (12048 → 12071 → 12094 across turns
  of a ~54k-char prompt); ``cached_tokens`` is a sub-detail only.
* DeepSeek / MiniMax (Anthropic-compatible): raw ``input_tokens``
  reports the uncached REMAINDER (12039 → 151), but the backend adds
  ``cache_read_input_tokens`` (≈11.9K) back, so
  ``last_total_context_tokens`` reconstructs the full prompt.
* Gemini: ``prompt_token_count`` covers the full prompt even on
  server-side ``cached_content`` suffix calls (12015 → 12027 →
  12027).
* Anthropic: ``input_tokens`` excludes cache fields, which the backend
  adds back (standard semantics).
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agents import Agent  # noqa: E402
from agents.backends.gemini_backend import GeminiBackend  # noqa: E402
from agents.backends.openai_backend import OpenAIBackend  # noqa: E402
from agents.backends.cerebras_backend import CerebrasBackend  # noqa: E402


# ── Shared agent fixtures ────────────────────────────────────────────

_FAKE_CONFIG = {
    "system_prompt": "IMMUTABLE SYSTEM PROMPT",
    "overbudget": "over budget",
    "provider": "kimi",
}


def _make_client(mock_backend):
    """Give the mocked backend the attributes _iterate / guards read."""
    client = mock_backend.return_value
    client.display_name = "MockModel"
    client.context_window_size = 200_000
    client.cost = 0.0
    client.cost_without_cache = 0.0
    client.peak_context_tokens = 0
    client.last_input_tokens = 0
    client.last_output_tokens = 0
    client.last_total_context_tokens = 0
    client.call_count = 0
    client.total_output_tokens = 0
    client.total_call_duration = 0.0
    client.output_rate_tokens_per_sec = None
    client.cost_per_hour = None
    return client


def _patched_agent(config=None, session_id="tst1"):
    """Construct a minimal Agent with config/memory/backend mocked out."""
    with mock.patch.object(
            agents_module, "read_configuration",
            return_value=config or _FAKE_CONFIG), \
         mock.patch.object(agents_module, "format_memory_view",
                           return_value=""), \
         mock.patch.object(agents_module, "notes_need_compact",
                           return_value=False), \
         mock.patch.object(agents_module, "create_backend") as mock_backend, \
         mock.patch.object(agents_module, "print_banner"), \
         mock.patch.object(agents_module, "print_iteration_header"), \
         mock.patch.object(agents_module, "print_error"):
        return Agent("fake.yaml", "do the thing", session_id=session_id), mock_backend


# Main-conversation values the fake generates report (largish, so the
# contrast with the one-shot's tiny conversation is unambiguous).
_MAIN_IN, _MAIN_OUT, _MAIN_TOTAL = 90_000, 10_000, 100_000


def _two_context_gen(client):
    """Fake generate_response for a main + one-shot conversation.

    Distinguishes the one-shot by its system prompt (the summarize /
    squash tools use summarizer prompts), and clobbers the tracking
    fields the way a REAL backend would for the smaller conversation —
    which is exactly what the fix must undo.
    """
    def gen(system_prompt, context):
        if "summarizer" in system_prompt.lower():
            # One-shot side channel: small fresh conversation.
            client.last_input_tokens = 300
            client.last_output_tokens = 100
            client.last_total_context_tokens = 400
            client.call_count += 1
            client.cost += 0.05
            return "SUMMARY TEXT"
        # Main conversation turn.
        client.last_input_tokens = _MAIN_IN
        client.last_output_tokens = _MAIN_OUT
        client.last_total_context_tokens = _MAIN_TOTAL
        client.last_call_duration = 1.0
        client.call_count += 1
        client.cost += 0.5
        return 'Command: run_console_command "echo ok"'
    return gen


# ── 1. Mid-turn one-shot calls must not clobber tracking ────────────

class TestOneShotSummarizeDoesNotClobberTracking(unittest.TestCase):
    """A summarize command mid-turn keeps the main conversation's
    tracked context intact for the usage guard and the next header."""

    def test_summarize_command_keeps_main_tracking_for_guard(self):
        agent, mock_backend = _patched_agent()
        client = _make_client(mock_backend)
        client.generate_response = _two_context_gen(client)

        from agents.tools import summarize as _sum_mod

        def fake_process_content(response, blocked_commands=None):
            # What the real dispatch does for 'Command: summarize …':
            # call the registered one-shot LLM with a small fresh
            # conversation (the registered closure must restore the
            # main conversation's tracking afterwards).
            _sum_mod._ensure_llm()("You are a summarizer.", "file content")
            return ("summarizer output", [])

        with mock.patch.object(
                agents_module, "process_content",
                side_effect=fake_process_content), \
             mock.patch.object(agents_module, "filter_content",
                               side_effect=lambda s: s), \
             mock.patch.object(agents_module, "print_clipped"), \
             mock.patch.object(agents_module, "safe_console_print"), \
             mock.patch.object(agents_module, "print_context_used"), \
             mock.patch.object(agents_module, "print_budget_warning"):
            running = agent._iterate()

        self.assertTrue(running)

        # The one-shot ran (call_count includes it) and its spend was
        # kept — real LLM usage is still accounted.
        self.assertEqual(client.call_count, 2)
        self.assertAlmostEqual(client.cost, 0.55)

        # …but the main conversation's tracking survived, so the next
        # header and the session file record the true size.
        self.assertEqual(client.last_input_tokens, _MAIN_IN)
        self.assertEqual(client.last_output_tokens, _MAIN_OUT)
        self.assertEqual(client.last_total_context_tokens, _MAIN_TOTAL)

        # And the guard measured the MAIN conversation: 100K of a 200K
        # window is exactly the 50% informational threshold.  With the
        # bug, the guard would have measured the one-shot's 400 tokens
        # (0.2%) and stayed silent.
        tool_results = [m["content"][0]["text"] for m in agent.context
                        if m["role"] == "user"][-1]
        self.assertIn("Informational", tool_results)
        self.assertIn(f"{_MAIN_TOTAL:,} of {client.context_window_size:,}",
                      tool_results)
        self.assertEqual(agent._context_guard_fired, {"info"})


class TestOneShotSquashDoesNotClobberTracking(unittest.TestCase):
    """The end-of-session episode-squash one-shot (inside run_agent)
    must leave the main conversation's tracking intact as well."""

    def test_squash_after_session_keeps_main_tracking(self):
        config = {**_FAKE_CONFIG, "squash_prompt": "SQUASH PROMPT"}

        def _one_turn(self):
            # Exactly one real command turn — run_agent saves the context
            # BEFORE the post-run episode turns, so the main conversation
            # must have produced the tracking values.  A plain function
            # (not a Mock side_effect) becomes a bound method when set as
            # the class attribute, so it receives self.
            return self._iterate()

        with mock.patch.object(
                agents_module, "read_configuration",
                return_value=config), \
             mock.patch.object(agents_module, "format_memory_view",
                               return_value=""), \
             mock.patch.object(agents_module, "notes_need_compact",
                               return_value=False), \
             mock.patch.object(agents_module, "create_backend") as mock_backend, \
             mock.patch.object(agents_module, "print_banner"), \
             mock.patch.object(agents_module, "print_iteration_header"), \
             mock.patch.object(agents_module, "print_error"), \
             mock.patch.object(agents_module, "filter_content",
                               side_effect=lambda s: s), \
             mock.patch.object(agents_module, "save_session") as mock_save, \
             mock.patch.object(agents_module, "add_episode"), \
             mock.patch.object(agents_module, "get_episode_count",
                               return_value=999), \
             mock.patch.object(agents_module, "squash_episodes") as mock_squash, \
             mock.patch.object(agents_module, "is_git_repo",
                               return_value=False), \
             mock.patch.object(Agent, "run", _one_turn), \
             mock.patch.object(Agent, "_seed_first_turn"), \
             mock.patch.object(agents_module, "process_content",
                               return_value=("ok output", [])):
            agent = Agent("fake.yaml", "do the thing", session_id="tstsq")
            client = _make_client(mock_backend)

            def gen(system_prompt, context):
                if "summarizer" in system_prompt.lower():
                    client.last_input_tokens = 300
                    client.last_output_tokens = 100
                    client.last_total_context_tokens = 400
                    return "SQUASHED"
                client.last_input_tokens = _MAIN_IN
                client.last_output_tokens = _MAIN_OUT
                client.last_total_context_tokens = _MAIN_TOTAL
                client.last_call_duration = 1.0
                return "SESSION SUMMARY"

            client.generate_response = gen

            def fake_squash(fn):
                # The real driver hands its episodes to the closure.
                fn("episode one\nepisode two")

            mock_squash.side_effect = fake_squash

            agents_module.run_agent(
                "fake.yaml", "do the thing", 2.0,
                session_id="tstsq",
            )

        # The squash ran through the real closure.
        mock_squash.assert_called_once()

        # Session file carries the main conversation's size, not the
        # squash one-shot's 400.
        saved = mock_save.call_args[0][2]
        self.assertEqual(saved["total_context_tokens"], _MAIN_TOTAL)

        # And the in-memory tracking survived as well.
        self.assertEqual(client.last_input_tokens, _MAIN_IN)
        self.assertEqual(client.last_output_tokens, _MAIN_OUT)
        self.assertEqual(client.last_total_context_tokens, _MAIN_TOTAL)


# ── 2. No-usage responses keep the last known tracking ───────────────

class TestOpenAICompatNoUsageKeepsLastValues(unittest.TestCase):
    """openai_compat (the base for Cerebras, Kimi, local servers): a
    stream without a usage chunk must not zero the tracking."""

    def test_missing_usage_chunk_keeps_last_tracking(self):
        with mock.patch.dict(os.environ, {"CEREBRAS_API_KEY": "sk-test"}):
            with mock.patch("cerebras.cloud.sdk.Cerebras") as client_cls:
                backend = CerebrasBackend(model="qwen-3.8-27b")

        ctx = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

        # Turn 1: usage present.
        client_cls.return_value.chat.completions.create.return_value = iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="a", tool_calls=None))], usage=None),
            SimpleNamespace(choices=[], usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5,
                prompt_tokens_details=None)),
        ])
        self.assertEqual(backend.generate_response("s", ctx), "a")
        self.assertEqual(backend.last_total_context_tokens, 15)
        cost_after_one = backend.cost
        self.assertGreater(cost_after_one, 0)

        # Turn 2: no usage chunk at all (e.g. local server ignored
        # stream_options), bigger conversation.
        client_cls.return_value.chat.completions.create.return_value = iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
                reasoning=None, content="b", tool_calls=None))], usage=None),
        ])
        self.assertEqual(backend.generate_response("s", ctx), "b")

        # Tracking keeps the last known value instead of collapsing to 0…
        self.assertEqual(backend.last_input_tokens, 10)
        self.assertEqual(backend.last_output_tokens, 5)
        self.assertEqual(backend.last_total_context_tokens, 15)
        self.assertEqual(backend.peak_context_tokens, 15)
        # …the call was still counted (it really happened)…
        self.assertEqual(backend.call_count, 2)
        # …and no phantom cost was invented for the unmeasured call.
        self.assertAlmostEqual(backend.cost, cost_after_one)


class TestOpenAIResponsesNoUsageKeepsLastValues(unittest.TestCase):
    """openai (Responses API): a stream without response.completed /
    usage must not zero the tracking."""

    def test_missing_completion_event_keeps_last_tracking(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with mock.patch("openai.OpenAI") as client_cls:
                backend = OpenAIBackend(model="gpt-5.2")

        ctx = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        usage = SimpleNamespace(input_tokens=10, output_tokens=5,
                                input_tokens_details=None)

        client_cls.return_value.responses.create.return_value = iter([
            SimpleNamespace(type="response.output_text.delta", delta="a"),
            SimpleNamespace(type="response.completed",
                            response=SimpleNamespace(usage=usage)),
        ])
        self.assertEqual(backend.generate_response("s", ctx), "a")
        self.assertEqual(backend.last_total_context_tokens, 15)
        cost_after_one = backend.cost
        self.assertGreater(cost_after_one, 0)

        # Turn 2: stream ends without the response.completed usage event.
        client_cls.return_value.responses.create.return_value = iter([
            SimpleNamespace(type="response.output_text.delta", delta="b"),
        ])
        self.assertEqual(backend.generate_response("s", ctx), "b")

        self.assertEqual(backend.last_input_tokens, 10)
        self.assertEqual(backend.last_output_tokens, 5)
        self.assertEqual(backend.last_total_context_tokens, 15)
        self.assertEqual(backend.peak_context_tokens, 15)
        self.assertEqual(backend.call_count, 2)
        self.assertAlmostEqual(backend.cost, cost_after_one)


class TestGeminiNoUsageKeepsLastValues(unittest.TestCase):
    """gemini: a stream without usage_metadata must not zero the
    tracking."""

    def test_missing_usage_metadata_keeps_last_tracking(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "k"}):
            with mock.patch("google.genai.Client") as client_cls:
                backend = GeminiBackend(model="gemini-3-flash-preview")

        ctx = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        stream = client_cls.return_value.models.generate_content_stream

        stream.return_value = iter([
            SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=100, candidates_token_count=5,
                    cached_content_token_count=0),
                text="a", function_calls=None),
        ])
        self.assertEqual(backend.generate_response("s", ctx), "a")
        self.assertEqual(backend.last_total_context_tokens, 105)
        cost_after_one = backend.cost
        self.assertGreater(cost_after_one, 0)

        # Turn 2: no usage_metadata on any chunk.
        stream.return_value = iter([
            SimpleNamespace(usage_metadata=None, text="b",
                            function_calls=None),
        ])
        self.assertEqual(backend.generate_response("s", ctx), "b")

        self.assertEqual(backend.last_input_tokens, 100)
        self.assertEqual(backend.last_output_tokens, 5)
        self.assertEqual(backend.last_total_context_tokens, 105)
        self.assertEqual(backend.peak_context_tokens, 105)
        self.assertEqual(backend.call_count, 2)
        self.assertAlmostEqual(backend.cost, cost_after_one)


if __name__ == "__main__":
    unittest.main()
