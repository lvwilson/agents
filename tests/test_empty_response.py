"""Regression tests for blank-response feedback and Kimi reasoning streaming.

Bug 1: a model turn containing only thinking tokens (no visible text)
raised "No text content found in model response", which propagated out
of the agent loop and ended the session.  The loop must instead feed
the failure back to the model so it can retry — commands written into
reasoning blocks are invisible to the harness.

Bug 2: the Kimi K3 backend discarded ``reasoning_content`` entirely.
Thinking tokens must stream through the reasoning hooks so the UI can
render them dimmed (the Anthropic backend already does this).
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agents import Agent  # noqa: E402
from agents.llm_backend import EmptyResponseError, StreamHandler  # noqa: E402
from agents.backends.kimi_backend import KimiBackend  # noqa: E402

_FAKE_CONFIG = {
    "system_prompt": "IMMUTABLE SYSTEM PROMPT",
    "overbudget": "over budget",
    "provider": "kimi",
}


def _make_agent(task="do the thing"):
    """Construct a minimal Agent with config/memory/backend mocked out."""
    with mock.patch.object(agents_module, "read_configuration", return_value=_FAKE_CONFIG), \
         mock.patch.object(agents_module, "format_memory_view", return_value=""), \
         mock.patch.object(agents_module, "notes_need_compact", return_value=False), \
         mock.patch.object(agents_module, "create_backend") as mock_backend, \
         mock.patch.object(agents_module, "print_banner"), \
         mock.patch.object(agents_module, "print_iteration_header"), \
         mock.patch.object(agents_module, "print_error"):
        client = mock_backend.return_value
        client.display_name = "MockModel"
        client.context_window_size = 200_000
        client.cost = 0.0
        client.cost_without_cache = 0.0
        client.peak_context_tokens = 0
        client.last_input_tokens = 0
        client.last_output_tokens = 0
        client.last_total_context_tokens = 0
        return Agent("fake.yaml", task, session_id="tst1")


class TestEmptyResponseFeedback(unittest.TestCase):
    """An EmptyResponseError from the backend must be fed back to the
    model as an instruction, not treated as session end."""

    def test_empty_response_injects_feedback_and_continues(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            side_effect=EmptyResponseError("No text content found in model response")
        )
        running = agent._iterate()
        self.assertTrue(running, "blank response must not end the session")
        self.assertEqual(agent._empty_response_count, 1)
        last = agent.context[-1]
        self.assertEqual(last["role"], "user")
        feedback = last["content"][0]["text"]
        self.assertIn("no text content", feedback)
        # Feedback must tell the model how to avoid the failure.
        self.assertIn("Command:", feedback)
        self.assertIn("thinking", feedback)

    def test_empty_response_recovers_on_next_iteration(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            side_effect=[
                EmptyResponseError("No text content found in model response"),
                "no commands here",
                "still no commands",
            ]
        )
        self.assertTrue(agent._iterate())
        # "no commands here" has no completion block either, so the
        # no-output reminder fires once before the session may end.
        self.assertTrue(agent._iterate())
        running = agent._iterate()
        self.assertFalse(running)  # second content-free turn → End.
        self.assertEqual(agent._empty_response_count, 0)
        # The successful reply was appended as an assistant message.
        roles = [m["role"] for m in agent.context]
        self.assertIn("assistant", roles)

    def test_three_consecutive_empties_reraise(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            side_effect=EmptyResponseError("No text content found in model response")
        )
        agent._iterate()
        agent._iterate()
        with self.assertRaises(EmptyResponseError):
            agent._iterate()
        self.assertEqual(agent._empty_response_count, 3)

    def test_feedback_appended_each_retry(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            side_effect=EmptyResponseError("No text content found in model response")
        )
        agent._iterate()
        agent._iterate()
        user_msgs = [m for m in agent.context if m["role"] == "user"]
        # First message (guard+task) plus one feedback per blank turn.
        self.assertEqual(len(user_msgs), 3)


class _RecordingHandler(StreamHandler):
    """StreamHandler that records the order of every hook invocation."""

    def __init__(self):
        super().__init__()
        self.events = []

    def on_stream_reasoning_start(self):
        super().on_stream_reasoning_start()
        self.events.append(("reasoning_start",))

    def on_stream_reasoning_token(self, token):
        super().on_stream_reasoning_token(token)
        self.events.append(("reasoning_token", token))

    def on_stream_reasoning_end(self):
        super().on_stream_reasoning_end()
        self.events.append(("reasoning_end",))

    def on_stream_token(self, token):
        super().on_stream_token(token)
        self.events.append(("token", token))


def _reasoning_event(text):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            delta=SimpleNamespace(reasoning_content=text, content=None))],
    )


def _content_event(text):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            delta=SimpleNamespace(reasoning_content=None, content=text))],
    )


def _usage_event():
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                              prompt_tokens_details=None),
        choices=[],
    )


def _make_kimi(events):
    """Build a KimiBackend with __init__ bypassed and a fake client."""
    backend = KimiBackend.__new__(KimiBackend)
    backend.model = "kimi-k3"
    backend.base_url = None
    backend.is_local = False
    backend.stream_handler = _RecordingHandler()
    backend.temperature = 1.0
    backend.cost = 0.0
    backend.cost_without_cache = 0.0
    backend.call_count = 0
    backend.last_input_tokens = 0
    backend.last_output_tokens = 0
    backend.last_total_context_tokens = 0
    backend.peak_context_tokens = 0
    backend._client = mock.Mock()
    backend._client.chat.completions.create.return_value = iter(events)
    return backend


class TestKimiReasoningStreaming(unittest.TestCase):
    """Kimi ``reasoning_content`` must stream through the reasoning hooks
    (rendered dim in the UI) and never leak into the content text."""

    def test_reasoning_streams_before_content(self):
        backend = _make_kimi([
            _reasoning_event("let me "),
            _reasoning_event("think…"),
            _content_event("visible "),
            _content_event("answer"),
            _usage_event(),
        ])
        text, _usage = backend._get_response("sys", [])
        self.assertEqual(text, "visible answer")
        self.assertEqual(
            backend.stream_handler.events,
            [
                ("reasoning_start",),
                ("reasoning_token", "let me "),
                ("reasoning_token", "think…"),
                ("reasoning_end",),
                ("token", "visible "),
                ("token", "answer"),
            ],
        )

    def test_reasoning_only_response_raises_empty(self):
        backend = _make_kimi([
            _reasoning_event("commands written to thoughts"),
            _usage_event(),
        ])
        with self.assertRaises(EmptyResponseError) as ctx:
            backend.generate_response("sys", [])
        self.assertIn("No text content found in model response", str(ctx.exception))
        # Reasoning was still displayed before the failure surfaced.
        self.assertIn(("reasoning_token", "commands written to thoughts"),
                      backend.stream_handler.events)


if __name__ == "__main__":
    unittest.main()
