"""Regression tests for free-form internal iterations.

Internal one-shot LLM calls — the episode summary
(:meth:`Agent._request_episode_summary`) and the commit-message
request (:meth:`Agent.request_commit_message`) — ask the model for
plain prose ("Do not include any commands").  They previously reused
the full :meth:`Agent._iterate` pipeline, so a correct prose reply
containing neither a ``Command:`` line nor a completion block
triggered the accidental-stop guard: the harness printed "Model
response contained neither commands nor a completion block",
injected a spurious NO_OUTPUT_REMINDER, and (on the second internal
call) complained again and "ended the session".  These calls now run
with ``free_form=True``, which skips command parsing and all guards.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agents import Agent, NO_OUTPUT_REMINDER  # noqa: E402

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


def _user_texts(agent):
    return [m["content"][0]["text"] for m in agent.context
            if m["role"] == "user"]


class TestFreeFormIterate(unittest.TestCase):
    """free_form=True must treat the reply as plain prose."""

    def test_plain_prose_no_reminder_no_tool_results(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value="I summarised the session in plain prose."
        )
        running = agent._iterate(free_form=True)
        self.assertFalse(running, "free-form turns never continue the loop")
        self.assertFalse(agent._no_output_reminded)
        self.assertNotIn(NO_OUTPUT_REMINDER, _user_texts(agent))
        # No framed tool-results ("End.") message was appended.
        self.assertFalse(any(agents_module.TOOL_RESULTS_HEADER in t
                             for t in _user_texts(agent)))
        # The prose reply itself is in context as an assistant message.
        self.assertEqual(agent.context[-1]["role"], "assistant")
        self.assertIn("plain prose", agent.context[-1]["content"][0]["text"])

    def test_command_like_text_is_not_executed(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value='Command: run_console_command "echo pwned"'
        )
        with mock.patch.object(agents_module, "process_content") as pc:
            running = agent._iterate(free_form=True)
        self.assertFalse(running)
        pc.assert_not_called()

    def test_identical_replies_do_not_trip_loop_guard(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value="the exact same sentence"
        )
        self.assertFalse(agent._iterate(free_form=True))
        # Must not raise RuntimeError despite the identical repeat.
        self.assertFalse(agent._iterate(free_form=True))
        self.assertEqual(agent._loop_count, 0)

    def test_reminder_state_left_untouched(self):
        agent = _make_agent()
        # Simulate a pending reminder from the main loop.
        agent._no_output_reminded = True
        agent.client.generate_response = mock.Mock(return_value="prose")
        agent._iterate(free_form=True)
        self.assertTrue(
            agent._no_output_reminded,
            "free-form turns must neither set nor clear the reminder flag",
        )


class TestInternalCallSitesUseFreeForm(unittest.TestCase):
    """Both internal one-shot calls must opt into free_form mode."""

    def test_episode_summary_uses_free_form(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value="Did some useful work this session."
        )
        summary = agent._request_episode_summary()
        self.assertEqual(summary, "Did some useful work this session.")
        self.assertFalse(agent._no_output_reminded)
        self.assertNotIn(NO_OUTPUT_REMINDER, _user_texts(agent))
        self.assertFalse(any(agents_module.TOOL_RESULTS_HEADER in t
                             for t in _user_texts(agent)))

    def test_commit_message_uses_free_form(self):
        agent = _make_agent()
        bt = "`" * 5
        agent.client.generate_response = mock.Mock(
            return_value=f"{bt}\nfix spurious no-output complaint\n{bt}"
        )
        msg = agent.request_commit_message()
        self.assertEqual(msg, "fix spurious no-output complaint")
        self.assertFalse(agent._no_output_reminded)
        self.assertNotIn(NO_OUTPUT_REMINDER, _user_texts(agent))

    def test_back_to_back_internal_calls_no_complaint(self):
        """The reported bug: both steps in sequence each complained."""
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            side_effect=[
                "Session summary in prose.",
                "concise commit message",
            ]
        )
        with mock.patch.object(agents_module, "print_error") as pe:
            summary = agent._request_episode_summary()
            commit = agent.request_commit_message()
        self.assertEqual(summary, "Session summary in prose.")
        self.assertEqual(commit, "concise commit message")
        pe.assert_not_called()
        self.assertNotIn(NO_OUTPUT_REMINDER, _user_texts(agent))


if __name__ == "__main__":
    unittest.main()
