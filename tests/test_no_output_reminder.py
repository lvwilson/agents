"""Regression tests for the accidental-stop reminder guard.

A model response containing neither a ``Command:`` line nor a
completion block previously ended the session immediately — even
though the model almost certainly did not intend to stop.  The loop
must now inject ONE reminder (explaining both the command mechanism
and the completion block) and give the model one chance to
self-correct; only a second consecutive content-free response ends
the session.  A response with a completion block but no commands is
an intentional stop and ends normally with no reminder.
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


def _completion_response():
    # extract_completion requires "Completion:" immediately after the
    # opening fence (no newline in between).
    bt = "`" * 5
    return ("All done.\n" + bt + "Completion: finished the task\n"
            "Success: True\n" + bt)


class TestNoOutputReminder(unittest.TestCase):
    """No commands + no completion block → one reminder, one chance."""

    def test_reminder_injected_and_loop_continues(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value="just some plain chatter, no commands"
        )
        running = agent._iterate()
        self.assertTrue(running, "first content-free response must not end the session")
        self.assertTrue(agent._no_output_reminded)
        last = agent.context[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"][0]["text"], NO_OUTPUT_REMINDER)

    def test_second_consecutive_blank_output_ends_session(self):
        agent = _make_agent()
        # Distinct strings per turn: identical consecutive responses are
        # caught by the anti-loop guard before the reminder logic runs.
        agent.client.generate_response = mock.Mock(
            side_effect=["nothing here", "still nothing"]
        )
        self.assertTrue(agent._iterate())
        running = agent._iterate()
        self.assertFalse(running, "second consecutive content-free response must end the session")
        self.assertTrue(agent._no_output_reminded)
        # Exactly one reminder was ever injected.
        reminders = [m for m in agent.context
                     if m["role"] == "user"
                     and m["content"][0]["text"] == NO_OUTPUT_REMINDER]
        self.assertEqual(len(reminders), 1)

    def test_recovery_with_command_rearms_reminder(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            side_effect=[
                "chatter",                       # no commands → reminder
                "Command: run_console_command \"echo hi\"",  # recovers
                "more chatter",                  # new incident → new reminder
                "still more chatter",            # second strike → end
            ]
        )
        self.assertTrue(agent._iterate())    # reminder 1
        self.assertTrue(agent._no_output_reminded)
        self.assertTrue(agent._iterate())    # command ran; flag reset
        self.assertFalse(agent._no_output_reminded)
        self.assertTrue(agent._iterate())    # reminder 2 (per-incident)
        self.assertTrue(agent._no_output_reminded)
        self.assertFalse(agent._iterate())   # ends
        reminders = [m for m in agent.context
                     if m["role"] == "user"
                     and m["content"][0]["text"] == NO_OUTPUT_REMINDER]
        self.assertEqual(len(reminders), 2, "each incident gets its own reminder")

    def test_completion_only_response_ends_without_reminder(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value=_completion_response()
        )
        running = agent._iterate()
        self.assertFalse(running, "completion-only response is an intentional stop")
        self.assertFalse(agent._no_output_reminded)
        self.assertNotIn(NO_OUTPUT_REMINDER,
                         [m["content"][0]["text"] for m in agent.context
                          if m["role"] == "user"])

    def test_completion_then_content_free_gets_fresh_reminder(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            side_effect=[
                _completion_response(),   # intentional stop candidate (ends)
                "chatter",                # would be a new incident
            ]
        )
        self.assertFalse(agent._iterate())
        self.assertFalse(agent._no_output_reminded)
        # A later content-free response (e.g. during completion-retry
        # iterations) gets a fresh reminder — the flag was reset.
        self.assertTrue(agent._iterate())
        self.assertTrue(agent._no_output_reminded)

if __name__ == "__main__":
    unittest.main()
