"""Regression tests for raw <tool_call> detection.

A model trained on the OpenAI tool-calling format may emit a raw
tool-call block (an open tag, a JSON function payload, and a close
tag) instead of this harness's ``Command:`` lines.  Nothing in the
harness parses that format, so such a response used to hit the generic
accidental-stop guard with no explanation of what went wrong.  The
loop now detects the raw open tag and injects a targeted reminder
explaining that this harness does not support <tool_call> tool-call
syntax, while sharing the same one-reminder, one-second-strike budget
as the generic no-output guard.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agents import (  # noqa: E402
    Agent,
    NO_OUTPUT_REMINDER,
    TOOL_CALL_MARKER,
    TOOL_CALL_REMINDER,
)

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


def _tool_call_response(tag="command"):
    """A turn in which the model emits a raw OpenAI tool-call block."""
    return (
        "I should list the files first.\n"
        "<tool_call>\n"
        "<function=run_console_command>\n"
        f"<parameter=arguments>ls -l {tag}</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def _user_texts(agent):
    return [m["content"][0]["text"] for m in agent.context
            if m["role"] == "user"]


class TestToolCallReminder(unittest.TestCase):
    """Raw tool-call block + no Command + no completion → targeted reminder."""

    def test_targeted_reminder_injected_and_loop_continues(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value=_tool_call_response()
        )
        with mock.patch.object(agents_module, "process_content",
                               return_value=("End.", [])):
            running = agent._iterate()
        self.assertTrue(running,
                        "tool-call-only response must not end the session")
        self.assertTrue(agent._no_output_reminded)
        last = agent.context[-1]
        self.assertEqual(last["role"], "user")
        self.assertEqual(last["content"][0]["text"], TOOL_CALL_REMINDER)
        # The generic reminder must NOT be the one injected here.
        self.assertNotIn(NO_OUTPUT_REMINDER, _user_texts(agent))

    def test_reminder_text_names_the_unsupported_format(self):
        self.assertIn(TOOL_CALL_MARKER, TOOL_CALL_REMINDER)
        self.assertIn("not support", TOOL_CALL_REMINDER)
        self.assertIn("Command: name args", TOOL_CALL_REMINDER)

    def test_command_block_suppresses_tool_call_reminder(self):
        # A real Command line alongside a stray tool-call block is a
        # command response — no reminder of any kind.
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value=_tool_call_response() + "\nCommand: read_file pyproject.toml"
        )
        with mock.patch.object(agents_module, "process_content",
                               return_value=("name = agents\n", [])):
            running = agent._iterate()
        self.assertTrue(running)
        self.assertFalse(agent._no_output_reminded)
        self.assertNotIn(TOOL_CALL_REMINDER, _user_texts(agent))
        self.assertNotIn(NO_OUTPUT_REMINDER, _user_texts(agent))

    def test_completion_block_suppresses_tool_call_reminder(self):
        # A completion block is an intentional stop — no reminder.
        agent = _make_agent()
        bt = "`" * 5
        completion = (
            f"All done.\n{bt}Completion: finished the task\n"
            "Success: True\n" + bt
        )
        agent.client.generate_response = mock.Mock(
            return_value=_tool_call_response() + "\n" + completion
        )
        with mock.patch.object(agents_module, "process_content",
                               return_value=("End.", [])):
            running = agent._iterate()
        self.assertFalse(running, "completion block is an intentional stop")
        self.assertFalse(agent._no_output_reminded)
        self.assertNotIn(TOOL_CALL_REMINDER, _user_texts(agent))

    def test_second_content_free_response_still_ends_session(self):
        # The targeted reminder shares the one-reminder budget: a second
        # consecutive command-free response ends the session.
        agent = _make_agent()
        # Distinct strings per turn: identical consecutive responses are
        # caught by the anti-loop guard before the reminder logic runs.
        agent.client.generate_response = mock.Mock(
            side_effect=[
                _tool_call_response(),
                _tool_call_response("second").replace(
                    "command</parameter>", "second</parameter>"
                ),
            ]
        )
        with mock.patch.object(agents_module, "process_content",
                               return_value=("End.", [])):
            self.assertTrue(agent._iterate())
            running = agent._iterate()
        self.assertFalse(running,
                         "second content-free response must end the session")
        reminders = [text for text in _user_texts(agent)
                     if text in (TOOL_CALL_REMINDER, NO_OUTPUT_REMINDER)]
        self.assertEqual(len(reminders), 1)
        self.assertEqual(reminders[0], TOOL_CALL_REMINDER)

    def test_plain_chatter_still_gets_generic_reminder(self):
        """No tool-call tag → behaviour is unchanged (regression guard)."""
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value="just some plain chatter, no commands"
        )
        with mock.patch.object(agents_module, "process_content",
                               return_value=("End.", [])):
            running = agent._iterate()
        self.assertTrue(running)
        self.assertEqual(agent.context[-1]["content"][0]["text"],
                         NO_OUTPUT_REMINDER)

    def test_marker_constant_is_the_open_tag(self):
        self.assertEqual(TOOL_CALL_MARKER, "<tool_call>")


if __name__ == "__main__":
    unittest.main()
