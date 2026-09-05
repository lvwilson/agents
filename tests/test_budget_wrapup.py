"""Regression tests for the budget wrap-up turn.

When the session's compute budget is exceeded in :meth:`Agent.run`,
the agent must not stop dead mid-task: it gets exactly ONE final
free-form turn — in which no ``Command:`` line is ever parsed or
executed — to record the work done so far and emit the completion
block, after which the session ends through the normal pipeline
(completion extraction, save, …).
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agents import Agent, BUDGET_WRAPUP_PROMPT, _find_latest_completion  # noqa: E402

_FAKE_CONFIG = {
    "system_prompt": "IMMUTABLE SYSTEM PROMPT",
    "overbudget": "over budget",
    "provider": "kimi",
}

BT = "`" * 5

COMPLETION_DESC = "Wrapped up after budget overrun"


def _completion(success="False"):
    return (
        "Summarising what was achieved so far.\n"
        f"{BT}Completion: {COMPLETION_DESC}\nSuccess: {success}\n{BT}"
    )


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


def _run(agent, stdout_isatty):
    """Run ``agent.run()`` with UI and real stdout patched.

    Returns ``(mock_stdout, print_budget_exceeded mock)``.
    """
    mock_stdout = mock.Mock()
    mock_stdout.isatty.return_value = stdout_isatty
    with mock.patch.object(agents_module, "print_budget_exceeded") as pbx, \
         mock.patch.object(agents_module, "print_summary"), \
         mock.patch.object(agents_module, "print_error"), \
         mock.patch.object(agents_module, "safe_console_print"), \
         mock.patch.object(sys, "stdout", mock_stdout):
        agent.run()
    return mock_stdout, pbx


def _written_text(mock_stdout):
    return "".join(
        c.args[0] for c in mock_stdout.write.call_args_list
        if c.args and isinstance(c.args[0], str)
    )


class TestBudgetWrapUp(unittest.TestCase):
    """run() must give one command-free final turn at budget overrun."""

    def test_overrun_triggers_exactly_one_free_form_wrap_up(self):
        agent = _make_agent()
        agent.client.cost = 3.0  # default budget is $2.00
        free_form_calls = []

        def fake_iterate(free_form=False):
            free_form_calls.append(free_form)
            agent.context.append({
                "role": "assistant",
                "content": [{"type": "text", "text": _completion()}],
            })
            return False if free_form else True

        agent._iterate = mock.Mock(side_effect=fake_iterate)
        mock_stdout, pbx = _run(agent, stdout_isatty=False)

        # One main-loop turn, then the budget check fires: exactly one
        # additional FREE-FORM turn, and the loop ends after it.
        self.assertEqual(free_form_calls, [False, True])
        pbx.assert_called_once()
        # The wrap-up directive reached the model exactly once.
        self.assertEqual(
            sum(t == BUDGET_WRAPUP_PROMPT for t in _user_texts(agent)), 1)
        # The session still ends normally: the completion block in the
        # wrap-up reply is found by the usual post-run scan.
        result = _find_latest_completion(agent.context)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, COMPLETION_DESC)
        self.assertFalse(result.success)
        # The wrap-up reply was dumped to the (piped) stdout so a
        # sub-agent's parent still receives the final answer.
        self.assertIn(COMPLETION_DESC, _written_text(mock_stdout))

    def test_no_wrap_up_when_budget_holds(self):
        agent = _make_agent()
        agent.client.cost = 0.5
        calls = []

        def fake_iterate(free_form=False):
            calls.append(free_form)
            return False  # normal stop: no command, loop ends

        agent._iterate = mock.Mock(side_effect=fake_iterate)
        _, pbx = _run(agent, stdout_isatty=True)

        self.assertEqual(calls, [False])
        pbx.assert_not_called()
        self.assertNotIn(BUDGET_WRAPUP_PROMPT, _user_texts(agent))

    def test_wrap_up_iteration_failure_does_not_crash(self):
        agent = _make_agent()
        agent.client.cost = 5.0

        def fake_iterate(free_form=False):
            if free_form:
                raise RuntimeError("looping error")
            return True

        agent._iterate = mock.Mock(side_effect=fake_iterate)
        # Must not propagate: run() finishes cleanly.
        _run(agent, stdout_isatty=True)

    def test_no_stdout_dump_when_on_terminal(self):
        agent = _make_agent()
        agent.client.cost = 3.0

        def fake_iterate(free_form=False):
            if free_form:
                agent.context.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": _completion()}],
                })
            return False if free_form else True

        agent._iterate = mock.Mock(side_effect=fake_iterate)
        mock_stdout, _ = _run(agent, stdout_isatty=True)
        mock_stdout.write.assert_not_called()


class TestWrapUpProcessesNoCommands(unittest.TestCase):
    """The wrap-up turn really processes no further commands."""

    def test_command_lines_in_wrap_up_reply_are_ignored(self):
        agent = _make_agent()
        reply = (
            'Command: run_console_command "echo pwned"\n'
            + _completion()
        )
        agent.client.generate_response = mock.Mock(return_value=reply)
        mock_stdout = mock.Mock()
        mock_stdout.isatty.return_value = False
        with mock.patch.object(agents_module, "process_content") as pc, \
             mock.patch.object(agents_module, "print_budget_exceeded"), \
             mock.patch.object(agents_module, "safe_console_print"), \
             mock.patch.object(agents_module, "print_iteration_header"), \
             mock.patch.object(agents_module, "print_clipped"), \
             mock.patch.object(sys, "stdout", mock_stdout):
            agent._run_budget_wrap_up()
        # Free-form mode bypasses the command pipeline entirely.
        pc.assert_not_called()
        self.assertIn(BUDGET_WRAPUP_PROMPT, _user_texts(agent))
        # The reply (including the ignored Command line) lands in
        # context as an ordinary assistant message and is salvaged to
        # piped stdout.
        self.assertIn("echo pwned", agent.context[-1]["content"][0]["text"])
        self.assertIn(COMPLETION_DESC, _written_text(mock_stdout))


if __name__ == "__main__":
    unittest.main()
