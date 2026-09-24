"""Regression tests for the context-window usage guard.

As the model's context window fills, the loop tells the model where
it stands: an INFORMATIONAL notice at the lower threshold (default
50% of the window) and a WRAP-UP warning at the higher one (default
80%) — a window that fills mid-task would otherwise die on a hard
provider 400 with no chance to record the work.  Each threshold
fires at most once per session, the notices ride the current turn's
tool-results message (the same delivery path as the budget overage
prompt), and the fired set persists with the session file so a
resume never re-warns.  Thresholds and messages are configurable
via the agent YAML ``context_guard`` block.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agents import (  # noqa: E402
    Agent,
    DEFAULT_CONTEXT_GUARD_INFO_MESSAGE,
    DEFAULT_CONTEXT_GUARD_WARN_MESSAGE,
)

_FAKE_CONFIG = {
    "system_prompt": "IMMUTABLE SYSTEM PROMPT",
    "overbudget": "over budget",
    "provider": "kimi",
}


def _cmd_response(n):
    # Distinct per turn: identical consecutive replies are caught by
    # the anti-loop guard before the notice logic runs.
    return f'Command: run_console_command "echo probe{n}"'


def _make_agent(config=None, task="do the thing"):
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
        return Agent("fake.yaml", task, session_id="tst1")


def _user_texts(agent):
    return [m["content"][0]["text"] for m in agent.context
            if m["role"] == "user"]


def _iterate_with_process(agent, response):
    """Run one _iterate() turn whose commands 'execute' to 'ok output'."""
    with mock.patch.object(
            agents_module, "process_content", return_value=("ok output", [])), \
         mock.patch.object(agents_module, "safe_console_print") as psc, \
         mock.patch.object(agents_module, "print_context_used") as pcu, \
         mock.patch.object(agents_module, "print_budget_warning"):
        agent.client.generate_response = mock.Mock(return_value=response)
        running = agent._iterate()
    return running, psc, pcu


class TestThresholdFiring(unittest.TestCase):
    """Default 50% informational / 80% wrap-up thresholds."""

    def test_info_notice_at_50_percent(self):
        agent = _make_agent()
        agent.client.last_total_context_tokens = 100_000  # exactly 50%
        running, psc, pcu = _iterate_with_process(
            agent, _cmd_response(1))
        self.assertTrue(running)
        tool_results = _user_texts(agent)[-1]
        self.assertTrue(tool_results.startswith("=== Tool Results ==="))
        self.assertIn("ok output", tool_results)
        expected_info = (
            DEFAULT_CONTEXT_GUARD_INFO_MESSAGE
            .replace("{pct}", "50")
            .replace("{used}", "100,000")
            .replace("{window}", "200,000")
        )
        self.assertIn(expected_info, tool_results)
        self.assertNotIn("Start wrapping up now", tool_results)
        self.assertEqual(agent._context_guard_fired, {"info"})
        # Terminal notice for the informational stage, no warn panel.
        self.assertEqual(psc.call_count, 1)
        self.assertIn("50%", psc.call_args.args[0])
        pcu.assert_not_called()

    def test_both_stages_fire_when_usage_jumps_past_both(self):
        agent = _make_agent()
        agent.client.last_total_context_tokens = 160_000  # 80%
        running, psc, pcu = _iterate_with_process(
            agent, _cmd_response(1))
        self.assertTrue(running)
        tool_results = _user_texts(agent)[-1]
        expected_info = (
            DEFAULT_CONTEXT_GUARD_INFO_MESSAGE
            .replace("{pct}", "80")
            .replace("{used}", "160,000")
            .replace("{window}", "200,000")
        )
        expected_warn = (
            DEFAULT_CONTEXT_GUARD_WARN_MESSAGE
            .replace("{pct}", "80")
            .replace("{used}", "160,000")
            .replace("{window}", "200,000")
        )
        self.assertIn(expected_info, tool_results)
        self.assertIn(expected_warn, tool_results)
        self.assertEqual(agent._context_guard_fired,
                         {"info", "warn"})
        self.assertEqual(psc.call_count, 2)
        pcu.assert_called_once_with(160_000, 200_000)

    def test_no_notice_below_thresholds(self):
        agent = _make_agent()
        agent.client.last_total_context_tokens = 50_000  # 25%
        running, psc, pcu = _iterate_with_process(
            agent, _cmd_response(1))
        self.assertTrue(running)
        self.assertEqual(_user_texts(agent)[-1],
                         "=== Tool Results ===\nok output\n=== End Tool Results ===")
        self.assertNotIn("Informational", _user_texts(agent)[-1])
        self.assertEqual(agent._context_guard_fired, set())
        psc.assert_not_called()
        pcu.assert_not_called()

    def test_no_notice_when_session_is_ending(self):
        agent = _make_agent()
        agent.client.last_total_context_tokens = 190_000  # 95%
        bt = "`" * 5
        completion = (
            f"Done.\n{bt}Completion: finished\nSuccess: True\n{bt}"
        )
        with mock.patch.object(
                agents_module, "process_content",
                return_value=("End.", [])), \
             mock.patch.object(agents_module, "safe_console_print") as psc, \
             mock.patch.object(agents_module, "print_context_used") as pcu:
            agent.client.generate_response = mock.Mock(
                return_value=completion)
            running = agent._iterate()
        self.assertFalse(running)
        # A terminating turn has no next turn to receive a notice.
        self.assertEqual(agent._context_guard_fired, set())
        psc.assert_not_called()
        pcu.assert_not_called()


class TestOncePerSession(unittest.TestCase):
    """Each threshold fires at most once, ever (across turns)."""

    def test_notice_delivered_once_across_turns(self):
        agent = _make_agent()
        agent.client.last_total_context_tokens = 110_000  # 55%
        for n in range(1, 4):
            running, psc, _ = _iterate_with_process(
                agent, _cmd_response(n))
            self.assertTrue(running)
        # Only the FIRST turn's tool results carry the notice.
        with_notice = [t for t in _user_texts(agent)
                       if "Informational" in t]
        self.assertEqual(len(with_notice), 1)
        self.assertIn("Informational", with_notice[0])
        self.assertEqual(agent._context_guard_fired, {"info"})


class TestConfigurability(unittest.TestCase):
    """YAML context_guard block: custom thresholds and messages."""

    def test_custom_thresholds_and_messages(self):
        config = {
            **_FAKE_CONFIG,
            "context_guard": {
                "info": 10,
                "warn": 20,
                "info_message": "CUSTOM INFO at {pct}%",
                "warn_message": "CUSTOM WARN at {pct}%",
            },
        }
        agent = _make_agent(config=config)
        self.assertEqual(agent.context_guard_info, 10.0)
        self.assertEqual(agent.context_guard_warn, 20.0)
        agent.client.last_total_context_tokens = 50_000  # 25%: past both
        running, _, _ = _iterate_with_process(agent, _cmd_response(1))
        self.assertTrue(running)
        tool_results = _user_texts(agent)[-1]
        self.assertIn("CUSTOM INFO at 25%", tool_results)
        self.assertIn("CUSTOM WARN at 25%", tool_results)
        self.assertEqual(agent._context_guard_fired, {"info", "warn"})

    def test_zero_threshold_disables_that_level(self):
        config = {
            **_FAKE_CONFIG,
            "context_guard": {"info": 0},
        }
        agent = _make_agent(config=config)
        agent.client.last_total_context_tokens = 160_000  # 80%
        running, psc, _ = _iterate_with_process(agent, _cmd_response(1))
        self.assertTrue(running)
        tool_results = _user_texts(agent)[-1]
        self.assertNotIn("Informational", tool_results)
        self.assertIn("Start wrapping up now", tool_results)
        self.assertEqual(agent._context_guard_fired, {"warn"})
        self.assertEqual(psc.call_count, 1)

    def test_invalid_threshold_falls_back_to_default(self):
        config = {**_FAKE_CONFIG, "context_guard": {"info": "lots"}}
        agent = _make_agent(config=config)
        self.assertEqual(agent.context_guard_info, 50.0)
        agent.client.last_total_context_tokens = 100_000  # 50%
        running, _, _ = _iterate_with_process(agent, _cmd_response(1))
        self.assertTrue(running)
        self.assertIn("Informational", _user_texts(agent)[-1])

    def test_message_with_literal_braces_does_not_crash(self):
        config = {
            **_FAKE_CONFIG,
            "context_guard": {
                "info": 50,
                "info_message": "Heads up {pct}% — braces {literal} intact.",
            },
        }
        agent = _make_agent(config=config)
        agent.client.last_total_context_tokens = 100_000
        running, _, _ = _iterate_with_process(agent, _cmd_response(1))
        self.assertTrue(running)
        self.assertIn("Heads up 50% — braces {literal} intact.",
                      _user_texts(agent)[-1])

    def test_no_notice_when_window_unknown(self):
        agent = _make_agent()
        agent.client.context_window_size = 0  # custom backend: no table entry
        agent.client.last_total_context_tokens = 999_999
        running, psc, pcu = _iterate_with_process(agent, _cmd_response(1))
        self.assertTrue(running)
        self.assertNotIn("Informational", _user_texts(agent)[-1])
        psc.assert_not_called()
        pcu.assert_not_called()


class TestResumePersistence(unittest.TestCase):
    """The fired set survives save/resume — no re-warn on a new leg."""

    def test_save_persists_fired_set_sorted(self):
        agent = _make_agent()
        agent._context_guard_fired = {"warn", "info"}
        with mock.patch.object(agents_module, "save_session") as mock_save:
            agent.save_context()
        state = mock_save.call_args[0][2]
        self.assertEqual(state["context_guard_fired"], ["info", "warn"])

    def test_load_restores_fired_set(self):
        agent1 = _make_agent()
        agent1._context_guard_fired = {"info"}

        with mock.patch.object(agents_module, "save_session") as mock_save:
            agent1.save_context()
        state = mock_save.call_args[0][2]

        agent2 = _make_agent()
        with mock.patch.object(
                agents_module, "load_session", return_value=state):
            agent2.load_context()
        self.assertEqual(agent2._context_guard_fired, {"info"})

        # The resumed leg at 50% must NOT re-emit the info notice.
        agent2.client.last_total_context_tokens = 100_000
        running, psc, _ = _iterate_with_process(agent2, _cmd_response(1))
        self.assertTrue(running)
        self.assertNotIn("Informational", _user_texts(agent2)[-1])
        self.assertEqual(agent2._context_guard_fired, {"info"})
        psc.assert_not_called()

    def test_legacy_session_without_key_gets_fresh_set(self):
        agent1 = _make_agent()
        with mock.patch.object(agents_module, "save_session") as mock_save:
            agent1.save_context()
        state = mock_save.call_args[0][2]
        state.pop("context_guard_fired")  # simulate an old session file

        agent2 = _make_agent()
        with mock.patch.object(
                agents_module, "load_session", return_value=state):
            agent2.load_context()
        self.assertEqual(agent2._context_guard_fired, set())


if __name__ == "__main__":
    unittest.main()
