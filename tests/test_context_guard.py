"""Regression tests for the immutable system prompt + context guard.

Architecture invariant: the system prompt must never contain dynamic
content (timestamps, working directory, memory).  All per-run context
lives in a "context guard" block prepended to the first user message.
Resuming a session must not mutate the restored prefix in any way —
the new task is appended at the tail and nothing is re-injected.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agents import (  # noqa: E402
    Agent,
    CONTEXT_GUARD_HEADER,
    build_context_guard,
)

_FAKE_CONFIG = {
    "system_prompt": "IMMUTABLE SYSTEM PROMPT",
    "overbudget": "over budget",
    "provider": "anthropic",
}


def _make_agent(task="do the thing", memory_view="", notes_compact=False):
    """Construct a minimal Agent with config/memory/backend mocked out."""
    with mock.patch.object(agents_module, "read_configuration", return_value=_FAKE_CONFIG), \
         mock.patch.object(agents_module, "format_memory_view", return_value=memory_view), \
         mock.patch.object(agents_module, "notes_need_compact", return_value=notes_compact), \
         mock.patch.object(agents_module, "create_backend") as mock_backend, \
         mock.patch.object(agents_module, "print_banner"):
        client = mock_backend.return_value
        client.display_name = "MockModel"
        client.context_window_size = 200_000
        client.cost = 0.0
        client.last_input_tokens = 0
        client.last_output_tokens = 0
        client.last_total_context_tokens = 0
        return Agent("fake.yaml", task, session_id="tst1")


class TestImmutableSystemPrompt(unittest.TestCase):

    def test_system_prompt_is_verbatim_config(self):
        agent = _make_agent(memory_view="some memory")
        self.assertEqual(agent.system_prompt, "IMMUTABLE SYSTEM PROMPT")

    def test_system_prompt_has_no_dynamic_content(self):
        agent = _make_agent(memory_view="EPISODE MEM", notes_compact=True)
        for needle in ("Working Directory", "System Date", "Operating System",
                       "Shell", "EPISODE MEM", "note rewrite", CONTEXT_GUARD_HEADER):
            self.assertNotIn(needle, agent.system_prompt, needle)


class TestContextGuardOnNewSession(unittest.TestCase):

    def test_guard_prepended_to_first_user_message(self):
        agent = _make_agent(memory_view="=== Folder Memory: Episodes ===\nEpisode 0: X")
        first = agent.context[0]
        self.assertEqual(first["role"], "user")
        text = first["content"][0]["text"]
        self.assertTrue(text.startswith(CONTEXT_GUARD_HEADER))
        self.assertIn(f"Working Directory: {os.getcwd()}", text)
        self.assertIn("System Date:", text)
        self.assertIn("Operating System:", text)
        self.assertIn("Shell:", text)
        self.assertIn("=== Folder Memory: Episodes ===", text)
        self.assertTrue(text.endswith("do the thing"))

    def test_memory_loaded_at_session_start(self):
        agent = _make_agent(memory_view="=== Folder Memory: Notes ===\nIMPORTANT NOTE")
        text = agent.context[0]["content"][0]["text"]
        self.assertIn("IMPORTANT NOTE", text)

    def test_compact_hint_lives_in_guard_not_system_prompt(self):
        agent = _make_agent(notes_compact=True)
        text = agent.context[0]["content"][0]["text"]
        self.assertIn("note rewrite", text)
        self.assertNotIn("note rewrite", agent.system_prompt)

    def test_build_context_guard_contents(self):
        with mock.patch.object(agents_module, "format_memory_view", return_value="MEM"), \
             mock.patch.object(agents_module, "notes_need_compact", return_value=True):
            guard = build_context_guard()
        self.assertIn(CONTEXT_GUARD_HEADER, guard)
        self.assertIn(f"Working Directory: {os.getcwd()}", guard)
        self.assertIn("System Date:", guard)
        self.assertIn("MEM", guard)
        self.assertIn("note rewrite", guard)


class TestCachePureResume(unittest.TestCase):

    def _resume(self, saved_context, saved_prompt, new_task="new task"):
        agent = _make_agent(task=new_task)
        state = {
            "context": [dict(m) for m in saved_context],
            "system_prompt": saved_prompt,
        }
        with mock.patch.object(agents_module, "load_session", return_value=state):
            agent.load_context("sid1")
        return agent

    def test_resume_does_not_mutate_prefix(self):
        saved_prompt = "IMMUTABLE SYSTEM PROMPT"
        guard_msg = {
            "role": "user",
            "content": [{"type": "text", "text":
                         CONTEXT_GUARD_HEADER +
                         "\nWorking Directory: /old/dir\n\nold task"}],
        }
        asst = {"role": "assistant",
                "content": [{"type": "text", "text": "working…"}]}
        trailing_user = {"role": "user",
                         "content": [{"type": "text", "text": "End."}]}
        saved = [guard_msg, asst, trailing_user]

        agent = self._resume(saved, saved_prompt)

        # System prompt restored verbatim — never regenerated.
        self.assertEqual(agent.system_prompt, saved_prompt)
        # First message (context guard) untouched — stale cwd and all,
        # no fresh guard or memory re-injected.
        self.assertEqual(agent.context[0], guard_msg)
        self.assertEqual(agent.context[1], asst)
        # Trailing user message replaced with the new task, guard-free.
        new_msg = agent.context[-1]
        self.assertEqual(new_msg["content"][0]["text"], "new task")
        self.assertNotIn(CONTEXT_GUARD_HEADER, new_msg["content"][0]["text"])
        # Cache annotation applied only to the new tail message.
        agent.client.mark_for_caching.assert_called_once_with(new_msg)

    def test_resume_legacy_session_logs_and_continues(self):
        saved_prompt = "IMMUTABLE SYSTEM PROMPT"
        legacy_first = {"role": "user",
                        "content": [{"type": "text", "text": "old task"}]}
        trailing_user = {"role": "user",
                         "content": [{"type": "text", "text": "End."}]}
        saved = [legacy_first, trailing_user]

        with self.assertLogs(level="INFO") as logs:
            agent = self._resume(saved, saved_prompt)

        self.assertTrue(any("context-guard" in m for m in logs.output))
        # Still no guard injected — prefix left exactly as saved.
        self.assertEqual(agent.context[0], legacy_first)
        self.assertEqual(agent.context[-1]["content"][0]["text"], "new task")


if __name__ == "__main__":
    unittest.main()
