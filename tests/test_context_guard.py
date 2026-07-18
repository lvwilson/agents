"""Regression tests for the immutable system prompt + context guard.

Architecture invariant: the system prompt must never contain dynamic
content (timestamps, working directory, memory).  All per-run context
lives in a "context guard" block prepended to the first user message.
Resuming a session must not mutate the restored prefix in any way —
the new task is appended at the tail and nothing is re-injected.
"""

import copy
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


class TestResumePayloadPurity(unittest.TestCase):
    """After a save→resume round-trip, the payload sent to the backend must
    be byte-identical to the pre-resume payload up to the new task appended
    at the tail.  This is the concrete condition that lets the Anthropic
    prompt cache hit; if anything earlier shifts, the whole prefix is
    re-processed at full cost.
    """

    def _run_one_iteration(self, agent):
        """Drive exactly one _iterate() with a canned backend reply."""
        agent.client.generate_response = mock.Mock(return_value="ok, working on it")
        agent._iterate()

    def test_resume_payload_identical_up_to_new_task(self):
        # ── Session 1: build a real multi-turn context ──────────────
        agent = _make_agent(task="original task")
        saved_prompt = agent.system_prompt
        # Simulate a couple of turns: assistant reply + user command result.
        agent.context.append({
            "role": "assistant",
            "content": [{"type": "text", "text": "step one complete"}],
        })
        agent.context.append({
            "role": "user",
            "content": [{"type": "text", "text": "Command output: file contents here"}],
        })
        # load_context pops the trailing user message (the last command
        # result) and replaces it with the new task — so the invariant
        # prefix is everything BEFORE that trailing user message.
        pre_resume_prefix = copy.deepcopy(agent.context[:-1])

        # ── Save, then resume into a fresh Agent with a new task ────
        state = {"context": copy.deepcopy(agent.context),
                 "system_prompt": saved_prompt}
        resumed = _make_agent(task="resume task")
        with mock.patch.object(agents_module, "load_session", return_value=state):
            resumed.load_context("sidX")

        # ── Capture the payload sent on the first post-resume call ──
        payloads = []
        resumed.client.generate_response = mock.Mock(
            side_effect=lambda sp, ctx: payloads.append((sp, copy.deepcopy(ctx))) or "resumed work"
        )
        resumed._iterate()

        self.assertTrue(payloads, "no payload captured on resume")
        system_prompt, context = payloads[0]

        # System prompt must be byte-identical.
        self.assertEqual(system_prompt, saved_prompt)

        # Every message before the appended task must equal the saved
        # prefix exactly — same count, same order, same content.
        self.assertEqual(context[:-1], pre_resume_prefix,
                         "context prefix changed across resume")
        # The only difference is the new task appended at the tail.
        self.assertEqual(context[-1]["role"], "user")
        self.assertEqual(context[-1]["content"][0]["text"], "resume task")
        self.assertEqual(len(context), len(pre_resume_prefix) + 1)

    def test_resume_does_not_inject_new_guard_or_memory(self):
        """On resume, the only new message is the bare task at the tail.
        No fresh guard is injected and no fresh memory leaks into the
        restored conversation."""
        saved_context = [
            {"role": "user", "content": [{"type": "text", "text":
                CONTEXT_GUARD_HEADER + "\nWorking Directory: /old\n\norig task"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "did work"}]},
            {"role": "user", "content": [{"type": "text", "text": "End."}]},
        ]
        state = {"context": copy.deepcopy(saved_context),
                 "system_prompt": "IMMUTABLE SYSTEM PROMPT"}
        # The new process sees DIFFERENT fresh memory — it must not leak in.
        resumed = _make_agent(task="resume task", memory_view="FRESH MEMORY XYZ")
        with mock.patch.object(agents_module, "load_session", return_value=state):
            resumed.load_context("sidY")

        # Prefix (everything before the popped trailing user msg) restored
        # byte-identically.
        self.assertEqual(resumed.context[:-1], saved_context[:-1])
        # Exactly one message appended — the bare new task.
        self.assertEqual(len(resumed.context), len(saved_context))
        self.assertEqual(resumed.context[-1]["content"][0]["text"], "resume task")
        # Fresh memory from the new process never entered the context.
        joined = " ".join(m["content"][0].get("text", "") for m in resumed.context)
        self.assertNotIn("FRESH MEMORY XYZ", joined)
        # Only the original (first) message carries a guard header.
        guard_msgs = [m for m in resumed.context
                      if CONTEXT_GUARD_HEADER in m["content"][0].get("text", "")]
        self.assertEqual(len(guard_msgs), 1)
        self.assertIs(guard_msgs[0], resumed.context[0])


if __name__ == "__main__":
    unittest.main()
