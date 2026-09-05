"""Regression tests for the explicit end_session command.

The session now ends *only* via an explicit ``Command: end_session``
issued as the sole command of a response, carrying the completion
note in a 5-backtick payload block.  A legacy completion block with
no sentinel still ends the session, and its text is still extracted
in the same-line fence form.

When end_session is issued alongside other commands the end attempt
is REJECTED: the other commands run and a notice is prepended to the
result.  The transcript (stored context) keeps the full turn —
sentinel line and payload intact — so a clean ending remains
extractable by _find_latest_completion() and reports as success.
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
    _find_latest_completion,
    _form_message,
    extract_completion,
)
from agents.tools.parser import (  # noqa: E402
    END_SESSION_COMMAND,
    END_SESSION_REJECTED_NOTICE,
    process_content,
    strip_end_session,
)

BT = "`" * 5


def _sentinel_payload(text="did the work", success="True"):
    """Canonical ending: sentinel line + payload block on its own fence."""
    return (
        f"Command: {END_SESSION_COMMAND}\n"
        f"{BT}\nCompletion: {text}\nSuccess: {success}\n{BT}"
    )


# ── strip_end_session ────────────────────────────────────────────────

class TestStripEndSession(unittest.TestCase):
    """strip_end_session removes the sentinel line + attached payload."""

    def test_removes_sentinel_and_payload(self):
        response = "All done.\n" + _sentinel_payload()
        cleaned, found = strip_end_session(response)
        self.assertTrue(found)
        self.assertEqual(cleaned, "All done.\n")

    def test_removes_bare_sentinel(self):
        cleaned, found = strip_end_session("Command: end_session")
        self.assertTrue(found)
        self.assertEqual(cleaned.strip(), "")

    def test_absent_is_unchanged(self):
        response = 'Command: run_console_command "echo hi"'
        cleaned, found = strip_end_session(response)
        self.assertFalse(found)
        self.assertEqual(cleaned, response)

    def test_case_insensitive(self):
        cleaned, found = strip_end_session(
            "prologue\nCommand: END_SESSION\nepilogue"
        )
        self.assertTrue(found)
        self.assertEqual(cleaned, "prologue\nepilogue")


# ── process_content ──────────────────────────────────────────────────

class TestProcessContentEndSession(unittest.TestCase):
    """The end_session sentinel resolves in the pipeline, never dispatches."""

    def test_alone_yields_end_sentinel(self):
        with mock.patch(
            "agents.tools.parser._execute_command"
        ) as fake_exec:
            result, images = process_content(_sentinel_payload())
        self.assertEqual(result, "End.")
        self.assertEqual(images, [])
        self.assertFalse(
            fake_exec.called, "end_session alone must never dispatch to a tool"
        )

    def test_alongside_other_command_is_rejected_but_runs(self):
        with mock.patch(
            "agents.tools.parser._execute_command"
        ) as fake_exec:
            fake_exec.return_value = "hi"
            content = (
                'Command: run_console_command "echo hi"\n'
                + _sentinel_payload()
            )
            result, _images = process_content(content)
        self.assertTrue(
            result.startswith(END_SESSION_REJECTED_NOTICE),
            f"expected notice prefix, got: {result[:80]!r}",
        )
        self.assertIn("hi", result, "the other command must still run")
        fake_exec.assert_called_once()

    def test_rejected_notice_only_when_other_commands_present(self):
        result, _ = process_content("Command: end_session")
        self.assertNotIn(END_SESSION_REJECTED_NOTICE, result)


# ── extract_completion ───────────────────────────────────────────────

class TestExtractCompletionFenceForms(unittest.TestCase):
    """Both fence placements must parse: fence on its own line (the
    canonical prompt form) and 'Completion:' immediately after the
    opening fence (legacy form)."""

    def test_fence_on_own_line(self):
        result = extract_completion(_sentinel_payload("shipped the fix"))
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "shipped the fix")
        self.assertTrue(result.success)

    def test_legacy_inline_fence(self):
        block = BT + "Completion: legacy text\nSuccess: False\n" + BT
        result = extract_completion("done\n" + block)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "legacy text")
        self.assertFalse(result.success)


# ── _find_latest_completion ──────────────────────────────────────────

class TestFindLatestCompletionEndSession(unittest.TestCase):
    """Completion extraction across the transcript, incl. the sentinel."""

    def test_clean_ending_payload_is_reported(self):
        # Guards the stripped-transcript bug: the payload must be
        # visible in the stored assistant message to be extractable.
        context = [
            _form_message("user", "task"),
            _form_message("assistant", "working\nCommand: read_file x"),
            _form_message("user", "file contents"),
            _form_message("assistant", "All done.\n" + _sentinel_payload("shipped")),
            _form_message("user", "End."),
        ]
        result = _find_latest_completion(context)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "shipped")
        self.assertTrue(result.success)

    def test_bare_sentinel_in_newest_message_is_success(self):
        context = [
            _form_message("user", "task"),
            _form_message("assistant", "working"),
            _form_message("assistant", "Command: end_session"),
        ]
        result = _find_latest_completion(context)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(result.text, "Task ended via end_session.")

    def test_rejected_sentinel_in_older_message_is_not_a_completion(self):
        # A REJECTED end_session (sentinel queued alongside another
        # command) lingers in the transcript of an earlier turn.  If
        # the session ends later via the content-free path, the
        # fallback must not resurrect the old attempt as a success.
        rejected_turn = (
            'Command: run_console_command "echo hi"\n'
            "Command: end_session"
        )
        context = [
            _form_message("user", "task"),
            _form_message("assistant", rejected_turn),
            _form_message(
                "user",
                "=== Tool Results ===\nend_session REJECTED "
                "...\nhi\n=== End Tool Results ===",
            ),
            _form_message("assistant", "just chatter"),
        ]
        self.assertIsNone(_find_latest_completion(context))

    def test_bare_sentinel_still_beats_old_rejected_turn(self):
        rejected_turn = (
            'Command: run_console_command "echo hi"\n'
            "Command: end_session"
        )
        context = [
            _form_message("user", "task"),
            _form_message("assistant", rejected_turn),
            _form_message("user", "rejected notice + hi"),
            _form_message("assistant", "Command: end_session"),
        ]
        result = _find_latest_completion(context)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)


# ── Agent._iterate end-to-end (mocked backend) ──────────────────────

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
        return Agent("fake.yaml", task, session_id="end1")


class TestIterateEndSession(unittest.TestCase):

    def test_lone_sentinel_ends_loop_and_keeps_payload_in_context(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value="All done.\n" + _sentinel_payload("shipped the feature")
        )
        running = agent._iterate()
        self.assertFalse(running, "a lone end_session must end the loop")
        # No accidental-stop reminder may have been injected.
        user_texts = [
            m["content"][0]["text"] for m in agent.context
            if m["role"] == "user"
        ]
        self.assertNotIn(NO_OUTPUT_REMINDER, user_texts)
        # The stored assistant turn keeps the full form (sentinel line
        # + payload) so run_agent can extract the completion from it.
        last_assistant = [
            m for m in agent.context if m["role"] == "assistant"
        ][-1]
        text = last_assistant["content"][0]["text"]
        self.assertIn("Command: end_session", text)
        self.assertIn("shipped the feature", text)
        self.assertIsNotNone(extract_completion(text))

    def test_sentinel_with_other_command_continues_with_notice(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value=(
                'Command: run_console_command "echo hi"\n'
                + _sentinel_payload("done")
            )
        )
        with mock.patch(
            "agents.tools.parser._execute_command"
        ) as fake_exec:
            fake_exec.return_value = "hi"
            running = agent._iterate()
        self.assertTrue(running, "a rejected end_session must not end the loop")
        last_user = [m for m in agent.context
                     if m["role"] == "user"][-1]
        text = last_user["content"][0]["text"]
        self.assertIn(END_SESSION_REJECTED_NOTICE, text)
        self.assertIn("hi", text, "the other command must have run")
        fake_exec.assert_called_once()

    def test_legacy_completion_block_without_sentinel_still_ends(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(
            return_value="All done.\n" + BT + "Completion: legacy end\nSuccess: True\n" + BT
        )
        running = agent._iterate()
        self.assertFalse(running,
                         "a legacy completion block still ends the session")
        user_texts = [
            m["content"][0]["text"] for m in agent.context
            if m["role"] == "user"
        ]
        self.assertNotIn(NO_OUTPUT_REMINDER, user_texts)


# ── run_agent reporting (mocked Agent) ───────────────────────────────

class TestRunAgentEndSessionReporting(unittest.TestCase):

    @mock.patch.object(agents_module, "Agent")
    def test_clean_end_reports_success_payload_text(self, MockAgent):
        agent = MockAgent.return_value
        agent.session_id = "endsess1"
        agent._request_episode_summary.return_value = None
        agent.context = [
            _form_message("user", "task"),
            _form_message("assistant", "working\nCommand: read_file x"),
            _form_message("user", "file contents"),
            _form_message("assistant",
                          "All done.\n" + _sentinel_payload("shipped it")),
            _form_message(
                "user",
                "=== Tool Results ===\nEnd.\n=== End Tool Results ===",
            ),
        ]
        completion, success, sid = agents_module.run_agent(
            "basic_agent.yaml", "do stuff", 1.0, save=False, nogit=True
        )
        self.assertTrue(success)
        self.assertEqual(completion, "shipped it")
        self.assertEqual(sid, "endsess1")

    @mock.patch.object(agents_module, "Agent")
    def test_bare_sentinel_end_reports_success(self, MockAgent):
        agent = MockAgent.return_value
        agent.session_id = "endsess2"
        agent._request_episode_summary.return_value = None
        agent.context = [
            _form_message("user", "task"),
            _form_message("assistant", "Command: end_session"),
            _form_message(
                "user",
                "=== Tool Results ===\nEnd.\n=== End Tool Results ===",
            ),
        ]
        completion, success, _sid = agents_module.run_agent(
            "basic_agent.yaml", "do stuff", 1.0, save=False, nogit=True
        )
        self.assertTrue(success)
        self.assertEqual(completion, "Task ended via end_session.")


# ── system prompts must teach the explicit ending ────────────────────

class TestPromptsTeachEndSession(unittest.TestCase):
    """All three agent prompts must instruct the explicit ending (a
    prior pass patched basic + sub_agent but silently skipped the
    manipulator prompt)."""

    def _prompt_of(self, name):
        import yaml
        from agents.agents import script_dir
        with open(os.path.join(script_dir, name)) as fh:
            return yaml.safe_load(fh)["system_prompt"]

    def test_all_prompts_mention_end_session(self):
        for name in ("basic_agent.yaml", "sub_agent.yaml",
                     "manipulator_agent.yaml"):
            with self.subTest(name=name):
                prompt = self._prompt_of(name)
                self.assertIn("Command: end_session", prompt)
                self.assertIn("REJECTED", prompt)
                # The implicit-stop convention must be gone — it
                # contradicts the explicit ending.
                self.assertNotIn(
                    "when no command is given it is assumed that you "
                    "have finished",
                    prompt,
                )


if __name__ == "__main__":
    unittest.main()
