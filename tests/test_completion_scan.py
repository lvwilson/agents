"""Regression tests for completion detection.

Bug report: "all tasks seem to be failing despite the completion
being correctly written."  Root cause: run_agent() only inspected
context[-2] (the assistant message immediately before the final user
message) for the completion block.  When the agent's final assistant
message did not itself contain the block — e.g. the completion was
written one turn earlier and the last reply ended with a Worklog line
or a trailing Command — the task was reported as failed even though a
valid completion had been written.  The fix scans recent assistant
messages newest-to-oldest via _find_latest_completion().
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agents import (  # noqa: E402
    _find_latest_completion,
    _form_message,
    extract_completion,
)

# Built programmatically: a literal 5-backtick line in this file would
# collide with the find_and_replace/write_file delimiters.
FENCE = "`" * 5
SUCCESS_BLOCK = FENCE + "Completion: did the thing\nSuccess: True\n" + FENCE
FAILURE_BLOCK = FENCE + "Completion: tried the thing\nSuccess: False\n" + FENCE


class TestExtractCompletion(unittest.TestCase):
    """Baseline behaviour of the single-message extractor."""

    def test_parses_wrapped_block(self):
        result = extract_completion("some prose\n" + SUCCESS_BLOCK)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "did the thing")
        self.assertTrue(result.success)

    def test_success_false(self):
        result = extract_completion(FAILURE_BLOCK)
        self.assertIsNotNone(result)
        self.assertFalse(result.success)

    def test_unwrapped_block_is_ignored(self):
        # Completion blocks must be wrapped in the fence; bare text
        # mentioning the fields is not a completion.
        self.assertIsNone(extract_completion("Completion: x\nSuccess: True"))

    def test_no_block_returns_none(self):
        self.assertIsNone(extract_completion("just some text"))


class TestFindLatestCompletion(unittest.TestCase):
    """The conversation scan used by run_agent()."""

    def test_completion_in_earlier_assistant_message(self):
        # The reported bug: completion written, then the agent replies
        # once more (trailing Worklog/Command) so the completion is no
        # longer at context[-2].
        context = [
            _form_message("user", "task"),
            _form_message("assistant", "done\n" + SUCCESS_BLOCK),
            _form_message("user", "cmd output"),
            _form_message(
                "assistant",
                'Worklog: wrapped up.\nCommand: run_console_command "git status"',
            ),
            _form_message("user", "End."),
        ]
        result = _find_latest_completion(context)
        self.assertIsNotNone(result)
        self.assertEqual(result.text, "did the thing")
        self.assertTrue(result.success)

    def test_newest_completion_wins(self):
        context = [
            _form_message("user", "task"),
            _form_message("assistant", FAILURE_BLOCK),
            _form_message("user", "keep going"),
            _form_message("assistant", SUCCESS_BLOCK),
            _form_message("user", "End."),
        ]
        result = _find_latest_completion(context)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)
        self.assertEqual(result.text, "did the thing")

    def test_user_messages_are_ignored(self):
        # Tool output may echo the completion instructions; it must not
        # be mistaken for the agent's own completion.
        context = [
            _form_message("user", "task"),
            _form_message("assistant", "working on it\nCommand: read_file x"),
            _form_message("user", "file contents:\n" + SUCCESS_BLOCK),
        ]
        self.assertIsNone(_find_latest_completion(context))

    def test_scan_limit_bounds_the_search(self):
        context = [
            _form_message("user", "task"),
            _form_message("assistant", SUCCESS_BLOCK),
        ]
        for _ in range(5):
            context.append(_form_message("user", "more"))
            context.append(_form_message("assistant", "still no block"))
        context.append(_form_message("user", "End."))
        # The completion is the 6th assistant message back — beyond the
        # default scan limit of 5.
        self.assertIsNone(_find_latest_completion(context))
        # ...but is found with a larger limit.
        self.assertIsNotNone(_find_latest_completion(context, scan_limit=6))

    def test_malformed_messages_are_skipped(self):
        context = [
            {"role": "assistant", "content": []},
            {"role": "assistant", "content": [{"type": "image"}]},
            _form_message("assistant", SUCCESS_BLOCK),
        ]
        result = _find_latest_completion(context)
        self.assertIsNotNone(result)
        self.assertTrue(result.success)


class TestRunAgentCompletionDetection(unittest.TestCase):
    """End-to-end (mocked) check that run_agent reports success."""

    @mock.patch.object(agents_module, "Agent")
    def test_success_despite_trailing_assistant_message(self, MockAgent):
        agent = MockAgent.return_value
        agent.session_id = "testsession"
        agent._request_episode_summary.return_value = None
        agent.context = [
            _form_message("user", "task"),
            _form_message("assistant", "done\n" + SUCCESS_BLOCK),
            _form_message("user", "cmd output"),
            _form_message("assistant", "Worklog: trailing reply without a block."),
            _form_message("user", "End."),
        ]

        completion, success, sid = agents_module.run_agent(
            "basic_agent.yaml", "do stuff", 1.0, save=False, nogit=True
        )

        self.assertTrue(success)
        self.assertEqual(completion, "did the thing")
        self.assertEqual(sid, "testsession")
        agent.request_completion.assert_not_called()

    @mock.patch.object(agents_module, "Agent")
    def test_failure_when_no_completion_anywhere(self, MockAgent):
        agent = MockAgent.return_value
        agent.session_id = "testsession"
        agent._request_episode_summary.return_value = None
        agent.context = [
            _form_message("user", "task"),
            _form_message("assistant", "no block"),
            _form_message("user", "End."),
        ]

        completion, success, _sid = agents_module.run_agent(
            "basic_agent.yaml", "do stuff", 1.0, save=False, nogit=True
        )

        self.assertFalse(success)
        self.assertEqual(completion, "Error")
        # request_completion is no longer called — the system prompt
        # explicitly requires a completion block, so absence is a
        # definitive failure with no retry.
        agent.request_completion.assert_not_called()

if __name__ == "__main__":
    unittest.main()
