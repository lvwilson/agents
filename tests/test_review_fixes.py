"""Regression tests for code-review bug fixes.

Covers:
- parser: deep_read inner-command arg forwarding, bare deep_read guard,
  tuple-result normalization (report items #3, #5, #6)
- agent_pool: --nogit flag and model propagation to sub-agents (#1)
- agents: commit-message fence stripping and crash protection (#7)
- git_utils: MM status classification, pathspec commits, '--' guards
- commit_cli: author identity respects -m (#9)
- docs: egg-info glob and .jar extension nits
"""
import os
import subprocess
import tempfile
import unittest
from unittest import mock

from agents.tools import parser
from agents.tools.parser import process_content, _execute_command
from agents import git_utils
from agents.cli import commit_cli
from agents.agent_pool import AgentPool


class TestDeepReadConsoleCommand(unittest.TestCase):
    """Bug #3: deep_read run_console_command "…" dropped its inner args."""

    def test_deep_read_console_executes_command(self):
        result = _execute_command('run_console_command "echo deep_read_works"', None, None)
        self.assertIn("deep_read_works", result)

    def test_deep_read_console_via_process_content(self):
        response, _ = process_content('Command: deep_read run_console_command "echo via_process"')
        self.assertIn("via_process", response)

    def test_plain_console_command_still_works(self):
        response, _ = process_content('Command: run_console_command "echo plain_works"')
        self.assertIn("plain_works", response)

    def test_deep_read_read_file_still_works(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("file_contents_here")
            path = f.name
        try:
            result = _execute_command(f"read_file {path}", None, None)
            self.assertIn("file_contents_here", result)
        finally:
            os.unlink(path)


class TestBareDeepRead(unittest.TestCase):
    """Bug #5: bare deep_read raised IndexError instead of a clean error."""

    def test_bare_deep_read_returns_error_string(self):
        result = _execute_command("", None, None)
        self.assertIn("Error", result)
        self.assertIn("inner command", result)

    def test_bare_deep_read_via_process_content_no_crash(self):
        # process_slice stores arguments as "" for a bare command
        response, _ = process_content("Command: deep_read")
        self.assertIn("Error", response)


class TestDeepReadTupleResult(unittest.TestCase):
    """Bug #6: deep_read view_page crashed concatenating a tuple with str."""

    def test_tuple_result_is_normalized(self):
        fake_result = ("page text here", "/tmp/screenshot.png")
        with mock.patch.object(parser.functions, "view_page", return_value=fake_result, create=True):
            response, _ = process_content("Command: deep_read view_page https://example.com")
        self.assertIn("page text here", response)


class TestAgentPoolCommandLine(unittest.TestCase):
    """Bug #1: sub-agents need --nogit and the parent's model."""

    def _captured_cmd(self, pool, monkey_run):
        pool.create("tester", "do things")
        pool.run("tester", "the task")
        assert monkey_run.called
        return monkey_run.call_args[0][0]

    @staticmethod
    def _args_after_nogit(cmd):
        """Everything after '--nogit' (avoids matching python's own -m)."""
        return cmd[cmd.index("--nogit") + 1:]

    def test_cmd_includes_nogit(self):
        pool = AgentPool()
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout="done", returncode=0)
            cmd = self._captured_cmd(pool, mock_run)
        self.assertIn("--nogit", cmd)

    def test_known_online_model_propagated(self):
        pool = AgentPool()
        pool.model = "kimi-k3"
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout="done", returncode=0)
            cmd = self._captured_cmd(pool, mock_run)
        tail = self._args_after_nogit(cmd)
        self.assertEqual(tail, ["-m", "kimi-k3"])

    def test_unknown_model_not_propagated(self):
        # Unknown (local) models propagate via inherited LOCAL_MODEL env,
        # not via -m (which would lose custom host/port settings).
        pool = AgentPool()
        pool.model = "my-local-model-9000"
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout="done", returncode=0)
            cmd = self._captured_cmd(pool, mock_run)
        self.assertEqual(self._args_after_nogit(cmd), [])

    def test_no_model_no_flag(self):
        pool = AgentPool()
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(stdout="done", returncode=0)
            cmd = self._captured_cmd(pool, mock_run)
        self.assertEqual(self._args_after_nogit(cmd), [])


class TestCommitMessageHandling(unittest.TestCase):
    """Bug #7 + sanitization: request_commit_message must not crash the
    CLI and must strip markdown fences from the model's reply."""

    def _make_agent(self):
        from agents.agents import Agent
        agent = Agent.__new__(Agent)
        agent.context = []
        agent.client = mock.Mock(cost=0.0)
        agent.compute_budget = 1.0
        agent._iterate = mock.Mock()
        return agent

    def test_fences_stripped(self):
        agent = self._make_agent()

        def fake_iterate(free_form=False):
            # Internal one-shot calls must run in free-form mode so the
            # accidental-stop guard doesn't fire on plain-prose replies.
            assert free_form, "commit-message iteration must be free_form"
            agent.context.append({
                "role": "assistant",
                "content": [{"type": "text", "text": "```\nfix the thing\n```"}],
            })
        agent._iterate.side_effect = fake_iterate
        self.assertEqual(agent.request_commit_message(), "fix the thing")

    def test_iterate_failure_returns_none(self):
        agent = self._make_agent()
        agent._iterate.side_effect = RuntimeError("looping error")
        self.assertIsNone(agent.request_commit_message())

    def test_episode_summary_failure_returns_none(self):
        agent = self._make_agent()
        agent._iterate.side_effect = RuntimeError("looping error")
        self.assertIsNone(agent._request_episode_summary())


class TestGitUtils(unittest.TestCase):
    """git status classification and pathspec-safe commits."""

    def setUp(self):
        self.repo = tempfile.TemporaryDirectory()
        self.addCleanup(self.repo.cleanup)
        env = dict(os.environ)
        for args in (
            ["init", "-q"],
            ["config", "user.email", "test@example.com"],
            ["config", "user.name", "Test"],
        ):
            subprocess.run(["git"] + args, cwd=self.repo.name, check=True,
                           capture_output=True, env=env)
        # Initial commit so tracked modifications are possible
        with open(os.path.join(self.repo.name, "tracked.txt"), "w") as f:
            f.write("v1\n")
        subprocess.run(["git", "add", "-A"], cwd=self.repo.name, check=True,
                       capture_output=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=self.repo.name,
                       check=True, capture_output=True)

    def test_mm_status_reported_as_both(self):
        path = os.path.join(self.repo.name, "tracked.txt")
        with open(path, "a") as f:
            f.write("staged\n")
        subprocess.run(["git", "add", "tracked.txt"], cwd=self.repo.name,
                       check=True, capture_output=True)
        with open(path, "a") as f:
            f.write("unstaged\n")
        clean, msg = git_utils.check_git_clean(self.repo.name)
        self.assertFalse(clean)
        self.assertIn("staged: tracked.txt", msg)
        self.assertIn("modified: tracked.txt", msg)

    def test_commit_with_files_uses_pathspec(self):
        # Pre-stage an unrelated file; it must not be committed.
        with open(os.path.join(self.repo.name, "unrelated.txt"), "w") as f:
            f.write("staged elsewhere\n")
        subprocess.run(["git", "add", "unrelated.txt"], cwd=self.repo.name,
                       check=True, capture_output=True)

        with open(os.path.join(self.repo.name, "tracked.txt"), "a") as f:
            f.write("work\n")

        ok, err = git_utils.git_add_and_commit(
            "work on tracked", path=self.repo.name, files=["tracked.txt"],
        )
        self.assertTrue(ok, err)
        # unrelated.txt must still be staged, not committed
        out = subprocess.run(
            ["git", "status", "--porcelain"], cwd=self.repo.name,
            capture_output=True, text=True, check=True,
        ).stdout
        self.assertIn("A  unrelated.txt", out)

    def test_commit_without_files_commits_all(self):
        with open(os.path.join(self.repo.name, "tracked.txt"), "a") as f:
            f.write("work\n")
        ok, err = git_utils.git_add_and_commit("all", path=self.repo.name)
        self.assertTrue(ok, err)
        clean, _ = git_utils.check_git_clean(self.repo.name)
        self.assertTrue(clean)


class TestCommitCliAuthor(unittest.TestCase):
    """Bug #9: author identity must respect the -m flag."""

    def test_model_flag_used_for_author(self):
        name, email = commit_cli._get_agent_author("claude-fable-5")
        self.assertEqual(name, "claude-fable-5")
        self.assertTrue(email.startswith("agent@"))

    def test_env_fallback_preserved(self):
        with mock.patch.dict(os.environ, {"AGENT_MODEL": "env-model"}, clear=False):
            os.environ.pop("LOCAL_MODEL", None)
            name, _ = commit_cli._get_agent_author()
        self.assertEqual(name, "env-model")


class TestDocsNits(unittest.TestCase):
    """docs.py skip-list fixes."""

    def test_egg_info_glob_replaced_by_suffix(self):
        from agents.tools import docs
        self.assertNotIn("*.egg-info", docs.SKIP_DIRS)
        self.assertIn(".egg-info", docs.SKIP_DIR_SUFFIXES)
        self.assertTrue("foo.egg-info".endswith(docs.SKIP_DIR_SUFFIXES))

    def test_jar_not_indexed_as_text(self):
        from agents.tools import docs
        self.assertNotIn(".jar", docs.TEXT_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
