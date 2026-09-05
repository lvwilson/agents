"""Tests for the ``.agent`` configuration files (``agents.config``).

Config files pin the backend (provider / model / base_url /
temperature):

* **project** — ``.agent``, searched upward from the cwd
* **global**  — ``~/.agents/agent_config.yaml`` (canonical), with the
  legacy ``~/.agent`` home-dir file as fallback; the first CLI run
  after the switch moves the legacy file to the new location
  (one-time, never deleted, never overwritten).

Resolution order (highest wins): project ``.agent`` (nearest
ancestor) > ``~/.agents/agent_config.yaml`` > legacy ``~/.agent`` >
environment variables.
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.config import (  # noqa: E402
    AGENT_FILENAME,
    HOME_AGENT_FILENAME,
    HOME_AGENT_MOVED_NOTICE,
    HOME_AGENT_MULTI_WARNING,
    HOME_AGENT_SQUAT_WARNING,
    LEGACY_HOME_AGENT_FILENAME,
    agent_config_path,
    home_agent_status,
    load_agent_config,
    migrate_legacy_home_agent,
    report_home_config,
)

_ENV_KEYS = ("AGENT_MODEL_PROVIDER", "AGENT_MODEL",
             "AGENT_BASE_URL", "AGENT_TEMPERATURE")

_LEGACY_SAMPLE = "provider: cerebras\nmodel: qwen-3.8-27b\n"


class _CleanEnv:
    """Context manager that clears the agent env vars for a test."""

    def __enter__(self):
        self._saved = {k: os.environ.pop(k, None) for k in _ENV_KEYS}
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _FakeHome:
    """Redirect ``~`` (via ``expanduser``) to an empty temp home dir.

    These tests must be hermetic with respect to the developer's real
    global config (``~/.agent`` / ``~/.agents/agent_config.yaml``) —
    without this mock, the machine's actual backend pin leaks into
    assertions about "nothing configured".
    """

    def __enter__(self):
        self._home = tempfile.mkdtemp()
        self._patcher = mock.patch(
            "agents.config.os.path.expanduser", return_value=self._home
        )
        self._patcher.start()
        return self._home

    def __exit__(self, *exc):
        self._patcher.stop()
        shutil.rmtree(self._home, ignore_errors=True)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class TestLoadAgentConfig(unittest.TestCase):
    def test_empty_when_nothing_configured(self):
        with _CleanEnv(), _FakeHome():
            with tempfile.TemporaryDirectory() as d:
                # tempdir is deep in the tree; ensure no ancestor .agent.
                self.assertEqual(load_agent_config(d), {})
                self.assertIsNone(agent_config_path(d))

    def test_project_file_parsed(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, AGENT_FILENAME), "w") as f:
                    f.write("provider: cerebras\n"
                            "model: qwen-3.8-27b\n"
                            "temperature: 0.3\n"
                            "base_url: http://localhost:8000/v1\n")
                cfg = load_agent_config(d)
                self.assertEqual(cfg, {
                    "provider": "cerebras",
                    "model": "qwen-3.8-27b",
                    "temperature": 0.3,
                    "base_url": "http://localhost:8000/v1",
                })
                self.assertEqual(agent_config_path(d),
                                 os.path.join(d, AGENT_FILENAME))

    def test_temperature_coerced_to_float(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, AGENT_FILENAME), "w") as f:
                    f.write("temperature: 0.5\n")
                self.assertEqual(load_agent_config(d)["temperature"], 0.5)

    def test_upward_search_from_subdirectory(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, AGENT_FILENAME), "w") as f:
                    f.write("provider: openai\nmodel: gpt-5.3\n")
                sub = os.path.join(d, "a", "b", "c")
                os.makedirs(sub)
                cfg = load_agent_config(sub)
                self.assertEqual(cfg["provider"], "openai")
                self.assertEqual(agent_config_path(sub),
                                 os.path.join(d, AGENT_FILENAME))

    def test_project_file_overrides_env(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                os.environ["AGENT_MODEL_PROVIDER"] = "openai"
                os.environ["AGENT_MODEL"] = "gpt-5.3"
                with open(os.path.join(d, AGENT_FILENAME), "w") as f:
                    f.write("provider: cerebras\nmodel: qwen-3.8-27b\n")
                cfg = load_agent_config(d)
                self.assertEqual(cfg["provider"], "cerebras")
                self.assertEqual(cfg["model"], "qwen-3.8-27b")

    def test_env_only_when_no_file(self):
        with _CleanEnv(), _FakeHome():
            with tempfile.TemporaryDirectory() as d:
                os.environ["AGENT_MODEL_PROVIDER"] = "gemini"
                os.environ["AGENT_MODEL"] = "gemini-3.1-pro-preview"
                cfg = load_agent_config(d)
                self.assertEqual(cfg["provider"], "gemini")
                self.assertEqual(cfg["model"], "gemini-3.1-pro-preview")

    def test_home_file_overrides_env(self):
        """The global config at ``~/.agents/agent_config.yaml`` wins over env."""
        with _CleanEnv(), _FakeHome() as home:
            with tempfile.TemporaryDirectory() as d:
                os.environ["AGENT_MODEL_PROVIDER"] = "openai"
                _write(os.path.join(home, ".agents", HOME_AGENT_FILENAME),
                       "provider: cerebras\n")
                cfg = load_agent_config(d)
                self.assertEqual(cfg["provider"], "cerebras")

    def test_project_file_overrides_home(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                home = os.path.join(d, "home")
                _write(os.path.join(home, ".agents", HOME_AGENT_FILENAME),
                       "provider: openai\nmodel: gpt-5.3\n")
                project = os.path.join(d, "proj")
                os.makedirs(project)
                with open(os.path.join(project, AGENT_FILENAME), "w") as f:
                    f.write("provider: cerebras\n")
                with mock.patch("agents.config.os.path.expanduser",
                                return_value=home):
                    cfg = load_agent_config(project)
                    # Provider comes from the project file.
                    self.assertEqual(cfg["provider"], "cerebras")
                    # Model comes from the global config (project only
                    # set provider).
                    self.assertEqual(cfg["model"], "gpt-5.3")

    def test_unknown_keys_ignored(self):
        with _CleanEnv(), _FakeHome():
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, AGENT_FILENAME), "w") as f:
                    f.write("provider: cerebras\napi_key: secret\nfoo: bar\n")
                cfg = load_agent_config(d)
                self.assertEqual(cfg, {"provider": "cerebras"})

    def test_invalid_yaml_returns_empty(self):
        with _CleanEnv(), _FakeHome():
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, AGENT_FILENAME), "w") as f:
                    f.write(":\n  - not: [valid: yaml\n")
                self.assertEqual(load_agent_config(d), {})

    def test_legacy_home_file_still_loads(self):
        """Without migration, the legacy ``~/.agent`` is still read."""
        with _CleanEnv(), _FakeHome() as home:
            with tempfile.TemporaryDirectory() as d:
                _write(os.path.join(home, LEGACY_HOME_AGENT_FILENAME),
                       _LEGACY_SAMPLE)
                cfg = load_agent_config(d)
                self.assertEqual(cfg, {"provider": "cerebras",
                                       "model": "qwen-3.8-27b"})
                self.assertEqual(agent_config_path(d),
                                 os.path.join(home,
                                              LEGACY_HOME_AGENT_FILENAME))

    def test_new_global_wins_over_legacy(self):
        """With both files present, the new location takes precedence."""
        with _CleanEnv(), _FakeHome() as home:
            with tempfile.TemporaryDirectory() as d:
                _write(os.path.join(home, LEGACY_HOME_AGENT_FILENAME),
                       "provider: openai\nmodel: gpt-5.3\n")
                _write(os.path.join(home, ".agents", HOME_AGENT_FILENAME),
                       "provider: cerebras\n")
                cfg = load_agent_config(d)
                self.assertEqual(cfg["provider"], "cerebras")
                # Model only exists in the legacy file…
                self.assertNotIn("model", cfg)
                self.assertEqual(agent_config_path(d),
                                 os.path.join(home, ".agents",
                                              HOME_AGENT_FILENAME))


class TestHomeAgentStatus(unittest.TestCase):
    """Status of the global config paths (new + legacy)."""

    def test_status_missing(self):
        with _CleanEnv(), _FakeHome():
            self.assertEqual(home_agent_status(), "missing")

    def test_status_ok_new_location(self):
        with _CleanEnv(), _FakeHome() as home:
            self.assertEqual(home_agent_status(), "missing")
            _write(os.path.join(home, ".agents", HOME_AGENT_FILENAME),
                   "provider: cerebras\n")
            self.assertEqual(home_agent_status(), "ok")

    def test_status_ok_legacy_location(self):
        with _CleanEnv(), _FakeHome() as home:
            _write(os.path.join(home, LEGACY_HOME_AGENT_FILENAME),
                   _LEGACY_SAMPLE)
            self.assertEqual(home_agent_status(), "ok")

    def test_status_multi_when_both_files(self):
        with _CleanEnv(), _FakeHome() as home:
            _write(os.path.join(home, LEGACY_HOME_AGENT_FILENAME),
                   _LEGACY_SAMPLE)
            _write(os.path.join(home, ".agents", HOME_AGENT_FILENAME),
                   "provider: cerebras\n")
            self.assertEqual(home_agent_status(), "multi")

    def test_status_squat_on_directory(self):
        """A directory named ``~/.agent`` (no new-location file) squats.

        The 2024 incident: a stale venv directory squatted ``~/.agent``
        and the global pin silently disappeared.  With no file in the
        new location, the status must still surface that state.
        """
        with _CleanEnv(), _FakeHome() as home:
            os.makedirs(os.path.join(home, AGENT_FILENAME))  # a dir!
            self.assertEqual(home_agent_status(), "squat")
            # The loader must ignore the squatting directory.
            proj = os.path.join(home, "proj")
            os.makedirs(proj)
            self.assertEqual(load_agent_config(proj), {})
            self.assertIn("~/.agent", HOME_AGENT_SQUAT_WARNING)

    def test_squat_hidden_by_ok_file(self):
        """A squatted legacy path is harmless once the new location is used."""
        with _CleanEnv(), _FakeHome() as home:
            os.makedirs(os.path.join(home, AGENT_FILENAME))  # squat
            _write(os.path.join(home, ".agents", HOME_AGENT_FILENAME),
                   "provider: cerebras\n")
            self.assertEqual(home_agent_status(), "ok")

    def test_upward_search_skips_directory_named_dot_agent(self):
        """A squatting directory must not stop the upward project search.

        (``openclaw`` legitimately keeps an ``.agent/`` *directory* of
        tracked workflow files in-tree; the search must continue past it
        to a real file in an ancestor.)
        """
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                base = os.path.join(d, "base")
                os.makedirs(base)
                with open(os.path.join(base, AGENT_FILENAME), "w") as f:
                    f.write("provider: openai\nmodel: gpt-5.3\n")
                proj = os.path.join(base, "mid", "proj")
                os.makedirs(proj)
                os.makedirs(os.path.join(proj, AGENT_FILENAME))  # squat in cwd
                cfg = load_agent_config(proj)
                self.assertEqual(cfg["provider"], "openai")
                self.assertEqual(cfg["model"], "gpt-5.3")
                self.assertEqual(agent_config_path(proj),
                                 os.path.join(base, AGENT_FILENAME))


class TestLegacyMigration(unittest.TestCase):
    """One-time ``~/.agent`` → ``~/.agents/agent_config.yaml`` move."""

    def test_migrate_moves_legacy_file(self):
        with _CleanEnv(), _FakeHome() as home:
            legacy = os.path.join(home, LEGACY_HOME_AGENT_FILENAME)
            new_path = os.path.join(home, ".agents", HOME_AGENT_FILENAME)
            _write(legacy, _LEGACY_SAMPLE)
            self.assertEqual(migrate_legacy_home_agent(),
                             HOME_AGENT_MOVED_NOTICE)
            self.assertFalse(os.path.exists(legacy))
            self.assertTrue(os.path.isfile(new_path))
            with open(new_path) as f:
                self.assertEqual(f.read(), _LEGACY_SAMPLE)
            self.assertEqual(home_agent_status(), "ok")

    def test_migrated_file_is_used_by_loader(self):
        with _CleanEnv(), _FakeHome() as home:
            _write(os.path.join(home, LEGACY_HOME_AGENT_FILENAME),
                   _LEGACY_SAMPLE)
            migrate_legacy_home_agent()
            with tempfile.TemporaryDirectory() as d:
                cfg = load_agent_config(d)
                self.assertEqual(cfg, {"provider": "cerebras",
                                       "model": "qwen-3.8-27b"})
                self.assertEqual(agent_config_path(d),
                                 os.path.join(home, ".agents",
                                              HOME_AGENT_FILENAME))

    def test_migrate_noop_when_nothing_there(self):
        with _CleanEnv(), _FakeHome():
            self.assertIsNone(migrate_legacy_home_agent())

    def test_migrate_noop_when_new_location_used(self):
        """Never overwrite: leftover legacy file stays, MULTI warning."""
        with _CleanEnv(), _FakeHome() as home:
            legacy = os.path.join(home, LEGACY_HOME_AGENT_FILENAME)
            new_path = os.path.join(home, ".agents", HOME_AGENT_FILENAME)
            _write(legacy, "provider: gemini\n")
            _write(new_path, "provider: cerebras\n")
            self.assertEqual(migrate_legacy_home_agent(),
                             HOME_AGENT_MULTI_WARNING)
            self.assertTrue(os.path.isfile(legacy))      # untouched
            with open(new_path) as f:
                self.assertEqual(f.read(), "provider: cerebras\n")
            self.assertEqual(home_agent_status(), "multi")
            self.assertIn("~/.agents/agent_config.yaml",
                          HOME_AGENT_MULTI_WARNING)
            self.assertIn("~/.agent", HOME_AGENT_MULTI_WARNING)

    def test_migrate_skips_squatting_directory(self):
        with _CleanEnv(), _FakeHome() as home:
            os.makedirs(os.path.join(home, LEGACY_HOME_AGENT_FILENAME))
            self.assertIsNone(migrate_legacy_home_agent())
            self.assertTrue(os.path.isdir(
                os.path.join(home, LEGACY_HOME_AGENT_FILENAME)))
            self.assertEqual(home_agent_status(), "squat")

    def test_migrate_is_idempotent(self):
        with _CleanEnv(), _FakeHome() as home:
            _write(os.path.join(home, LEGACY_HOME_AGENT_FILENAME),
                   _LEGACY_SAMPLE)
            self.assertEqual(migrate_legacy_home_agent(),
                             HOME_AGENT_MOVED_NOTICE)
            self.assertIsNone(migrate_legacy_home_agent())

    def test_report_home_config_migrates_silently(self):
        with _CleanEnv(), _FakeHome() as home:
            _write(os.path.join(home, LEGACY_HOME_AGENT_FILENAME),
                   _LEGACY_SAMPLE)
            # Side effect (migration) happens even when output is muted.
            report_home_config(verbose=False)
            self.assertFalse(
                os.path.exists(os.path.join(home,
                                            LEGACY_HOME_AGENT_FILENAME)))
            self.assertTrue(os.path.isfile(
                os.path.join(home, ".agents", HOME_AGENT_FILENAME)))

    def test_report_home_config_does_not_touch_missing(self):
        """No global config anywhere: nothing is created or reported."""
        with _CleanEnv(), _FakeHome() as home:
            report_home_config(verbose=False)
            self.assertFalse(os.path.exists(
                os.path.join(home, ".agents", HOME_AGENT_FILENAME)))


if __name__ == "__main__":
    unittest.main()
