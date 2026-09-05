"""Tests for the ``.agent`` configuration files (``agents.config``).

Config files pin the backend (provider / model / base_url /
temperature):

* **project** — ``.agent``, searched upward from the cwd
* **global**  — ``~/.agents/agent_config.yaml``, the single canonical
  cross-project pin location

The retired legacy location (a bare ``~/.agent`` file in the home
directory) is no longer read at all — see
``test_legacy_home_file_is_ignored``.

Resolution order (highest wins): project ``.agent`` (nearest
ancestor) > ``~/.agents/agent_config.yaml`` > environment variables.
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
    HOME_AGENT_SQUAT_WARNING,
    agent_config_path,
    home_agent_status,
    load_agent_config,
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
    global config (``~/.agents/agent_config.yaml``) — without this
    mock, the machine's actual backend pin leaks into assertions about
    "nothing configured".
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


def _global_path(home):
    return os.path.join(home, ".agents", HOME_AGENT_FILENAME)


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
                _write(_global_path(home), "provider: cerebras\n")
                cfg = load_agent_config(d)
                self.assertEqual(cfg["provider"], "cerebras")

    def test_project_file_overrides_home(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                home = os.path.join(d, "home")
                _write(_global_path(home),
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

    def test_legacy_home_file_is_ignored(self):
        """The retired ``~/.agent`` location is no longer read, period.

        Regression guard for the debt removal: a plausible-looking
        ``~/.agent`` file must neither pin the backend, shadow the
        global config path, nor be reported as it.
        """
        with _CleanEnv(), _FakeHome() as home:
            with tempfile.TemporaryDirectory() as d:
                _write(os.path.join(home, AGENT_FILENAME), _LEGACY_SAMPLE)
                # The file is right there, on disk — and completely
                # dead: no pin, no path, no status impact.
                self.assertTrue(os.path.isfile(
                    os.path.join(home, AGENT_FILENAME)))
                self.assertEqual(load_agent_config(d), {})
                self.assertIsNone(agent_config_path(d))
                self.assertEqual(home_agent_status(), "missing")

    def test_legacy_home_file_cannot_shadow_global(self):
        """Even alongside the live global config, the legacy one is a no-op."""
        with _CleanEnv(), _FakeHome() as home:
            with tempfile.TemporaryDirectory() as d:
                _write(os.path.join(home, AGENT_FILENAME),
                       "provider: openai\nmodel: gpt-5.3\n")
                _write(_global_path(home), "provider: cerebras\n")
                cfg = load_agent_config(d)
                # Everything comes from the global location alone.
                self.assertEqual(cfg, {"provider": "cerebras"})
                self.assertEqual(agent_config_path(d), _global_path(home))


class TestHomeAgentStatus(unittest.TestCase):
    """Status of the canonical global config path."""

    def test_status_missing(self):
        with _CleanEnv(), _FakeHome():
            self.assertEqual(home_agent_status(), "missing")

    def test_status_ok(self):
        with _CleanEnv(), _FakeHome() as home:
            self.assertEqual(home_agent_status(), "missing")
            _write(_global_path(home), "provider: cerebras\n")
            self.assertEqual(home_agent_status(), "ok")

    def test_status_squat_on_directory(self):
        """A directory squatting the global config path is reported.

        The 2024 incident: a stale venv directory squatted the old
        ``~/.agent`` name (a file path in the home dir) and the global
        pin silently disappeared.  Squatting the now-canonical path
        must surface the same way: status ``squat`` and the warning —
        while the loader ignores the directory.
        """
        with _CleanEnv(), _FakeHome() as home:
            os.makedirs(_global_path(home))  # a dir!
            self.assertEqual(home_agent_status(), "squat")
            # The loader must ignore the squatting directory.
            proj = os.path.join(home, "proj")
            os.makedirs(proj)
            self.assertEqual(load_agent_config(proj), {})
            self.assertRegex(HOME_AGENT_SQUAT_WARNING,
                             r"~/\.agents/agent_config\.yaml")

    def test_squatting_legacy_name_is_just_leftover_junk(self):
        """A non-file at the retired ``~/.agent`` location is no status at
        all — it cannot squat the global config anymore."""
        with _CleanEnv(), _FakeHome() as home:
            os.makedirs(os.path.join(home, AGENT_FILENAME))  # a dir!
            self.assertEqual(home_agent_status(), "missing")
            with tempfile.TemporaryDirectory() as d:
                self.assertEqual(load_agent_config(d), {})
                self.assertIsNone(agent_config_path(d))

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


if __name__ == "__main__":
    unittest.main()
