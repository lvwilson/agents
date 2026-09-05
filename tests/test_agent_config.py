"""Tests for the ``.agent`` configuration file (``agents.config``).

The ``.agent`` file is the primary way to pin a backend (provider / model
/ base_url / temperature) for a project or globally.  Resolution order
(highest wins): project ``.agent`` (nearest ancestor) > home ``~/.agent``
> environment variables.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.config import (  # noqa: E402
    load_agent_config,
    agent_config_path,
    AGENT_FILENAME,
)

_ENV_KEYS = ("AGENT_MODEL_PROVIDER", "AGENT_MODEL",
             "AGENT_BASE_URL", "AGENT_TEMPERATURE")


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


class TestLoadAgentConfig(unittest.TestCase):
    def test_empty_when_nothing_configured(self):
        with _CleanEnv():
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
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                os.environ["AGENT_MODEL_PROVIDER"] = "gemini"
                os.environ["AGENT_MODEL"] = "gemini-3.1-pro-preview"
                cfg = load_agent_config(d)
                self.assertEqual(cfg["provider"], "gemini")
                self.assertEqual(cfg["model"], "gemini-3.1-pro-preview")

    def test_home_file_overrides_env(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                os.environ["AGENT_MODEL_PROVIDER"] = "openai"
                home = os.path.join(d, "home")
                os.makedirs(home)
                with open(os.path.join(home, AGENT_FILENAME), "w") as f:
                    f.write("provider: cerebras\n")
                with mock.patch("agents.config.os.path.expanduser",
                                return_value=home):
                    cfg = load_agent_config(d)
                self.assertEqual(cfg["provider"], "cerebras")

    def test_project_file_overrides_home(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                home = os.path.join(d, "home")
                os.makedirs(home)
                with open(os.path.join(home, AGENT_FILENAME), "w") as f:
                    f.write("provider: openai\nmodel: gpt-5.3\n")
                project = os.path.join(d, "proj")
                os.makedirs(project)
                with open(os.path.join(project, AGENT_FILENAME), "w") as f:
                    f.write("provider: cerebras\n")
                with mock.patch("agents.config.os.path.expanduser",
                                return_value=home):
                    cfg = load_agent_config(project)
                self.assertEqual(cfg["provider"], "cerebras")
                # Model comes from home (project only set provider).
                self.assertEqual(cfg["model"], "gpt-5.3")

    def test_unknown_keys_ignored(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, AGENT_FILENAME), "w") as f:
                    f.write("provider: cerebras\napi_key: secret\nfoo: bar\n")
                cfg = load_agent_config(d)
                self.assertEqual(cfg, {"provider": "cerebras"})

    def test_invalid_yaml_returns_empty(self):
        with _CleanEnv():
            with tempfile.TemporaryDirectory() as d:
                with open(os.path.join(d, AGENT_FILENAME), "w") as f:
                    f.write(":\n  - not: [valid: yaml\n")
                self.assertEqual(load_agent_config(d), {})


if __name__ == "__main__":
    unittest.main()
