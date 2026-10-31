"""Hermetic integration tests for the persistent default-model tool."""

from pathlib import Path

import pytest
import yaml

from agents import agents as app
from agents.config import load_agent_config
from agents.tools import functions, parser

CATALOG = {
    "qwen-3.8-27b": "cerebras",
    "claude-opus-4-6": "anthropic",
}
LOCAL = "http://localhost:8000/v1"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(project)
    for key in (
        "AGENT_MODEL", "AGENT_MODEL_PROVIDER", "AGENT_BASE_URL",
        "AGENT_TEMPERATURE", "AGENT_EFFORT", "LOCAL_MODEL",
        "LOCAL_LLM_HOST", "LOCAL_LLM_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(app, "_online_models", lambda: CATALOG)
    return home / ".agents" / "agent_config.yaml", project / ".agent"


def saved_target():
    config = load_agent_config()
    return app.resolve_backend_target(
        config_model=config.get("model"),
        config_provider=config.get("provider"),
        config_base_url=config.get("base_url"),
    )


@pytest.mark.parametrize("args,expected", [
    (("qwen-3.8-27b",), ("cerebras", "qwen-3.8-27b", None)),
    (("claude-opus-4-6", "--online"), ("anthropic", "claude-opus-4-6", None)),
    (("custom-local",), ("openai", "custom-local", LOCAL)),
    (("qwen-3.8-27b", "--local"), ("openai", "qwen-3.8-27b", LOCAL)),
    (("custom", "--local", "--host", "::1", "--port", "11434"),
     ("openai", "custom", "http://[::1]:11434/v1")),
    (("custom", "--local", "--host", "192.168.1.50"),
     ("openai", "custom", "http://192.168.1.50:8000/v1")),
    (("custom", "--local", "--provider", "anthropic"),
     ("anthropic", "custom", LOCAL)),
    (("new-cloud", "--online", "--provider", "CEREBRAS"),
     ("cerebras", "new-cloud", None)),
    (("custom", "--provider", "openai", "--base-url", "https://proxy.example/prefix/v1"),
     ("openai", "custom", "https://proxy.example/prefix/v1")),
    (("qwen-3.8-27b", "--base-url", "https://proxy.example/custom"),
     ("cerebras", "qwen-3.8-27b", "https://proxy.example/custom")),
    (("qwen-3.8-27b", "--base-url", "http://127.0.0.2:8080/v1"),
     ("openai", "qwen-3.8-27b", "http://127.0.0.2:8080/v1")),
])
def test_selections_round_trip(workspace, args, expected):
    result = functions.set_default_model(*args)
    assert result.startswith("Saved global default"), result
    assert saved_target() == expected
    assert "running session" in result
    assert not workspace[1].exists()


def test_hosted_switch_clears_url_and_preserves_unrelated_text(workspace, monkeypatch):
    global_path, _ = workspace
    global_path.parent.mkdir()
    global_path.write_text(
        "# my config\nprovider: openai # protocol\nmodel: old\n"
        "base_url: http://localhost:11434/v1 # endpoint\n"
        "temperature: 0.7\neffort: high\ncustom:\n  model: do-not-touch\n"
    )
    monkeypatch.setenv("AGENT_BASE_URL", "http://localhost:9000/v1")
    result = functions.set_default_model("qwen-3.8-27b")
    assert result.startswith("Saved"), result
    config = yaml.safe_load(global_path.read_text())
    assert config["base_url"] == ""
    assert config["temperature"] == 0.7
    assert config["effort"] == "high"
    assert config["custom"]["model"] == "do-not-touch"
    assert "# protocol" in global_path.read_text()
    assert "# endpoint" in global_path.read_text()
    assert saved_target() == ("cerebras", "qwen-3.8-27b", None)


def test_project_scope_overrides_global_endpoint_without_changing_global(workspace):
    global_path, project_path = workspace
    functions.set_default_model("old-local")
    original = global_path.read_bytes()
    result = functions.set_default_model("qwen-3.8-27b", "--project")
    assert result.startswith("Saved project"), result
    assert project_path.exists()
    assert global_path.read_bytes() == original
    assert saved_target() == ("cerebras", "qwen-3.8-27b", None)


def test_project_scope_updates_nearest_ancestor(workspace, monkeypatch):
    global_path, project_path = workspace
    project_path.write_text("# parent\nmodel: old\n")
    child = project_path.parent / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    result = functions.set_default_model("custom", "--scope", "project")
    assert str(project_path) in result
    assert not (child / ".agent").exists()
    assert not global_path.exists()
    assert saved_target() == ("openai", "custom", LOCAL)


def test_global_warns_about_project_and_local_model(workspace, monkeypatch):
    global_path, project_path = workspace
    project_path.write_text("model: project-model\n")
    monkeypatch.setenv("LOCAL_MODEL", "env-model")
    original = project_path.read_bytes()
    result = functions.set_default_model("qwen-3.8-27b", "--global")
    assert "project config" in result and "takes precedence" in result
    assert "LOCAL_MODEL" in result and "unset it" in result
    assert project_path.read_bytes() == original
    assert yaml.safe_load(global_path.read_text())["model"] == "qwen-3.8-27b"


def test_local_environment_destination(workspace, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_HOST", "192.168.1.90")
    monkeypatch.setenv("LOCAL_LLM_PORT", "11434")
    functions.set_default_model("custom")
    assert saved_target() == ("openai", "custom", "http://192.168.1.90:11434/v1")


@pytest.mark.parametrize("args", [
    (), ("",), ("bad\nmodel",), ("model", "--bogus"),
    ("model", "--scope", "other"), ("model", "--global", "--project"),
    ("model", "--local", "--online"), ("model", "--provider", "typo"),
    ("model", "--online"), ("model", "--local", "--provider", "cerebras"),
    ("model", "--port", "wrong"), ("model", "--port", "0"),
    ("model", "--port", "65536"), ("qwen-3.8-27b", "--host", "server"),
    ("model", "--base-url", "https://proxy.example", "--port", "8000"),
    ("model", "--base-url", "file:///tmp/test"),
    ("model", "--base-url", "http://"),
    ("model", "--base-url", "http://localhost:abc/v1"),
    ("model", "--base-url", "http://user:pass@server/v1"),
    ("model", "--base-url", "http://server/v1?secret=yes"),
    ("model", "--base-url", "http://server/v1#fragment"),
    ("model", "--online", "--provider", "openai", "--base-url", LOCAL),
    ("model", "--local", "--host", "server/path"),
    ("qwen-3.8-27b", "--provider", "openai"),
    ("qwen-3.8-27b", "--local", "--host", "192.168.1.50"),
])
def test_invalid_selection_never_writes(workspace, args):
    global_path, project_path = workspace
    result = functions.set_default_model(*args)
    assert result.startswith("Error:"), result
    assert not global_path.exists()
    assert not project_path.exists()


def test_bad_local_environment_does_not_affect_hosted_selection(workspace, monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_PORT", "bad")
    assert functions.set_default_model("custom").startswith("Error:")
    assert functions.set_default_model("qwen-3.8-27b").startswith("Saved")


def test_malformed_config_left_untouched(workspace):
    global_path, _ = workspace
    global_path.parent.mkdir()
    global_path.write_text("model: [broken\n")
    original = global_path.read_bytes()
    assert functions.set_default_model("custom").startswith("Error:")
    assert global_path.read_bytes() == original


def test_project_directory_not_clobbered(workspace):
    _, project_path = workspace
    project_path.mkdir()
    result = functions.set_default_model("custom", "--project")
    assert result.startswith("Error:"), result
    assert project_path.is_dir()


def test_actual_dispatch_and_planning_guard(workspace):
    global_path, _ = workspace
    command = 'Command: set_default_model "qwen-3.8-27b" --global'
    for prefix in ("", "deep_read "):
        result, images = parser.process_content(
            command.replace("Command: ", "Command: " + prefix),
            blocked_commands=app.PLANNING_MODE_BLOCKED_COMMANDS,
        )
        assert "BLOCKED" in result
        assert not global_path.exists()
        assert not images
    result, images = parser.process_content(command)
    assert "Saved global default" in result
    assert saved_target() == ("cerebras", "qwen-3.8-27b", None)


@pytest.mark.parametrize("name", ["basic_agent", "sub_agent", "manipulator_agent"])
def test_packaged_prompts_document_tool(name):
    source = Path(app.__file__).parent / f"{name}.yaml"
    prompt = yaml.safe_load(source.read_text())["system_prompt"]
    assert "set_default_model MODEL" in prompt
    assert "future sessions" in prompt
    assert "user requests" in prompt


def test_non_bmp_model_survives_persistence(workspace):
    model = "custom-🧪"
    assert functions.set_default_model(model).startswith("Saved")
    assert saved_target() == ("openai", model, LOCAL)


@pytest.mark.parametrize("old_url", [LOCAL, "https://stale-proxy.example/v1"])
def test_actual_agent_startup_respects_cleared_url(workspace, old_url):
    from unittest import mock

    assert functions.set_default_model("qwen-3.8-27b").startswith("Saved")
    config = {
        "system_prompt": "system", "overbudget": "stop",
        "base_url": old_url, "provider": "openai",
    }
    with mock.patch.object(app, "read_configuration", return_value=config), \
            mock.patch.object(app, "build_context_guard", return_value=""), \
            mock.patch.object(app, "print_banner"), \
            mock.patch.object(app.Agent, "_register_summarize_backend"), \
            mock.patch.object(app.Agent, "_init_agent_pool"), \
            mock.patch.object(app, "create_backend") as backend:
        agent = app.Agent("fake.yaml", "task", session_id="default-model-test")
    assert agent.provider == "cerebras"
    assert agent.model_name == "qwen-3.8-27b"
    assert agent.base_url is None
    assert backend.call_args.args == ("cerebras",)
    assert backend.call_args.kwargs["base_url"] is None


@pytest.mark.parametrize("old_url", [LOCAL, "https://stale-proxy.example/v1"])
def test_commit_startup_respects_cleared_url(workspace, old_url):
    from unittest import mock
    from agents.cli import commit_cli

    assert functions.set_default_model("qwen-3.8-27b").startswith("Saved")
    with mock.patch.object(commit_cli, "read_configuration",
                           return_value={"base_url": old_url}), \
            mock.patch.object(commit_cli, "create_backend") as backend:
        backend.return_value.generate_response.return_value = "Test commit"
        assert commit_cli._generate_commit_message(
            "diff", ["file.py"], online=True
        ) == ("Test commit", "qwen-3.8-27b")
    backend.assert_called_once_with(
        "cerebras", model="qwen-3.8-27b", base_url=None
    )
