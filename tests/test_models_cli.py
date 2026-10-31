"""Hermetic shell-CLI coverage for listing and persistent default selection."""

import json
from pathlib import Path
import tomllib

import pytest
import yaml

from agents import agents as app
from agents.cli.models_cli import main
from agents.config import load_agent_config


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AGENT_MODEL_CATALOG", str(home / "catalog.json"))
    monkeypatch.chdir(project)
    for key in (
        "AGENT_MODEL", "AGENT_MODEL_PROVIDER", "AGENT_BASE_URL",
        "LOCAL_MODEL", "LOCAL_LLM_HOST", "LOCAL_LLM_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    return home / ".agents" / "agent_config.yaml", project / ".agent"


def test_executable_aliases_registered():
    config = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )
    scripts = config["project"]["scripts"]
    assert scripts["agent-model"] == scripts["agent-models"] == "agents.cli.models_cli:main"


@pytest.mark.parametrize("provider", [None, "openai", "gemini"])
def test_list_offline_no_config_writes(workspace, capsys, monkeypatch, provider):
    def no_discovery(*args, **kwargs):
        pytest.fail("offline listing must not query provider APIs")

    monkeypatch.setattr("agents.cli.models_cli.discover_models", no_discovery)
    args = ["list"] + (["--provider", provider] if provider else [])
    assert main(args) == 0
    entries = json.loads(capsys.readouterr().out)
    assert entries
    assert all("model" in entry for entry in entries)
    if provider:
        assert {entry["provider"] for entry in entries} == {provider}
    assert not any(path.exists() for path in workspace)


def test_table_listing(workspace, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr("agents.cli.model_table.print_model_table", calls.append)
    assert main(["list", "--provider", "openai", "--table"]) == 0
    assert calls == ["openai"]
    assert not any(path.exists() for path in workspace)


@pytest.mark.parametrize("command", ["set-default", "set"])
def test_hosted_default_round_trip(workspace, capsys, command):
    assert main([command, "gpt-6.1-sol"]) == 0
    config = load_agent_config()
    assert app.resolve_backend_target(
        config_model=config["model"],
        config_provider=config["provider"],
        config_base_url=config["base_url"],
    ) == ("openai", "gpt-6.1-sol", None)
    assert config["base_url"] == ""
    assert "Saved global default" in capsys.readouterr().out
    assert not workspace[1].exists()


@pytest.mark.parametrize("scope", [
    ["--project"], ["--scope", "project"],
])
def test_project_local_selection(workspace, capsys, scope):
    assert main([
        "set-default", "my-local", *scope, "--local",
        "--host", "::1", "--port", "11434",
    ]) == 0
    assert yaml.safe_load(workspace[1].read_text()) == {
        "model": "my-local", "provider": "openai",
        "base_url": "http://[::1]:11434/v1",
    }
    assert not workspace[0].exists()
    assert "Saved project default" in capsys.readouterr().out


def test_hosted_clears_stale_endpoint_preserves_config(workspace):
    path, _ = workspace
    path.parent.mkdir()
    path.write_text(
        "# settings\nmodel: old\nprovider: openai\n"
        "base_url: http://localhost:9000/v1 # endpoint\n"
        "temperature: 0.3\ncustom:\n  keep: true\n"
    )
    assert main(["set-default", "gpt-6.1-sol", "--global", "--online"]) == 0
    text = path.read_text()
    data = yaml.safe_load(text)
    assert data["base_url"] == ""
    assert data["temperature"] == 0.3
    assert data["custom"] == {"keep": True}
    assert "# settings" in text and "# endpoint" in text


def test_uncatalogued_online_proxy(workspace):
    assert main([
        "set-default", "custom-cloud", "--online", "--provider", "openai",
        "--base-url", "https://proxy.example/v1",
    ]) == 0
    assert load_agent_config()["base_url"] == "https://proxy.example/v1"


@pytest.mark.parametrize("options", [
    ["typo", "--online"],
    ["custom", "--provider", "invalid"],
    ["custom", "--base-url", "not-a-url"],
    ["custom", "--local", "--port", "0"],
    ["gpt-6.1-sol", "--host", "example.com"],
])
def test_invalid_selection_nonzero_without_write(workspace, capsys, options):
    assert main(["set-default", *options]) == 1
    output = capsys.readouterr()
    assert "Default model error:" in output.err
    assert not output.out
    assert not any(path.exists() for path in workspace)


@pytest.mark.parametrize("options", [
    [], ["--local", "--online"], ["--global", "--project"],
    ["--port", "invalid"],
])
def test_argparse_errors(workspace, options):
    with pytest.raises(SystemExit) as exc:
        main(["set-default", *options])
    assert exc.value.code == 2
    assert not any(path.exists() for path in workspace)


def test_unsafe_yaml_preserved(workspace, capsys):
    path, _ = workspace
    path.parent.mkdir()
    path.write_text("model: [broken\n")
    original = path.read_bytes()
    assert main(["set-default", "gpt-6.1-sol"]) == 1
    assert path.read_bytes() == original
    assert "error:" in capsys.readouterr().err


def test_precedence_warnings(workspace, monkeypatch, capsys):
    workspace[1].write_text("model: project-local\n")
    monkeypatch.setenv("LOCAL_MODEL", "environment-local")
    assert main(["set-default", "gpt-6.1-sol"]) == 0
    output = capsys.readouterr().out
    assert "project config" in output
    assert "LOCAL_MODEL" in output


def test_help_documents_selection(workspace, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["set-default", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--project", "--scope", "--local", "--online", "--base-url"):
        assert option in help_text
    assert not any(path.exists() for path in workspace)
