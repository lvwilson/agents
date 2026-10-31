"""LLM generation and real-Git regression tests for agent-commit."""

import subprocess
from unittest import mock

import pytest

from agents.cli import commit_cli

FENCE = "`" * 5


@pytest.fixture
def backend(monkeypatch):
    for name in ("LOCAL_MODEL", "AGENT_MODEL", "AGENT_MODEL_PROVIDER",
                 "AGENT_BASE_URL", "LOCAL_LLM_HOST", "LOCAL_LLM_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(commit_cli, "read_configuration",
                        lambda _: {})
    monkeypatch.setattr(commit_cli, "load_agent_config", lambda: {})
    factory = mock.Mock()
    factory.return_value.generate_response.return_value = (
        f"{FENCE}\nDescribe actual changes\n{FENCE}"
    )
    monkeypatch.setattr(commit_cli, "create_backend", factory)
    return factory


def git(*args):
    result = subprocess.run(
        ["git", *args], text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Do not depend on (or modify) the user's Git identity or hooks.
    for name in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME",
                 "GIT_COMMITTER_EMAIL", "GIT_DIR", "GIT_WORK_TREE",
                 "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    git("init", "-q")
    git("config", "user.name", "Repository User")
    git("config", "user.email", "user@example.test")
    git("config", "commit.gpgsign", "false")
    git("config", "core.hooksPath", str(tmp_path / "no-hooks"))
    (tmp_path / "file.txt").write_text("before\n")
    git("add", "file.txt")
    git("commit", "-qm", "Initial")
    (tmp_path / "file.txt").write_text("after\n")
    return tmp_path


def invoke(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["agent-commit", *args])
    commit_cli.main()


def test_default_configuration_calls_llm(backend):
    message, model = commit_cli._generate_commit_message("actual diff", ["file.txt"])
    assert message == "Describe actual changes"
    assert model == "claude-opus-4-6"
    backend.assert_called_once_with("anthropic", model=model, base_url=None)
    system, context = backend.return_value.generate_response.call_args.args
    assert system == commit_cli.COMMIT_SYSTEM_PROMPT
    assert "actual diff" in context[0]["content"][0]["text"]


def test_configured_model_without_flags_calls_llm(backend, monkeypatch):
    monkeypatch.setattr(commit_cli, "load_agent_config",
                        lambda: {"model": "qwen-3.8-27b"})
    assert commit_cli._generate_commit_message("diff", ["file.txt"]) == (
        "Describe actual changes", "qwen-3.8-27b"
    )
    backend.assert_called_once_with(
        "cerebras", model="qwen-3.8-27b", base_url=None
    )


def test_local_environment_calls_llm(backend, monkeypatch):
    monkeypatch.setenv("LOCAL_MODEL", "local-model")
    monkeypatch.setenv("LOCAL_LLM_PORT", "invalid")
    assert commit_cli._generate_commit_message("diff", ["file.txt"])[1] == "local-model"
    backend.assert_called_once_with(
        "openai", model="local-model", base_url="http://localhost:8000/v1"
    )


@pytest.mark.parametrize("response", [
    f"{FENCE}\nA useful message\n{FENCE}",
    "```\nA useful message\n```",
    "Task: scaffolding\nDetailed thoughts and Plans: scaffolding\nA useful message",
    "A useful message",
])
def test_shared_message_extraction(backend, response):
    backend.return_value.generate_response.return_value = response
    assert commit_cli._generate_commit_message("diff", ["file.txt"])[0] == "A useful message"


@pytest.mark.parametrize("response", ["", "   ", "Task: scaffolding\nWorklog: scaffolding"])
def test_unusable_response_is_not_replaced(backend, response):
    backend.return_value.generate_response.return_value = response
    with pytest.raises(ValueError, match="no usable commit message"):
        commit_cli._generate_commit_message("diff", ["file.txt"])


@pytest.mark.parametrize("args", [
    ("--tracked", "-m", "qwen-3.8-27b"),
    ("--all", "-m", "qwen-3.8-27b"),
    ("-m", "qwen-3.8-27b", "file.txt"),
    ("--tracked",),
])
def test_real_commit_uses_llm_message_and_resolved_author(
    repo, backend, monkeypatch, args
):
    monkeypatch.setattr(commit_cli, "load_agent_config",
                        lambda: {"model": "qwen-3.8-27b"})
    if "-m" in args:
        monkeypatch.setenv("LOCAL_MODEL", "stale-local-model")
        monkeypatch.setenv("AGENT_MODEL", "stale-env-model")
    monkeypatch.setattr(commit_cli.platform, "node", lambda: "test-host")
    invoke(monkeypatch, *args)
    assert git("log", "-1", "--format=%s%n%an%n%ae").splitlines() == [
        "Describe actual changes", "qwen-3.8-27b", "agent@test-host",
    ]
    assert git("status", "--porcelain") == ""
    backend.assert_called_once_with(
        "cerebras", model="qwen-3.8-27b", base_url=None
    )
    prompt = backend.return_value.generate_response.call_args.args[1][0]["content"][0]["text"]
    assert "-before" in prompt
    assert "+after" in prompt


@pytest.mark.parametrize("failure", ["construction", "generation", "empty", "routing"])
def test_generation_failure_does_not_commit(
    repo, backend, monkeypatch, capsys, failure
):
    args = ["--tracked", "-m", "qwen-3.8-27b"]
    if failure == "construction":
        backend.side_effect = RuntimeError("backend unavailable")
    elif failure == "generation":
        backend.return_value.generate_response.side_effect = RuntimeError("API unavailable")
    elif failure == "empty":
        backend.return_value.generate_response.return_value = ""
    else:
        args = ["--tracked", "-o", "-m", "unknown-online-model"]
    before = git("rev-parse", "HEAD")
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, *args)
    assert error.value.code == 1
    assert git("rev-parse", "HEAD") == before
    assert git("diff", "--cached", "--name-only") == "file.txt"
    assert "No commit was created; changes remain staged." in capsys.readouterr().err
    if failure == "routing":
        backend.assert_not_called()


def test_unrelated_staged_files_still_rejected(repo, backend, monkeypatch):
    (repo / "other.txt").write_text("unrelated\n")
    git("add", "other.txt")
    with pytest.raises(SystemExit) as error:
        invoke(monkeypatch, "file.txt")
    assert error.value.code == 1
    backend.assert_not_called()
    assert git("log", "-1", "--format=%s") == "Initial"


def test_no_changes_does_not_call_llm(repo, backend, monkeypatch):
    git("restore", "file.txt")
    invoke(monkeypatch, "file.txt")
    backend.assert_not_called()
