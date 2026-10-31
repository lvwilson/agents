"""Folder memory retains its legacy identity when a project is moved."""

import hashlib
import logging
import stat
import subprocess

import pytest
import yaml

from agents import config, memory


@pytest.fixture
def folder(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(memory, "_MEMORY_DIR", tmp_path / "memory")
    monkeypatch.setattr(memory, "_tables_ensured", False)
    monkeypatch.setattr(memory, "_resolve_folder_cache", {})
    monkeypatch.setattr(memory, "_identity_warnings", set())
    # Most tests exercise plain directories, independent of the host's Git setup.
    monkeypatch.setattr(memory, "_resolve_folder_uncached", lambda cwd: cwd)
    return root


def legacy_hash(path):
    return hashlib.sha256(str(path).encode()).hexdigest()


def saved_hash(root):
    return yaml.safe_load((root / ".agent").read_text())["folder_hash"]


def test_first_startup_creates_identity_without_any_memory(folder):
    from agents.agents import build_context_guard

    assert not (folder / ".agent").exists()
    build_context_guard()
    assert saved_hash(folder) == legacy_hash(folder)
    assert stat.S_IMODE((folder / ".agent").stat().st_mode) == 0o600
    assert memory.get_notes() == ""
    assert memory.get_episodes() == []


def test_existing_legacy_database_is_reused(folder):
    conn = memory._get_connection()
    fh = legacy_hash(folder)
    conn.execute(
        "INSERT INTO notes (folder_hash, content, updated_at) VALUES (?, ?, 0)",
        (fh, "legacy notes"),
    )
    for table in ("episodes", "episodes_raw"):
        conn.execute(
            f"INSERT INTO {table} "
            "(folder_hash, episode_index, summary, session_id, created_at) "
            "VALUES (?, 0, 'legacy episode', 'old-session', 0)",
            (fh,),
        )
    conn.commit()
    conn.close()

    assert memory.get_notes() == "legacy notes"
    assert saved_hash(folder) == fh
    assert memory.get_episodes()[0]["summary"] == "legacy episode"
    assert memory.get_raw_episodes()[0]["session_id"] == "old-session"
    assert memory.add_episode("next episode") == 1


@pytest.mark.parametrize("git_repo", [False, True])
def test_memory_survives_move(folder, monkeypatch, git_repo):
    if git_repo:
        # Use the production Git-root resolver in this integration case.
        monkeypatch.setattr(memory, "_resolve_folder_uncached", GIT_RESOLVER)
        subprocess.run(["git", "init", "-q", str(folder)], check=True)
    child = folder / "nested"
    child.mkdir()
    if git_repo:
        monkeypatch.chdir(child)
    memory.note_append("remember this")
    memory.add_episode("before move", session_id="session")
    identity = memory.folder_hash()

    moved = folder.with_name("renamed-project")
    folder.rename(moved)
    monkeypatch.chdir(moved / "nested" if git_repo else moved)
    # Simulate a fresh process, not just an in-process cached lookup.
    memory._resolve_folder_cache.clear()

    assert memory.folder_hash() == identity == legacy_hash(folder)
    assert identity != legacy_hash(moved)
    assert memory.folder_path() == str(moved)
    assert memory.get_notes() == "remember this"
    assert memory.get_episodes()[0]["summary"] == "before move"
    assert memory.get_raw_episodes()[0]["session_id"] == "session"
    assert memory.add_episode("after move") == 1
    assert memory.note_append("still here") == "remember this\nstill here"
    assert saved_hash(moved) == identity
    assert not (moved / "nested" / ".agent").exists()


GIT_RESOLVER = memory._resolve_folder_uncached


def test_git_root_identity_wins_over_nested_config(folder, monkeypatch):
    monkeypatch.setattr(memory, "_resolve_folder_uncached", GIT_RESOLVER)
    subprocess.run(["git", "init", "-q", str(folder)], check=True)
    root_identity = memory.folder_hash()
    child = folder / "nested"
    child.mkdir()
    (child / ".agent").write_text(f'folder_hash: {"b" * 64}\nmodel: child\n')
    monkeypatch.chdir(child)
    assert memory.folder_hash() == root_identity
    assert memory.folder_path() == str(folder)
    assert saved_hash(child) == "b" * 64


def test_existing_settings_comments_and_permissions_preserved(folder):
    path = folder / ".agent"
    original = (
        '# Project settings 🧪\nprovider: openai # comment\nmodel: custom\n'
        'extra:\n  values: [one, two]\n'
    )
    path.write_text(original)
    path.chmod(0o640)
    fh = memory.folder_hash()
    assert path.read_text() == original + f'folder_hash: "{fh}"\n'
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    config.write_agent_config_keys(str(path), set_keys={"model": "new"})
    assert saved_hash(folder) == fh
    assert memory.folder_hash() == fh


def test_saved_identity_is_used_without_rewriting(folder, monkeypatch):
    path = folder / ".agent"
    original = f'# Saved elsewhere\nfolder_hash: "{"a" * 64}"\n'
    path.write_text(original)
    before = path.stat()

    def unexpected_write(*args, **kwargs):
        pytest.fail("an existing identity must not be rewritten")

    monkeypatch.setattr(memory, "write_agent_config_keys", unexpected_write)
    assert memory.folder_hash() == "a" * 64
    assert path.read_text() == original
    assert path.stat().st_mtime_ns == before.st_mtime_ns


@pytest.mark.parametrize("content", [
    "folder_hash: invalid\n",
    "folder_hash: null\n",
    "folder_hash: 123\n",
    "folder_hash: [value]\n",
    "model: [invalid yaml\n",
    "- not a mapping\n",
    "model: one\nmodel: two\n",
    "model: &model example\nextra: *model\n",
    "{model: example}\n",
])
def test_unsafe_or_invalid_config_is_not_overwritten(folder, caplog, content):
    path = folder / ".agent"
    path.write_text(content)
    with caplog.at_level(logging.WARNING):
        assert memory.folder_hash() == legacy_hash(folder)
        assert memory.folder_hash() == legacy_hash(folder)
    assert path.read_text() == content
    assert len(caplog.records) == 1
    assert "memory may not survive a move" in caplog.text


def test_unwritable_config_falls_back_and_can_retry(folder, monkeypatch, caplog):
    writer = memory.write_agent_config_keys

    def denied(*args, **kwargs):
        raise ValueError("permission denied")

    monkeypatch.setattr(memory, "write_agent_config_keys", denied)
    assert memory.folder_hash() == legacy_hash(folder)
    assert not (folder / ".agent").exists()
    assert "permission denied" in caplog.text
    monkeypatch.setattr(memory, "write_agent_config_keys", writer)
    assert memory.folder_hash() == legacy_hash(folder)
    assert saved_hash(folder) == legacy_hash(folder)


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_nonregular_config_is_not_modified(folder, caplog, kind):
    path = folder / ".agent"
    target = folder / "other.yaml"
    target.write_text("model: original\n")
    if kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(target)
    assert memory.folder_hash() == legacy_hash(folder)
    assert (path.is_dir() if kind == "directory" else path.is_symlink())
    assert target.read_text() == "model: original\n"
    assert "Cannot persist folder memory identity" in caplog.text


def test_failed_write_rereads_concurrently_created_identity(folder, monkeypatch):
    def competing_writer(*args, **kwargs):
        (folder / ".agent").write_text(f'folder_hash: "{"c" * 64}"\n')
        raise ValueError("config changed while writing")

    monkeypatch.setattr(memory, "write_agent_config_keys", competing_writer)
    assert memory.folder_hash() == "c" * 64


def test_identity_is_local_but_backend_settings_remain_inherited(folder, monkeypatch):
    parent_config = folder.parent / ".agent"
    parent_config.write_text(
        f'model: inherited-model\nfolder_hash: "{"d" * 64}"\n'
    )
    monkeypatch.setattr(config, "_home_agent", lambda: None)
    for env in config._ENV_MAP.values():
        monkeypatch.delenv(env, raising=False)
    before = config.load_agent_config()
    assert before == {"model": "inherited-model"}
    assert memory.folder_hash() == legacy_hash(folder)
    assert config.load_agent_config() == before
    assert config.agent_config_path() == str(parent_config)
    # Model-setting tools should still target the actual project's .agent.
    assert config.project_config_path() == str(folder / ".agent")


def test_hash_is_not_a_backend_option(folder, monkeypatch):
    monkeypatch.setattr(config, "_home_agent", lambda: None)
    for env in config._ENV_MAP.values():
        monkeypatch.delenv(env, raising=False)
    (folder / ".agent").write_text(
        f'model: my-model\nfolder_hash: "{"e" * 64}"\n'
    )
    assert config.load_agent_config() == {"model": "my-model"}
    assert config.agent_config_path() == str(folder / ".agent")


def test_different_folders_have_independent_identities(folder, monkeypatch):
    first = memory.folder_hash()
    other = folder.with_name("other")
    other.mkdir()
    monkeypatch.chdir(other)
    assert memory.folder_hash() == legacy_hash(other) != first
    assert saved_hash(folder) == first
