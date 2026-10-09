"""Conservative config editing must preserve text and fail without data loss."""

import os
import stat
from unittest import mock

import pytest
import yaml

from agents import config


def write(path, text):
    path.write_bytes(text.encode("utf-8"))
    return path


def update(path, **kwargs):
    return config.write_agent_config_keys(str(path), **kwargs)


def test_create_missing_file_and_parents(tmp_path):
    path = tmp_path / "missing" / "nested" / ".agent"
    result = update(path, set_keys={"MODEL": "yes", "provider": "openai",
                                   "temperature": "0.3"})
    assert result == (str(path), ("provider", "model", "temperature"), ())
    assert yaml.safe_load(path.read_text()) == {
        "provider": "openai", "model": "yes", "temperature": 0.3,
    }
    assert path.read_text().startswith(config.DEFAULT_CONFIG_COMMENT + "\n")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_append_multiple_keys_and_update_existing(tmp_path):
    path = write(tmp_path / ".agent", "# header\nmodel: old # keep\nextra: retained")
    result = update(path, set_keys={"provider": "openai", "model": "new",
                                   "base_url": "http://localhost:8000/v1"})
    assert result[1] == ("provider", "model", "base_url")
    assert path.read_text() == (
        '# header\nmodel: "new" # keep\nextra: retained\nprovider: "openai"\n'
        'base_url: "http://localhost:8000/v1"\n'
    )


def test_quoted_keys_comments_and_nested_keys_unchanged(tmp_path):
    original = (
        '# model: example\n"model" : old  # model note\n'
        "'provider': old # provider note\n"
        "extra:\n  model: nested # untouched\n  base_url: nested\n"
        "  sequence: [{model: inside}]\n"
        "notes: |\n  model: prose\n  more text\n"
    )
    path = write(tmp_path / ".agent", original)
    update(path, set_keys={"model": "next", "provider": "openai", "base_url": "url"})
    assert path.read_text() == original.replace(
        '"model" : old', '"model" : "next"'
    ).replace("'provider': old", "'provider': \"openai\"") + 'base_url: "url"\n'


@pytest.mark.parametrize("value", [
    "yes", "null", "123", "a: b # c", 'quote"and\\slash', "two\nlines",
    "", "☃", "custom-🧪", "a\x85\u2028\u2029z", True, False, 7, 0.125,
])
def test_scalar_round_trip(tmp_path, value):
    path = write(tmp_path / ".agent", "model: old # keep\n")
    update(path, set_keys={"model": value})
    assert yaml.safe_load(path.read_text())["model"] == value
    assert len(path.read_text().splitlines()) == 1
    assert path.read_text().endswith(" # keep\n")


@pytest.mark.parametrize("text", ["model:\n", "model: # keep\n", "model: null # keep\n"])
def test_null_scalar_assignment(tmp_path, text):
    path = write(tmp_path / ".agent", text)
    update(path, set_keys={"model": "next"})
    assert yaml.safe_load(path.read_text()) == {"model": "next"}
    if "# keep" in text:
        assert path.read_text().endswith("# keep\n")


def test_removal_retains_comments_and_nested_values(tmp_path):
    path = write(tmp_path / ".agent",
                 '# header\n"model": old  # model note\n'
                 "provider: old\nbase_url: # URL note\n"
                 "extra:\n  model: nested\n")
    result = update(path, set_keys={"MODEL": None},
                    remove_keys=["provider", "base_url", "model", "effort"])
    assert result == (str(path), (), ("model", "provider", "base_url"))
    assert path.read_text() == (
        "# header\n# model note\n# URL note\nextra:\n  model: nested\n"
    )


def test_assignment_wins_over_removal(tmp_path):
    path = write(tmp_path / ".agent", "model: old # keep\n")
    result = update(path, set_keys={"model": "new"}, remove_keys=["MODEL", "model"])
    assert result[1:] == (("model",), ("model",))
    assert path.read_text() == 'model: "new" # keep\n'


def test_remove_only_key(tmp_path):
    path = write(tmp_path / ".agent", "model: old\n")
    update(path, set_keys={"model": None})
    assert path.read_bytes() == b""


@pytest.mark.parametrize("text", [
    "", "# comment\n", "# comment", "\n# header\n\n",
])
def test_empty_or_comment_only_documents(tmp_path, text):
    path = write(tmp_path / ".agent", text)
    update(path, set_keys={"model": "new", "effort": "high"})
    assert yaml.safe_load(path.read_text()) == {"model": "new", "effort": "high"}
    if text:
        assert path.read_text().startswith(text)


def test_crlf_and_no_final_newline_preserved(tmp_path):
    path = write(tmp_path / ".agent", '# header\r\nmodel: old # keep\r\nother: x')
    update(path, set_keys={"model": "new"})
    assert path.read_bytes() == b'# header\r\nmodel: "new" # keep\r\nother: x'
    update(path, set_keys={"provider": "openai"})
    assert path.read_bytes().endswith(b'other: x\r\nprovider: "openai"\r\n')


@pytest.mark.parametrize("text", [
    "model: [broken\n", "- list\n", "null\n", "42\n", "plain string\n",
    "{model: old}\n", "{}\n",
    "model: old\nmodel: duplicate\n", 'model: old\n"model": duplicate\n',
    "extra:\n  same: one\n  same: two\n", "1: value\n",
    "? [a, b]\n: value\n", "? model\n: old\n",
    "  model: old\n", '"mo\\\n  del": old\n',
    "model: &pin old\n", "extra: &pin old\nmodel: *pin\n",
    "extra: &pin {model: old}\n<<: *pin\n",
    "model: !!str old\n", "model: !custom old\n",
    "---\nmodel: old\n", "model: old\n...\n",
    "---\nmodel: old\n---\nmodel: other\n",
    "%YAML 1.1\n---\nmodel: old\n",
    "model: |\n  multiline\n", "model: >\n  multiline\n",
    'model: "multi\n  line"\n', "model:\n  nested: value\n",
    "model: [one, two]\n", "model: {nested: value}\n",
    "model:\n  old\n", "\ufeffmodel: old\n", "model: old\u2028other: x\n",
])
def test_unsupported_yaml_is_not_modified(tmp_path, text):
    path = write(tmp_path / ".agent", text)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        update(path, set_keys={"model": "new"})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("text", [
    "model: |\n  multiline\n", "model: [one]\n", "model:\n  child: x\n",
])
def test_unsupported_removal_does_not_truncate_values(tmp_path, text):
    path = write(tmp_path / ".agent", text)
    with pytest.raises(ValueError):
        update(path, remove_keys=["model"])
    assert path.read_text() == text


@pytest.mark.parametrize("kwargs", [
    {"set_keys": {"api_key": "secret"}},
    {"set_keys": {1: "wrong"}},
    {"remove_keys": ["not_known"]},
    {"remove_keys": [None]},
    {"set_keys": {"MODEL": "one", "model": "two"}},
    {"set_keys": {"model": ["not", "scalar"]}},
    {"set_keys": {"model": {"not": "scalar"}}},
    {"set_keys": {"model": b"bytes"}},
    {"set_keys": [("model", "value")]},
    {"remove_keys": "model"},
])
def test_invalid_requests_do_not_create_parent(tmp_path, kwargs):
    path = tmp_path / "missing" / ".agent"
    with pytest.raises(ValueError):
        update(path, **kwargs)
    assert list(tmp_path.iterdir()) == []


def test_noop_and_absent_removals_do_not_create_file(tmp_path):
    path = tmp_path / "missing" / ".agent"
    assert update(path) == (str(path), (), ())
    assert update(path, remove_keys=["model"]) == (str(path), (), ())
    assert not path.parent.exists()


def test_absent_removal_leaves_existing_file_and_inode_unchanged(tmp_path):
    path = write(tmp_path / ".agent", "# only comment\n")
    before = path.stat()
    assert update(path, remove_keys=["model"])[2] == ()
    assert path.stat() == before
    assert path.read_text() == "# only comment\n"


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644, 0o754])
def test_existing_permissions_preserved(tmp_path, mode):
    path = write(tmp_path / ".agent", "model: old\n")
    path.chmod(mode)
    update(path, set_keys={"model": "new"})
    assert stat.S_IMODE(path.stat().st_mode) == mode


def test_directory_target_rejected(tmp_path):
    path = tmp_path / ".agent"
    path.mkdir()
    with pytest.raises(ValueError):
        update(path, set_keys={"model": "new"})
    assert path.is_dir()


@pytest.mark.parametrize("dangling", [False, True])
def test_target_symlink_rejected(tmp_path, dangling):
    target = tmp_path / "real"
    if not dangling:
        write(target, "model: old\n")
    path = tmp_path / ".agent"
    path.symlink_to(target)
    with pytest.raises(ValueError):
        update(path, set_keys={"model": "new"})
    assert path.is_symlink()
    if dangling:
        assert not target.exists()
    else:
        assert target.read_text() == "model: old\n"


def test_symlink_parent_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    parent = tmp_path / "link"
    parent.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError):
        update(parent / "missing" / ".agent", set_keys={"model": "new"})
    assert list(real.iterdir()) == []


def test_non_directory_parent_rejected(tmp_path):
    parent = write(tmp_path / "file", "untouched")
    with pytest.raises(ValueError):
        update(parent / ".agent", set_keys={"model": "new"})
    assert parent.read_text() == "untouched"


def test_fifo_target_rejected_without_blocking(tmp_path):
    path = tmp_path / ".agent"
    os.mkfifo(path)
    with pytest.raises(ValueError):
        update(path, set_keys={"model": "new"})
    assert stat.S_ISFIFO(path.lstat().st_mode)


@pytest.mark.parametrize("failure", ["mkstemp", "fsync", "fchmod", "replace"])
def test_atomic_failure_keeps_original_and_cleans_temp(tmp_path, failure):
    path = write(tmp_path / ".agent", "model: old # keep\n")
    path.chmod(0o640)
    module = config.tempfile if failure == "mkstemp" else config.os
    with mock.patch.object(module, failure, side_effect=OSError("injected failure")):
        with pytest.raises(ValueError, match="injected failure"):
            update(path, set_keys={"model": "new"})
    assert path.read_text() == "model: old # keep\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert list(tmp_path.iterdir()) == [path]


def test_unique_same_directory_temp_not_fixed_name(tmp_path):
    path = write(tmp_path / ".agent", "model: old\n")
    victim = write(tmp_path / "victim", "untouched")
    fixed = tmp_path / ".agent.tmp"
    fixed.symlink_to(victim)
    actual_replace = os.replace
    sources = []

    def replace(source, destination):
        assert os.path.dirname(source) == str(tmp_path)
        assert source != str(fixed)
        assert stat.S_IMODE(os.stat(source).st_mode) == stat.S_IMODE(path.stat().st_mode)
        assert path.read_text().startswith("model:")
        sources.append(source)
        actual_replace(source, destination)

    with mock.patch.object(config.os, "replace", side_effect=replace):
        update(path, set_keys={"model": "one"})
        update(path, set_keys={"model": "two"})
    assert len(set(sources)) == 2
    assert fixed.is_symlink()
    assert victim.read_text() == "untouched"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".agent", ".agent.tmp", "victim"]


def test_concurrent_edit_detected(tmp_path):
    path = write(tmp_path / ".agent", "model: old\n")
    fsync = os.fsync

    def modify(fd):
        fsync(fd)
        write(path, "model: concurrent\n")

    with mock.patch.object(config.os, "fsync", side_effect=modify):
        with pytest.raises(ValueError, match="changed while writing"):
            update(path, set_keys={"model": "new"})
    assert path.read_text() == "model: concurrent\n"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_utf8_unchanged(tmp_path):
    path = tmp_path / ".agent"
    path.write_bytes(b"model: \xff\n")
    with pytest.raises(ValueError):
        update(path, set_keys={"model": "new"})
    assert path.read_bytes() == b"model: \xff\n"


def test_public_path_helpers_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert config.global_config_path() == str(tmp_path / "home" / ".agents" / "agent_config.yaml")
    root = tmp_path / "project"
    root.mkdir()
    path = write(root / ".agent", "model: old\n")
    child = root / "child"
    child.mkdir()
    assert config.project_config_path(str(child)) == str(path)


@pytest.mark.parametrize("text", ["model: old\rother: keep\r",
                                  "model: old\r\nother: keep\r"])
def test_bare_carriage_returns_rejected_without_data_loss(tmp_path, text):
    path = write(tmp_path / ".agent", text)
    with pytest.raises(ValueError, match="carriage-return"):
        update(path, remove_keys=["model"])
    assert path.read_bytes() == text.encode()


def test_escaped_quoted_key_and_hash_in_value(tmp_path):
    path = write(tmp_path / ".agent", '"\\u006dodel": "old # value" # actual comment\n')
    update(path, set_keys={"model": "new # value"})
    assert path.read_text() == '"\\u006dodel": "new # value" # actual comment\n'
    update(path, remove_keys=["model"])
    assert path.read_text() == "# actual comment\n"


def test_removal_preserves_final_comment_without_newline(tmp_path):
    path = write(tmp_path / ".agent", "model: old # final")
    update(path, remove_keys=["model"])
    assert path.read_text() == "# final"


def test_relative_path_supported(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config.write_agent_config_keys(
        ".agent", set_keys={"model": "new"}
    ) == (".agent", ("model",), ())
    assert yaml.safe_load((tmp_path / ".agent").read_text()) == {"model": "new"}


def test_write_failure_keeps_original_and_cleans_temp(tmp_path):
    from contextlib import contextmanager

    path = write(tmp_path / ".agent", "model: original\n")
    fdopen = os.fdopen

    @contextmanager
    def failing_writer(fd, mode, **kwargs):
        with fdopen(fd, mode, **kwargs) as stream:
            if mode == "w":
                wrapped = mock.Mock(wraps=stream)
                wrapped.write.side_effect = OSError("write failed")
                yield wrapped
            else:
                yield stream

    with mock.patch.object(config.os, "fdopen", side_effect=failing_writer):
        with pytest.raises(ValueError, match="write failed"):
            update(path, set_keys={"model": "new"})
    assert path.read_text() == "model: original\n"
    assert list(tmp_path.iterdir()) == [path]


def test_replace_failure_on_new_file_leaves_no_file(tmp_path):
    path = tmp_path / ".agent"
    with mock.patch.object(config.os, "replace", side_effect=OSError("failed")):
        with pytest.raises(ValueError):
            update(path, set_keys={"model": "new"})
    assert list(tmp_path.iterdir()) == []


def test_target_swapped_to_symlink_before_read_is_not_followed(tmp_path):
    path = write(tmp_path / ".agent", "model: original\n")
    victim = write(tmp_path / "victim", "do not read or modify")
    original_open = os.open

    def swap(target, flags, *args, **kwargs):
        if target == str(path):
            path.unlink()
            path.symlink_to(victim)
        return original_open(target, flags, *args, **kwargs)

    with mock.patch.object(config.os, "open", side_effect=swap):
        with pytest.raises(ValueError):
            update(path, set_keys={"model": "new"})
    assert path.is_symlink()
    assert victim.read_text() == "do not read or modify"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".agent", "victim"]


def test_target_swapped_to_symlink_before_replace_is_not_overwritten(tmp_path):
    path = write(tmp_path / ".agent", "model: original\n")
    victim = write(tmp_path / "victim", "do not modify")
    fsync = os.fsync

    def swap(fd):
        fsync(fd)
        path.unlink()
        path.symlink_to(victim)

    with mock.patch.object(config.os, "fsync", side_effect=swap):
        with pytest.raises(ValueError, match="not a regular file"):
            update(path, set_keys={"model": "new"})
    assert path.is_symlink()
    assert victim.read_text() == "do not modify"
    assert sorted(p.name for p in tmp_path.iterdir()) == [".agent", "victim"]
