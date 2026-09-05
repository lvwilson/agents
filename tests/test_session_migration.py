"""Regression tests for the one-time /tmp/agents-$USER -> ~/.agents/sessions migration.

Closes open item M12 from untracked/report_complete.md: a session that was
live when storage moved from /tmp to ~/.agents/sessions must still be
migrated and resumable with -r.
"""

import json
import time

import agents.session as sess


def _make_state(sid, working_dir="/home/user/project"):
    return {
        'session_id': sid,
        'working_dir': working_dir,
        'timestamp': time.time(),
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}],
    }


def _patch_dirs(monkeypatch, tmp_path):
    """Point the session store and the legacy store at fresh temp dirs."""
    new_dir = tmp_path / "sessions"
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    monkeypatch.setattr(sess, '_sessions_dir', lambda: str(new_dir))
    monkeypatch.setattr(sess, '_legacy_sessions_dir', lambda: str(legacy_dir))
    return new_dir, legacy_dir


def test_legacy_session_files_are_copied(tmp_path, monkeypatch):
    new_dir, legacy_dir = _patch_dirs(monkeypatch, tmp_path)
    (legacy_dir / 'abcd.json').write_text(json.dumps(_make_state('abcd')))
    (legacy_dir / 'notes.txt').write_text('not a session file')

    sess.save_session('efgh', '/home/user/project', {'messages': []})

    assert (new_dir / 'abcd.json').exists()          # migrated
    assert (new_dir / 'efgh.json').exists()          # the new save landed
    assert not (new_dir / 'notes.txt').exists()      # non-session files ignored


def test_legacy_index_entries_merged_so_r_resumes(tmp_path, monkeypatch):
    new_dir, legacy_dir = _patch_dirs(monkeypatch, tmp_path)
    (legacy_dir / 'abcd.json').write_text(
        json.dumps(_make_state('abcd', '/home/user/old-project')))
    (legacy_dir / 'index.json').write_text(json.dumps({
        '/home/user/old-project': {'session_id': 'abcd', 'timestamp': 123.0},
    }))

    sess.save_session('efgh', '/home/user/project', {'messages': []})

    assert sess.get_latest_session_for_dir('/home/user/old-project') == 'abcd'


def test_newer_session_is_never_overwritten(tmp_path, monkeypatch):
    new_dir, legacy_dir = _patch_dirs(monkeypatch, tmp_path)
    (legacy_dir / 'abcd.json').write_text(json.dumps(_make_state('abcd')))
    new_dir.mkdir(parents=True, exist_ok=True)
    (new_dir / 'abcd.json').write_text(json.dumps({'winner': 'current'}))

    sess.save_session('efgh', '/home/user/project', {'messages': []})

    assert json.loads((new_dir / 'abcd.json').read_text()) == {'winner': 'current'}


def test_index_entry_for_missing_session_file_is_skipped(tmp_path, monkeypatch):
    new_dir, legacy_dir = _patch_dirs(monkeypatch, tmp_path)
    (legacy_dir / 'index.json').write_text(json.dumps({
        '/home/user/gone-project': {'session_id': 'zzzz', 'timestamp': 1.0},
    }))

    sess.save_session('efgh', '/home/user/project', {'messages': []})

    assert sess.get_latest_session_for_dir('/home/user/gone-project') is None


def test_migration_is_a_noop_without_legacy_dir(tmp_path, monkeypatch):
    new_dir = tmp_path / 'sessions'
    absent = tmp_path / 'no-legacy'
    monkeypatch.setattr(sess, '_sessions_dir', lambda: str(new_dir))
    monkeypatch.setattr(sess, '_legacy_sessions_dir', lambda: str(absent))

    sess.save_session('efgh', '/home/user/project', {'messages': []})

    assert (new_dir / 'efgh.json').exists()
