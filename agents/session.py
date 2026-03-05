"""
Session persistence for the agents package.

Sessions are stored as JSON files under a per-user temporary directory
(``/tmp/agents-<username>/``).  A lightweight index file maps working
directories to the most-recently-used session ID so that ``-r`` can
resume without an explicit ID.

Data retention is inherently limited: ``/tmp`` is cleared on reboot,
and stale sessions older than ``MAX_SESSION_AGE_DAYS`` are pruned on
every save.
"""

import json
import os
import random
import re
import string
import time


# ── Configuration ────────────────────────────────────────────────────

MAX_SESSION_AGE_DAYS = 7
_DIR_PERMISSIONS = 0o700
_FILE_PERMISSIONS = 0o600


# ── Paths ────────────────────────────────────────────────────────────

def _sessions_dir():
    """Return the per-user sessions directory path."""
    user = os.environ.get("USER", "unknown")
    return os.path.join("/tmp", f"agents-{user}")


def _index_path():
    """Return the path to the directory → session index file."""
    return os.path.join(_sessions_dir(), "index.json")


def _session_path(session_id):
    """Return the JSON file path for a given session ID."""
    return os.path.join(_sessions_dir(), f"{session_id}.json")


def _ensure_sessions_dir():
    """Create the sessions directory with restricted permissions."""
    d = _sessions_dir()
    os.makedirs(d, exist_ok=True)
    # Tighten permissions even if the directory already existed (e.g.
    # created by an older version without the chmod).
    os.chmod(d, _DIR_PERMISSIONS)


# ── Session ID helpers ───────────────────────────────────────────────

def generate_session_id(length=4):
    """Generate a short random alphanumeric session ID.

    Returns a lowercase string of *length* characters (default 4).
    The ID is claimed atomically via ``O_CREAT | O_EXCL`` to avoid
    TOCTOU races.
    """
    _ensure_sessions_dir()
    chars = string.ascii_lowercase + string.digits
    for _ in range(100):
        sid = ''.join(random.choices(chars, k=length))
        path = _session_path(sid)
        try:
            # Atomically create the file — fails if it already exists.
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _FILE_PERMISSIONS)
            os.close(fd)
            return sid
        except FileExistsError:
            continue
    # Extremely unlikely fallback: use a longer ID.
    return ''.join(random.choices(chars, k=length + 4))


def validate_session_id(sid):
    """Validate that a session ID is safe and within length limits.

    Raises ``ValueError`` if invalid.
    """
    if not sid:
        raise ValueError("Session ID cannot be empty")
    if len(sid) > 10:
        raise ValueError(f"Session ID must be at most 10 characters, got {len(sid)}")
    if not re.match(r'^[a-zA-Z0-9_-]+$', sid):
        raise ValueError(
            f"Session ID may only contain alphanumeric characters, "
            f"hyphens, and underscores: {sid!r}"
        )


# ── Index management ─────────────────────────────────────────────────

def _read_index():
    """Read the directory → session index, returning an empty dict on error."""
    path = _index_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_index(index):
    """Atomically write the directory → session index."""
    _ensure_sessions_dir()
    path = _index_path()
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(index, f, indent=2)
    os.chmod(tmp_path, _FILE_PERMISSIONS)
    os.replace(tmp_path, path)


def _update_index(working_dir, session_id):
    """Record that *session_id* is the latest session for *working_dir*."""
    index = _read_index()
    index[working_dir] = {
        "session_id": session_id,
        "timestamp": time.time(),
    }
    _write_index(index)


def get_latest_session_for_dir(working_dir):
    """Return the most recent session ID used in *working_dir*, or ``None``."""
    index = _read_index()
    entry = index.get(working_dir)
    if entry is None:
        return None
    sid = entry.get("session_id")
    if sid and os.path.exists(_session_path(sid)):
        return sid
    return None


# ── Save / Load ──────────────────────────────────────────────────────

def save_session(session_id, working_dir, state):
    """Persist *state* dict to the session file for *session_id*.

    Also updates the directory index and prunes stale sessions.
    """
    _ensure_sessions_dir()
    state = {
        'session_id': session_id,
        'working_dir': working_dir,
        'timestamp': time.time(),
        **state,
    }
    path = _session_path(session_id)
    tmp_path = path + ".tmp"
    with open(tmp_path, 'w') as f:
        json.dump(state, f, indent=2)
    os.chmod(tmp_path, _FILE_PERMISSIONS)
    os.replace(tmp_path, path)

    _update_index(working_dir, session_id)
    _prune_stale_sessions()


def load_session(session_id):
    """Load and return the state dict for *session_id*.

    Raises ``FileNotFoundError`` if the session file does not exist.
    """
    path = _session_path(session_id)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No session file found for session '{session_id}' at {path}"
        )
    with open(path, 'r') as f:
        return json.load(f)


# ── Cleanup ──────────────────────────────────────────────────────────

def _prune_stale_sessions():
    """Remove session files older than ``MAX_SESSION_AGE_DAYS``."""
    cutoff = time.time() - MAX_SESSION_AGE_DAYS * 86400
    d = _sessions_dir()
    pruned_sids = set()
    for fname in os.listdir(d):
        if not fname.endswith('.json') or fname == 'index.json':
            continue
        path = os.path.join(d, fname)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                pruned_sids.add(fname[:-5])  # session id
        except OSError:
            continue

    # Clean stale entries from the index.
    if pruned_sids:
        index = _read_index()
        changed = False
        for wd, entry in list(index.items()):
            if entry.get("session_id") in pruned_sids:
                del index[wd]
                changed = True
        if changed:
            _write_index(index)
