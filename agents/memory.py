"""
Hierarchical memory system backed by SQLite.

Memory is indexed per folder (hashed cwd).  If the current directory is
inside a git repository, memory is pinned to the repository root so that
all sub-directories share the same memory context.

Two kinds of memory are supported:

* **Episodes** — short summaries captured at the end of each agent session.
  After 8 episodes are stored the oldest 4 are squashed together by the
  LLM to form 1 compressed episode.  Original episodes are kept verbatim
  in a separate append-only table for future analysis.

* **Notes** — free-form notes maintained by the agent itself via
  append / replace / rewrite commands.  When notes exceed a configurable
  character limit the agent is prompted to rewrite them more compactly.

A CLI endpoint ``agent-memory view`` lets a human inspect the memory
as the LLM will see it.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional


# ── Configuration ────────────────────────────────────────────────────

_AGENTS_DIR = Path.home() / ".agents"
_MEMORY_DIR = _AGENTS_DIR / "memory"
_MEMORY_DB_NAME = "memory.db"
MAX_NOTES_CHARS = 10_000
EPISODES_BEFORE_SQUASH = 8
SQUASH_KEEP = 4  # oldest N episodes are squashed into 1


def _db_path() -> Path:
    """Return the path to the SQLite memory database."""
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    return _MEMORY_DIR / _MEMORY_DB_NAME


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the memory database, creating tables if needed."""
    db = _db_path()
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create memory tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_hash TEXT NOT NULL,
            episode_index INTEGER NOT NULL,
            summary TEXT NOT NULL,
            session_id TEXT,
            created_at REAL NOT NULL,
            UNIQUE(folder_hash, episode_index)
        );

        CREATE TABLE IF NOT EXISTS episodes_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_hash TEXT NOT NULL,
            episode_index INTEGER NOT NULL,
            summary TEXT NOT NULL,
            session_id TEXT,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            folder_hash TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL,
            UNIQUE(folder_hash)
        );
    """)
    conn.commit()


# ── Folder resolution ───────────────────────────────────────────────

def _resolve_folder() -> str:
    """Return the folder path to use for memory indexing.

    If the current working directory is inside a git repository, the
    repository root is returned.  Otherwise the cwd itself is used.
    """
    cwd = os.getcwd()
    # Try to find git root
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, cwd=cwd, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return cwd


def folder_hash() -> str:
    """Return a stable hash for the current folder (git root or cwd)."""
    folder = _resolve_folder()
    return hashlib.sha256(folder.encode()).hexdigest()


def folder_path() -> str:
    """Return the resolved folder path (git root or cwd)."""
    return _resolve_folder()


# ── Episode operations ──────────────────────────────────────────────

def add_episode(
    summary: str,
    session_id: Optional[str] = None,
) -> int:
    """Store an episode summary and return its index.

    Original text is also stored in the append-only ``episodes_raw`` table.
    """
    conn = _get_connection()
    fh = folder_hash()
    # Determine next index
    row = conn.execute(
        "SELECT COALESCE(MAX(episode_index), -1) + 1 AS next_idx FROM episodes WHERE folder_hash = ?",
        (fh,),
    ).fetchone()
    idx = row["next_idx"]

    now = time.time()
    conn.execute(
        "INSERT INTO episodes (folder_hash, episode_index, summary, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (fh, idx, summary, session_id, now),
    )
    conn.execute(
        "INSERT INTO episodes_raw (folder_hash, episode_index, summary, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (fh, idx, summary, session_id, now),
    )
    conn.commit()
    conn.close()
    return idx


def get_episodes() -> list[dict]:
    """Return all compressed episodes for the current folder, ordered by index."""
    conn = _get_connection()
    fh = folder_hash()
    rows = conn.execute(
        "SELECT episode_index, summary, session_id, created_at FROM episodes "
        "WHERE folder_hash = ? ORDER BY episode_index",
        (fh,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_episode_count() -> int:
    """Return the number of compressed episodes for the current folder."""
    conn = _get_connection()
    fh = folder_hash()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM episodes WHERE folder_hash = ?",
        (fh,),
    ).fetchone()
    conn.close()
    return row["cnt"]


def get_raw_episodes() -> list[dict]:
    """Return all raw (append-only) episodes for the current folder."""
    conn = _get_connection()
    fh = folder_hash()
    rows = conn.execute(
        "SELECT episode_index, summary, session_id, created_at FROM episodes_raw "
        "WHERE folder_hash = ? ORDER BY episode_index",
        (fh,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def squash_episodes(squash_fn) -> Optional[str]:
    """Squash the oldest episodes into one compressed episode.

    When the number of episodes reaches EPISODES_BEFORE_SQUASH, the
    oldest SQUASH_KEEP episodes are selected, their summaries are
    concatenated and passed to *squash_fn* (a callable that takes a
    string and returns a compressed summary string).

    The squashed episodes are removed from the compressed table and
    replaced by a single new episode.  A ``[SQUASHED]`` entry is added
    to the raw table for audit purposes.

    Returns the new compressed summary, or None if no squash was needed.
    """
    conn = _get_connection()
    fh = folder_hash()

    count = conn.execute(
        "SELECT COUNT(*) AS cnt FROM episodes WHERE folder_hash = ?",
        (fh,),
    ).fetchone()["cnt"]

    if count < EPISODES_BEFORE_SQUASH:
        conn.close()
        return None

    # Get the oldest SQUASH_KEEP episodes
    rows = conn.execute(
        "SELECT episode_index, summary FROM episodes "
        "WHERE folder_hash = ? ORDER BY episode_index LIMIT ?",
        (fh, SQUASH_KEEP),
    ).fetchall()

    indices = [r["episode_index"] for r in rows]
    summaries = [r["summary"] for r in rows]

    # Build input for the LLM
    parts = []
    for idx, summary in zip(indices, summaries):
        parts.append(f"Episode {idx}:\n{summary}")
    input_text = "\n\n".join(parts)

    # Compress via LLM
    compressed = squash_fn(input_text)

    # Determine the new index: min of squashed indices
    new_idx = min(indices)

    # Remove old episodes
    placeholders = ",".join("?" * len(indices))
    conn.execute(
        f"DELETE FROM episodes WHERE folder_hash = ? AND episode_index IN ({placeholders})",
        [fh] + indices,
    )

    # Insert the squashed episode
    conn.execute(
        "INSERT INTO episodes (folder_hash, episode_index, summary, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (fh, new_idx, compressed, None, time.time()),
    )

    # Also store in raw table for audit trail
    conn.execute(
        "INSERT INTO episodes_raw (folder_hash, episode_index, summary, session_id, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (fh, new_idx, f"[SQUASHED] {compressed}", None, time.time()),
    )

    conn.commit()
    conn.close()
    return compressed


# ── Notes operations ────────────────────────────────────────────────

def get_notes() -> str:
    """Return the current notes content for the current folder."""
    conn = _get_connection()
    fh = folder_hash()
    row = conn.execute(
        "SELECT content FROM notes WHERE folder_hash = ?",
        (fh,),
    ).fetchone()
    conn.close()
    if row and row["content"]:
        return row["content"]
    return ""


def note_append(text: str) -> str:
    """Append *text* to the notes.  Returns the updated notes."""
    conn = _get_connection()
    fh = folder_hash()
    row = conn.execute(
        "SELECT content FROM notes WHERE folder_hash = ?",
        (fh,),
    ).fetchone()
    current = row["content"] if row else ""

    if current and not current.endswith("\n"):
        current += "\n"
    new_content = current + text.strip()

    conn.execute(
        "INSERT INTO notes (folder_hash, content, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(folder_hash) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
        (fh, new_content, time.time()),
    )
    conn.commit()
    conn.close()
    return new_content


def note_replace(pattern: str, replacement: str) -> str:
    """Replace all occurrences of *pattern* in notes with *replacement*.

    Returns the updated notes.
    """
    conn = _get_connection()
    fh = folder_hash()
    row = conn.execute(
        "SELECT content FROM notes WHERE folder_hash = ?",
        (fh,),
    ).fetchone()
    current = row["content"] if row else ""

    new_content = current.replace(pattern, replacement)

    conn.execute(
        "INSERT INTO notes (folder_hash, content, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(folder_hash) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
        (fh, new_content, time.time()),
    )
    conn.commit()
    conn.close()
    return new_content


def note_rewrite(new_content: str) -> str:
    """Replace the entire notes content with *new_content*.

    Returns the new notes content.
    """
    conn = _get_connection()
    fh = folder_hash()
    conn.execute(
        "INSERT INTO notes (folder_hash, content, updated_at) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(folder_hash) DO UPDATE SET content = excluded.content, updated_at = excluded.updated_at",
        (fh, new_content.strip(), time.time()),
    )
    conn.commit()
    conn.close()
    return new_content.strip()


def notes_need_compact() -> bool:
    """Return True if notes exceed the character limit."""
    return len(get_notes()) > MAX_NOTES_CHARS


# ── View / rendering ────────────────────────────────────────────────

def format_memory_view() -> str:
    """Return the memory as the LLM will see it (appended to system prompt).

    This is also what ``agent-memory view`` displays to the human.
    """
    parts: list[str] = []

    episodes = get_episodes()
    if episodes:
        parts.append("=== Folder Memory: Episodes ===")
        for ep in episodes:
            parts.append(f"Episode {ep['episode_index']}: {ep['summary']}")
        parts.append("")

    notes = get_notes()
    if notes:
        parts.append("=== Folder Memory: Notes ===")
        parts.append(notes)
        parts.append("")

    if not parts:
        return ""

    return "\n".join(parts)


def format_raw_view() -> str:
    """Return all raw episodes (append-only) for analysis."""
    episodes = get_raw_episodes()
    if not episodes:
        return "(No raw episodes stored.)"
    lines = [f"=== Raw Episodes ({len(episodes)} total) ==="]
    for ep in episodes:
        sid = ep.get("session_id") or "?"
        lines.append(f"Episode {ep['episode_index']} (session {sid}):")
        lines.append(ep["summary"])
        lines.append("")
    return "\n".join(lines)
