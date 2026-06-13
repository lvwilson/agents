"""
Docs tool — index, search, and browse local document/code folders.

Two interfaces:
1. **CLI** (run as script): ``python -m agents.tools.docs <subcommand> ...``
   Used by the user to manage indexes.
2. **Agent harness** (called as ``docs <subcommand> ...``):
   Used by the LLM to search and browse indexed sources.

Index storage
~~~~~~~~~~~~~
All indexes are stored in a SQLite database:
    ``~/.agents/docs_index.db``

Two tables:
    *sources* — one row per indexed folder (name, path, alias, indexed_at)
    *files*   — one row per file (source, rel_path, size, content)

Each source maps to a folder on disk.  When indexed, every readable text
file is stored with its relative path and full content (up to a size
limit), enabling fast fuzzy search without touching the filesystem.
"""

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone

# ── Configuration ───────────────────────────────────────────────────

INDEX_DIR = os.path.expanduser("~/.agents")
INDEX_FILE = os.path.join(INDEX_DIR, "docs_index.db")

# Maximum file size to index (bytes). Larger files are recorded but
# their content is not stored to keep the index lean.
MAX_FILE_SIZE = 256 * 1024  # 256 KB

# File extensions considered as text/code (case-insensitive)
TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".css", ".scss", ".sass",
    ".html", ".htm", ".xml", ".json", ".yaml", ".yml", ".toml",
    ".md", ".rst", ".txt", ".cfg", ".ini", ".env", ".sh", ".bash",
    ".zsh", ".fish", ".rb", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".swift", ".kt", ".scala", ".php", ".sql",
    ".graphql", ".proto", ".dockerfile", ".makefile", ".cmake",
    ".gitignore", ".gitattributes", ".editorconfig", ".prettierrc",
    ".eslintrc", ".babelrc", ".vimrc", ".tmux", ".lua", ".r",
    ".ipynb", ".csv", ".log", ".diff", ".patch",
}

# Directories to skip when indexing
SKIP_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules", ".venv",
    "venv", "env", ".env", ".tox", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", ".idea", ".vscode", ".vs", "build", "dist",
    "target", "out", ".next", ".nuxt", ".output", "coverage",
    ".eggs", "*.egg-info", ".cache", ".DS_Store",
}


# ── Database helpers ────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    """Return a connection to the index database, creating it if needed."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn = sqlite3.connect(INDEX_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            name        TEXT PRIMARY KEY,
            path        TEXT NOT NULL,
            alias       TEXT,
            indexed_at  TEXT NOT NULL,
            file_count  INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT NOT NULL,
            rel_path    TEXT NOT NULL,
            size        INTEGER DEFAULT 0,
            content     TEXT,
            FOREIGN KEY (source) REFERENCES sources(name) ON DELETE CASCADE,
            UNIQUE(source, rel_path)
        )
    """)
    # Index for fast search and tree queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_files_source ON files(source)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_files_rel_path ON files(rel_path)
    """)
    conn.commit()


def _is_text_file(filepath: str) -> bool:
    """Check if a file should be indexed based on extension or name."""
    _, ext = os.path.splitext(filepath)
    if ext.lower() in TEXT_EXTENSIONS:
        return True
    # Index files with no extension that are common config/readme files
    basename = os.path.basename(filepath).lower()
    if basename in (
        "readme", "license", "changelog", "makefile", "dockerfile",
        "justfile", "taskfile",
    ):
        return True
    return False


def _scan_folder(folder_path: str) -> dict:
    """Walk *folder_path* and return a dict of relative_path -> content."""
    files = {}
    folder_path = os.path.abspath(folder_path)

    for root, dirs, filenames in os.walk(folder_path):
        # Prune skip directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in sorted(filenames):
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, folder_path)

            if not _is_text_file(full_path):
                continue

            try:
                size = os.path.getsize(full_path)
                if size > MAX_FILE_SIZE:
                    # Record the file but don't store content
                    files[rel_path] = {
                        "size": size,
                        "content": None,
                        "truncated": True,
                    }
                    continue

                with open(full_path, "r", errors="replace") as f:
                    content = f.read()
                files[rel_path] = {
                    "size": size,
                    "content": content,
                    "truncated": False,
                }
            except (IOError, OSError):
                continue

    return files


def _get_source_name(folder_path: str, alias: str = None) -> str:
    """Derive the source name: alias if given, else folder basename."""
    if alias:
        return alias
    return os.path.basename(os.path.abspath(folder_path))


# ── Core operations ─────────────────────────────────────────────────

def index_folder(folder_path: str, alias: str = None) -> str:
    """Index a folder and store it in the index database."""
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        return f"Error: '{folder_path}' is not a directory."

    source_name = _get_source_name(folder_path, alias)
    conn = _get_db()

    # Check if source already exists
    row = conn.execute("SELECT name FROM sources WHERE name = ?", (source_name,)).fetchone()
    if row:
        conn.close()
        return (f"Source '{source_name}' already exists. "
                f"Use 'refresh' to re-index or 'delete' first.")

    files = _scan_folder(folder_path)
    now = datetime.now(timezone.utc).isoformat()

    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO sources (name, path, alias, indexed_at, file_count) VALUES (?, ?, ?, ?, ?)",
        (source_name, folder_path, alias, now, len(files)),
    )

    # Bulk insert files
    insert_many = """
        INSERT OR REPLACE INTO files (source, rel_path, size, content)
        VALUES (?, ?, ?, ?)
    """
    file_data = [
        (source_name, rel_path, info["size"], info["content"])
        for rel_path, info in files.items()
    ]
    conn.executemany(insert_many, file_data)
    conn.commit()
    conn.close()

    total_size = sum(f.get("size", 0) for f in files.values())
    size_str = f"{total_size / 1024:.1f} KB" if total_size < 1024 * 1024 else f"{total_size / (1024*1024):.1f} MB"
    return (f"Indexed '{source_name}' ({folder_path}).\n"
            f"  {len(files)} files, {size_str} total.")


def delete_source(source_name: str) -> str:
    """Delete an indexed source."""
    conn = _get_db()
    row = conn.execute("SELECT name FROM sources WHERE name = ?", (source_name,)).fetchone()
    if not row:
        conn.close()
        return f"Error: Source '{source_name}' not found."

    conn.execute("DELETE FROM files WHERE source = ?", (source_name,))
    conn.execute("DELETE FROM sources WHERE name = ?", (source_name,))
    conn.commit()
    conn.close()
    return f"Deleted source '{source_name}'."


def rename_source(old_name: str, new_name: str) -> str:
    """Rename an indexed source."""
    conn = _get_db()
    row = conn.execute("SELECT name FROM sources WHERE name = ?", (old_name,)).fetchone()
    if not row:
        conn.close()
        return f"Error: Source '{old_name}' not found."

    existing = conn.execute("SELECT name FROM sources WHERE name = ?", (new_name,)).fetchone()
    if existing:
        conn.close()
        return f"Error: Source '{new_name}' already exists."

    conn.execute("UPDATE sources SET name = ? WHERE name = ?", (new_name, old_name))
    conn.execute("UPDATE files SET source = ? WHERE source = ?", (new_name, old_name))
    conn.commit()
    conn.close()
    return f"Renamed '{old_name}' to '{new_name}'."


def refresh_source(source_name: str) -> str:
    """Re-index a source (delete + re-index)."""
    conn = _get_db()
    row = conn.execute("SELECT path, alias FROM sources WHERE name = ?", (source_name,)).fetchone()
    if not row:
        conn.close()
        return f"Error: Source '{source_name}' not found."

    folder_path = row["path"]
    alias = row["alias"]

    # Remove old files and source entry
    conn.execute("DELETE FROM files WHERE source = ?", (source_name,))
    conn.execute("DELETE FROM sources WHERE name = ?", (source_name,))
    conn.commit()
    conn.close()

    # Re-index
    files = _scan_folder(folder_path)
    conn = _get_db()
    now = datetime.now(timezone.utc).isoformat()

    conn.execute("BEGIN")
    conn.execute(
        "INSERT INTO sources (name, path, alias, indexed_at, file_count) VALUES (?, ?, ?, ?, ?)",
        (source_name, folder_path, alias, now, len(files)),
    )
    file_data = [
        (source_name, rel_path, info["size"], info["content"])
        for rel_path, info in files.items()
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO files (source, rel_path, size, content) VALUES (?, ?, ?, ?)",
        file_data,
    )
    conn.commit()
    conn.close()

    return (f"Refreshed '{source_name}' ({folder_path}).\n"
            f"  {len(files)} files indexed.")


def list_sources() -> str:
    """List all indexed sources with summary info."""
    conn = _get_db()
    rows = conn.execute(
        "SELECT name, path, alias, file_count, indexed_at FROM sources ORDER BY name"
    ).fetchall()
    conn.close()

    if not rows:
        return "No indexed sources. Use 'docs index <folder>' to add one."

    lines = ["Indexed sources:"]
    for row in rows:
        alias_str = f" (alias: {row['alias']})" if row["alias"] else ""
        indexed_at = row["indexed_at"][:19].replace("T", " ")
        lines.append(f"  {row['name']}{alias_str}")
        lines.append(f"    path: {row['path']}")
        lines.append(f"    files: {row['file_count']}, indexed: {indexed_at}")
        lines.append("")
    return "\n".join(lines)


def search_sources(query: str) -> str:
    """Fuzzy search across all indexed sources.

    Searches both file paths and file contents.
    Returns matching files with enough context to identify them.
    """
    conn = _get_db()

    # Check if any sources exist
    count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    if count == 0:
        conn.close()
        return "No indexed sources to search."

    tokens = query.lower().split()

    # Fetch all files with content
    rows = conn.execute(
        "SELECT f.source, f.rel_path, f.size, f.content FROM files f"
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        rel_path = row["rel_path"]
        content = row["content"]

        path_score = _match_tokens(rel_path.lower(), tokens)
        content_score = 0
        content_line = None

        if content:
            content_lower = content.lower()
            content_score = _match_tokens(content_lower, tokens)
            # Find the first line that matches any token
            for line in content.split("\n"):
                if any(t in line.lower() for t in tokens):
                    content_line = line.strip()
                    break

        total_score = path_score + content_score
        if total_score > 0:
            results.append({
                "source": row["source"],
                "path": rel_path,
                "score": total_score,
                "size": row["size"],
                "content_line": content_line,
                "has_content": content is not None,
            })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    if not results:
        return f"No matches found for: {query}"

    lines = [f"Search results for: {query}\n"]
    for i, r in enumerate(results[:30], 1):  # Limit to 30 results
        source = r["source"]
        path = r["path"]
        size = r["size"]
        size_str = f"{size / 1024:.1f} KB" if size >= 1024 else f"{size} B"

        lines.append(f"  {i}. [{source}] {path}")
        lines.append(f"     size: {size_str}, score: {r['score']}")
        if r["content_line"] and r["has_content"]:
            cl = r["content_line"][:200]
            if len(r["content_line"]) > 200:
                cl += "..."
            lines.append(f"     → {cl}")
        elif not r["has_content"]:
            lines.append(f"     (content not indexed — file too large)")
        lines.append("")

    if len(results) > 30:
        lines.append(f"  ... and {len(results) - 30} more results")

    return "\n".join(lines)


def tree_source(source_name: str, scope: str = None, depth: int = None) -> str:
    """View the directory tree of an indexed source.

    *scope* limits the tree to a subdirectory prefix.
    *depth* limits the nesting level shown.
    """
    conn = _get_db()
    row = conn.execute("SELECT name FROM sources WHERE name = ?", (source_name,)).fetchone()
    if not row:
        available = [r["name"] for r in conn.execute("SELECT name FROM sources ORDER BY name").fetchall()]
        conn.close()
        return f"Error: Source '{source_name}' not found.\nAvailable: {', '.join(available)}"

    # Fetch all relative paths for this source
    rows = conn.execute(
        "SELECT rel_path FROM files WHERE source = ? ORDER BY rel_path", (source_name,)
    ).fetchall()
    conn.close()

    if not rows:
        return f"Source '{source_name}' has no indexed files."

    # Build a dict of rel_path -> True (content not needed for tree)
    files = {r["rel_path"]: True for r in rows}

    # Build a directory tree from file paths
    tree = _build_tree(files, scope, depth)

    lines = [f"Tree of '{source_name}'"]
    if scope:
        lines.append(f"  (scoped to: {scope})")
    if depth is not None:
        lines.append(f"  (depth: {depth})")
    lines.append("")
    lines.append(tree)
    lines.append("")
    return "\n".join(lines)


def view_document(source_name: str, file_path: str) -> str:
    """View the full content of an indexed document.

    *source_name* is the indexed source.
    *file_path* is the relative path within that source.
    """
    conn = _get_db()
    row = conn.execute("SELECT name, path FROM sources WHERE name = ?", (source_name,)).fetchone()
    if not row:
        available = [r["name"] for r in conn.execute("SELECT name FROM sources ORDER BY name").fetchall()]
        conn.close()
        return f"Error: Source '{source_name}' not found.\nAvailable: {', '.join(available)}"

    source_path = row["path"]

    # Try exact match first
    file_row = conn.execute(
        "SELECT rel_path, size, content FROM files WHERE source = ? AND rel_path = ?",
        (source_name, file_path),
    ).fetchone()

    if not file_row:
        # Try partial match
        matches = conn.execute(
            "SELECT rel_path FROM files WHERE source = ? AND (rel_path LIKE ? OR rel_path LIKE ?)",
            (source_name, f"%{file_path}%", f"%/{file_path}"),
        ).fetchall()
        conn.close()

        match_paths = [r["rel_path"] for r in matches]
        if len(match_paths) == 1:
            file_path = match_paths[0]
            # Re-fetch content
            conn = _get_db()
            file_row = conn.execute(
                "SELECT rel_path, size, content FROM files WHERE source = ? AND rel_path = ?",
                (source_name, file_path),
            ).fetchone()
        elif match_paths:
            conn.close()
            return (f"Multiple matches for '{file_path}':\n"
                    + "\n".join(f"  - {m}" for m in match_paths[:10]))
        else:
            conn.close()
            return f"Error: File '{file_path}' not found in source '{source_name}'."

    content = file_row["content"]
    size = file_row["size"]
    conn.close()

    if content is None:
        return (f"[{source_name}] {file_path}\n"
                f"(Content not indexed — file too large: "
                f"{size} bytes)\n"
                f"To view: read_file {source_path}/{file_path}")

    return f"[{source_name}] {file_path}\n\n{content}"


# ── Helper functions ────────────────────────────────────────────────

def _match_tokens(text: str, tokens: list) -> int:
    """Score how well *tokens* match *text*. Returns count of matched tokens."""
    return sum(1 for t in tokens if t in text)


def _build_tree(files: dict, scope: str = None, depth: int = None) -> str:
    """Build a visual tree string from a dict of relative file paths."""
    # Collect all paths, optionally scoped
    paths = sorted(files.keys())
    if scope:
        scope = scope.rstrip("/")
        # Match exact directory prefix: "agents" should match "agents/foo.py"
        # but NOT "agents.egg-info/foo.txt"
        scope_prefix = scope + "/"
        paths = [p for p in paths if p.startswith(scope_prefix) or p == scope]

    # Build a nested dict structure
    root = {}
    for path in paths:
        parts = path.split("/")
        if scope:
            # Remove scope prefix from parts
            scope_parts = scope.split("/")
            if len(parts) > len(scope_parts):
                parts = parts[len(scope_parts):]
            elif parts[0] == scope:
                parts = parts[1:]
        if depth is not None:
            parts = parts[:depth]

        node = root
        for part in parts:
            if part not in node:
                node[part] = {}
            node = node[part]

    # Render tree
    lines = []
    if scope:
        _render_tree_node(scope, root, "", True, lines, depth, 0)
    else:
        sorted_items = sorted(root.items())
        for i, (name, children) in enumerate(sorted_items):
            is_last = i == len(sorted_items) - 1
            _render_tree_node(name, children, "", is_last, lines, depth, 0)

    return "\n".join(lines)


def _render_tree_node(name: str, children: dict, prefix: str,
                      is_last: bool, lines: list,
                      max_depth: int = None, current_depth: int = 0):
    """Recursively render a tree node."""
    connector = "└── " if is_last else "├── "
    lines.append(f"{prefix}{connector}{name}")

    if max_depth is not None and current_depth + 1 >= max_depth:
        return

    new_prefix = prefix + ("    " if is_last else "│   ")
    sorted_children = sorted(children.items())
    for i, (child_name, grand_children) in enumerate(sorted_children):
        child_is_last = i == len(sorted_children) - 1
        _render_tree_node(
            child_name, grand_children, new_prefix,
            child_is_last, lines, max_depth, current_depth + 1,
        )


# ── Agent harness entry point ───────────────────────────────────────

def docs(*args):
    """Docs tool for the agent harness.

    Subcommands:
        list              — list all indexed sources
        search <query>    — fuzzy search across all sources
        tree <source> [scope] [depth] — view tree of a source
        view <source> <file> — view a document
    """
    if not args:
        return list_sources()

    cmd = args[0].lower()

    if cmd == "list":
        return list_sources()

    elif cmd == "search":
        if len(args) < 2:
            return "Error: search requires a query string."
        # Join remaining args as the query (preserves multi-word queries)
        query = " ".join(args[1:])
        return search_sources(query)

    elif cmd == "tree":
        if len(args) < 2:
            return "Error: tree requires a source name."
        source_name = args[1]
        scope = None
        depth = None
        if len(args) >= 3:
            scope = args[2]
        if len(args) >= 4:
            try:
                depth = int(args[3])
            except ValueError:
                return f"Error: depth must be an integer, got '{args[3]}'."
        return tree_source(source_name, scope=scope, depth=depth)

    elif cmd == "view":
        if len(args) < 3:
            return "Error: view requires a source name and file path."
        source_name = args[1]
        file_path = args[2]
        return view_document(source_name, file_path)

    else:
        return (f"Unknown docs subcommand: '{cmd}'\n"
                f"Available: list, search, tree, view")


# ── CLI entry point ─────────────────────────────────────────────────

def main():
    """CLI entry point for managing document indexes."""
    parser = argparse.ArgumentParser(
        prog="docs",
        description="Index and search local document/code folders.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # index
    idx_parser = subparsers.add_parser("index", help="Index a folder")
    idx_parser.add_argument("folder", help="Path to the folder to index")
    idx_parser.add_argument("-a", "--alias", help="Alias for the source")

    # tree
    tree_parser = subparsers.add_parser("tree", help="View indexed folder tree")
    tree_parser.add_argument("source", help="Source name")
    tree_parser.add_argument("scope", nargs="?", default=None, help="Scope to subdirectory")
    tree_parser.add_argument("--depth", "-d", type=int, default=None, help="Max depth")

    # delete
    del_parser = subparsers.add_parser("delete", help="Delete an index")
    del_parser.add_argument("source", help="Source name to delete")

    # rename
    ren_parser = subparsers.add_parser("rename", help="Rename an index")
    ren_parser.add_argument("old_name", help="Current source name")
    ren_parser.add_argument("new_name", help="New source name")

    # refresh
    ref_parser = subparsers.add_parser("refresh", help="Refresh an index")
    ref_parser.add_argument("source", help="Source name to refresh")

    # list
    subparsers.add_parser("list", help="List all indexed sources")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "index":
        result = index_folder(args.folder, alias=args.alias)
    elif args.command == "tree":
        result = tree_source(args.source, scope=args.scope, depth=args.depth)
    elif args.command == "delete":
        result = delete_source(args.source)
    elif args.command == "rename":
        result = rename_source(args.old_name, args.new_name)
    elif args.command == "refresh":
        result = refresh_source(args.source)
    elif args.command == "list":
        result = list_sources()
    else:
        result = f"Unknown command: {args.command}"

    print(result)


if __name__ == "__main__":
    main()
