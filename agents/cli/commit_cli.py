"""
CLI entry point for ``agent-commit``.

Commits files using the agent's git author identity.  The commit
message is auto-generated from the list of changed files.

Usage
-----
    agent-commit --tracked              Commit tracked modified files
    agent-commit --all                  Commit all changed files (including untracked)
    agent-commit file1.py file2.py      Commit specific files
"""

import argparse
import os
import platform
import sys

from ..git_utils import (
    is_git_repo,
    get_all_changed_files,
    get_staged_files,
    get_tracked_modified_files,
    git_add_and_commit,
)


def _get_agent_author():
    """Return (author_name, author_email) matching the agent's auto-commit identity."""
    model = os.environ.get("LOCAL_MODEL", os.environ.get("AGENT_MODEL", "agent"))
    hostname = platform.node()
    return model, f"agent@{hostname}"


def _auto_message(files: list[str]) -> str:
    """Generate a concise commit message from the list of changed files.

    Uses the basename of each file.  If there's a single file the message
    is ``"update <file>"``; for multiple files it lists them all.
    """
    names = [os.path.basename(f) for f in files]
    if len(names) == 1:
        return f"update {names[0]}"
    return f"update {', '.join(names)}"


def main():
    """Parse arguments and perform the commit."""
    parser = argparse.ArgumentParser(
        prog="agent-commit",
        description="Commit files using the agent's git author identity.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--tracked",
        action="store_true",
        help="Commit all tracked files with changes (modified, deleted, renamed).",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Commit all changed files including untracked ones.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific files to commit (used when --tracked and --all are omitted).",
    )

    args = parser.parse_args()

    if not is_git_repo("."):
        print("Error: not a git repository.", file=sys.stderr)
        sys.exit(1)

    author_name, author_email = _get_agent_author()

    if args.tracked:
        files = sorted(set(get_tracked_modified_files(".")) | set(get_staged_files(".")))
        if not files:
            print("No tracked files with changes to commit.")
            sys.exit(0)
    elif args.all:
        files = get_all_changed_files(".")
        if not files:
            print("No files to commit.")
            sys.exit(0)
    else:
        if not args.files:
            parser.error("Provide --tracked, --all, or specific file paths.")
        files = args.files

    message = _auto_message(files)

    ok, err = git_add_and_commit(
        message,
        author_name=author_name,
        author_email=author_email,
        files=files,
    )

    if ok:
        print(f"Committed {len(files)} file(s) as {author_name} <{author_email}>:")
        for f in files:
            print(f"  {f}")
        print(f'  "{message}"')
    else:
        print(f"Commit failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
