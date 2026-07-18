"""Git utility functions for the agent harness.

All git operations are performed silently by the harness — the agent
itself is never made aware of them.
"""
import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def _run_git(*args, check=False, cwd=None):
    """Run a git command and return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            text=True,
            check=check,
            cwd=cwd,
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except FileNotFoundError:
        return "", "git not found", 127


def is_git_repo(path: str = ".") -> bool:
    """Return True if *path* is inside a git repository."""
    _, _, rc = _run_git("rev-parse", "--git-dir", cwd=path)
    return rc == 0


def check_git_clean(path: str = ".") -> tuple[bool, str]:
    """Check whether the git working tree is clean.

    Returns
    -------
    (is_clean: bool, message: str)
        *is_clean* is True when there are no staged, unstaged, or
        untracked changes.  *message* is a human-readable description
        of the dirty state (empty when clean).
    """
    stdout, _, rc = _run_git("status", "--porcelain", "-b", cwd=path)
    if rc != 0:
        return True, ""

    # The first line starts with '## branch' — skip it for the diff
    lines = stdout.splitlines()
    branch_line = lines[0] if lines else ""
    status_lines = lines[1:] if len(lines) > 1 else []

    if not status_lines:
        return True, ""

    # Classify changes
    staged = []
    unstaged = []
    untracked = []
    for line in status_lines:
        if line.startswith("??"):
            untracked.append(line[3:].strip())
            continue
        x, y = line[0], line[1]
        fname = line[3:].strip()
        # A file can appear in both index (x) and worktree (y), e.g. "MM"
        if x in ("M", "A", "D", "R", "C"):
            staged.append(fname)
        if y in ("M", "A", "D", "R", "C"):
            unstaged.append(fname)

    parts: list[str] = []
    if staged:
        parts.append(f"staged: {', '.join(staged)}")
    if unstaged:
        parts.append(f"modified: {', '.join(unstaged)}")
    if untracked:
        parts.append(f"untracked: {', '.join(untracked)}")

    return False, "Git working tree is not clean (" + "; ".join(parts) + ")"


def get_diff_summary(path: str = ".") -> str:
    """Return a concise summary of all uncommitted changes.

    Includes staged, unstaged, and untracked files with their diffs.
    """
    stdout, _, rc = _run_git("diff", "--stat", cwd=path)
    if rc != 0:
        return ""

    # Also get the full diff
    diff_out, _, _ = _run_git("diff", cwd=path)

    # Get staged diff
    staged_diff, _, _ = _run_git("diff", "--cached", cwd=path)

    # Get untracked files
    status_out, _, _ = _run_git("status", "--porcelain", cwd=path)
    untracked = [
        line[3:].strip()
        for line in status_out.splitlines()
        if line.startswith("??")
    ]

    parts: list[str] = []
    if staged_diff:
        parts.append("=== Staged changes ===\n" + staged_diff)
    if diff_out:
        parts.append("=== Unstaged changes ===\n" + diff_out)
    if untracked:
        parts.append("=== Untracked files ===\n" + "\n".join(untracked))

    return "\n\n".join(parts) if parts else stdout


def git_add_and_commit(
    message: str,
    path: str = ".",
    author_name: str = "",
    author_email: str = "",
    files: list[str] | None = None,
) -> tuple[bool, str]:
    """Stage changes and commit with the given message.

    Args:
        message: The commit message.
        path: Working directory for git commands.
        author_name: Git author name (uses system default if empty).
        author_email: Git author email (uses system default if empty).
        files: If given, stage only these specific files.  If None,
               stage all changes (``git add -A``).

    Returns (success, error_message).
    """
    if files:
        # '--' guards against filenames that start with a dash
        _, stderr, rc = _run_git("add", "--", *files, cwd=path)
    else:
        _, stderr, rc = _run_git("add", "-A", cwd=path)
    if rc != 0:
        return False, f"git add failed: {stderr}"

    # Build commit args.  When a file list is given, commit with a
    # pathspec so unrelated pre-staged changes are not swept in.
    commit_args = ["commit", "-m", message]
    if author_name:
        commit_args += ["--author", f"{author_name} <{author_email}>"]
    if files:
        commit_args += ["--"] + list(files)

    # Commit
    _, stderr, rc = _run_git(*commit_args, cwd=path)
    if rc != 0:
        # Might be "nothing to commit" — not an error
        if "nothing to commit" in stderr.lower():
            return True, ""
        return False, f"git commit failed: {stderr}"

    return True, ""


def get_tracked_modified_files(path: str = ".") -> list[str]:
    """Return a list of tracked files that have been modified (not staged).

    Includes modified, deleted, and renamed tracked files.
    Excludes untracked files.
    """
    stdout, _, rc = _run_git("diff", "--name-only", cwd=path)
    if rc != 0:
        return []
    return [f for f in stdout.splitlines() if f]


def get_staged_files(path: str = ".") -> list[str]:
    """Return a list of staged files."""
    stdout, _, rc = _run_git("diff", "--cached", "--name-only", cwd=path)
    if rc != 0:
        return []
    return [f for f in stdout.splitlines() if f]


def get_all_changed_files(path: str = ".") -> list[str]:
    """Return all changed files: tracked (modified/deleted) + untracked + staged."""
    tracked_modified = set(get_tracked_modified_files(path))
    staged = set(get_staged_files(path))

    # Also get untracked files
    status_out, _, _ = _run_git("status", "--porcelain", cwd=path)
    untracked = {
        line[3:].strip()
        for line in status_out.splitlines()
        if line.startswith("??")
    }

    return sorted(tracked_modified | staged | untracked)
