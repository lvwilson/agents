"""
File and folder summarization using an LLM backend.

This module provides summarization capabilities that can be used as a tool
within the agents system. The LLM backend is injected at runtime via
``register_llm`` to avoid circular imports between the tools and agents layers.

Usage
-----
::

    from agents.tools.summarize import register_llm, summarize_file, summarize_folder

    # Register an LLM callable (done once by the agents layer)
    register_llm(my_generate_fn)

    # Summarize a single file
    result = summarize_file("path/to/file.py")

    # Summarize a folder
    result = summarize_folder("path/to/dir", filter_pattern="*.py", recursive=True)
"""

from __future__ import annotations

import fnmatch
import os
import re
from typing import Callable, Optional

# Module-level LLM callable.  Signature:
#   generate(system_prompt: str, user_message: str) -> str
_llm_generate: Optional[Callable[[str, str], str]] = None

SYSTEM_PROMPT = (
    "You are a concise code/file summarizer. Given file content, produce a brief summary covering: "
    "purpose, key components (classes/functions/endpoints), dependencies, and notable patterns. "
    "Be factual and terse — bullet points preferred. Skip boilerplate details."
)


def register_llm(generate_fn: Callable[[str, str], str]) -> None:
    """Register the LLM generation function.

    Parameters
    ----------
    generate_fn : callable
        A function with signature ``(system_prompt: str, user_message: str) -> str``
        that sends a single-turn request to an LLM and returns the response text.
    """
    global _llm_generate
    _llm_generate = generate_fn


def _ensure_llm() -> Callable[[str, str], str]:
    """Return the registered LLM callable or raise a clear error."""
    if _llm_generate is None:
        raise RuntimeError(
            "No LLM backend registered for summarization. "
            "Call agents.tools.summarize.register_llm() first — this is normally "
            "done automatically by the agents layer during Agent.__init__."
        )
    return _llm_generate


def summarize_text(
    content: str,
    filename: str = "",
    instruction: str = "",
) -> str:
    """Summarize a piece of text using the registered LLM.

    Parameters
    ----------
    content : str
        The text to summarize.
    filename : str
        Optional filename for context (helps the LLM understand file type).
    instruction : str
        Optional additional instruction to fine-tune the summary.

    Returns
    -------
    str
        The LLM-generated summary.
    """
    generate = _ensure_llm()

    system = SYSTEM_PROMPT
    if instruction:
        system += f"\n\nAdditional instruction: {instruction}"

    header = f"File: {filename}\n\n" if filename else ""
    user_message = f"{header}{content}"

    # Truncate very large files to avoid blowing context limits.
    # 200k chars ≈ ~50-70k tokens — safe for most models.
    max_chars = 200_000
    if len(user_message) > max_chars:
        user_message = user_message[:max_chars] + "\n\n[... truncated ...]"

    return generate(system, user_message)


_MTIME_HEADER_RE = re.compile(r"^<!-- source_mtime: ([\d.]+) -->$")


def _read_cached_summary(summary_path: str) -> tuple[Optional[float], str]:
    """Read a cached ``.summary`` file and extract the stored source mtime.

    Returns
    -------
    tuple[Optional[float], str]
        ``(stored_mtime, summary_text)`` — *stored_mtime* is ``None`` if the
        header is missing or unparseable.  *summary_text* is the summary body
        without the metadata header.
    """
    try:
        with open(summary_path, "r") as f:
            first_line = f.readline()
            match = _MTIME_HEADER_RE.match(first_line.strip())
            stored_mtime = float(match.group(1)) if match else None
            # Skip the blank separator line after the header.
            if match:
                f.readline()
            body = f.read()
            return stored_mtime, body
    except (OSError, ValueError):
        return None, ""


def _is_summary_current(file_path: str, summary_path: str) -> tuple[bool, str]:
    """Check whether an existing summary is still up-to-date.

    Compares the source file's current mtime against the mtime stored in the
    summary file's metadata header.

    Returns
    -------
    tuple[bool, str]
        ``(is_current, cached_summary_text)``
    """
    if not os.path.exists(summary_path):
        return False, ""

    try:
        source_mtime = os.path.getmtime(file_path)
    except OSError:
        return False, ""

    stored_mtime, cached_body = _read_cached_summary(summary_path)
    if stored_mtime is not None and stored_mtime == source_mtime and cached_body:
        return True, cached_body

    return False, ""


def summarize_file(file_path: str, instruction: str = "") -> str:
    """Read and summarize a single file.

    Writes the summary to a companion file named ``<filename>.<ext>.summary``
    in the same directory as the input file.  A metadata header records the
    source file's modification time so that unchanged files can be skipped on
    subsequent runs.

    Parameters
    ----------
    file_path : str
        Path to the file to summarize.
    instruction : str
        Optional instruction to fine-tune the summary.

    Returns
    -------
    str
        The summary, or an error message.
    """
    summary_path = file_path + ".summary"

    # Fast path: reuse the cached summary if the source file hasn't changed.
    is_current, cached_body = _is_summary_current(file_path, summary_path)
    if is_current:
        return f"## {file_path}\n{cached_body}\n\n(Cached — file unchanged since last summary)"

    try:
        with open(file_path, "r") as f:
            content = f.read()
    except Exception as e:
        return f"Error reading {file_path}: {e}"

    if not content.strip():
        return f"{file_path}: (empty file)"

    try:
        summary = summarize_text(content, filename=file_path, instruction=instruction)
    except Exception as e:
        return f"Error summarizing {file_path}: {e}"

    # Record the source file's mtime so we can skip re-summarizing next time.
    try:
        source_mtime = os.path.getmtime(file_path)
    except OSError:
        source_mtime = 0.0

    # Write the summary with a metadata header to the companion file.
    try:
        with open(summary_path, "w") as f:
            f.write(f"<!-- source_mtime: {source_mtime} -->\n\n")
            f.write(summary)
    except Exception as e:
        return f"## {file_path}\n{summary}\n\n(Warning: failed to write summary file {summary_path}: {e})"

    return f"## {file_path}\n{summary}\n\n(Summary written to {summary_path})"


def summarize_folder(
    folder_path: str,
    filter_pattern: str = "*",
    recursive: bool = False,
    instruction: str = "",
) -> str:
    """Summarize all matching files in a folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder.
    filter_pattern : str
        Glob pattern to filter files (e.g. ``"*.py"``). Default: all files.
    recursive : bool
        If True, recurse into subdirectories.
    instruction : str
        Optional instruction to fine-tune each file's summary.

    Returns
    -------
    str
        Concatenated summaries of all matching files.
    """
    if not os.path.isdir(folder_path):
        return f"Error: {folder_path} is not a directory."

    # Collect matching files
    matched_files: list[str] = []

    if recursive:
        for root, _dirs, files in os.walk(folder_path):
            for fname in sorted(files):
                if fname.endswith(".summary"):
                    continue
                if fnmatch.fnmatch(fname, filter_pattern):
                    matched_files.append(os.path.join(root, fname))
    else:
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".summary"):
                continue
            full_path = os.path.join(folder_path, fname)
            if os.path.isfile(full_path) and fnmatch.fnmatch(fname, filter_pattern):
                matched_files.append(full_path)

    if not matched_files:
        return f"No files matching '{filter_pattern}' found in {folder_path}."

    summaries: list[str] = []
    for fpath in matched_files:
        # Skip binary files
        try:
            with open(fpath, "r") as f:
                f.read(512)
        except (UnicodeDecodeError, PermissionError):
            summaries.append(f"## {fpath}\n(skipped — binary or unreadable)")
            continue

        summary = summarize_file(fpath, instruction=instruction)
        summaries.append(summary)

    header = (
        f"# Folder summary: {folder_path}\n"
        f"Filter: {filter_pattern} | Files: {len(matched_files)} | Recursive: {recursive}\n\n"
    )
    return header + "\n\n".join(summaries)
