"""
CLI entry point for ``agent-memory``.

Provides commands to view and manage the hierarchical memory store.

Usage
-----
    agent-memory view       Show memory as the LLM will see it
    agent-memory raw        Show all raw (append-only) episodes
    agent-memory notes      Show current notes
"""

import argparse
import sys

from ..memory import format_memory_view, format_raw_view, get_notes


def main():
    """Parse arguments and display memory."""
    parser = argparse.ArgumentParser(
        prog="agent-memory",
        description="View and manage agent memory for the current folder.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("view", help="Show memory as the LLM will see it (episodes + notes)")
    sub.add_parser("raw", help="Show all raw (append-only) episodes")
    sub.add_parser("notes", help="Show current notes only")

    args = parser.parse_args()

    if args.command == "raw":
        print(format_raw_view())
    elif args.command == "notes":
        notes = get_notes()
        if notes:
            print(notes)
        else:
            print("(No notes stored.)")
    else:
        # Default: view
        print(format_memory_view())


if __name__ == "__main__":
    main()
