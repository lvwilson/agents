#!/usr/bin/env python3
"""
Utility helpers shared across the agent system.

The ``ClaudeClient`` that used to live here has been replaced by the
backend classes in ``backends/``.  Only format-conversion helpers remain.
"""


def convert_string_to_dict(string):
    """Convert a plain string into the internal content-block format.

    Returns a list containing a single text block::

        [{"type": "text", "text": string}]

    Cache-control annotations are *not* added here — they are an
    Anthropic-specific concern handled by the Anthropic backend.
    """
    return [{"type": "text", "text": string}]
