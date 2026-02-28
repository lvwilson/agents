#!/usr/bin/env python3
"""
Utility helpers shared across the agent system.

The ``ClaudeClient`` that used to live here has been replaced by the
backend classes in ``backends/``.  Only format-conversion helpers remain.
"""


def convert_string_to_dict(string, cache=False):
    result = []
    if cache:
        result.append({
            "type": "text",
            "text": string,
            "cache_control": {"type": "ephemeral"}
        })
    else:
        result.append({
            "type": "text",
            "text": string,
        })

    return result
