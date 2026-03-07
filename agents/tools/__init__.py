"""
tools — command parsing and execution toolkit for the agent system.

This subpackage provides the tooling layer: parsing LLM output for
structured commands, executing them against the real world (filesystem,
shell, images, web browser), and returning results as plain strings.

Public API
----------
- ``process_content(text)`` — parse and execute commands from LLM output
- ``filter_content(text)`` — trim output when the LLM queues multiple read commands
- ``terminate_process()`` — terminate any running subprocess
- ``get_default_shell()`` — return the current user's default shell
- ``register_llm(fn)`` — register an LLM backend for the summarize tool
"""

from .parser import process_content, filter_content, terminate_process
from .functions import get_default_shell, register_pool
from .summarize import register_llm
