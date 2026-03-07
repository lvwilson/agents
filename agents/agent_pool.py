"""
Sub-agent pool — create, register, and run sub-agents.

Sub-agents run as **subprocesses** using the existing ``agents`` CLI.
This gives each sub-agent a fully isolated context, proper tool support,
and the same system prompt as the parent.  The sub-agent communicates
results back via stdout (using the ``stdout`` tool), while its UI and
streaming output go to /dev/tty.

Cost is controlled by passing a budget flag (``-b``) to the subprocess.
The sub-agent inherits the parent's environment (API keys, model
settings, etc.) automatically.

The pool is session-scoped and injected into the tools layer via
``register_pool()``, following the same pattern as ``summarize.register_llm()``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Instruction appended to every sub-agent task so it knows to use stdout.
_STDOUT_INSTRUCTION = (
    "\n\nIMPORTANT: When you have finished your task, write your final result "
    "using the stdout tool. This is how your output is returned to the caller. "
    "Do not use a completion block — just write your findings to stdout and then stop."
)


@dataclass
class SubAgentConfig:
    """Configuration for a sub-agent."""
    name: str
    role_prompt: str
    description: str = ""


class AgentPool:
    """Session-scoped registry of sub-agent configurations.

    The pool holds agent configs (not live agents).  Each ``run()`` call
    spawns a subprocess running ``python -m agents`` that inherits the
    parent's environment.
    """

    def __init__(self):
        self._agents: dict[str, SubAgentConfig] = {}

    def create(
        self,
        name: str,
        role_prompt: str,
        description: str = "",
    ) -> str:
        """Register a sub-agent configuration.

        Args:
            name: Unique name for the sub-agent.
            role_prompt: Role / persona instructions prepended to the task.
            description: Short description of what this agent does.

        Returns:
            Status message.
        """
        if not name or not name.strip():
            return "Error: agent name cannot be empty."
        name = name.strip()

        action = "updated" if name in self._agents else "created"
        self._agents[name] = SubAgentConfig(
            name=name,
            role_prompt=role_prompt,
            description=description or f"Sub-agent: {name}",
        )
        return f"Agent '{name}' {action}. Use `run_agent {name}` to execute it."

    def list(self) -> str:
        """List all registered sub-agents."""
        if not self._agents:
            return "No sub-agents registered. Use create_agent to create one."
        lines = [f"Registered sub-agents ({len(self._agents)}):"]
        for name, config in self._agents.items():
            lines.append(f"  \u2022 {name}: {config.description}")
        return "\n".join(lines)

    def run(
        self,
        name: str,
        task: str,
        budget: float = 1.00,
        timeout: int = 300,
    ) -> str:
        """Run a sub-agent on a task and return its stdout output.

        Spawns ``python -m agents`` as a subprocess.  The role prompt
        and task are combined into the command argument.  The sub-agent
        writes its result to stdout via the ``stdout`` tool.

        Only stdout is read back.  Stderr is captured and discarded —
        the sub-agent is expected to handle its own errors.  UI and
        streaming output go to ``/dev/tty`` as usual.

        Args:
            name: Name of a registered sub-agent.
            task: The task to give the sub-agent.
            budget: Compute budget in dollars (default $1.00).
            timeout: Maximum wall-clock seconds (default 300).

        Returns:
            The sub-agent's stdout output, or an error message.
        """
        name = name.strip()
        if name not in self._agents:
            available = ", ".join(self._agents.keys()) if self._agents else "(none)"
            return f"Error: agent '{name}' not found. Available: {available}"

        config = self._agents[name]

        # Build the task string with role context and stdout instruction
        parts = []
        if config.role_prompt:
            parts.append(f"Role: {config.role_prompt}")
        parts.append(f"Task: {task}")
        parts.append(_STDOUT_INSTRUCTION)
        full_task = "\n\n".join(parts)

        # Spawn the sub-agent as a subprocess.
        # stdout is piped to read the result; stderr is captured and
        # discarded (the sub-agent handles its own errors).
        # TTY output from the sub-agent goes directly to /dev/tty.
        cmd = [
            sys.executable, "-m", "agents",
            full_task,
            "-b", str(budget),
            "-a", "sub_agent.yaml",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )
        except subprocess.TimeoutExpired:
            return f"Sub-agent '{name}' timed out after {timeout} seconds."
        except Exception as e:
            return f"Sub-agent '{name}' failed to start: {e}"

        stdout_output = result.stdout.strip()
        if stdout_output:
            return f"[Sub-agent '{name}' output]\n{stdout_output}"
        else:
            # No stdout output — the sub-agent may not have used the stdout tool
            return (
                f"[Sub-agent '{name}' finished with no stdout output]\n"
                f"Exit code: {result.returncode}\n"
                f"(The sub-agent may not have used the stdout tool to return results.)"
            )
