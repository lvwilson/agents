# agents

Autonomous AI software engineering agents powered by LLMs. This is the **reasoning layer** of a two-layer system — it owns the conversation loop, the LLM client, and the decision of when to start and stop. It pairs with [llmide](https://github.com/lvwilson/llmide), the **tooling layer**, which handles command parsing and execution.

## Architecture

The core loop is deliberately minimal: **generate → parse → execute → feed back**.

1. The LLM produces text containing embedded `Command:` directives with optional backtick-delimited payloads.
2. Commands are parsed and executed by `llmide` against the real filesystem, shell, and image tools.
3. Results become the next user message in the conversation.
4. If no commands are found, the agent is finished.

There is no planner, no task graph, no state machine — the LLM's own reasoning, guided by the system prompt, *is* the control flow.

### Integration Surface

The boundary between `agents` and `llmide` is two functions:

- `process_content()` — parse and execute commands from LLM output
- `filter_content()` — trim output when the LLM queues multiple read commands

This is the entire integration surface. `llmide` knows nothing about Claude, conversation history, or budgets.

## Design Philosophy

- **Configuration Over Code** — Agent behavior is defined in YAML files, not Python. Each config specifies a model, a system prompt, and an over-budget warning. New agent archetypes are created by writing prose, not code.
- **The LLM as the Only Moving Part** — No hardcoded task decomposition, no retry logic, no verification beyond what the LLM chooses to do. The infrastructure faithfully executes commands and stays out of the way.
- **Context as Conversation** — All state lives in the message history. No external database, no structured memory. Context can be serialized to disk and restored for pause/resume.
- **Cost Awareness** — Token usage and dollar cost are tracked in real time. The agent is warned at 75% budget and terminated at 100%, making autonomous operation safe and bounded.

### Available Tools (via llmide)

| Category | Capabilities |
|---|---|
| **File I/O** | Read, write, append — with diffs reported back for self-verification |
| **AST Code Manipulation** | Address-based read/replace/insert/remove on Python syntax trees |
| **Text Code Manipulation** | Find-and-replace blocks, line-based cut/insert operations |
| **Shell Access** | Full pseudo-terminal command execution with output capture |
| **Image Handling** | Load, resize, encode images for vision LLMs; generate via external API |

## Installation

### Prerequisites

Clone and install llmide:

    git clone https://github.com/lvwilson/llmide
    cd llmide
    pip install -e .

### Setup

    pip install -r requirements.txt

PEP workaround if needed:

    pip install --break-system-packages -r requirements.txt

### API Keys

Add the following to your `.bashrc` (Linux):

    export CLAUDE_API_KEY='your_api_key'
