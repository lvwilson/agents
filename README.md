# agents

Autonomous AI software engineering agents powered by LLMs. The system is organized in two layers within a single package:

- **Reasoning layer** (`agents.agents`) — owns the conversation loop, the LLM client, and the decision of when to start and stop.
- **Tooling layer** (`agents.tools`) — handles command parsing and execution against the filesystem, shell, images, and web browser.

## Architecture

The core loop is deliberately minimal: **generate → parse → execute → feed back**.

1. The LLM produces text containing embedded `Command:` directives with optional backtick-delimited payloads.
2. Commands are parsed and executed by `agents.tools` against the real filesystem, shell, and image tools.
3. Results become the next user message in the conversation.
4. If no commands are found, the agent is finished.

There is no planner, no task graph, no state machine — the LLM's own reasoning, guided by the system prompt, *is* the control flow.

### Internal Boundary

The boundary between the reasoning and tooling layers is three functions:

- `process_content()` — parse and execute commands from LLM output
- `filter_content()` — trim output when the LLM queues multiple read commands
- `terminate_process()` — terminate any running subprocess

The tooling layer knows nothing about Claude, conversation history, or budgets.

## Design Philosophy

- **Model Agnostic** — Supports multiple LLM providers (Anthropic, OpenAI, Gemini) and local models out of the box. Providers are lazy-loaded, so you only need the SDKs for the models you actually use.
- **Configuration Over Code** — Agent behavior is defined in YAML files, not Python. Each config specifies a provider, model, system prompt, and an over-budget warning. New agent archetypes are created by writing prose, not code.
- **The LLM as the Only Moving Part** — No hardcoded task decomposition, no retry logic, no verification beyond what the LLM chooses to do. The infrastructure faithfully executes commands and stays out of the way.
- **Context as Conversation** — All state lives in the message history. No external database, no structured memory. Sessions are persisted as JSON files and can be resumed across invocations.
- **Cost Awareness** — Token usage and dollar cost are tracked in real time, including prompt caching discounts. The agent is warned at 75% budget and terminated at 100%, making autonomous operation safe and bounded.

### Available Tools

| Category | Capabilities |
|---|---|
| **File I/O** | Read, write, append — with diffs reported back for self-verification |
| **AST Code Manipulation** | Address-based read/replace/insert/remove on Python syntax trees |
| **Text Code Manipulation** | Find-and-replace blocks, line-based cut/insert operations |
| **Shell Access** | Full pseudo-terminal command execution with output capture |
| **Image Handling** | Load, resize, encode images for vision LLMs; generate via external API |
| **Web Browser** | Playwright-powered headless browser for navigation, reading, clicking, screenshots |
| **Summarization** | LLM-powered file and folder summarization with caching |

## Installation

### Setup

Install the package with all dependencies:

    pip install -e .

For optional provider support:

    pip install -e '.[openai]'     # OpenAI models
    pip install -e '.[gemini]'     # Google Gemini models
    pip install -e '.[browser]'    # Playwright web browser
    pip install -e '.[all]'        # Everything

### API Keys

Add the relevant keys for the providers you intend to use to your `.bashrc` (Linux):

    export CLAUDE_API_KEY='your_anthropic_api_key'
    export OPENAI_API_KEY='your_openai_api_key'
    export GEMINI_API_KEY='your_gemini_api_key'

## Usage

Agents are configured via YAML files (e.g., `basic_agent.yaml`). You can override the provider and model using environment variables:

    AGENT_MODEL_PROVIDER=openai AGENT_MODEL=gpt-4o agents "Write a python script to calculate fibonacci numbers"

### Session Management

Every invocation is assigned a short session ID (e.g. `a7x2`). The full conversation context, system prompt, and token/cost accounting are saved as a JSON file under `/tmp/agents-<username>/` when the agent finishes.

To resume the most recent session for the current working directory:

    agents -r "Continue where you left off"

To resume a specific session by ID:

    agents -r -s a7x2 "Fix the remaining test failures"

To start a new session with a chosen ID:

    agents -s mysession "Refactor the parser module"

**How it works:**

- Sessions are stored at `/tmp/agents-<username>/<session_id>.json` with `0600` permissions.
- An index file maps each working directory to its most recently used session, so `-r` works without specifying an ID.
- Sessions older than 7 days are automatically pruned on each save.
- `/tmp` is cleared on reboot, so sessions are inherently ephemeral. For long-lived persistence, copy the JSON file elsewhere.

When a session is restored, the original system prompt is reused verbatim so that provider-side prompt caches (e.g. Anthropic's cache) remain valid.

### Local Models

You can run against local OpenAI-compatible servers (like Ollama, vLLM, or llama.cpp) by using the `--local` flag and setting the `LOCAL_MODEL` environment variable. By default, it connects to `http://localhost:8000`.

    LOCAL_MODEL=llama3.1 agents --local "Explain quantum mechanics"

You can change the port using the `-p` or `--port` flag:

    LOCAL_MODEL=qwen2.5 agents --local -p 11434 "Write a haiku"
