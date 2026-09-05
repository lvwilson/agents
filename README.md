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

- **Model Agnostic** — Supports multiple LLM providers (Anthropic, OpenAI, Cerebras, Gemini, DeepSeek, Kimi, MiniMax) and local models out of the box. Providers are lazy-loaded, so you only need the SDKs for the models you actually use.
- **Configuration Over Code** — Agent behavior is defined in YAML files, not Python. Each config specifies a provider, model, system prompt, and an over-budget warning. New agent archetypes are created by writing prose, not code.
- **The LLM as the Only Moving Part** — No hardcoded task decomposition, no retry logic, no verification beyond what the LLM chooses to do. The infrastructure faithfully executes commands and stays out of the way.
- **Context as Conversation** — All state lives in the message history. No external database, no structured memory. Sessions are persisted as JSON files and can be resumed across invocations.
- **Cost Awareness** — Token usage and dollar cost are tracked in real time, including prompt caching discounts. Each step's info line shows the completed step's output rate (tokens/second) as soon as it is measured, and the session closes with a metrics panel reporting the whole task's salient numbers — cost, steps, duration, peak context, total output tokens, overall output rate — plus an estimated cost per hour at the observed generation pace. The agent is warned at 80% of its budget, and when 100% is hit it is given one final turn that processes no commands — a wrap-up in which it records the work done so far and emits its completion block — before the session ends, keeping autonomous operation safe and bounded.

### Available Tools

| Category | Capabilities |
|---|---|
| **File I/O** | Read, write, append — with diffs reported back for self-verification |
| **AST Code Manipulation** | Address-based read/replace/insert/remove on Python syntax trees |
| **Text Code Manipulation** | Find-and-replace blocks, line-based cut/insert operations |
| **Shell Access** | Full pseudo-terminal command execution with output capture |
| **Image Handling** | Load, resize, encode images for vision LLMs; generate via external API |
| **Web Browser** | Playwright-powered headless browser for navigation, reading, clicking, screenshots — stealth-hardened and proxy-capable (see Web browsing & stealth below) |
| **Summarization** | LLM-powered file and folder summarization with caching |

## Installation

### Setup

Install the package with all dependencies:

    pip install -e .

For optional provider support:

    pip install -e '.[openai]'     # OpenAI models
    pip install -e '.[gemini]'     # Google Gemini models
    pip install -e '.[cerebras]'   # Cerebras Inference models
    pip install -e '.[browser]'    # Playwright web browser
    pip install -e '.[all]'        # Everything

### API Keys

Add the relevant keys for the providers you intend to use to your `.bashrc` (Linux):

    export CLAUDE_API_KEY='your_anthropic_api_key'
    export OPENAI_API_KEY='your_openai_api_key'
    export GEMINI_API_KEY='your_gemini_api_key'
    export CEREBRAS_API_KEY='your_cerebras_api_key'

Optional configuration for local/remote LLM servers:

    export LOCAL_MODEL='qwen3.8-27b'        # Model name for --local mode
    export LOCAL_LLM_PORT=11434          # Port for local API server (default: 8000)
    export LOCAL_LLM_HOST='192.168.1.50' # Hostname for remote LLM server (default: localhost)

## Usage

Agents are configured via YAML files (e.g., `basic_agent.yaml`). You can override the provider and model using environment variables:

    AGENT_MODEL_PROVIDER=openai AGENT_MODEL=gpt-4o agents "Write a python script to calculate fibonacci numbers"

### Backend Configuration (the `.agent` file)

The easiest way to pick a backend is a small **`.agent`** YAML file.  Put one in your
project directory (it is picked up from any subdirectory of that directory).  For a
global default applicable to every project, write it to
**`~/.agents/agent_config.yaml`** — the same state directory that holds sessions,
memory and the browser profile.  Only the keys you need are required:

```yaml
# .agent — project pin, e.g. use Cerebras' Qwen 3.8 27B
provider: cerebras
model: qwen-3.8-27b
```

```yaml
# ~/.agents/agent_config.yaml — any OpenAI-compatible server (vLLM, llama.cpp, Ollama, …)
provider: openai
model: qwen3.8-27b
base_url: http://localhost:8000/v1
```

```yaml
# .agent — pin a temperature too
provider: cerebras
model: gpt-oss-120b
temperature: 0.3
```

| Key | Meaning |
|---|---|
| `provider` | Backend: `anthropic`, `openai`, `cerebras`, `gemini`, `kimi`, `deepseek`, `minimax`. |
| `model` | Model name.  Omit to use the provider's default. |
| `base_url` | Custom endpoint (local / self-hosted / proxy). |
| `temperature` | Sampling temperature (per-backend default if omitted). |

**Resolution order** (highest wins, per key): `--provider` / `--model` flags → project
`.agent` → global `~/.agents/agent_config.yaml` → `AGENT_MODEL_PROVIDER` /
`AGENT_MODEL` / `AGENT_BASE_URL` env vars → the agent YAML → provider default.
API keys always come from the environment (`CLAUDE_API_KEY`, `OPENAI_API_KEY`,
`CEREBRAS_API_KEY`, …) — never from the files.

Note: the old global location `~/.agent` (a bare file in the home directory) is
deprecated.  On the first CLI run it is moved to `~/.agents/agent_config.yaml`;
until then it is still read, and if both files exist the new location wins and a
warning tells you to clean up the leftover.

### Web browsing & stealth

The web browser tool (`read_page`, `read_page_html`, `page_links`, `view_page`,
`browse_open`, `browse_read`, `browse_click`, `browse_type`, `browse_js`) runs a
headless Chromium/Chrome that is stealth-hardened (Plan:
`untracked/web_stealth_plan.md`): a real-Chrome user agent, consistent
locale/timezone/viewport fingerprints, a small vendored init script that fixes
the classic headless tells (`navigator.webdriver`, empty `plugins`, missing
`window.chrome`), and gentle jittered pacing between actions. All settings are
environment variables (tool-level config lives in the shell env, never in
`.agent`); bad values warn and fall back to safe defaults — they never crash
the agent.

| Variable | Default | Meaning |
|---|---|---|
| `WEB_PROXY` | – | Single proxy `scheme://[user:pass@]host:port` for the browser (`http`, `https`, `socks5`). Also feeds `web_search` (DDGS). `WEB_PROXY_FILE` wins if both are set. |
| `WEB_PROXY_FILE` | – | File with one proxy URL per line (`#` comments and blank lines ignored); round-robin per navigation. Mutually exclusive with a persistent profile (file wins). |
| `WEB_BROWSER_PROFILE` | off | `1`/`0` or a directory path. `1` enables a persistent browser profile at `~/.agents/browser_profile/` so cookies/storage survive across runs (repeat visits look like a returning visitor). |
| `WEB_CHANNEL` | `auto` | `auto` (drive the installed Google Chrome when present, fall back to bundled Chromium; an explicit `WEB_CHANNEL=chrome` falls back the same way on launch failure), `chrome` (installed Chrome), or `chromium` (force the bundled build). |
| `WEB_USER_AGENT` | auto-built | Full user-agent override. Auto-built = the real Chrome UA for this platform from a single version constant (never `HeadlessChrome`). |
| `WEB_LOCALE` | `en-US` | Context locale **and** the matching `Accept-Language` header (`en-US,en;q=0.9`). |
| `WEB_TIMEZONE` | `Etc/UTC` | Context `timezone_id`. Set it to match your proxy's geography — an IP in one city with a mismatched timezone is a red flag that no UA can hide. |
| `WEB_REQUEST_DELAY` | `0.5` | Mean seconds of jittered delay after navigation / before interactive actions (actual wait is `uniform(0.5×d, 1.5×d)`); `0` disables it. |
| `WEB_STEALTH` | `1` | `1` = apply the stealth init script and the `--disable-blink-features=AutomationControlled` launch arg; `0` = skip both (for debugging/diffing). |

Notes:

- `WEB_PROXY` also feeds `web_search` (DDGS); `DDGS_PROXY` remains as a legacy
  search-only fallback.
- A rotating proxy file and a persistent profile are mutually exclusive — the
  proxy file wins (the browser warns on launch).
- The three-variable trio `WEB_STEALTH=0` + `WEB_REQUEST_DELAY=0` +
  `WEB_CHANNEL=chromium` reproduces the pre-hardening configuration (stealth
  patches and launch flag removed, no pacing, bundled Chromium). With one
  nuance: the pre-hardening browser passed `--no-sandbox` unconditionally,
  whereas today it only passes it under root/`sudo`; so under non-root the
  launch args are `["--disable-gpu"]` (root: `["--no-sandbox",
  "--disable-gpu"]`), which is closer to a normal desktop than the old
  blanket `--no-sandbox` was.

So to switch this project to Cerebras you just write the two-line `.agent` above and run:

    agents "your task"

or override without editing the file:

    agents -P cerebras -m qwen-3.8-27b "your task"

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

### Interrupt Behaviour (Ctrl+C)

The agent supports a three-tier interrupt system so you can pause, redirect, or stop work without losing context:

| Press | Effect |
|---|---|
| **First Ctrl+C** | The current step finishes normally, then the agent pauses and prompts for feedback. |
| **Second Ctrl+C** | Current work is stopped immediately and the agent prompts for feedback. |
| **Third Ctrl+C** | The agent exits. |

In feedback mode you can type new instructions to redirect the agent, or press Ctrl+C to exit. Partial LLM output from an interrupted stream is preserved in the conversation context so nothing is lost.

### Git Integration

When the working directory is a Git repository, the harness provides two safety features:

**Pre-flight clean check** — Before the agent starts, the harness runs `git status --porcelain` to verify the working tree is clean. If there are uncommitted changes, the agent is **not started** and a descriptive error is printed. This prevents the agent from running on top of uncommitted work.

**Auto-commit** — When the agent finishes and there are uncommitted changes, the harness automatically asks the LLM for a commit message, stages all changes, and commits them. The agent itself is not made aware of this process — it simply sees a feedback prompt asking for a commit message.

To disable both features, pass `--nogit`:

    agents --nogit "Refactor the parser module"

If the directory is **not** a Git repository, the harness behaves normally with no git operations attempted.

### Local and Remote Models

You can run against OpenAI-compatible servers (like Ollama, vLLM, or llama.cpp) by using the `--local` flag and setting the `LOCAL_MODEL` environment variable. By default, it connects to `http://localhost:8000`.

    LOCAL_MODEL=qwen3.8-27b agents --local "Explain quantum mechanics"

You can change the port using the `-p` or `--port` flag, or by setting the `LOCAL_LLM_PORT` environment variable:

    LOCAL_MODEL=qwen2.5 agents --local -p 11434 "Write a haiku"

    # Or set the port via environment variable (useful in .bashrc)
    export LOCAL_LLM_PORT=11434
    LOCAL_MODEL=qwen2.5 agents --local "Write a haiku"

To connect to a remote LLM server, use the `-H` or `--host` flag, or set the `LOCAL_LLM_HOST` environment variable:

    LOCAL_MODEL=qwen3.8-27b agents --local -H 192.168.1.50 "Explain quantum mechanics"

    # Or set the host via environment variable (useful in .bashrc)
    export LOCAL_LLM_HOST=192.168.1.50
    export LOCAL_LLM_PORT=8000
    LOCAL_MODEL=qwen3.8-27b agents --local "Explain quantum mechanics"

The `-H` flag takes precedence over `LOCAL_LLM_HOST`, and `-p` takes precedence over `LOCAL_LLM_PORT`, when both are specified.
