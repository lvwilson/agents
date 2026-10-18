# agent

An autonomous AI software engineering agent powered by LLMs. The system is organized in two layers within a single package:

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
- **Cost Awareness** — Token usage and dollar cost are tracked in real time, including prompt caching discounts. Each step's info line shows the completed step's output rate (tokens/second) as soon as it is measured, and the session closes with a metrics panel reporting the whole task's salient numbers — cost, steps, duration, peak context, total output tokens, overall output rate — plus an estimated cost per hour at the observed generation pace. The agent is warned at 80% of its budget, and when 100% is hit it is given one final turn that processes no commands — a wrap-up in which it records the work done so far and emits its completion block — before the session ends, keeping autonomous operation safe and bounded. The same care applies to the model's context window: once the conversation reaches 50% of it, the agent receives an informational reminder to keep command outputs small, and at 80% a wrap-up warning prompting it to record outstanding work before the window fills — both thresholds and messages are configurable in the agent YAML (`context_guard` block), and each fires at most once per session.

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

This installs the main command `agent`, plus the helper CLIs
`agent-docs`, `agent-memory`, and `agent-commit`.

The base install includes the stealth web browser (Playwright).  A
browser binary is also required — either install Google Chrome (it is
driven automatically when present) or:

    python -m playwright install chromium

For optional provider and tool support:

    pip install -e '.[openai]'     # OpenAI models
    pip install -e '.[gemini]'     # Google Gemini models
    pip install -e '.[cerebras]'   # Cerebras Inference models
    pip install -e '.[search]'     # DDGS web search tool
    pip install -e '.[images]'     # image generation (create_image)
    pip install -e '.[mcp]'        # MCP client tools
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

    AGENT_MODEL_PROVIDER=openai AGENT_MODEL=gpt-4o agent "Write a python script to calculate fibonacci numbers"

Use `agent -o -m MODEL "task"` to require online routing. With `-o` /
`--online`, unknown model names (for example, a typo) cannot fall back to a
local server. It also ignores `LOCAL_MODEL` and inherited loopback endpoints.
An unlisted hosted model can still be selected with `--provider NAME` or a
remote `--base-url`; an explicit local `--base-url` conflicts with `-o`.
Without `-o`, unknown names retain their normal local fallback behavior.
The same online-routing rule applies to `agent-commit -o`.

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
| `effort` | Reasoning/thinking effort: `none`, `low`, `medium`, `high`, `xhigh`, `max` (clamped to what the provider accepts; per-model default if omitted). |

**Resolution order** (highest wins, per key): `--provider` / `--model` flags → project
`.agent` → global `~/.agents/agent_config.yaml` → `AGENT_MODEL_PROVIDER` /
`AGENT_MODEL` / `AGENT_BASE_URL` env vars → the agent YAML → provider default.
API keys always come from the environment (`CLAUDE_API_KEY`, `OPENAI_API_KEY`,
`CEREBRAS_API_KEY`, …) — never from the files.

**Thinking / reasoning effort.** Some providers think on every call (Cerebras,
DeepSeek, Kimi, MiniMax), Anthropic's extended thinking is opt-in
(`CLAUDE_THINKING_ENABLED=true`), and OpenAI/Gemini leave it to the server.
One knob covers all of them: `-e/--effort` (or the `effort` key above) —
`none`, `low`, `medium`, `high`, `xhigh`, `max` — normalised once and clamped
per provider, so a request never 400s (e.g. `-e off` on Cerebras qwen turns
reasoning off; `-e max` on Kimi is just `max`). Omit it and each provider keeps
its per-model default. Which wire shape each backend actually sends, and the
output ceilings reasoning can consume: structure.md → "Thinking / reasoning
controls".

Note: the old global location `~/.agent` (a bare file in the home directory) is
retired and no longer read — move any remaining pin from it to
`~/.agents/agent_config.yaml`, then delete the stale file.

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

    agent "your task"

or override without editing the file:

    agent -P cerebras -m qwen-3.8-27b "your task"

### Session Management

Every invocation is assigned a short session ID (e.g. `a7x2`). The full conversation context, system prompt, and token/cost accounting are saved as a JSON file under `/tmp/agents-<username>/` when the agent finishes.

To resume the most recent session for the current working directory:

    agent -r "Continue where you left off"

To resume a specific session by ID:

    agent -r -s a7x2 "Fix the remaining test failures"

To start a new session with a chosen ID:

    agent -s mysession "Refactor the parser module"

**How it works:**

- Sessions are stored at `/tmp/agents-<username>/<session_id>.json` with `0600` permissions.
- An index file maps each working directory to its most recently used session, so `-r` works without specifying an ID.
- Sessions older than 7 days are automatically pruned on each save.
- `/tmp` is cleared on reboot, so sessions are inherently ephemeral. For long-lived persistence, copy the JSON file elsewhere.
- When a session is restored, its step count and the whole-task metrics rollup are restored too — the per-step header and the final `Steps:` count keep numbering across resumed legs instead of restarting at zero.

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

    agent --nogit "Refactor the parser module"

If the directory is **not** a Git repository, the harness behaves normally with no git operations attempted.

### Model discovery and updates

Model updates no longer require editing backend Python dictionaries. The effective
catalog combines backend defaults, reviewed package metadata in
`agents/model_catalog.json`, and a user overlay at `~/.agents/model_catalog.json`.
Normal startup and `agent --list-models` remain offline. Updates do **not** change
your default model, endpoint configuration, or active backend.

```bash
# Preview IDs available to your account (no writes or inference calls).
agent-models discover openai
agent-models discover openai --model gpt-6.1-sol

# Explicitly register selected IDs. Repeat --model for multiple selections.
agent-models discover openai --model gpt-6.1-sol --apply

# Preview, then apply reviewed metadata (prices, limits, request options).
agent-models import reviewed-models.json
agent-models import reviewed-models.json --apply

# Inspect effective metadata and use the model.
agent-models list --provider openai
agent --list-models openai
agent -o -m gpt-6.1-sol --effort max "Your task"
```

After updating an editable installation, reinstall with `pip install -e .` to
register the new `agent-models` executable. Alternatively, use
`python -m agents.cli.models_cli` with the same arguments without reinstalling.

**Discovery is not a capability or pricing oracle.** Provider model listings
confirm IDs, but may include embeddings, audio models, or models your account
cannot run. Therefore `discover --apply` requires exact `--model` selections.
No prices, context sizes, or reasoning settings are guessed from names.
ID-only additions show unknown prices/limits; the backend warns that unpriced
models cannot be constrained by the dollar budget. Import verified pricing
before budgeted use. Inference compatibility still needs a small live check.

Discovery adapters: OpenAI (`OPENAI_API_KEY`), Anthropic (`CLAUDE_API_KEY`),
Cerebras (`CEREBRAS_API_KEY`), DeepSeek (`DEEPSEEK_API_KEY`), Gemini
(`GEMINI_API_KEY`), and Kimi (`MOONSHOT_API_KEY`). Gemini discovery excludes
models without `generateContent`. MiniMax currently supports metadata imports
only. Discovery uses fixed official HTTPS endpoints, not configured proxy/local
URLs, and refuses redirects. Credentials are never saved in the catalog.

A reviewed import is a versioned JSON document, for example:

```json
{
  "version": 1,
  "models": [{
    "provider": "openai",
    "model": "gpt-6.1-sol",
    "display": "GPT-6.1 Sol",
    "context": 1050000,
    "max_output": 128000,
    "effort_levels": ["low", "medium", "high", "xhigh", "max"],
    "pricing": {
      "input_token_cost": 2.0,
      "output_token_cost": 10.0,
      "cache_read_cost": 0.1,
      "cache_write_cost": 2.5,
      "long_context_threshold": 272000,
      "long_input_multiplier": 2.0,
      "long_output_multiplier": 1.5
    },
    "source": "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
    "verified_at": "2026-10-18"
  }]
}
```

Prices are USD per million tokens, using standard processing rates. Sol's bundled
record includes its long-context tier: **over** 272K input tokens, input/cache
rates double and output rates multiply by 1.5 for the full request. Cache-write
pricing is applied when the API reports cache-creation usage. Special service-tier,
regional, Batch, and Flex adjustments are not modeled.

All providers support `display`, `context`, and complete `pricing` overrides.
Pricing requires input, output, and cache-read rates; Gemini additionally supports
`cache_storage_cost_per_hour`. Request options `max_output`, `effort_levels`,
and `send_temperature`, plus tier/cache-write pricing, currently support OpenAI
only. An empty `effort_levels` list suppresses the reasoning parameter.
Unspecified request options retain backend defaults, not discovered guarantees.

Imports merge fields per `(provider, model)`; a `pricing` object replaces that
model's pricing completely. Discovery provenance (`discovery_source`,
`discovered_at`) is separate from reviewed metadata provenance (`source`,
`verified_at`). Updates are validated and written atomically; failures preserve
the previous catalog, and missing IDs are never automatically deleted.
Cross-provider duplicate IDs are rejected to prevent silent routing changes;
use an explicit `--provider` without registering a duplicate instead.
To undo an overlay, remove the relevant record (or the overlay file); package
defaults then apply. Already-created backend instances keep their metadata.
`AGENT_MODEL_CATALOG=/path/catalog.json` selects an alternate overlay for testing
or deployment. No project-local catalog is loaded.

**Sol validation (2026-10-18):** its exact ID was returned by OpenAI's live model
listing, and the actual streaming backend returned `SOL_OK` with 23 input and
6 output tokens (standard-rate estimate: $0.000106). Automated tests cover
routing, metadata overlays, effort mapping including `max`, output limits,
cache pricing, long-context thresholds, and discovery/update failures.

### Local and Remote Models

You can run against OpenAI-compatible servers (like Ollama, vLLM, or llama.cpp) by using the `--local` flag and setting the `LOCAL_MODEL` environment variable. By default, it connects to `http://localhost:8000/v1`.

    LOCAL_MODEL=qwen3.8-27b agent --local "Explain quantum mechanics"

Local mode uses the OpenAI-compatible **chat-completions** backend, even
when an inherited configuration pins a hosted provider such as Cerebras.
The model can also be supplied with `--local -m NAME` or a configured
`model:`. An explicit `--provider` selects a different wire protocol;
a configured `provider: anthropic` is also retained for Anthropic-compatible
local servers.

Use `--base-url URL` to override the configured endpoint exactly (include
`/v1` for OpenAI-compatible APIs). Endpoint pins are used verbatim with
`--local`; use `--base-url` to replace an inherited proxy URL.
For example:

    agent --local -m local-model --base-url http://localhost:11434/v1 "Write a haiku"

Without forced local mode, catalogued models select their hosted provider
unless `--provider` is explicit. Custom proxy URLs are retained for both
`-m` and configured models. Loopback endpoint pins force local routing;
LAN servers should use `--local` or `--provider openai` when overriding a
cloud-provider pin. Sub-agents inherit the resolved provider, model, and
exact endpoint, including scheme and path.


You can change the port using the `--port` flag, or by setting the `LOCAL_LLM_PORT` environment variable:

    LOCAL_MODEL=qwen2.5 agent --local --port 11434 "Write a haiku"

    # Or set the port via environment variable (useful in .bashrc)
    export LOCAL_LLM_PORT=11434
    LOCAL_MODEL=qwen2.5 agent --local "Write a haiku"

To connect to a remote LLM server, use the `-H` or `--host` flag, or set the `LOCAL_LLM_HOST` environment variable:

    LOCAL_MODEL=qwen3.8-27b agent --local -H 192.168.1.50 "Explain quantum mechanics"

    # Or set the host via environment variable (useful in .bashrc)
    export LOCAL_LLM_HOST=192.168.1.50
    export LOCAL_LLM_PORT=8000
    LOCAL_MODEL=qwen3.8-27b agent --local "Explain quantum mechanics"

The `-H` flag takes precedence over `LOCAL_LLM_HOST`, and `--port` takes precedence over `LOCAL_LLM_PORT`, when both are specified.

### Saving a default model

Ask the agent to change your default model; it can use the persistent tool:

```text
Command: set_default_model qwen-3.8-27b
Command: set_default_model my-local-model --project --local --port 11434
Command: set_default_model custom-model --provider openai --base-url https://proxy.example/v1
```

This is an **agent tool**, not a standalone shell command. Its syntax is:

```text
set_default_model MODEL [--global | --project | --scope global|project]
    [--local | --online] [--provider NAME]
    [--base-url URL | --host HOST --port PORT]
```

- **Global** (default): writes `~/.agents/agent_config.yaml`.
- **Project**: updates the nearest existing `.agent` searched upward from the
  working directory, or creates `.agent` in that directory if none exists.
- A known hosted model selects its hosting provider. An unknown name defaults
  to a local OpenAI-compatible server; use `--online --provider NAME` for an
  uncatalogued hosted model. `--local` also allows local deployments to reuse
  hosted model names. Local protocols are `openai` (default) or `anthropic`.
- Local destinations use `--host`/`--port`, then `LOCAL_LLM_HOST`/
  `LOCAL_LLM_PORT`, then `localhost:8000`, with `/v1` appended.
  `--base-url` supplies the exact endpoint instead (including its API path).
  **Existing URLs are not implicitly retained:** pass a proxy URL explicitly.
  A direct hosted selection writes `base_url: ""` to suppress stale lower-level
  endpoints, rather than deleting the key and accidentally inheriting one.
- The tool validates that the saved target will resolve identically at startup.
  The current resolver prefers the catalogue over a file-based provider pin
  at non-loopback endpoints. Consequently, a conflicting provider for a
  catalogued name (including a LAN OpenAI-compatible deployment of another
  provider's model) is rejected without writing. Use a distinct server model
  alias or an explicit CLI `--provider` for that case.
- Applies to **future sessions only**. The running session and its sub-agents
  keep their current backend. Project settings still override global settings;
  CLI flags and `LOCAL_MODEL` can also override the saved selection. If
  `LOCAL_MODEL` is set, unset it or use `agent --online` (`-o`).
- Comments, unrelated keys, temperature, and effort are preserved. Writes are
  atomic; existing permissions are retained and new files are private.
  Malformed or ambiguous YAML, unsupported advanced YAML constructs
  (such as anchors, root flow mappings, and multiline values being edited),
  and symlinked paths are refused without replacing the original file.
  No API keys are stored and no inference request is made.
- Use only for a user-requested persistent change. The tool is blocked in
  planning mode. `set_default_model --help` lists its options.

### Reasoning context and local replay

Reasoning is kept separate from visible assistant text and command parsing.
Completed turns retain provider-native replay data in saved sessions:
Claude signed/redacted thinking blocks, DeepSeek/MiniMax native thinking blocks,
Cerebras `reasoning`, Kimi `reasoning_content`, OpenAI Responses reasoning items
(including encrypted content), and Gemini ordered thought/signature parts.
Replay is restricted to the matching model and wire format. Failed, interrupted,
and unfinished streams do not publish replay data. Native function calls remain
logging-only; this harness executes textual `Command:` lines.

For **Strata or another local OpenAI-compatible server**, explicitly opt in:

```bash
agent -l -m qwen3.8-flash-next-iq3_s --preserve-reasoning "Your task"
```

This sends previous assistant reasoning in `reasoning_content`; it does not
insert `<think>` text into visible messages or enable thinking on the server.
Local replay is **off by default**, leaving existing llama.cpp behavior unchanged.
The setting is saved with the session and inherited by sub-agents. On resume,
`--preserve-reasoning` / `--no-preserve-reasoning` explicitly overrides the saved
setting. The disabling flag controls generic/local replay, not hosted providers'
native reasoning policies. Old sessions without reasoning cannot reconstruct it.

Preservation increases session-file size and can increase input-token costs and
context pressure. Treat session files as sensitive: they can now contain reasoning
and opaque provider state. Provider-reported output totals already include reasoning
for Anthropic and OpenAI-compatible APIs; Gemini's separate `thoughts_token_count`
is added once for cost, throughput, and context accounting. The context meter is
the last reported prompt-plus-generation usage, not an exact prediction of the
next prompt: providers may strip prior reasoning or cache it differently.

Verification: mocked SDK regressions cover all backend families. A live Strata
two-turn comparison confirmed that opt-in replay reached the server (122 prompt
tokens versus 96 without replay). Hosted reasoning replay was not live-tested.
