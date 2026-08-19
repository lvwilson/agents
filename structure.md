# Project Structure — `agents`

> **Purpose of this file:** Quick-reference map for AI agents (and humans) working on this codebase. Describes where things live, how they connect, and what to watch out for.

---

## Overview

`agents` is an autonomous AI software engineering system organized in two layers within a single package. The **reasoning layer** (`agents.agents`) owns the conversation loop, LLM client management, cost tracking, and start/stop decisions. The **tooling layer** (`agents.tools`) handles command parsing and execution against the filesystem, shell, image tools, and web browser.

**Core loop:** `generate → parse → execute → feed back`. The LLM's own reasoning (guided by the system prompt) *is* the control flow — there is no planner, task graph, or state machine.

---

## Directory Layout

```
agents/                          ← repo root
├── structure.md                 ← THIS FILE
├── README.md                    ← User-facing docs, installation, usage
├── pyproject.toml               ← Package config & dependencies
├── requirements.txt             ← Pinned dependencies (mirrors pyproject.toml)
├── LICENSE
│
├── agents/                      ← Python source package (reasoning layer)
│   ├── agents.py                ← ENTRY POINT — Agent class, main(), CLI arg parsing
│   ├── llm_backend.py           ← Abstract base class: LLMBackend, StreamHandler, retry logic
│   ├── ui.py                    ← All Rich console output (banners, headers, spinners, RichStreamHandler)
│   ├── session.py               ← Session persistence (JSON files in /tmp)
│   │
│   ├── backends/                ← Provider-specific LLM implementations (lazy-loaded)
│   │   ├── __init__.py          ← Backend registry & factory: create_backend()
│   │   ├── anthropic_backend.py ← AnthropicBackend — Claude models, prompt caching
│   │   ├── openai_backend.py    ← OpenAIBackend — GPT models, Responses API
│   │   └── gemini_backend.py    ← GeminiBackend — Gemini models, server-side context caching
│   │
│   ├── tools/                   ← Tooling layer (command parsing & execution)
│   │   ├── __init__.py          ← Public API: process_content, filter_content, terminate_process
│   │   ├── parser.py            ← Command parser — extracts commands from LLM output
│   │   ├── functions.py         ← Tool implementations (file I/O, shell, code manipulation, web)
│   │   ├── codemanipulator.py   ← AST-based Python code manipulation
│   │   ├── code_scissors.py     ← Line-based text cutting operations
│   │   ├── findreplace.py       ← Search/replace block parsing
│   │   ├── web_browser.py       ← Playwright headless browser singleton
│   │   └── summarize.py         ← LLM-powered file/folder summarization with caching
│   │
│   ├── basic_agent.yaml         ← DEFAULT config — file I/O, find-and-replace, shell, images
│   └── manipulator_agent.yaml   ← ALT config — AST-based Python code manipulation commands
│
└── tests/                       ← Test suite
    ├── test_code_scissors.py
    ├── test_code_scissors_extended.py
    ├── test_find_replace.py
    └── test_manipulator.py
```

---

## Key Files in Detail

### `agents/agents.py` — Entry Point & Agent Orchestrator

- **`Agent` class** — The central orchestrator. Holds conversation context, system prompt, LLM backend, and budget.
  - `__init__()` — Loads YAML config, resolves provider/model (env vars override config), creates backend via `create_backend()`, builds system prompt with OS/shell/date/user info, displays startup banner.
  - `_iterate()` — One turn of the conversation loop: calls `generate_response()`, runs `filter_content()` and `process_content()` (from `agents.tools`), appends results, checks budget, marks large messages for caching.
  - `run()` — Loops `_iterate()` until no commands returned, budget exceeded, KeyboardInterrupt, or error.
  - `save_context()` / `load_context()` — Pickle-based pause/resume of full conversation state including token counts and costs.
  - `LARGE_MESSAGE_CACHE_THRESHOLD = 10_000` — Character threshold for requesting backend caching of a user message.
- **`run_agent()`** — High-level function: creates Agent, optionally restores context, runs, extracts completion block, retries once if no completion found.
- **`extract_completion()`** — Parses the YAML completion block from the LLM's final response (wrapped in 5 backticks).
- **`main()`** — CLI entry point with argparse. Supports `-b` budget, `-r` restore, `-l` local mode, `-p` port. Reads piped stdin.

**Integration with tools subpackage** (the internal boundary):
- `process_content(response)` → parses commands from LLM output, executes them, returns `(text_result, image_tuples)`
- `filter_content(response)` → trims output when LLM queues multiple read commands
- `terminate_process()` → kills any running subprocess (used in SIGTERM handler)
- `get_default_shell()` → used in system prompt construction

### `agents/llm_backend.py` — Abstract Base Class

- **`StreamHandler`** — Callback protocol for streaming events (`on_stream_start`, `on_stream_token`, `on_stream_end`, `on_retry`, `on_error`). Default is silent no-op (headless mode).
- **`NullStreamHandler`** — Alias for the base `StreamHandler` (no-op).
- **`RATE_LIMIT` / `TRANSIENT`** — Error classification constants.
- **`LLMBackend` (ABC)** — Unified interface all providers implement:
  - **Retry template:** `_run_with_retries(attempt_fn)` — exponential backoff with jitter for rate limits, fixed retry count for transient errors. Manages `on_stream_start`/`on_stream_end` lifecycle.
  - **Abstract:** `generate_response(system_prompt, context)` → `str`, `display_name` property.
  - **Virtual (with defaults):** `context_window_size` (256K default), `mark_for_caching()`, `trim_cache_blocks()`, `calculate_cost()`.
  - **Subclass hooks:** `_classify_error(e)` → `RATE_LIMIT`|`TRANSIENT`, `_extract_retry_after(e)` → optional seconds.
  - **State tracking:** `cost`, `cost_without_cache`, `call_count`, `last_input_tokens`, `last_output_tokens`, `last_total_context_tokens`, `peak_context_tokens`.
  - **Retry config constants:** `RETRY_TIMEOUT=300`, `RETRY_BASE_DELAY=1`, `RETRY_MAX_DELAY=60`, `RETRY_BACKOFF_FACTOR=2`, `MAX_ERROR_RETRIES=3`, `TRANSIENT_RETRY_DELAY=2`.

### `agents/ui.py` — Rich Console UI

- Owns the `Console` instance (writes to `/dev/tty` to keep stdout clean).
- **Theme:** `agent_theme` with styles for stream, info, success, warning, error, cost, muted.
- **Display functions:** `print_banner()`, `print_iteration_header()`, `print_summary()`, `print_completion_result()`, `print_budget_warning()`, `print_budget_exceeded()`, `print_error()`, `print_interrupted()`, `print_sigterm()`, `print_clipped()`.
- **Helpers:** `build_budget_bar()`, `build_context_bar()`, `format_tokens()`, `safe_console_print()`, `create_spinner()`.
- **`RichStreamHandler(StreamHandler)`** — Interactive terminal implementation of the stream callback protocol. Manages spinner → streaming text transition. This is passed to backends to decouple them from Rich.

### `agents/backends/__init__.py` — Backend Registry & Factory

- **`_REGISTRY`** — Maps provider name → `(module_path, class_name)`: `"anthropic"`, `"openai"`, `"gemini"`.
- **`create_backend(provider, model=, base_url=, cache_step=, stream_handler=)`** — Lazy-imports the provider module on first use, instantiates and returns the backend. Keeps startup fast and avoids hard SDK dependencies.

### `agents/backends/anthropic_backend.py` — Claude Provider

- **`AnthropicBackend(LLMBackend)`** — Uses `anthropic` SDK.
- **Prompt caching:** Anthropic-specific `cache_control: {"type": "ephemeral"}` annotations. `mark_for_caching()` and `trim_cache_blocks()` manage up to 2 active cache blocks. System prompt is always wrapped as a cacheable content block. Cache blocks placed every `cache_step` calls (default 2).
- **Pricing:** Per-model dicts for input/output/cache-creation(1.25×)/cache-read(0.1×) costs.
- **Context windows:** Per-model, all currently 200K.
- **MiniMax routing:** Special API key validation for MiniMax models (prefix check).
- **Error classification:** `anthropic.RateLimitError` → `RATE_LIMIT`, everything else → `TRANSIENT`.
- **Streaming:** Uses `client.messages.stream()` context manager, iterates `text_stream`.
- **max_tokens:** 64000 (highest of all backends). Extra headers for output-128k beta and prompt-caching beta.
- **Response handling:** Skips `ThinkingBlock` objects, returns first `TextBlock`. Falls back to error message if no text content.

### `agents/backends/openai_backend.py` — OpenAI Provider

- **`OpenAIBackend(LLMBackend)`** — Uses `openai` SDK with the **Responses API** (not Chat Completions).
- **Message translation:** `_format_messages()` converts internal format to Responses API format (`input_text`, `input_image`, `output_text`). Includes `_validate_responses_input()` for role/content-type validation.
- **Pricing:** Per-model dicts. Cache read at 50% input cost.
- **Context windows:** Per-model, all currently 128K.
- **Error classification:** `openai.RateLimitError` → `RATE_LIMIT`.
- **Streaming:** Uses `responses.create(stream=True)`, handles `response.output_text.delta` and `response.completed` events.
- **max_tokens:** 16384.
- **No prompt caching support** (uses base class no-op `mark_for_caching`/`trim_cache_blocks`).

### `agents/backends/gemini_backend.py` — Google Gemini Provider

- **`GeminiBackend(LLMBackend)`** — Uses `google-genai` unified SDK.
- **Server-side context caching:** Creates/manages server-side caches with TTL (300s). Caches all messages except the last user message. Charges storage cost upfront. Auto-invalidates and retries on cache errors.
- **Message translation:** `_translate_messages()` converts to Gemini `Content`/`Part` objects. Maps `"assistant"` → `"model"` role.
- **Pricing:** Per-model dicts with explicit `cache_read_cost` and `cache_storage_cost_per_hour`.
- **Context windows:** Per-model, up to 2M tokens (Pro) or 1M (Flash).
- **Error classification:** Checks `google.api_core.exceptions.ResourceExhausted`/`TooManyRequests`, falls back to string matching for `"429"`/`"RESOURCE_EXHAUSTED"`.
- **Streaming:** Uses `generate_content_stream()`.
- **max_tokens:** 16384.

### `agents/tools/` — Tooling Layer

The `tools` subpackage handles all command parsing and execution. It knows nothing about LLM providers, conversation history, or budgets.

- **`__init__.py`** — Public API: re-exports `process_content`, `filter_content`, `terminate_process`, `get_default_shell`, `register_llm`.
- **`parser.py`** — Command parser. Extracts `Command:` directives and backtick-delimited payloads from LLM output. `process_content()` dispatches to tool functions. `filter_content()` trims output when multiple read commands are queued.
- **`functions.py`** — All tool implementations: file I/O (`read_file`, `write_file`, `append_to_file`), find-and-replace, line-based text operations (code scissors wrappers), AST code manipulation wrappers, shell execution (`run_console_command` with PTY), `stdout`, `summarize`, and all web browser command wrappers.
- **`codemanipulator.py`** — AST-based Python code manipulation. Uses `ast.NodeTransformer` to read/replace/insert/remove code at dot-separated addresses (e.g. `ClassName.method_name`). Formats output with `black`.
- **`code_scissors.py`** — Line-based text cutting operations: `insert_before`, `insert_after`, `replace_before`, `replace_after`, `replace_between`.
- **`findreplace.py`** — Parses SEARCH/REPLACE blocks and performs string replacement.
- **`web_browser.py`** — Playwright-powered headless browser singleton. Provides navigation, text/HTML reading, clicking, typing, screenshots, JavaScript execution, and element waiting.
- **`summarize.py`** — LLM-powered file/folder summarization with mtime-based caching. The LLM backend is injected via `register_llm()` (called by `Agent.__init__`).

### `agents/basic_agent.yaml` — Default Agent Configuration

- System prompt defines the agent persona, response format, available commands, examples, and completion protocol.
- `overbudget` message injected at 80% budget.
- **This is the config used by `main()` via `run_agent('basic_agent.yaml', ...)`.**

### `agents/manipulator_agent.yaml` — AST Manipulation Configuration

- Alternative system prompt with AST-based Python code manipulation commands (read_code_at_address, replace_code_at_address, add_code_before/after_address, read_code_signatures_and_docstrings, replace_docstring_at_address).
- Uses dot-separated address notation (e.g., `ClassName.method_name`).
- Only supports one command per response (no batching).
- References an older model default (`claude-3-5-sonnet-20240620`).

---

## Data Flow

```
User CLI input
    │
    ▼
Agent.__init__()  ←── YAML config + env vars → create_backend()
    │
    ▼
Agent.run() loop:
    │
    ├─► LLMBackend.generate_response(system_prompt, context)
    │       │
    │       ├─► _run_with_retries(attempt_fn)  [retry/backoff]
    │       │       │
    │       │       └─► Provider SDK streaming call
    │       │               │
    │       │               └─► StreamHandler.on_stream_token()  [UI output]
    │       │
    │       └─► Returns response text + updates cost/token tracking
    │
    ├─► filter_content(response)         [tools — trim multi-read]
    │
    ├─► process_content(response)        [tools — parse & execute commands]
    │       │
    │       └─► Returns (command_output, image_tuples)
    │
    ├─► Append assistant + user messages to context
    │
    ├─► Budget check (75% warning, 100% termination)
    │
    └─► If command_output == "End." → stop; else → next iteration
```

---

## Internal Message Format

All conversation state uses this Anthropic-derived format (other backends translate in/out):

```python
{
    "role": "user" | "assistant",
    "content": [
        {"type": "text", "text": "..."},
        # Optional image blocks:
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
        # Optional Anthropic cache annotation:
        {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}},
    ]
}
```

---

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `CLAUDE_API_KEY` | Anthropic API key | For Anthropic provider |
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI provider |
| `GEMINI_API_KEY` | Google Gemini API key | For Gemini provider |
| `AGENT_MODEL_PROVIDER` | Override provider (`anthropic`, `openai`, `gemini`) | No (defaults to config) |
| `AGENT_MODEL` | Override model name | No (defaults to provider default) |
| `LOCAL_MODEL` | Model name for local inference | Required with `--local` flag |

---

## CLI Usage

```bash
# Default (Anthropic)
python agents.py "task description"

# With budget
python agents.py -b 2.0 "task description"

# Resume previous session
python agents.py -r "continue the task"

# Override provider/model
AGENT_MODEL_PROVIDER=openai AGENT_MODEL=gpt-5.3-codex python agents.py "task"

# Local model
LOCAL_MODEL=qwen3.8-27b python agents.py --local -p 11434 "task"

# Pipe input
echo "file contents" | python agents.py "analyze this"
```

---

## Known Issues

Documented in `issues.md`. Summary:

1. **~~Backends own streaming I/O~~** — *Resolved.* StreamHandler protocol now decouples backends from Rich.
2. **~~Retry/backoff duplicated~~** — *Resolved.* `_run_with_retries()` template method in base class.
3. **~~Hardcoded context window~~** — *Resolved.* `context_window_size` property on each backend.
4. **~~ClaudeAgent naming~~** — *Partially resolved.* Renamed to `Agent`. Internal message format still Anthropic-derived.
5. **~~cache_control in ai_client.py~~** — *Resolved.* `convert_string_to_dict` no longer adds cache annotations; handled by Anthropic backend.

---

## Dependencies

- **`anthropic`** — Anthropic SDK (required by default)
- **`PyYAML`** — YAML config parsing
- **`rich`** — Terminal UI (panels, spinners, styled output)
- **`pillow`** — Image handling
- **`openai`** — OpenAI SDK (optional, install for OpenAI provider)
- **`google-genai`** — Google Gemini SDK (optional, install for Gemini provider)
- **`black`** — Code formatting (used by AST code manipulator)
- **`requests`** — HTTP requests (used by image generation)
- **`playwright`** — Headless browser (optional, install with `pip install -e '.[browser]'`)
