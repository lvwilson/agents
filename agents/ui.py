#!/usr/bin/env python3
"""
UI module — all Rich-based display logic for the agent system.

This module owns the console instance, theme, and every function that
renders styled output.  Other modules should not import Rich directly.
"""

import signal
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.theme import Theme

# ── Theme ────────────────────────────────────────────────────────────
agent_theme = Theme({
    "stream":     "bright_cyan",
    "stream.dim": "dim cyan",
    "info":       "bright_blue",
    "success":    "bright_green",
    "warning":    "bright_yellow",
    "error":      "bright_red",
    "cost":       "bright_magenta",
    "muted":      "dim white",
})

# ── Console (writes to /dev/tty so stdout stays clean) ───────────────
_tty = None


def _get_tty():
    """Return a writable file for /dev/tty, falling back to stderr."""
    global _tty
    if _tty is None:
        try:
            _tty = open("/dev/tty", "w")
        except OSError:
            _tty = sys.stderr
    return _tty


# Lazy console singleton — initialised on first access so that the
# /dev/tty open is deferred until the module is actually *used*.
_console = None


def _get_console():
    """Return the module-level Rich Console, creating it on first use."""
    global _console
    if _console is None:
        _console = Console(file=_get_tty(), theme=agent_theme)
    return _console


# Keep a module-level ``console`` property-like accessor.  Existing code
# references ``console`` directly (e.g. ``console.print(…)``), so we
# replace the module attribute with a lazy wrapper.
class _LazyConsole:
    """Proxy that forwards attribute access to the real Console."""
    def __getattr__(self, name):
        return getattr(_get_console(), name)

console = _LazyConsole()


def safe_console_print(text, style="default", end="\n"):
    """Print to the console, falling back to plain write on error."""
    try:
        _get_console().print(text, style=style, end=end)
    except Exception:
        print(text, file=_get_tty())


# ── Formatting helpers ───────────────────────────────────────────────

def build_budget_bar(spent, budget, width=20):
    """Return a Rich-markup progress bar for budget usage."""
    ratio = min(spent / budget, 1.0) if budget > 0 else 0
    filled = int(ratio * width)
    empty = width - filled

    if ratio < 0.5:
        color = "bright_green"
    elif ratio < 0.75:
        color = "bright_yellow"
    else:
        color = "bright_red"

    bar = f"[{color}]{'━' * filled}[/][dim]{'─' * empty}[/]"
    pct = f"{ratio * 100:.0f}%"
    return f"{bar} {pct}"


def build_context_bar(used_tokens, max_tokens, width=20):
    """Return a Rich-markup progress bar for context window usage."""
    ratio = min(used_tokens / max_tokens, 1.0) if max_tokens > 0 else 0
    filled = int(ratio * width)
    empty = width - filled

    if ratio < 0.5:
        color = "bright_green"
    elif ratio < 0.75:
        color = "bright_yellow"
    elif ratio < 0.9:
        color = "bright_red"
    else:
        color = "bold bright_red"

    bar = f"[{color}]{'━' * filled}[/][dim]{'─' * empty}[/]"
    pct = f"{ratio * 100:.0f}%"
    return f"{bar} {pct}"


def format_tokens(n):
    """Format a token count with K/M suffix."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_rate(toks_per_sec):
    """Format a tokens-per-second rate with one decimal.

    ``None`` / non-numeric renders as ``"—"`` so callers can pass an
    unmeasured rate without pre-checking before the header.
    """
    try:
        return f"{float(toks_per_sec):.1f} tok/s"
    except (TypeError, ValueError):
        return "—"


def format_duration(seconds):
    """Format a duration in seconds as e.g. ``"1m 42s"`` or ``"42s"``."""
    if seconds is None or seconds <= 0:
        return "0s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# ── Display functions ────────────────────────────────────────────────

def print_banner(display_name, compute_budget, platform_str, context_window_tokens,
                 planning_mode=False):
    """Display the startup banner."""
    info_line = (
        f"[muted]Model:[/] [bright_cyan]{display_name}[/]  "
        f"[muted]Budget:[/] [bright_green]${compute_budget:.2f}[/]  "
        f"[muted]System:[/] {platform_str}  "
        f"[muted]Context window:[/] {format_tokens(context_window_tokens)}"
    )
    if planning_mode:
        info_line += (
            f"\n[muted]Mode:[/] [warning]PLANNING — plan first; "
            "no state changes until you approve at the terminal[/]"
        )
    console.print(Panel(
        info_line,
        title="[bold bright_white]◈  Agent Initialized  ◈[/]",
        border_style="bright_blue",
        padding=(0, 1),
    ))


def _format_cache_savings(cost, cost_without_cache):
    """Return a Rich-markup string showing cache savings, or empty string."""
    if cost_without_cache > 0 and cost_without_cache > cost:
        pct = (cost_without_cache - cost) / cost_without_cache * 100
        return f" [success]({pct:.0f}% saved)[/]"
    return ""


def print_iteration_header(step, cost, compute_budget,
                           last_input_tokens=0, last_output_tokens=0,
                           last_total_context_tokens=0,
                           cost_without_cache=0.0,
                           context_window_tokens=256_000,
                           step_tokens_per_sec=None):
    """Display the iteration header with cost, budget, and context window info."""
    cost_str = f"${cost:.4f}"
    savings_str = _format_cache_savings(cost, cost_without_cache)
    budget_bar = build_budget_bar(cost, compute_budget)

    token_info = ""
    if last_input_tokens > 0:
        token_info = (
            f"  [muted]in:[/] {format_tokens(last_input_tokens)}"
            f"  [muted]out:[/] {format_tokens(last_output_tokens)}"
        )

    # Per-step generation rate (this step's output tokens over the
    # wall-clock time of its generation).  Shown once measured —
    # i.e. from the second step onwards, since step N's rate is only
    # known after step N-1's generation has completed.
    if step_tokens_per_sec is not None and step_tokens_per_sec > 0:
        token_info += f"  [muted]rate:[/] {format_rate(step_tokens_per_sec)}"

    context_bar = build_context_bar(last_total_context_tokens, context_window_tokens)
    context_info = (
        f"  [muted]Context:[/] {format_tokens(last_total_context_tokens)}"
        f"/{format_tokens(context_window_tokens)}  {context_bar}"
    )

    header_left = f"[bold bright_white]Step {step}[/]"
    header_right = f"[cost]{cost_str}[/]{savings_str}  {budget_bar}{token_info}{context_info}"

    console.print()
    console.print(Rule(style="dim bright_blue"))
    console.print(f"  {header_left}    {header_right}")
    console.print(Rule(style="dim bright_blue"))


def build_final_metrics(context, cost, steps, elapsed, compute_budget,
                        peak_context_tokens, cost_without_cache,
                        context_window_tokens,
                        total_output_tokens=0,
                        output_rate_tokens_per_sec=None,
                        cost_per_hour=None):
    """Build the final-session metrics panel (the "Session Complete" display).

    Shows the salient metrics of the whole task: cost (with cache
    savings), step count, duration, budget bar, peak context, total
    output tokens, overall output rate, and the estimated cost per hour
    if the task kept running at the observed pace.

    ``context`` is the Rich Console to print to — passed in rather than
    using the module global so tests can capture output with a plain
    in-memory Console.
    """
    console = context
    time_str = format_duration(elapsed)

    savings_str = _format_cache_savings(cost, cost_without_cache)
    metrics = [
        f"[muted]Cost:[/] [cost]${cost:.4f}[/]{savings_str}",
        f"[muted]Steps:[/] {steps}",
        f"[muted]Duration:[/] {time_str}",
        f"[muted]Budget:[/] {build_budget_bar(cost, compute_budget)}",
    ]
    if peak_context_tokens > 0:
        context_bar = build_context_bar(peak_context_tokens, context_window_tokens)
        metrics.append(
            f"[muted]Peak context:[/] "
            f"{format_tokens(peak_context_tokens)}/{format_tokens(context_window_tokens)}"
            f"  {context_bar}"
        )
    if total_output_tokens > 0:
        metrics.append(
            f"[muted]Output tokens:[/] {format_tokens(total_output_tokens)}"
        )
    metrics.append(
        f"[muted]Output rate:[/] "
        f"{format_rate(output_rate_tokens_per_sec)}"
    )
    if cost_per_hour is not None and cost_per_hour > 0:
        metrics.append(
            f"[muted]Est. cost/hour:[/] [cost]${cost_per_hour:.2f}[/]"
        )

    console.print()
    console.print(Panel(
        "\n".join(metrics),
        title="[bold bright_white]◈  Session Complete  ◈[/]",
        border_style="bright_blue",
        padding=(0, 1),
    ))
    console.print()


def print_summary(cost, steps, elapsed, compute_budget, peak_context_tokens=0,
                  cost_without_cache=0.0, context_window_tokens=256_000,
                  total_output_tokens=0,
                  output_rate_tokens_per_sec=None,
                  cost_per_hour=None):
    """Display the final session summary + metrics panel."""
    build_final_metrics(
        _get_console(),
        cost=cost,
        steps=steps,
        elapsed=elapsed,
        compute_budget=compute_budget,
        peak_context_tokens=peak_context_tokens,
        cost_without_cache=cost_without_cache,
        context_window_tokens=context_window_tokens,
        total_output_tokens=total_output_tokens,
        output_rate_tokens_per_sec=output_rate_tokens_per_sec,
        cost_per_hour=cost_per_hour,
    )


def print_completion_result(completion, success):
    """Display the final completion result in a styled panel."""
    if success:
        icon, style, title_style = "✓", "bright_green", "bold bright_green"
    else:
        icon, style, title_style = "✗", "bright_red", "bold bright_red"

    console.print(Panel(
        f"[{style}]{completion}[/]",
        title=f"[{title_style}]{icon}  {'Success' if success else 'Failed'}[/]",
        border_style=style,
        padding=(0, 1),
    ))


def print_budget_warning(cost, compute_budget):
    """Display a budget warning panel."""
    console.print()
    pct = cost / compute_budget * 100
    console.print(Panel(
        f"[warning]Budget at {pct:.0f}% (${cost:.4f} / ${compute_budget:.2f})[/]",
        title="[bold warning]⚠  Budget Warning[/]",
        border_style="bright_yellow",
        padding=(0, 1),
    ))


def print_budget_exceeded(cost, compute_budget):
    """Display a budget-exceeded panel."""
    console.print()
    console.print(Panel(
        f"[error]Spent ${cost:.4f} of ${compute_budget:.2f} budget[/]",
        title="[bold error]✗  Budget Exceeded[/]",
        border_style="bright_red",
        padding=(0, 1),
    ))


def print_error(exception, trace_str=None):
    """Display an error panel with an optional traceback.

    ``trace_str`` may be ``None`` (or empty) when there is no traceback to
    show — e.g. a loop termination, which is a controlled stop rather than
    an exception with a stack.
    """
    body = f"[error]{exception}[/]"
    if trace_str:
        body += f"\n[muted]{trace_str}[/]"
    console.print()
    console.print(Panel(
        body,
        title="[bold error]✗  Error[/]",
        border_style="bright_red",
        padding=(0, 1),
    ))


def print_interrupted():
    """Display an interruption notice."""
    console.print("\n  ⚠  Interrupted by user", style="warning")


def print_interrupt_feedback():
    """Display a notice that the agent is waiting for user feedback after interrupt."""
    console.print()
    console.print(Panel(
        "[warning]Agent paused. Enter feedback to continue, or press Ctrl+C to exit.[/]",
        title="[bold warning]⚠  Interrupted[/]",
        border_style="bright_yellow",
        padding=(0, 1),
    ))


def get_user_feedback() -> str | None:
    """Prompt the user for feedback input.

    Returns the user's input string, or ``None`` if the user presses
    Ctrl+C to exit.
    """
    try:
        tty_in = open("/dev/tty", "r")
    except OSError:
        tty_in = sys.stdin

    try:
        safe_console_print("[bright_yellow]Feedback:[/] ", end="")
        _get_tty().flush()
        line = tty_in.readline()
        if not line:
            return None
        return line.rstrip("\n")
    except (KeyboardInterrupt, EOFError):
        return None
    finally:
        if tty_in is not sys.stdin:
            tty_in.close()


def print_sigterm():
    """Display a SIGTERM notice."""
    console.print("\n  ⚠  SIGTERM received — terminating subprocess…", style="warning")


def print_clipped(clipped_chars, response_text):
    """Display a clipping notice and the filtered response."""
    console.print(f"\n  ✂  Clipped {clipped_chars} characters from response", style="warning")
    safe_console_print(response_text, style="stream")


def create_spinner(message="  ◌  Waiting for response…"):
    """Create a Rich Status spinner for display while awaiting LLM response.

    Returns a Status object that must be started with .start() and stopped
    with .stop().
    """
    return console.status(message, spinner="dots", spinner_style="bright_cyan")


# ── Stream handler (decouples backends from Rich) ────────────────────

from .llm_backend import StreamHandler


class RichStreamHandler(StreamHandler):
    """StreamHandler that renders to the Rich console.

    This is the interactive-terminal implementation.  Pass an instance to
    ``create_backend(…, stream_handler=RichStreamHandler())`` to get the
    same streaming UX the backends previously hard-coded.
    """

    def __init__(self):
        super().__init__()
        self._spinner = None

    def _stop_spinner(self) -> None:
        """Stop and clear the spinner if it is running."""
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

    def _start_spinner(self) -> None:
        """Create and start a fresh spinner."""
        self._stop_spinner()
        self._spinner = create_spinner()
        self._spinner.start()

    def on_stream_start(self) -> None:
        super().on_stream_start()
        self._start_spinner()

    def on_stream_token(self, token: str) -> None:
        super().on_stream_token(token)
        self._stop_spinner()
        safe_console_print(token, style="stream", end="")

    def on_stream_end(self) -> None:
        self._stop_spinner()

    # ── Reasoning token streaming ────────────────────────────────────

    def on_stream_reasoning_start(self) -> None:
        super().on_stream_reasoning_start()
        self._stop_spinner()
        safe_console_print("\n[dim]# Reasoning[/dim]\n", end="")

    def on_stream_reasoning_token(self, token: str) -> None:
        super().on_stream_reasoning_token(token)
        self._stop_spinner()
        safe_console_print(token, style="dim", end="")

    def on_stream_reasoning_end(self) -> None:
        super().on_stream_reasoning_end()
        safe_console_print("\n", end="")
        # Restart spinner while waiting for text content
        self._start_spinner()

    def on_tool_call(self, name: str, arguments: str = "") -> None:
        args_str = str(arguments)
        if len(args_str) > 120:
            args_str = args_str[:117] + "…"
        safe_console_print(
            f"  ⚠ Native tool call detected (not executed): "
            f"[bright_yellow]{name}[/]({args_str})",
            style="warning",
        )

    def on_retry(self, message: str) -> None:
        safe_console_print(f"\n  ⏳ {message}", style="warning")

    def on_error(self, message: str) -> None:
        safe_console_print(f"\n  ✗ {message}", style="error")
# ── Planning-mode approval gate ──────────────────────────────────────

@dataclass
class PlanDecision:
    """Outcome of the interactive plan-approval prompt."""
    approved: bool
    feedback: str = ""


def prompt_plan_approval() -> PlanDecision | None:
    """Ask the user, in the terminal, to approve the agent's plan.

    While waiting:
    * Enter / ``y`` / ``yes`` / ``ok`` → :class:`PlanDecision` with
      ``approved=True``.
    * Any other text → ``approved=False`` and the text kept as
      ``feedback`` (the caller feeds it back to the agent).
    * Ctrl+C or EOF → ``None`` — the caller should end the session
      cleanly (the plan and all state remain saved, resumable).

    The default SIGINT handler is installed for the duration of the
    read so that a Ctrl+C here is a plain KeyboardInterrupt (cancel the
    prompt) instead of the agent's three-tier interrupt machinery.

    Returns
    -------
    PlanDecision | None
    """
    console.print()
    console.print(Panel(
        "[bold]The agent has finished planning and is asking to continue.[/]\n"
        "Press [bright_green]Enter[/] (or type y / yes / ok) to approve the "
        "plan and start execution.\n"
        "Type [warning]anything else[/] to send it back to the agent as "
        "feedback — planning continues.\n"
        "Press [error]Ctrl+C[/] to end the session (the plan is saved; "
        "resume with -r).",
        title="[bold bright_yellow]⏸  PLANNING MODE — Awaiting your approval[/]",
        border_style="bright_yellow",
        padding=(0, 1),
    ))
    try:
        tty_in = open("/dev/tty", "r")
    except OSError:
        tty_in = sys.stdin

    original_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        safe_console_print("  Plan approval 👉 ", end="")
        _get_tty().flush()
        line = tty_in.readline()
        if not line:
            return None
        line = line.rstrip("\n")
        if line.strip().lower() in ("", "y", "yes", "ok"):
            return PlanDecision(approved=True)
        return PlanDecision(approved=False, feedback=line.strip())
    except (KeyboardInterrupt, EOFError):
        return None
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if tty_in is not sys.stdin:
            tty_in.close()
