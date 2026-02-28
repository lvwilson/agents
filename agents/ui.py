#!/usr/bin/env python3
"""
UI module — all Rich-based display logic for the agent system.

This module owns the console instance, theme, and every function that
renders styled output.  Neither agents.py nor ai_client.py should
import Rich directly.
"""

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
_tty = open("/dev/tty", "w")
console = Console(file=_tty, theme=agent_theme)


def safe_console_print(text, style="default", end="\n"):
    """Print to the console, falling back to plain write on error."""
    try:
        console.print(text, style=style, end=end)
    except Exception:
        print(text, file=_tty)


# ── Formatting helpers ───────────────────────────────────────────────

# TODO: This is Anthropic-specific (256K). OpenAI models use 128K, Gemini varies.
# Now that multiple backends exist, this should come from the backend instance.
CONTEXT_WINDOW_TOKENS = 256_000

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


def build_context_bar(used_tokens, max_tokens=CONTEXT_WINDOW_TOKENS, width=20):
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


# ── Display functions ────────────────────────────────────────────────

def print_banner(display_name, compute_budget, platform_str):
    """Display the startup banner."""
    info_line = (
        f"[muted]Model:[/] [bright_cyan]{display_name}[/]  "
        f"[muted]Budget:[/] [bright_green]${compute_budget:.2f}[/]  "
        f"[muted]System:[/] {platform_str}  "
        f"[muted]Context window:[/] {format_tokens(CONTEXT_WINDOW_TOKENS)}"
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
                           cost_without_cache=0.0):
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

    context_bar = build_context_bar(last_total_context_tokens)
    context_info = (
        f"  [muted]Context:[/] {format_tokens(last_total_context_tokens)}"
        f"/{format_tokens(CONTEXT_WINDOW_TOKENS)}  {context_bar}"
    )

    header_left = f"[bold bright_white]Step {step}[/]"
    header_right = f"[cost]{cost_str}[/]{savings_str}  {budget_bar}{token_info}{context_info}"

    console.print()
    console.print(Rule(style="dim bright_blue"))
    console.print(f"  {header_left}    {header_right}")
    console.print(Rule(style="dim bright_blue"))


def print_summary(cost, steps, elapsed, compute_budget, peak_context_tokens=0,
                  cost_without_cache=0.0):
    """Display the final session summary panel."""
    console.print()
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    savings_str = _format_cache_savings(cost, cost_without_cache)
    summary_line = (
        f"[muted]Cost:[/] [cost]${cost:.4f}[/]{savings_str}  "
        f"[muted]Steps:[/] {steps}  "
        f"[muted]Duration:[/] {time_str}  "
        f"[muted]Budget:[/] {build_budget_bar(cost, compute_budget)}"
    )

    if peak_context_tokens > 0:
        context_bar = build_context_bar(peak_context_tokens)
        summary_line += (
            f"  [muted]Peak context:[/] "
            f"{format_tokens(peak_context_tokens)}/{format_tokens(CONTEXT_WINDOW_TOKENS)}"
            f"  {context_bar}"
        )

    console.print(Panel(
        summary_line,
        title="[bold bright_white]◈  Session Complete  ◈[/]",
        border_style="bright_blue",
        padding=(0, 1),
    ))
    console.print()


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


def print_error(exception, trace_str):
    """Display an error panel with traceback."""
    console.print()
    console.print(Panel(
        f"[error]{exception}[/]\n[muted]{trace_str}[/]",
        title="[bold error]✗  Error[/]",
        border_style="bright_red",
        padding=(0, 1),
    ))


def print_interrupted():
    """Display an interruption notice."""
    console.print("\n  ⚠  Interrupted by user", style="warning")


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
