"""
``--list-models`` table renderer.

Renders the model metadata aggregated by
:func:`agents.backends.list_available_models` as a Rich table on
**stdout** so the output is pipeable (``--list-models | grep opus``)
and redirectable (``--list-models > models.txt``) like any other
``--list`` style CLI flag.  Rich automatically strips ANSI styling
when stdout is not a terminal, so redirected output stays clean.

This module exists to keep ``agents.py`` free of presentation code.
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from ..backends import list_available_models


def _fmt_money(value) -> str:
    """Format a per-million-token dollar price, or an em dash."""
    if isinstance(value, (int, float)):
        return f"${value:.4g}"
    return "—"


def print_model_table(provider_filter: str | None = None) -> None:
    """Print a Rich table of available models to stdout.

    Parameters
    ----------
    provider_filter : str or None
        If given, only show models from this provider.  An empty result
        prints a message to stderr instead of a table.
    """
    entries = list_available_models(provider_filter)
    if not entries:
        if provider_filter:
            sys.stderr.write(
                f"No models found for provider '{provider_filter}'.\n"
            )
        else:
            sys.stderr.write("No models found.\n")
        return

    # Default Console writes to stdout; colour is auto-disabled when
    # stdout is not a TTY (pipes, files, CI logs).
    console = Console()

    table = Table(title="Available Models", border_style="bright_blue")
    table.add_column("Provider", style="dim")
    # overflow="fold" wraps long raw model keys instead of truncating
    # them with an ellipsis (e.g. gemini-3.1-pro-preview-customtools).
    table.add_column("Model", style="bold", overflow="fold")
    table.add_column("Context", justify="right")
    table.add_column("Input $/M", justify="right")
    table.add_column("Cache Hit $/M", justify="right")
    table.add_column("Output $/M", justify="right")

    current_provider = None
    for e in entries:
        if e["provider"] != current_provider:
            current_provider = e["provider"]
            if table.row_count > 0:
                table.add_section()

        ctx = f"{e['context']:,}" if isinstance(e["context"], int) else "—"
        table.add_row(
            e["provider"],
            e["display"],
            ctx,
            _fmt_money(e["input_cost"]),
            _fmt_money(e.get("cache_read_cost")),
            _fmt_money(e["output_cost"]),
        )

    console.print()
    console.print(table)
    console.print()
