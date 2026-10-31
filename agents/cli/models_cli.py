"""List models, save defaults, and manage catalogs without starting an agent."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..model_catalog import apply_catalog, read_catalog
from ..model_discovery import discover_models
from ..tools.default_model import add_default_model_arguments, save_default_model


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="List supported models, set defaults, and explicitly update the model catalog.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List effective offline metadata")
    listing.add_argument("--provider")
    listing.add_argument("--table", action="store_true",
                         help="Show a readable table instead of JSON")
    selection = commands.add_parser(
        "set-default", aliases=["set"], allow_abbrev=False,
        help="Save a default model for future sessions (global unless --project)",
    )
    add_default_model_arguments(selection)
    discover = commands.add_parser("discover", help="Query a hosted provider's model listing")
    discover.add_argument("provider")
    discover.add_argument("--model", action="append", default=[],
                          help="Exact model ID; repeat to select multiple models")
    discover.add_argument("--apply", action="store_true",
                          help="Apply selected IDs; requires --model")
    importing = commands.add_parser("import", help="Preview/import a reviewed version-1 JSON catalog")
    importing.add_argument("file", type=Path)
    importing.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in ("set-default", "set"):
            print(save_default_model(args))
            return 0
        if args.command == "list":
            if args.table:
                from .model_table import print_model_table
                print_model_table(args.provider)
                return 0
            from ..backends import list_available_models
            print(json.dumps(list_available_models(args.provider), indent=2))
            return 0
        if args.command == "discover":
            if args.apply and not args.model:
                parser.error("discover --apply requires at least one exact --model ID")
            records = discover_models(args.provider)
            if args.model:
                wanted = set(args.model)
                found = {r["model"] for r in records}
                if wanted - found:
                    raise ValueError("models absent from provider listing: " +
                                     ", ".join(sorted(wanted - found)))
                records = [r for r in records if r["model"] in wanted]
        else:
            records = read_catalog(args.file)
        if args.apply:
            path = apply_catalog(records)
            print(f"Applied {len(records)} model record(s) to {path}", file=sys.stderr)
        else:
            print("Preview only; use --apply to persist these records.", file=sys.stderr)
        print(json.dumps({"version": 1, "models": records}, indent=2, ensure_ascii=False))
        print(
            "Discovery confirms IDs, not inference compatibility or pricing. "
            "Missing metadata stays unknown; import verified metadata before budgeted use.",
            file=sys.stderr,
        )
        return 0
    except (ValueError, OSError) as exc:
        label = "Default model" if args.command in ("set-default", "set") else "Model catalog"
        print(f"{label} error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
