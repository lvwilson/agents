"""Tests for --list-models: list_available_models + print_model_table."""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.backends import list_available_models  # noqa: E402
from agents.cli.model_table import print_model_table  # noqa: E402


class TestListAvailableModels(unittest.TestCase):
    """Metadata aggregation across all registered backends."""

    def test_aggregation_shape_and_inheritance(self):
        entries = list_available_models()
        providers = {e["provider"] for e in entries}
        self.assertEqual(
            providers, {"anthropic", "deepseek", "gemini", "kimi", "minimax", "openai"}
        )
        # Every entry exposes the documented keys.
        for e in entries:
            for key in ("provider", "model", "display", "input_cost",
                        "output_cost", "cache_read_cost", "context"):
                self.assertIn(key, e)
        # DeepSeekBackend subclasses AnthropicBackend — it must list
        # only its own models, not inherited Claude entries.
        deepseek = [e for e in entries if e["provider"] == "deepseek"]
        self.assertTrue(deepseek)
        self.assertTrue(all(e["model"].startswith("deepseek-") for e in deepseek))

    def test_provider_filter(self):
        entries = list_available_models("deepseek")
        self.assertTrue(entries)
        self.assertTrue(all(e["provider"] == "deepseek" for e in entries))

    def test_all_models_have_cache_read_cost(self):
        """Every model across all providers must have a positive cache_read_cost."""
        entries = list_available_models()
        for e in entries:
            self.assertIsNotNone(
                e["cache_read_cost"],
                f'{e["provider"]}/{e["model"]} missing cache_read_cost',
            )
            self.assertGreater(
                e["cache_read_cost"], 0,
                f'{e["provider"]}/{e["model"]} cache_read_cost is zero',
            )


class TestPrintModelTable(unittest.TestCase):
    """Stream routing: table on stdout, diagnostics on stderr."""

    def test_table_goes_to_stdout(self):
        # The original commit wrote to /dev/tty, which made the flag
        # unpipeable and silent in non-interactive environments.
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            print_model_table(None)
        text = out.getvalue()
        self.assertIn("Available Models", text)
        for provider in ("anthropic", "deepseek", "gemini", "kimi", "openai"):
            self.assertIn(provider, text)
        self.assertEqual(err.getvalue(), "")

    def test_unknown_provider_message_on_stderr(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            print_model_table("no-such-provider")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("no-such-provider", err.getvalue())


if __name__ == "__main__":
    unittest.main()
