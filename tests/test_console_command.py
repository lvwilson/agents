"""Tests for run_console_command quote/backtick handling."""
import unittest

from agents.tools import functions


class TestRunConsoleCommand(unittest.TestCase):
    """Backtick-block scripts must reach the shell verbatim."""

    def test_backtick_content_preserves_escaped_quotes(self):
        script = 'echo "outer \\"inner\\" done"'
        out = functions.run_console_command("", backtick_content=script)
        self.assertIn('outer "inner" done', out)

    def test_multiline_positional_is_verbatim(self):
        out = functions.run_console_command('echo "a \\"b\\" c"\necho line2')
        self.assertIn('a "b" c', out)
        self.assertIn("line2", out)

    def test_legacy_single_line_unescapes(self):
        out = functions.run_console_command('"echo \\"hello\\""')
        self.assertIn("hello", out)

    def test_backtick_heredoc_intact(self):
        script = 'cat <<\'HD\'\nliteral $VAR and "quotes" and \\backslash\nHD'
        out = functions.run_console_command("", backtick_content=script)
        self.assertIn('literal $VAR and "quotes"', out)


if __name__ == "__main__":
    unittest.main()
