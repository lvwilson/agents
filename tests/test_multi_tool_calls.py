"""Tests for multiple tool call parsing and execution.

These tests verify that the parser correctly identifies and dispatches
multiple tool calls in a single LLM response, and identify current
limitations that need to be addressed as a feature.

Key areas tested:
- process_slice: correct extraction of commands + backtick blocks
- process_content: correct dispatch of multiple commands
- find_replace: multiple SEARCH/REPLACE blocks in one call
- filter_content: read-command stacking behavior
"""
import os
import tempfile
import unittest

from agents.tools.parser import process_slice, process_content, filter_content
from agents.tools.findreplace import find_replace


class TestProcessSliceSingleCommand(unittest.TestCase):
    """Baseline: single command with a backtick block."""

    def test_command_with_backtick_on_next_line(self):
        content = (
            "Command: find_and_replace file.py\n"
            "`````<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n`````"
        )
        cmd, args, backtick, remaining = process_slice(content)
        self.assertEqual(cmd, "find_and_replace")
        self.assertEqual(args, "file.py")
        self.assertIsNotNone(backtick)
        self.assertIn("<<<<<<< SEARCH", backtick)
        self.assertEqual(remaining, "")

    def test_command_without_backtick(self):
        content = "Command: read_file /path/to/file.py"
        cmd, args, backtick, remaining = process_slice(content)
        self.assertEqual(cmd, "read_file")
        self.assertEqual(args, "/path/to/file.py")
        self.assertIsNone(backtick)
        self.assertEqual(remaining, "")

    def test_command_with_backtick_and_language_hint(self):
        content = (
            "Command: write_file file.py\n"
            "`````Python\nprint('hello')\n`````"
        )
        cmd, args, backtick, remaining = process_slice(content)
        self.assertEqual(cmd, "write_file")
        self.assertEqual(backtick, "print('hello')\n")

    def test_no_command_returns_all_none(self):
        content = "Just some text without any commands."
        cmd, args, backtick, remaining = process_slice(content)
        self.assertIsNone(cmd)
        self.assertIsNone(args)
        self.assertIsNone(backtick)

    def test_leading_text_before_command_is_skipped(self):
        content = (
            "Here is what I found:\n\n"
            "Command: read_file /path/to/file.py"
        )
        cmd, args, backtick, remaining = process_slice(content)
        self.assertEqual(cmd, "read_file")
        self.assertEqual(args, "/path/to/file.py")


class TestProcessSliceMultipleCommands(unittest.TestCase):
    """Multiple commands in sequence, each with optional backtick blocks."""

    def test_two_commands_each_with_backtick(self):
        content = (
            "Command: find_and_replace file1.py\n"
            "`````<<<<<<< SEARCH\nold1\n=======\nnew1\n>>>>>>> REPLACE\n`````\n"
            "Command: find_and_replace file2.py\n"
            "`````<<<<<<< SEARCH\nold2\n=======\nnew2\n>>>>>>> REPLACE\n`````"
        )
        cmd1, args1, bt1, rem1 = process_slice(content)
        self.assertEqual(cmd1, "find_and_replace")
        self.assertEqual(args1, "file1.py")
        self.assertIn("old1", bt1)
        self.assertNotIn("old2", bt1)

        cmd2, args2, bt2, rem2 = process_slice(rem1)
        self.assertEqual(cmd2, "find_and_replace")
        self.assertEqual(args2, "file2.py")
        self.assertIn("old2", bt2)
        self.assertEqual(rem2, "")

    def test_three_commands_with_backticks(self):
        content = (
            "Command: write_file a.py\n"
            "`````text\ncontent_a\n`````\n"
            "Command: find_and_replace b.py\n"
            "`````text\nblock_b\n`````\n"
            "Command: read_file c.py\n"
            "`````text\nblock_c\n`````"
        )
        cmd1, _, bt1, rem1 = process_slice(content)
        self.assertEqual(cmd1, "write_file")
        self.assertEqual(bt1, "content_a\n")

        cmd2, _, bt2, rem2 = process_slice(rem1)
        self.assertEqual(cmd2, "find_and_replace")
        self.assertEqual(bt2, "block_b\n")

        cmd3, _, bt3, rem3 = process_slice(rem2)
        self.assertEqual(cmd3, "read_file")
        self.assertEqual(bt3, "block_c\n")
        self.assertEqual(rem3, "")

    def test_command_without_backtick_then_command_with_backtick(self):
        content = (
            "Command: read_file /path/to/file.py\n"
            "Command: find_and_replace file2.py\n"
            "`````<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n`````"
        )
        cmd1, args1, bt1, rem1 = process_slice(content)
        self.assertEqual(cmd1, "read_file")
        self.assertIsNone(bt1)

        cmd2, args2, bt2, rem2 = process_slice(rem1)
        self.assertEqual(cmd2, "find_and_replace")
        self.assertIn("old", bt2)

    def test_two_commands_without_backticks(self):
        content = (
            "Command: read_file /path1.py\n"
            "Command: read_file /path2.py"
        )
        cmd1, args1, bt1, rem1 = process_slice(content)
        self.assertEqual(cmd1, "read_file")
        self.assertEqual(args1, "/path1.py")
        self.assertIsNone(bt1)

        cmd2, args2, bt2, rem2 = process_slice(rem1)
        self.assertEqual(cmd2, "read_file")
        self.assertEqual(args2, "/path2.py")
        self.assertIsNone(bt2)
        self.assertEqual(rem2, "")

    def test_trailing_text_after_last_command(self):
        content = (
            "Command: read_file /path.py\n"
            "\nAll done!"
        )
        cmd, args, bt, rem = process_slice(content)
        self.assertEqual(cmd, "read_file")
        self.assertIn("All done", rem)


class TestProcessSliceBacktickAttachment(unittest.TestCase):
    """The parser requires the backtick to be on the line immediately
    after the command. A blank line causes the backtick to be ignored."""

    def test_backtick_attached_with_no_blank_line(self):
        content = (
            "Command: find_and_replace file.py\n"
            "`````block\n`````"
        )
        cmd, args, backtick, _ = process_slice(content)
        self.assertEqual(cmd, "find_and_replace")
        self.assertIsNotNone(backtick)

    def test_backtick_ignored_when_blank_line_separates(self):
        """LEGITIMATE FAILURE: A blank line between command and backtick
        causes the backtick block to be silently ignored."""
        content = (
            "Command: find_and_replace file.py\n"
            "\n"
            "`````<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n`````"
        )
        cmd, args, backtick, remaining = process_slice(content)
        self.assertEqual(cmd, "find_and_replace")
        self.assertIsNone(backtick)

    def test_backtick_attached_with_inline_space(self):
        content = (
            "Command: find_and_replace file.py \n"
            "`````block\n`````"
        )
        cmd, args, backtick, _ = process_slice(content)
        self.assertEqual(cmd, "find_and_replace")
        self.assertIsNotNone(backtick)


class TestProcessContentMultipleCommands(unittest.TestCase):
    """process_content should parse and execute every command in sequence."""

    def test_multiple_read_file_commands(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f1:
            f1.write("alpha")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f2:
            f2.write("beta")
            path2 = f2.name

        try:
            content = (
                f"Command: read_file {path1}\n"
                f"Command: read_file {path2}"
            )
            result, images = process_content(content)
            self.assertIn("alpha", result)
            self.assertIn("beta", result)
            self.assertEqual(len(images), 0)
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_multiple_find_and_replace_on_same_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f:
            f.write("def foo():\n    pass\n\ndef bar():\n    pass\n")
            path = f.name

        try:
            content = (
                f"Command: find_and_replace {path}\n"
                "`````<<<<<<< SEARCH\ndef foo():\n    pass\n"
                "=======\ndef foo():\n    return 1\n"
                ">>>>>>> REPLACE\n`````\n"
                f"Command: find_and_replace {path}\n"
                "`````<<<<<<< SEARCH\ndef bar():\n    pass\n"
                "=======\ndef bar():\n    return 2\n"
                ">>>>>>> REPLACE\n`````"
            )
            result, _ = process_content(content)
            self.assertIn("successfully written", result)

            with open(path) as f:
                final = f.read()
            self.assertIn("return 1", final)
            self.assertIn("return 2", final)
        finally:
            os.unlink(path)

    def test_write_file_then_find_and_replace(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f:
            f.write("")
            path = f.name

        try:
            content = (
                f"Command: write_file {path}\n"
                "`````def hello():\n    print('hello')\n`````\n"
                f"Command: find_and_replace {path}\n"
                "`````<<<<<<< SEARCH\nprint('hello')\n"
                "=======\nprint('world')\n"
                ">>>>>>> REPLACE\n`````"
            )
            result, _ = process_content(content)

            with open(path) as f:
                final = f.read()
            self.assertIn("print('world')", final)
        finally:
            os.unlink(path)

    def test_multiple_find_and_replace_on_different_files(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f1:
            f1.write("OLD_A")
            path1 = f1.name
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                         delete=False) as f2:
            f2.write("OLD_B")
            path2 = f2.name

        try:
            content = (
                f"Command: find_and_replace {path1}\n"
                "`````<<<<<<< SEARCH\nOLD_A\n=======\nNEW_A\n"
                ">>>>>>> REPLACE\n`````\n"
                f"Command: find_and_replace {path2}\n"
                "`````<<<<<<< SEARCH\nOLD_B\n=======\nNEW_B\n"
                ">>>>>>> REPLACE\n`````"
            )
            result, _ = process_content(content)

            with open(path1) as f:
                self.assertIn("NEW_A", f.read())
            with open(path2) as f:
                self.assertIn("NEW_B", f.read())
        finally:
            os.unlink(path1)
            os.unlink(path2)

    def test_mixed_read_and_write_commands(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                         delete=False) as f:
            f.write("original")
            path = f.name

        try:
            content = (
                f"Command: read_file {path}\n"
                f"Command: write_file {path}\n"
                "`````text\nupdated\n`````\n"
                f"Command: read_file {path}"
            )
            result, _ = process_content(content)
            self.assertIn("original", result)
            self.assertIn("updated", result)
        finally:
            os.unlink(path)


class TestFindReplaceMultipleBlocks(unittest.TestCase):
    """findreplace.find_replace() uses re.search (single match), so only
    the first SEARCH/REPLACE block in a command string is processed.
    Additional blocks are silently ignored."""

    def setUp(self):
        self.source = (
            "def func_a():\n"
            "    return 1\n"
            "\n"
            "def func_b():\n"
            "    return 2\n"
            "\n"
            "def func_c():\n"
            "    return 3\n"
        )

    def test_single_block_replaces_correctly(self):
        command = (
            "<<<<<<< SEARCH\n"
            "    return 1\n"
            "=======\n"
            "    return 10\n"
            ">>>>>>> REPLACE"
        )
        result = find_replace(self.source, command)
        self.assertIn("return 10", result)
        self.assertIn("return 2", result)
        self.assertIn("return 3", result)

    def test_multiple_blocks_only_first_processed(self):
        """LEGITIMATE FAILURE: With two SEARCH/REPLACE blocks in one
        command string, only the first block is applied."""
        command = (
            "<<<<<<< SEARCH\n"
            "    return 1\n"
            "=======\n"
            "    return 10\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "    return 2\n"
            "=======\n"
            "    return 20\n"
            ">>>>>>> REPLACE"
        )
        result = find_replace(self.source, command)
        self.assertIn("return 10", result)
        self.assertIn("return 2", result)
        self.assertNotIn("return 20", result)

    def test_three_blocks_only_first_processed(self):
        """All three SEARCH/REPLACE blocks in one command are applied."""
        command = (
            "<<<<<<< SEARCH\n"
            "    return 1\n"
            "=======\n"
            "    return 10\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "    return 2\n"
            "=======\n"
            "    return 20\n"
            ">>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\n"
            "    return 3\n"
            "=======\n"
            "    return 30\n"
            ">>>>>>> REPLACE"
        )
        result = find_replace(self.source, command)
        self.assertIn("return 10", result)
        self.assertIn("return 20", result)
        self.assertIn("return 30", result)

    def test_multiple_blocks_with_overlapping_search_regions(self):
        source = "one two three four"
        command = (
            "<<<<<<< SEARCH\none\n=======\nONE\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\ntwo\n=======\nTWO\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\nthree\n=======\nTHREE\n>>>>>>> REPLACE\n"
            "<<<<<<< SEARCH\nfour\n=======\nFOUR\n>>>>>>> REPLACE"
        )
        result = find_replace(source, command)
        self.assertIn("ONE", result)
        self.assertIn("TWO", result)
        self.assertIn("THREE", result)
        self.assertIn("FOUR", result)


class TestFilterContent(unittest.TestCase):
    """filter_content should pass through consecutive read commands and
    stop at the first non-read command that follows them."""

    def test_all_read_commands_passed_through(self):
        content = (
            "Command: read_file /path1.py\n"
            "Command: read_file /path2.py"
        )
        result = filter_content(content)
        self.assertEqual(result, content)

    def test_stops_at_non_read_after_reads(self):
        content = (
            "Command: read_file /path1.py\n"
            "Command: read_file /path2.py\n"
            "Command: find_and_replace /path3.py\n"
            "`````block\n`````\n"
            "Command: read_file /path4.py"
        )
        result = filter_content(content)
        self.assertIn("read_file /path1.py", result)
        self.assertIn("read_file /path2.py", result)
        self.assertNotIn("find_and_replace /path3.py", result)
        self.assertNotIn("read_file /path4.py", result)

    def test_deep_read_is_stackable(self):
        content = (
            "Command: deep_read read_file /path1.py\n"
            "Command: read_page https://example.com"
        )
        result = filter_content(content)
        self.assertEqual(result, content)

    def test_non_read_first_then_read(self):
        content = (
            "Command: write_file /path.py\n"
            "`````content\n`````\n"
            "Command: read_file /other.py"
        )
        result = filter_content(content)
        self.assertIn("write_file", result)
        self.assertIn("read_file", result)

    def test_mixed_stackable_commands(self):
        content = (
            "Command: read_file /path1.py\n"
            'Command: web_search "query"\n'
            "Command: read_page_html https://example.com"
        )
        result = filter_content(content)
        self.assertIn("read_file", result)
        self.assertIn("web_search", result)
        self.assertIn("read_page_html", result)


class TestProcessSliceEdgeCases(unittest.TestCase):
    """Edge cases for command + backtick parsing."""

    def test_empty_backtick_block(self):
        content = "Command: stdout\n`````\n`````"
        cmd, args, backtick, remaining = process_slice(content)
        self.assertEqual(cmd, "stdout")
        self.assertEqual(backtick, "")

    def test_command_with_quoted_arguments(self):
        content = (
            'Command: run_console_command "echo hello"\n'
            "`````multiline script\n`````"
        )
        cmd, args, backtick, remaining = process_slice(content)
        self.assertEqual(cmd, "run_console_command")
        self.assertEqual(args, '"echo hello"')
        self.assertIsNotNone(backtick)

    def test_backtick_content_with_nested_backticks(self):
        content = (
            "Command: write_file code.py\n"
            "`````Python\n"
            "x = 1  # ```` inside\n"
            "`````"
        )
        cmd, args, backtick, remaining = process_slice(content)
        self.assertEqual(cmd, "write_file")
        self.assertIn("```` inside", backtick)

    def test_multiple_commands_with_trailing_text(self):
        content = (
            "Command: read_file /a.py\n"
            "Command: read_file /b.py\n"
            "\nAll done. No more commands."
        )
        cmd1, args1, _, rem1 = process_slice(content)
        self.assertEqual(cmd1, "read_file")
        self.assertEqual(args1, "/a.py")

        cmd2, args2, _, rem2 = process_slice(rem1)
        self.assertEqual(cmd2, "read_file")
        self.assertEqual(args2, "/b.py")

        cmd3, _, _, _ = process_slice(rem2)
        self.assertIsNone(cmd3)


if __name__ == "__main__":
    unittest.main()
