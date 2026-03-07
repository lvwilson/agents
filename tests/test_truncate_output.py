"""Tests for the output safety truncation system in parser.py."""

import pytest

from agents.tools.parser import truncate_output, _TRUNCATE_THRESHOLD, _TRUNCATE_KEEP


class TestTruncateOutput:
    """Tests for the truncate_output function."""

    def test_short_string_unchanged(self):
        """Strings under the threshold should pass through unchanged."""
        text = "Hello, world!"
        assert truncate_output(text) == text

    def test_exactly_at_threshold_unchanged(self):
        """A string exactly at the threshold should pass through unchanged."""
        text = "x" * _TRUNCATE_THRESHOLD
        assert truncate_output(text) == text

    def test_one_over_threshold_triggers_truncation(self):
        """A string one character over the threshold should be truncated."""
        text = "x" * (_TRUNCATE_THRESHOLD + 1)
        result = truncate_output(text)
        assert "[OUTPUT TRUNCATED" in result
        assert "[END OF TRUNCATED OUTPUT" in result
        # The actual content kept is 60,000 chars (30k head + 30k tail)
        # plus the notice text — so for barely-over-threshold inputs
        # the result may be slightly longer due to notice overhead.
        # For large inputs, the result will be much smaller.

    def test_empty_string_unchanged(self):
        """Empty string should pass through unchanged."""
        assert truncate_output("") == ""

    def test_non_string_unchanged(self):
        """Non-string values should pass through unchanged."""
        assert truncate_output(123) == 123
        assert truncate_output(None) is None
        assert truncate_output([1, 2, 3]) == [1, 2, 3]

    def test_truncated_output_contains_head(self):
        """The truncated output should start with the first 30,000 characters."""
        # Create a string with identifiable head and tail
        head = "HEAD" * 10_000  # 40,000 chars — first 30k will be kept
        tail = "TAIL" * 10_000  # 40,000 chars — last 30k will be kept
        text = head + tail  # 80,000 chars total

        result = truncate_output(text)

        # First 30,000 chars should be from the head
        expected_head = text[:_TRUNCATE_KEEP]
        assert result.startswith(expected_head)

    def test_truncated_output_contains_tail(self):
        """The truncated output should end with the last 30,000 characters (plus end notice)."""
        head = "A" * 40_000
        tail = "Z" * 40_000
        text = head + tail  # 80,000 chars

        result = truncate_output(text)

        expected_tail = text[-_TRUNCATE_KEEP:]
        # The tail should appear before the end notice
        assert expected_tail in result

    def test_truncated_output_has_middle_notice(self):
        """The truncated output should contain a middle clipping notice."""
        text = "x" * 100_000
        result = truncate_output(text)
        assert "[OUTPUT TRUNCATED" in result
        assert "characters clipped from middle" in result
        assert "100,000 total characters" in result

    def test_truncated_output_has_end_notice(self):
        """The truncated output should end with an end-of-truncation notice."""
        text = "x" * 100_000
        result = truncate_output(text)
        assert result.endswith("[END OF TRUNCATED OUTPUT — Showed first 30,000 and last 30,000 of 100,000 total characters]")

    def test_clipped_count_is_correct(self):
        """The number of clipped characters reported should be accurate."""
        total = 90_000
        text = "x" * total
        expected_clipped = total - (_TRUNCATE_KEEP * 2)  # 90000 - 60000 = 30000

        result = truncate_output(text)
        assert f"{expected_clipped:,} characters clipped" in result

    def test_preserves_content_boundaries(self):
        """Verify that the exact first 30k and last 30k characters are preserved."""
        # Build a string where each position is identifiable
        # Use a repeating pattern of digits
        pattern = "0123456789"
        repeats = 10_000  # 100,000 chars total
        text = pattern * repeats

        result = truncate_output(text)

        # Extract the head (first 30,000 chars of result)
        result_head = result[:_TRUNCATE_KEEP]
        expected_head = text[:_TRUNCATE_KEEP]
        assert result_head == expected_head

        # The tail in the result is the last 30,000 chars of original text
        expected_tail = text[-_TRUNCATE_KEEP:]
        # Find where the tail starts in the result (after middle notice)
        tail_marker = "...\n\n"
        middle_notice_end = result.index(tail_marker, _TRUNCATE_KEEP) + len(tail_marker)
        end_notice_start = result.rindex("\n\n[END OF TRUNCATED OUTPUT")
        result_tail = result[middle_notice_end:end_notice_start]
        assert result_tail == expected_tail

    def test_large_output_size_is_reasonable(self):
        """The truncated output should be roughly 60k + notices, not the original size."""
        text = "x" * 1_000_000  # 1 million chars
        result = truncate_output(text)

        # Should be approximately 60k + ~200 chars of notices
        assert len(result) < 65_000
        assert len(result) > 60_000

    def test_unicode_content(self):
        """Truncation should work correctly with unicode content."""
        text = "🎉" * 80_000  # Each emoji is 1 Python char but multi-byte in UTF-8
        result = truncate_output(text)
        assert "[OUTPUT TRUNCATED" in result
        assert result.startswith("🎉" * _TRUNCATE_KEEP)  # First 30k emojis


class TestTruncateInExecuteCommand:
    """Integration-style tests verifying truncation works through _execute_command."""

    def test_read_file_truncation(self, tmp_path):
        """Reading a large file should produce truncated output."""
        from agents.tools.parser import _execute_command

        # Create a large file
        large_file = tmp_path / "large.txt"
        content = "x" * 100_000
        large_file.write_text(content)

        result = _execute_command("read_file", str(large_file), None)
        assert "[OUTPUT TRUNCATED" in result
        assert "[END OF TRUNCATED OUTPUT" in result

    def test_small_file_no_truncation(self, tmp_path):
        """Reading a small file should not trigger truncation."""
        from agents.tools.parser import _execute_command

        small_file = tmp_path / "small.txt"
        content = "Hello, world!"
        small_file.write_text(content)

        result = _execute_command("read_file", str(small_file), None)
        assert result == content
        assert "[OUTPUT TRUNCATED" not in result
