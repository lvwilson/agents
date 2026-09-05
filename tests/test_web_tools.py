"""
Tests for the new web tool commands (core + interactive) and parser changes.

Tests use unittest.mock to avoid launching a real browser.
"""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import namedtuple

from agents.tools.web_browser import WebBrowser
from agents.tools.parser import filter_content, process_content, process_slice, CommandInfo, STACKABLE_READ_COMMANDS


# ── Helpers ─────────────────────────────────────────────────────────

def _make_browser_with_mocks():
    """Create a WebBrowser with mocked Playwright internals."""
    browser = WebBrowser()
    # Mock the page object
    mock_page = MagicMock()
    mock_page.url = "https://example.com"
    mock_page.title.return_value = "Example"
    mock_page.inner_text.return_value = "Hello World"
    mock_page.content.return_value = "<html><body>Hello World</body></html>"
    mock_page.is_closed.return_value = False
    mock_page.viewport_size = {"width": 1280, "height": 900}

    browser._page = mock_page
    browser._browser = MagicMock()
    browser._browser.is_connected.return_value = True
    browser._playwright = MagicMock()
    # This fixture wires browser state manually and bypasses _launch()
    # (where _cfg is set), so pin the Phase 3 pacing delay to 0 to keep
    # the tests from hitting the real jittered sleep.
    browser._cfg = {"request_delay": 0.0}

    return browser, mock_page


# ── Phase 1: Core tool tests ───────────────────────────────────────

class TestReadPage(unittest.TestCase):
    """Tests for WebBrowser.read_page()"""

    def test_read_page_combines_navigate_and_read(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.inner_text.return_value = "Page content here"
        mock_page.goto.return_value = MagicMock(status=200)

        result = browser.read_page("https://example.com")

        mock_page.goto.assert_called_once()
        self.assertIn("Page content here", result)

    def test_read_page_returns_error_on_navigation_failure(self):
        browser, mock_page = _make_browser_with_mocks()
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        mock_page.goto.side_effect = PlaywrightTimeout("Timeout")

        result = browser.read_page("https://slow-site.com")

        self.assertIn("Timeout", result)

    def test_read_page_with_selector(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.goto.return_value = MagicMock(status=200)
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Scoped content"
        mock_page.query_selector.return_value = mock_element

        result = browser.read_page("https://example.com", selector="#main")

        self.assertIn("Scoped content", result)


class TestReadPageHtml(unittest.TestCase):
    """Tests for WebBrowser.read_page_html()"""

    def test_read_page_html_combines_navigate_and_html(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.goto.return_value = MagicMock(status=200)
        mock_page.content.return_value = "<html><body>Test</body></html>"

        result = browser.read_page_html("https://example.com")

        mock_page.goto.assert_called_once()
        self.assertIn("<html>", result)

    def test_read_page_html_returns_error_on_navigation_failure(self):
        browser, mock_page = _make_browser_with_mocks()
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        mock_page.goto.side_effect = PlaywrightTimeout("Timeout")

        result = browser.read_page_html("https://slow-site.com")

        self.assertIn("Timeout", result)


class TestPageLinks(unittest.TestCase):
    """Tests for WebBrowser.page_links()"""

    def test_page_links_combines_navigate_and_links(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.goto.return_value = MagicMock(status=200)
        mock_page.eval_on_selector_all.return_value = [
            {"text": "Home", "href": "https://example.com/"},
            {"text": "About", "href": "https://example.com/about"},
        ]

        result = browser.page_links("https://example.com")

        mock_page.goto.assert_called_once()
        self.assertIn("Home", result)
        self.assertIn("About", result)
        self.assertIn("2 links", result)


class TestViewPage(unittest.TestCase):
    """Tests for WebBrowser.view_page()"""

    def test_view_page_includes_text_and_interactive_elements(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.goto.return_value = MagicMock(status=200)
        mock_page.inner_text.return_value = "Important page text"
        mock_page.evaluate.return_value = {
            "links": [{"text": "Click me", "href": "https://example.com/link", "selector": "a#link1"}],
            "buttons": [{"text": "Submit", "selector": "button#submit"}],
            "inputs": [{"type": "text", "name": "email", "placeholder": "Email", "value": "", "selector": "input#email"}],
            "selects": [],
            "textareas": []
        }
        mock_page.screenshot = MagicMock()

        rich_text, file_path = browser.view_page("https://example.com")

        self.assertIn("Important page text", rich_text)
        self.assertIn("Interactive Elements", rich_text)
        self.assertIn("Click me", rich_text)
        self.assertIn("Submit", rich_text)
        self.assertIn("email", rich_text)

    def test_view_page_returns_none_path_on_failure(self):
        browser, mock_page = _make_browser_with_mocks()
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        mock_page.goto.side_effect = PlaywrightTimeout("Timeout")

        rich_text, file_path = browser.view_page("https://slow-site.com")

        self.assertIsNone(file_path)
        self.assertIn("Timeout", rich_text)

    def test_view_page_generates_file_path(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.goto.return_value = MagicMock(status=200)
        mock_page.inner_text.return_value = "Text"
        mock_page.evaluate.return_value = {
            "links": [], "buttons": [], "inputs": [],
            "selects": [], "textareas": []
        }
        mock_page.screenshot = MagicMock()

        _, file_path = browser.view_page("https://example.com")

        self.assertIsNotNone(file_path)
        # TMPDIR-independent: check the basename prefix, not the absolute dir.
        import os
        self.assertTrue(os.path.basename(file_path).startswith("web_screenshot_"))
        self.assertTrue(file_path.endswith(".png"))


class TestGetInteractiveElements(unittest.TestCase):
    """Tests for WebBrowser.get_interactive_elements()"""

    def test_extracts_each_element_kind(self):
        """Each element kind is rendered with a count, label and selector."""
        cases = [
            ("links", [{"text": "Home", "href": "https://example.com/", "selector": "a#home"}],
             "Links (1)", "Home", "a#home"),
            ("buttons", [{"text": "Submit", "selector": "button#submit-btn"}],
             "Buttons (1)", "Submit", "button#submit-btn"),
            ("inputs", [{"type": "text", "name": "email", "placeholder": "Email", "value": "", "selector": "input#email"}],
             "Inputs (1)", "email", "input#email"),
        ]
        for kind, items, header, label, selector in cases:
            with self.subTest(kind=kind):
                browser, mock_page = _make_browser_with_mocks()
                mock_page.evaluate.return_value = {
                    "links": [], "buttons": [], "inputs": [],
                    "selects": [], "textareas": [], kind: items,
                }
                result = browser.get_interactive_elements()
                self.assertIn(header, result)
                self.assertIn(label, result)
                self.assertIn(selector, result)

    def test_returns_empty_string_when_no_elements(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.evaluate.return_value = {
            "links": [], "buttons": [], "inputs": [],
            "selects": [], "textareas": []
        }

        result = browser.get_interactive_elements()

        self.assertEqual(result, "")


# ── Phase 2: Interactive tool tests ─────────────────────────────────

class TestBrowseOpen(unittest.TestCase):
    """Tests for WebBrowser.browse_open()"""

    def test_browse_open_auto_reads(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.goto.return_value = MagicMock(status=200)
        mock_page.inner_text.return_value = "Welcome to the site"

        result = browser.browse_open("https://example.com")

        # Should return page text, not just navigation status
        self.assertIn("Welcome to the site", result)
        mock_page.goto.assert_called_once()


class TestBrowseClick(unittest.TestCase):
    """Tests for WebBrowser.browse_click()"""

    def test_browse_click_auto_reads(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.inner_text.return_value = "New page after click"

        result = browser.browse_click("button#submit")

        mock_page.click.assert_called_once_with("button#submit", timeout=5000)
        self.assertIn("New page after click", result)


class TestBrowseType(unittest.TestCase):
    """Tests for WebBrowser.browse_type()"""

    def test_browse_type_inline_enter(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.inner_text.return_value = "Search results"

        result = browser.browse_type("#search", "hello[Enter]")

        mock_page.fill.assert_called_once_with("#search", "hello", timeout=5000)
        mock_page.keyboard.press.assert_called_once_with("Enter")
        # Should auto-read because text ends with [Enter]
        self.assertIn("Search results", result)


    def test_browse_type_no_auto_read_on_tab(self):
        browser, mock_page = _make_browser_with_mocks()

        result = browser.browse_type("#email", "user@test.com[Tab]")

        mock_page.fill.assert_called_once_with("#email", "user@test.com", timeout=5000)
        mock_page.keyboard.press.assert_called_once_with("Tab")
        # Should NOT auto-read, just confirm
        self.assertIn("Typed into", result)

    def test_browse_type_plain_text(self):
        browser, mock_page = _make_browser_with_mocks()

        result = browser.browse_type("#name", "John Doe")

        mock_page.fill.assert_called_once_with("#name", "John Doe", timeout=5000)
        self.assertIn("Typed into", result)

    def test_browse_type_multiple_keys(self):
        browser, mock_page = _make_browser_with_mocks()

        result = browser.browse_type("#field", "text[Tab]more[Enter]")

        # Should have two fill calls and two key presses
        self.assertEqual(mock_page.fill.call_count, 2)
        self.assertEqual(mock_page.keyboard.press.call_count, 2)

    def test_browse_type_escape(self):
        browser, mock_page = _make_browser_with_mocks()

        result = browser.browse_type("#popup", "[Escape]")

        mock_page.keyboard.press.assert_called_once_with("Escape")
        # No text to fill, no Enter, so just confirmation
        self.assertIn("Typed into", result)


class TestBrowseRead(unittest.TestCase):
    """Tests for WebBrowser.browse_read()"""

    def test_browse_read_no_selector(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_page.inner_text.return_value = "Full page text"

        result = browser.browse_read()

        self.assertIn("Full page text", result)

    def test_browse_read_with_selector(self):
        browser, mock_page = _make_browser_with_mocks()
        mock_element = MagicMock()
        mock_element.inner_text.return_value = "Scoped text"
        mock_page.query_selector.return_value = mock_element
        mock_page.wait_for_selector.return_value = None

        result = browser.browse_read("#content")

        mock_page.wait_for_selector.assert_called_once()
        self.assertIn("Scoped text", result)


# ── Phase 1.4 / Phase 4.2: filter_content tests ────────────────────

class TestFilterContent(unittest.TestCase):
    """Tests for filter_content() bug fix and command stacking."""

    def test_filter_content_cuts_after_non_read(self):
        """A non-read command after read commands causes truncation."""
        content = (
            'Command: read_file file1.txt\n'
            'Command: run_console_command "ls"\n'
            'Command: read_file file2.txt\n'
        )
        result = filter_content(content)
        self.assertIn("file1.txt", result)
        # run_console_command should cause cut, so file2.txt should not be present
        self.assertNotIn("file2.txt", result)

    def test_filter_content_single_non_read_passes(self):
        """A single non-read command passes through unchanged."""
        content = 'Command: run_console_command "ls -l"\n'
        result = filter_content(content)
        self.assertIn("run_console_command", result)




# ── Phase 4.3: view_page parser image pipeline ─────────────────────

class TestViewPageParser(unittest.TestCase):
    """Tests for view_page handling in process_content()."""

    @patch('agents.tools.parser._execute_command')
    @patch('agents.tools.parser._load_and_resize_image')
    def test_view_page_image_display(self, mock_load_image, mock_execute):
        """process_content extracts file path from view_page tuple and loads image."""
        mock_execute.return_value = ("Page text content\nInteractive Elements:\n  ...", "/tmp/test_screenshot.png")
        mock_load_image.return_value = ("base64data", "image/png")

        with patch('os.path.exists', return_value=True):
            content = 'Command: view_page "https://example.com"\n'
            result_text, images = process_content(content)

        self.assertIn("Page text content", result_text)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0], ("base64data", "image/png"))

    @patch('agents.tools.parser._execute_command')
    def test_view_page_no_image_on_failure(self, mock_execute):
        """No image is loaded when view_page returns (error, None)."""
        mock_execute.return_value = ("Timeout navigating to https://slow.com", None)

        content = 'Command: view_page "https://slow.com"\n'
        result_text, images = process_content(content)

        self.assertIn("Timeout", result_text)
        self.assertEqual(len(images), 0)

    @patch('agents.tools.parser._execute_command')
    def test_view_page_plain_string_fallback(self, mock_execute):
        """If view_page returns a plain string (unexpected), handle gracefully."""
        mock_execute.return_value = "Some error string"

        content = 'Command: view_page "https://example.com"\n'
        result_text, images = process_content(content)

        self.assertIn("Some error string", result_text)
        self.assertEqual(len(images), 0)


# ── Phase 4.4: Command dispatch tests ──────────────────────────────

class TestCommandDispatch(unittest.TestCase):
    """Tests that new commands dispatch correctly via _execute_command."""

    def test_all_browser_commands_dispatch(self):
        """Every browser command reaches its method via _execute_command."""
        from agents.tools.parser import _execute_command
        commands = [
            ("read_page", '"https://example.com"', None),
            ("read_page_html", '"https://example.com"', None),
            ("page_links", '"https://example.com"', None),
            ("view_page", '"https://example.com"', None),
            ("browse_open", '"https://example.com"', None),
            ("browse_read", "", None),
            ("browse_click", '"button#submit"', None),
            ("browse_type", '"#input" "hello[Enter]"', None),
            ("browse_js", "", "return 40 + 2;"),
        ]
        for cmd_name, args, bt in commands:
            with patch("agents.tools.functions.get_browser") as mock_get_browser:
                mock_browser = MagicMock()
                mock_get_browser.return_value = mock_browser
                _execute_command(cmd_name, args, bt)
                # Each command should call exactly one browser method
                method_called = any(
                    getattr(mock_browser, m).called
                    for m in ("read_page", "read_page_html", "page_links", "view_page",
                               "browse_open", "browse_read", "browse_click",
                               "browse_type", "execute_js")
                )
                self.assertTrue(method_called, f"{cmd_name} did not call any browser method")



# ── STACKABLE_READ_COMMANDS constant test ───────────────────────────

class TestStackableCommands(unittest.TestCase):
    """Verify the STACKABLE_READ_COMMANDS set is correct."""

    def test_contains_expected_commands(self):
        expected = {'read_file', 'deep_read', 'read_page', 'read_page_html', 'page_links', 'view_page', 'web_search',
                    'request_approval'}
        self.assertEqual(STACKABLE_READ_COMMANDS, expected)

    def test_non_read_commands_not_stackable(self):
        non_stackable = ['run_console_command', 'write_file', 'browse_open',
                         'browse_click', 'browse_type', 'browse_js', 'browse_read']
        for cmd in non_stackable:
            self.assertNotIn(cmd, STACKABLE_READ_COMMANDS)


if __name__ == "__main__":
    unittest.main()
