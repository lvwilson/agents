"""Tests for Phase 3 (behavioral pacing) of the web browser stealth
hardening (plan: untracked/web_stealth_plan.md, "Commit 3").

Covers:
- the jittered ``_pacing_wait()`` value (default 0.5 -> [0.25, 0.75],
  WEB_REQUEST_DELAY=0.2 -> [0.1, 0.3], WEB_REQUEST_DELAY=0 -> no sleep);
- pacing is called after a successful navigation (read_page / view_page)
  and before interactive actions (browse_click / browse_type), and NOT on
  ``browse_read`` (a human re-read stays instant -- double-read test);
- ``browse_type`` paces exactly ONCE (before the first real fill), even
  when the text flushes in several chunks (a[Tab]b[Enter]);
- the "reproduces today exactly" combo WEB_STEALTH=0 + WEB_REQUEST_DELAY=0
  + WEB_CHANNEL=chromium;
- a malformed WEB_REQUEST_DELAY falls back to the 0.5 default (never
  crashes the agent).

Follows the mock patterns of tests/test_web_fingerprint.py /
tests/test_web_tools.py:
- fresh WebBrowser() instances per test,
- patch("agents.tools.web_browser.sync_playwright") for launch paths,
- patch.dict(os.environ, ..., clear=True) so no env bleeds between tests,
- patch("agents.tools.web_browser.time") to capture sleeps (``random`` is
  left REAL so the recorded sleep value is a genuine ``uniform`` sample
  whose range we can assert).
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from agents.tools import web_browser as wbmod
from agents.tools.web_browser import WebBrowser

# Every env var _env_config() reads -- cleared so nothing leaks in.
_STEALTH_ENV_VARS = (
    "WEB_PROXY", "WEB_PROXY_FILE", "WEB_BROWSER_PROFILE", "WEB_CHANNEL",
    "WEB_USER_AGENT", "WEB_LOCALE", "WEB_TIMEZONE", "WEB_REQUEST_DELAY",
    "WEB_STEALTH", "DDGS_PROXY",
)


def _clean_env(**extra):
    """Env with all stealth vars removed plus any explicit *extra*."""
    env = {k: v for k, v in os.environ.items()
           if k not in _STEALTH_ENV_VARS}
    env.update(extra)
    return env


def _make_page(url="https://example.com", text="Page body"):
    """A Page mock that behaves like a live, open page."""
    page = MagicMock()
    page.is_closed.return_value = False
    page.url = url
    page.title.return_value = "Example"
    page.inner_text.return_value = text
    page.content.return_value = "<html><body>Page body</body></html>"
    return page


def _make_playwright_mock():
    """Wire sync_playwright() -> playwright -> chromium -> browser -> context.

    ``browser.new_context`` returns a FRESH context+page on every call;
    every created page is collected in ``created["pages"]``.
    """
    sync = MagicMock()
    playwright = MagicMock()
    chromium = MagicMock()
    browser = MagicMock()
    browser.is_connected.return_value = True
    created = {"contexts": [], "pages": []}

    def _new_page():
        page = _make_page()
        created["pages"].append(page)
        return page

    def _new_context(**kwargs):
        ctx = MagicMock()
        ctx.is_closed.return_value = False
        ctx.new_page.side_effect = _new_page
        created["contexts"].append(ctx)
        return ctx

    browser.new_context.side_effect = _new_context
    chromium.launch.return_value = browser
    playwright.chromium = chromium
    playwright.stop.return_value = None
    # Code path is sync_playwright().start() -> Playwright.
    sync.return_value.start.return_value = playwright
    return sync, playwright, chromium, browser, created


def _sleep_values(mock_time):
    """Lone positional (seconds) of every captured time.sleep(...) call."""
    return [c.args[0] for c in mock_time.sleep.call_args_list if c.args]


# ── Pacing value ────────────────────────────────────────────────────

class TestPacingValue(unittest.TestCase):
    """_pacing_wait() jitter range + the WEB_REQUEST_DELAY knob."""

    def test_default_delay_after_read_page_in_025_075(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ, _clean_env(WEB_CHANNEL="chromium"),
                            clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        # Exactly one pacing wait (the navigation read).
        vals = _sleep_values(mock_time)
        self.assertEqual(len(vals), 1)
        self.assertGreaterEqual(vals[0], 0.25)
        self.assertLessEqual(vals[0], 0.75)

    def test_delay_02_in_01_03(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ, _clean_env(
                    WEB_CHANNEL="chromium", WEB_REQUEST_DELAY="0.2"),
                    clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        vals = _sleep_values(mock_time)
        self.assertEqual(len(vals), 1)
        self.assertGreaterEqual(vals[0], 0.1)
        self.assertLessEqual(vals[0], 0.3)

    def test_disabled_delay_no_sleep_anywhere(self):
        """WEB_REQUEST_DELAY=0 -> no sleep in any navigation/action cmd."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ, _clean_env(
                    WEB_CHANNEL="chromium", WEB_REQUEST_DELAY="0"),
                    clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
                wb.browse_read()
                wb.browse_click("button#submit")
                wb.browse_type("#email", "user@test.com[Tab]")
        mock_time.sleep.assert_not_called()

    def test_invalid_delay_falls_back_to_default(self):
        """Bad WEB_REQUEST_DELAY -> warn + 0.5 default, never crash."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ, _clean_env(
                    WEB_CHANNEL="chromium",
                    WEB_REQUEST_DELAY="banana"), clear=True):
                wb = WebBrowser()
                # Navigation still succeeds (no crash).
                result = wb.read_page("https://example.com")
        self.assertIn("Page body", result)
        vals = _sleep_values(mock_time)
        self.assertEqual(len(vals), 1)
        self.assertGreaterEqual(vals[0], 0.25)
        self.assertLessEqual(vals[0], 0.75)


# ── Call sites ──────────────────────────────────────────────────────

class TestPacingCallSites(unittest.TestCase):
    """Where the wait lives: after goto / before action; not on read."""

    def test_view_page_paces_once_after_goto(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                rich_text, file_path = wb.view_page("https://example.com")
        self.assertIsNotNone(file_path)
        self.assertIn("Page body", rich_text)
        # Exactly one pacing wait, sized by the default delay (0.5).
        mock_time.sleep.assert_called_once()
        [val] = _sleep_values(mock_time)
        self.assertGreaterEqual(val, 0.25)
        self.assertLessEqual(val, 0.75)

    def test_browse_click_paces_before_click(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        order = []
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                page = wb.page  # the live page mock
                page.click.side_effect = lambda *a, **k: order.append("click")
                mock_time.sleep.side_effect = lambda s: order.append("sleep")
                wb.browse_click("button#submit")
        self.assertEqual(order, ["sleep", "click"])
        page.click.assert_called_once()

    def test_browse_type_multiple_chunks_one_wait(self):
        """a[Tab]b[Enter] flushes twice but paces only before the 1st fill."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.browse_type("#search", "a[Tab]b[Enter]")
        mock_time.sleep.assert_called_once()  # one wait, not two

    def test_browse_type_plain_text_one_wait(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.browse_type("#name", "John Doe")
        mock_time.sleep.assert_called_once()

    def test_browse_read_does_not_pace(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                # Establish a page first (this navigation does pace)...
                mock_time.reset_mock()
                # ...then a re-read of the CURRENT page paces nothing.
                wb.browse_read()
        mock_time.sleep.assert_not_called()

    def test_double_read_instant(self):
        """browse_open (1 wait) then two browse_reads (0 extra) => 1 total.

        The "already-loaded is instant" guarantee: the wait lives in the
        navigation entry point, not in read_text, so repeated reads of the
        current page add no sleep.
        """
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.browse_open("https://example.com")   # navigation -> wait
                wb.browse_read()                        # re-read -> no wait
                wb.browse_read()                        # re-read -> no wait
        mock_time.sleep.assert_called_once()


# ── "Reproduces today exactly" combo ────────────────────────────────

class TestReproducesTodayCombo(unittest.TestCase):
    """WEB_STEALTH=0 + WEB_REQUEST_DELAY=0 + WEB_CHANNEL=chromium.

    Honest claim (documented in the docstring): this combo removes the
    automation blink-feature flag, disables pacing entirely, and drives
    bundled Chromium (channel=None).  The pre-Phase-2 args were literally
    ``["--no-sandbox", "--disable-gpu"]`` for EVERY user; today --no-sandbox
    is root-only, so under non-root the code produces ``["--disable-gpu"]``
    which is *closer to* a normal desktop than the old blanket --no-sandbox.
    """

    def test_combo_non_root(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=False), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ, _clean_env(
                    WEB_CHANNEL="chromium",
                    WEB_STEALTH="0",
                    WEB_REQUEST_DELAY="0"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        kwargs = chromium.launch.call_args.kwargs
        self.assertEqual(kwargs["channel"], None)          # bundled
        self.assertEqual(kwargs["args"], ["--disable-gpu"])  # no blink flag
        self.assertNotIn("--no-sandbox", kwargs["args"])
        mock_time.sleep.assert_not_called()                # pacing off

    def test_combo_root_keeps_no_sandbox(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=True), \
             patch("agents.tools.web_browser.time") as mock_time:
            with patch.dict(os.environ, _clean_env(
                    WEB_CHANNEL="chromium",
                    WEB_STEALTH="0",
                    WEB_REQUEST_DELAY="0"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        kwargs = chromium.launch.call_args.kwargs
        self.assertEqual(kwargs["channel"], None)
        # Under root/CI --no-sandbox is normally required, so it stays:
        self.assertEqual(kwargs["args"],
                         ["--no-sandbox", "--disable-gpu"])
        mock_time.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
