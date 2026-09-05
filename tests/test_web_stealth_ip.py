"""Tests for Phase 1 (IP protection) of the web browser stealth hardening:
fixed proxy (WEB_PROXY), rotating proxy pool (WEB_PROXY_FILE), opt-in
persistent profile (WEB_BROWSER_PROFILE), malformed-config fallbacks, the
profile/proxy-file conflict rule, and WEB_PROXY/DDGS_PROXY threading
into web_search.

Follows the mock patterns of tests/test_web_tools.py:
- fresh WebBrowser() instances per test,
- patch("agents.tools.web_browser.sync_playwright") for launch paths,
- patch.dict(os.environ, ..., clear=True) with explicit env dicts so
  nothing leaks into other tests.

Phase 2 (fingerprint hardening) updated these tests where the plan
prescribes new context/launch defaults: every context is now created
with the full fingerprint option dict (user agent, locale, timezone,
viewport, screen, device scale, Accept-Language) and the hardened
launch args.  The Phase 1 test cases below therefore pin
WEB_CHANNEL=chromium (bundled channel, exactly as this commit asserted)
and -- where launch args are asserted -- patch _running_as_root() to
make the args deterministic.  Fingerprint-specific assertions live in
tests/test_web_fingerprint.py.
"""

import os
import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agents.tools.web_browser import WebBrowser

try:
    import ddgs  # noqa: F401
    HAVE_DDGS = True
except ImportError:
    HAVE_DDGS = False

# The full set of stealth env vars -- tests start from a clean env that
# contains NONE of these unless they say otherwise (clear=True below).
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
    """Wire sync_playwright() -> playwright -> chromium -> browser.

    ``browser.new_context`` returns a fresh context+page on every call
    (side_effect), so rotation assertions can inspect each context's
    creation kwargs independently.
    """
    sync = MagicMock()
    playwright = MagicMock()
    chromium = MagicMock()
    browser = MagicMock()
    browser.is_connected.return_value = True

    def _new_context(**kwargs):
        ctx = MagicMock()
        ctx.is_closed.return_value = False
        ctx.new_page.return_value = _make_page()
        return ctx

    browser.new_context.side_effect = _new_context

    persistent_ctx = MagicMock()
    persistent_ctx.is_closed.return_value = False
    persistent_ctx.new_page.return_value = _make_page()
    chromium.launch_persistent_context.return_value = persistent_ctx

    chromium.launch.return_value = browser
    playwright.chromium = chromium
    playwright.stop.return_value = None
    # Code path is sync_playwright().start() -> Playwright.
    sync.return_value.start.return_value = playwright
    return sync, playwright, chromium, browser


def _expected_context_options(proxy=None):
    """The full Phase 2 context-option dict these launches now carry.

    Mirrors WebBrowser._get_context_options() at the Phase 2 defaults
    (locale en-US, Etc/UTC, auto UA) so Phase 1 tests keep asserting
    the exact call shape without re-deriving the UA template.
    """
    from agents.tools import web_browser as wbmod
    opts = {
        "user_agent": wbmod._build_auto_user_agent(),
        "locale": "en-US",
        "timezone_id": "Etc/UTC",
        "viewport": {"width": 1280, "height": 900},
        "screen": {"width": 1280, "height": 900},
        "device_scale_factor": 1,
        "extra_http_headers": {"Accept-Language": "en-US,en;q=0.9"},
    }
    if proxy is not None:
        opts["proxy"] = proxy
    return opts


def _expected_launch_args(root=False, stealth=True):
    """The Phase 2 launch args for the pinned (root, stealth) combo."""
    args = ["--no-sandbox"] if root else []
    args.append("--disable-gpu")
    if stealth:
        args.append("--disable-blink-features=AutomationControlled")
    return args


class TestFixedProxy(unittest.TestCase):
    """WEB_PROXY -> one context per session, created with the proxy dict."""

    def test_fixed_proxy_passes_exact_proxy_dict(self):
        sync, playwright, chromium, browser = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=False):
            with patch.dict(os.environ,
                            _clean_env(
                                WEB_PROXY="http://user:pass@p.example:3128",
                                WEB_CHANNEL="chromium"),
                            clear=True):
                wb = WebBrowser()
                result = wb.read_page("https://example.com")
                self.assertIn("Page body", result)
                browser.new_context.assert_called_once_with(
                    **_expected_context_options(
                        proxy={
                            "server": "http://p.example:3128",
                            "username": "user",
                            "password": "pass",
                        }
                    )
                )
                # The fixed proxy is created once, not per navigation.
                wb.read_page("https://example.com/again")
                self.assertEqual(browser.new_context.call_count, 1)
                # Launch: bundled channel, Phase 2 hardened args.
                chromium.launch.assert_called_once_with(
                    headless=True, channel=None,
                    args=_expected_launch_args(root=False))


class TestRotatingProxyFile(unittest.TestCase):
    """WEB_PROXY_FILE -> per-navigation round-robin context rebuild."""

    def _write_pool(self, root):
        pool_file = root / "proxies.txt"
        pool_file.write_text(
            "http://proxy-a.example:3128\n"
            "# a comment line that must be skipped\n"
            "\n"
            "http://proxy-b.example:8080\n"
        )
        return str(pool_file)

    def test_rotation_rebuilds_context_per_navigation(self):
        sync, playwright, chromium, browser = _make_playwright_mock()
        with tempfile.TemporaryDirectory() as tmp:
            pool_file = self._write_pool(pathlib.Path(tmp))
            with patch("agents.tools.web_browser.sync_playwright", sync):
                with patch.dict(os.environ,
                                _clean_env(WEB_PROXY_FILE=pool_file,
                                           WEB_CHANNEL="chromium"),
                                clear=True):
                    wb = WebBrowser()
                    result1 = wb.read_page("https://example.com")
                    self.assertIn("Page body", result1)
                    first = browser.new_context.call_args_list[0].kwargs.get("proxy")
                    self.assertEqual(first, {"server": "http://proxy-a.example:3128"})
                    self.assertEqual(wb._proxy_idx, 0)
                    self.assertEqual(wb._active_proxy,
                                     "http://proxy-a.example:3128")

                    # Second navigation: round-robin to the second proxy.
                    result2 = wb.read_page("https://example.com/other")
                    self.assertIn("Page body", result2)
                    self.assertEqual(browser.new_context.call_count, 2)
                    second = browser.new_context.call_args_list[1].kwargs.get("proxy")
                    self.assertEqual(second, {"server": "http://proxy-b.example:8080"})
                    self.assertEqual(wb._proxy_idx, 1)
                    self.assertEqual(wb._active_proxy,
                                     "http://proxy-b.example:8080")

                    # Third navigation wraps back to the first proxy.
                    wb.read_page("https://example.com/third")
                    self.assertEqual(browser.new_context.call_count, 3)
                    self.assertEqual(wb._proxy_idx, 0)

    def test_interactive_commands_do_not_rotate(self):
        sync, playwright, chromium, browser = _make_playwright_mock()
        with tempfile.TemporaryDirectory() as tmp:
            pool_file = self._write_pool(pathlib.Path(tmp))
            with patch("agents.tools.web_browser.sync_playwright", sync):
                with patch.dict(os.environ,
                                _clean_env(WEB_PROXY_FILE=pool_file,
                                           WEB_CHANNEL="chromium"),
                                clear=True):
                    wb = WebBrowser()
                    wb.browse_open("https://example.com")  # rotation #1
                    self.assertEqual(browser.new_context.call_count, 1)
                    wb.browse_read()  # interactive: no rotation
                    wb.browse_click("a#link")
                    wb.browse_type("#name", "x")
                    self.assertEqual(browser.new_context.call_count, 1)


class TestMalformedProxyConfig(unittest.TestCase):
    """Bad proxy config -> warn + direct egress, never crash."""

    def test_malformed_web_proxy_falls_back_to_direct(self):
        sync, playwright, chromium, browser = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_PROXY="notaurl",
                                       WEB_CHANNEL="chromium"),
                            clear=True):
                wb = WebBrowser()
                result = wb.read_page("https://example.com")
                self.assertIn("Page body", result)  # still functions
                self.assertIsNone(wb._cfg["proxy"])
                # Fingerprint options are still applied; no proxy kwarg.
                call = browser.new_context.call_args
                self.assertNotIn("proxy", call.kwargs)
                self.assertIn("user_agent", call.kwargs)

    def test_bare_new_context_when_no_proxy_configured(self):
        sync, playwright, chromium, browser = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
                call = browser.new_context.call_args
                # No proxy kwarg, but the full Phase 2 option dict.
                self.assertNotIn("proxy", call.kwargs)
                self.assertEqual(call.kwargs,
                                 _expected_context_options(proxy=None))


class TestPersistentProfile(unittest.TestCase):
    """WEB_BROWSER_PROFILE -> launch_persistent_context + single-handle close."""

    def test_persistent_profile_launch_and_close(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            profile_dir = pathlib.Path(tmp) / "profile"
            sync, playwright, chromium, browser = _make_playwright_mock()
            with patch("agents.tools.web_browser.sync_playwright", sync), \
                 patch("agents.tools.web_browser._running_as_root",
                       return_value=False):
                with patch.dict(os.environ,
                                _clean_env(WEB_BROWSER_PROFILE=str(profile_dir),
                                           WEB_CHANNEL="chromium"),
                                clear=True):
                    wb = WebBrowser()
                    result = wb.read_page("https://example.com")
                    self.assertIn("Page body", result)

                    chromium.launch_persistent_context.assert_called_once()
                    kwargs = chromium.launch_persistent_context.call_args.kwargs
                    self.assertEqual(kwargs["user_data_dir"], str(profile_dir))
                    self.assertTrue(kwargs["headless"])
                    self.assertEqual(kwargs["args"],
                                     _expected_launch_args(root=False))
                    self.assertEqual(kwargs["channel"], None)
                    self.assertNotIn("proxy", kwargs)  # no fixed proxy set
                    # Phase 2: the fingerprint options ride along at the
                    # launch level (persistent contexts take them there).
                    self.assertEqual(kwargs["user_agent"],
                                     _expected_context_options()["user_agent"])
                    self.assertEqual(kwargs["locale"], "en-US")
                    self.assertEqual(kwargs["timezone_id"], "Etc/UTC")

                    # The profile directory is (created) on disk.
                    self.assertTrue(profile_dir.is_dir())

                    # Persistent lifecycle: context closed EXACTLY once,
                    # playwright stopped EXACTLY once, no plain browser.
                    wb.close()

            persistent_ctx = chromium.launch_persistent_context.return_value
            self.assertEqual(persistent_ctx.close.call_count, 1)
            self.assertEqual(playwright.stop.call_count, 1)
            browser.launch.assert_not_called()

    def test_persistent_profile_default_location_for_one(self):
        sync, playwright, chromium, browser = _make_playwright_mock()
        with tempfile.TemporaryDirectory() as tmp:
            # HOME -> tmp so "~/.agents/browser_profile" resolves into the
            # sandbox (no real home-dir pollution from makedirs).
            with patch("agents.tools.web_browser.sync_playwright", sync):
                with patch.dict(os.environ,
                                _clean_env(WEB_BROWSER_PROFILE="1",
                                           WEB_CHANNEL="chromium",
                                           HOME=tmp), clear=True):
                    wb = WebBrowser()
                    wb.read_page("https://example.com")
                    kwargs = chromium.launch_persistent_context.call_args.kwargs
                    expected = os.path.expanduser("~/.agents/browser_profile")
                    self.assertEqual(kwargs["user_data_dir"], expected)
                    # Resolved into the sandbox HOME.
                    self.assertTrue(expected.startswith(tmp),
                                    f"{expected} should be under {tmp}")
                    wb.close()

    def test_profile_with_fixed_proxy_passes_proxy_to_launch(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            sync, playwright, chromium, browser = _make_playwright_mock()
            with patch("agents.tools.web_browser.sync_playwright", sync), \
                 patch("agents.tools.web_browser._running_as_root",
                       return_value=False):
                with patch.dict(os.environ, _clean_env(
                        WEB_BROWSER_PROFILE=str(pathlib.Path(tmp) / "p"),
                        WEB_CHANNEL="chromium",
                        WEB_PROXY="http://p.example:3128"), clear=True):
                    wb = WebBrowser()
                    wb.read_page("https://example.com")
                    kwargs = chromium.launch_persistent_context.call_args.kwargs
                    self.assertEqual(kwargs["proxy"],
                                     {"server": "http://p.example:3128"})
                    # Hardened launch args despite the profile/proxy combo.
                    self.assertEqual(kwargs["args"],
                                     _expected_launch_args(root=False))
                    wb.close()


class TestProxyFileProfileConflict(unittest.TestCase):
    """WEB_PROXY_FILE + WEB_BROWSER_PROFILE -> file wins, profile disabled."""

    def test_proxy_file_wins_over_profile(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pool_file = root / "proxies.txt"
            pool_file.write_text("http://proxy-a.example:3128\n")
            sync, playwright, chromium, browser = _make_playwright_mock()
            with patch("agents.tools.web_browser.sync_playwright", sync):
                with patch.dict(os.environ, _clean_env(
                        WEB_PROXY_FILE=str(pool_file),
                        WEB_CHANNEL="chromium",
                        WEB_BROWSER_PROFILE=str(root / "profile")), clear=True):
                    wb = WebBrowser()
                    result = wb.read_page("https://example.com")
                    self.assertIn("Page body", result)

                    # Persistent path NOT used; rotation path active.
                    chromium.launch_persistent_context.assert_not_called()
                    self.assertFalse(wb._persistent)
                    call = browser.new_context.call_args
                    self.assertEqual(call.kwargs,
                                     _expected_context_options(
                                         proxy={
                                             "server": "http://proxy-a.example:3128"}))
                    wb.close()


class TestWebSearchProxy(unittest.TestCase):
    """web_search threads WEB_PROXY / DDGS_PROXY into DDGS(proxy=...)."""

    @unittest.skipUnless(HAVE_DDGS, "ddgs package not installed")
    def test_web_proxy_threads_into_ddgs(self):
        from agents.tools.functions import web_search
        with patch("ddgs.DDGS") as mock_ddgs:
            inst = MagicMock()
            inst.text.return_value = [
                {"title": "T", "href": "https://example.com", "body": "B"}
            ]
            mock_ddgs.return_value = inst
            with patch.dict(os.environ,
                            _clean_env(WEB_PROXY="http://p.example:3128"),
                            clear=True):
                out = web_search("test query")
            mock_ddgs.assert_called_once_with(proxy="http://p.example:3128")
            self.assertIn("Search results for: test query", out)

    @unittest.skipUnless(HAVE_DDGS, "ddgs package not installed")
    def test_ddgs_proxy_fallback_when_no_web_proxy(self):
        from agents.tools.functions import web_search
        with patch("ddgs.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = []
            with patch.dict(os.environ,
                            _clean_env(DDGS_PROXY="http://legacy.example:9999"),
                            clear=True):
                web_search("test query")
            mock_ddgs.assert_called_once_with(proxy="http://legacy.example:9999")

    @unittest.skipUnless(HAVE_DDGS, "ddgs package not installed")
    def test_no_proxy_kwarg_when_both_unset(self):
        from agents.tools.functions import web_search
        with patch("ddgs.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text.return_value = []
            with patch.dict(os.environ, _clean_env(), clear=True):
                web_search("test query")
            mock_ddgs.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
