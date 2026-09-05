"""Tests for Phase 2 (fingerprint hardening) of the web browser stealth
hardening (plan: untracked/web_stealth_plan.md, "Commit 2").

Covers:
- context options on every launch path (auto UA / WEB_USER_AGENT,
  WEB_LOCALE, WEB_TIMEZONE, viewport-at-creation, screen,
  device_scale_factor, Accept-Language) -- defaults and overrides,
  including the persistent-context path;
- launch args: --disable-blink-features=AutomationControlled gated by
  WEB_STEALTH, --no-sandbox only under root (euid 0), --disable-gpu kept;
- WEB_CHANNEL "auto"/"chrome" trying the installed Chrome first with a
  warning + transparent fallback to bundled Chromium on launch failure
  (mocked), plus the explicit-channel pass-throughs;
- the vendored _STEALTH_INIT_JS: add_init_script called exactly once per
  page creation (fresh context + proxy-rotation pages), skipped when
  WEB_STEALTH=0, never crashing on failure; content assertions
  (webdriver defineProperty, plugins/mimeTypes, window.chrome,
  permissions prompt, languages pinned to the locale, ANGLE/SwiftShader
  pair);
- the auto UA derives from _CHROME_VERSION and never contains
  "HeadlessChrome".

Follows the mock patterns of tests/test_web_tools.py /
tests/test_web_stealth_ip.py:
- fresh WebBrowser() instances per test,
- patch("agents.tools.web_browser.sync_playwright") for launch paths,
- patch.dict(os.environ, ..., clear=True) with explicit env dicts so
  nothing leaks into other tests.
"""

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from agents.tools import web_browser as wbmod
from agents.tools.web_browser import WebBrowser, _build_auto_user_agent

_STEALTH_ENV_VARS = (
    "WEB_PROXY", "WEB_PROXY_FILE", "WEB_BROWSER_PROFILE", "WEB_CHANNEL",
    "WEB_USER_AGENT", "WEB_LOCALE", "WEB_TIMEZONE", "WEB_REQUEST_DELAY",
    "WEB_STEALTH", "DDGS_PROXY",
)


def _clean_env(**extra):
    """Env with all stealth vars removed plus any explicit *extra*."""
    env = {k: v for k, v in os.environ.items()
           if k not in _STEALTH_ENV_VARS}
    # Keep these launch-path tests fast: without a WEB_REQUEST_DELAY the
    # navigations would hit the real (jittered) pacing sleep.
    env["WEB_REQUEST_DELAY"] = "0"
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

    ``browser.new_context`` returns a FRESH context+page on every call
    (the side_effect below), so tests can inspect each context's
    creation kwargs AND each created page independently.  Every context
    and page created is collected in the returned ``created`` dict::

        created["contexts"]  -- list of BrowserContext mocks (launch order)
        created["pages"]     -- list of Page mocks (creation order)
        created["persistent_pages"] -- pages created by the persistent ctx
    """
    sync = MagicMock()
    playwright = MagicMock()
    chromium = MagicMock()
    browser = MagicMock()
    browser.is_connected.return_value = True
    created = {"contexts": [], "pages": [], "persistent_pages": []}

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

    persistent_ctx = MagicMock()
    persistent_ctx.is_closed.return_value = False

    def _persistent_new_page():
        page = _make_page()
        created["persistent_pages"].append(page)
        return page

    persistent_ctx.new_page.side_effect = _persistent_new_page
    chromium.launch_persistent_context.return_value = persistent_ctx

    chromium.launch.return_value = browser
    playwright.chromium = chromium
    playwright.stop.return_value = None
    # Code path is sync_playwright().start() -> Playwright.
    sync.return_value.start.return_value = playwright
    return sync, playwright, chromium, browser, created


def _expected_ctx_opts(user_agent=None, locale="en-US", timezone="Etc/UTC",
                       proxy=None):
    """The full Phase 2 context-option dict for the given overrides.

    Mirrors ``WebBrowser._get_context_options()`` so launch tests assert
    the exact call shape with a single source of truth for the shape.
    """
    if user_agent is None:
        user_agent = _build_auto_user_agent()
    if "-" in locale:
        accept_language = f"{locale},{locale.split('-', 1)[0]};q=0.9"
    else:
        accept_language = locale
    opts = {
        "user_agent": user_agent,
        "locale": locale,
        "timezone_id": timezone,
        "viewport": {"width": 1280, "height": 900},
        "screen": {"width": 1280, "height": 900},
        "device_scale_factor": 1,
        "extra_http_headers": {"Accept-Language": accept_language},
    }
    if proxy is not None:
        opts["proxy"] = proxy
    return opts


# ── Auto user agent ────────────────────────────────────────────────

class TestAutoUserAgent(unittest.TestCase):
    """_build_auto_user_agent(): real Chrome UA, version from the constant."""

    def test_linux_platform_template(self):
        with patch.object(wbmod, "sys") as mock_sys:
            mock_sys.platform = "linux"
            ua = _build_auto_user_agent()
        self.assertEqual(
            ua,
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

    def test_darwin_platform_template(self):
        with patch.object(wbmod, "sys") as mock_sys:
            mock_sys.platform = "darwin"
            ua = _build_auto_user_agent()
        self.assertIn("Macintosh; Intel Mac OS X 10_15_7", ua)
        self.assertIn("Chrome/138.0.0.0", ua)

    def test_windows_platform_template(self):
        with patch.object(wbmod, "sys") as mock_sys:
            mock_sys.platform = "win32"
            ua = _build_auto_user_agent()
        self.assertIn("Windows NT 10.0; Win64; x64", ua)
        self.assertIn("Chrome/138.0.0.0", ua)

    def test_unknown_platform_falls_back_to_windows_template(self):
        with patch.object(wbmod, "sys") as mock_sys:
            mock_sys.platform = "sunos5"
            ua = _build_auto_user_agent()
        self.assertIn("Windows NT 10.0", ua)

    def test_never_contains_headlesschrome(self):
        with patch.object(wbmod, "sys") as mock_sys:
            mock_sys.platform = "linux"
            ua = _build_auto_user_agent()
        self.assertNotIn("HeadlessChrome", ua)

    def test_version_constant_single_source(self):
        """Bumping _CHROME_VERSION is the one place the UA version changes."""
        with patch.object(wbmod.sys, "platform", "linux"), \
             patch.object(wbmod, "_CHROME_VERSION", "129.0.0.0"):
            ua = _build_auto_user_agent()
        self.assertIn("Chrome/129.0.0.0", ua)

    def test_cross_product_consistency(self):
        """A Chrome UA must never claim a WebKit-only engine
        ("Version/" token belongs to Safari; --disable-gpu + the
        ANGLE/SwiftShader pair belong to Chromium)."""
        for platform in ("linux", "darwin", "win32", "sunos5"):
            with self.subTest(platform=platform):
                with patch.object(wbmod.sys, "platform", platform):
                    self.assertNotIn("Version/", _build_auto_user_agent())


# ── Context options on every launch path ───────────────────────────

class TestContextOptions(unittest.TestCase):
    """Fingerprint context options at creation, on all paths."""

    def test_default_options_on_fixed_context_path(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        browser.new_context.assert_called_once_with(**_expected_ctx_opts())

    def test_options_with_env_overrides(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ, _clean_env(
                    WEB_CHANNEL="chromium",
                    WEB_USER_AGENT="Custom/1.0 TestAgent",
                    WEB_LOCALE="pt-BR",
                    WEB_TIMEZONE="America/Sao_Paulo"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        browser.new_context.assert_called_once_with(**_expected_ctx_opts(
            user_agent="Custom/1.0 TestAgent",
            locale="pt-BR",
            timezone="America/Sao_Paulo"))
        headers = browser.new_context.call_args.kwargs[
            "extra_http_headers"]
        self.assertEqual(headers["Accept-Language"], "pt-BR,pt;q=0.9")

    def test_options_on_rotating_proxy_contexts(self):
        """Every rotated context carries the fingerprint options too."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with tempfile.TemporaryDirectory() as tmp:
            pool = pathlib.Path(tmp) / "proxies.txt"
            pool.write_text(
                "http://proxy-a.example:3128\n"
                "http://proxy-b.example:8080\n")
            with patch("agents.tools.web_browser.sync_playwright", sync):
                with patch.dict(os.environ,
                                _clean_env(WEB_PROXY_FILE=str(pool),
                                           WEB_CHANNEL="chromium"),
                                clear=True):
                    wb = WebBrowser()
                    wb.read_page("https://example.com")
                    wb.read_page("https://example.com/other")
                self.assertEqual(browser.new_context.call_count, 2)
                # Rotation tears contexts down via close() only --
                # BrowserContext has no is_closed() (live Playwright),
                # so this must never be called on the context mocks.
                for ctx in created["contexts"]:
                    ctx.is_closed.assert_not_called()
                for idx, server in enumerate(
                        ("http://proxy-a.example:3128",
                         "http://proxy-b.example:8080")):
                    call = browser.new_context.call_args_list[idx]
                    expected = _expected_ctx_opts(
                        proxy={"server": server})
                    self.assertEqual(call.kwargs, expected,
                                     f"context rebuild #{idx + 1}")

    def test_options_ride_along_on_persistent_launch(self):
        """launch_persistent_context accepts the options at launch level."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with tempfile.TemporaryDirectory() as tmp:
            profile = pathlib.Path(tmp) / "profile"
            with patch("agents.tools.web_browser.sync_playwright", sync):
                with patch.dict(os.environ,
                                _clean_env(WEB_BROWSER_PROFILE=str(profile),
                                           WEB_CHANNEL="chromium",
                                           WEB_LOCALE="pt-BR",
                                           WEB_TIMEZONE="America/Sao_Paulo"),
                                clear=True):
                    wb = WebBrowser()
                    wb.read_page("https://example.com")
                    wb.close()
        call = chromium.launch_persistent_context.call_args
        for key, value in _expected_ctx_opts(
                locale="pt-BR", timezone="America/Sao_Paulo").items():
            self.assertEqual(call.kwargs.get(key), value,
                             f"persistent launch kwarg {key!r}")

    def test_viewport_at_creation_not_set_after(self):
        """Viewport is a context-creation kwarg; set_viewport_size gone."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        self.assertEqual(browser.new_context.call_args.kwargs["viewport"],
                         {"width": 1280, "height": 900})
        for i, page in enumerate(created["pages"]):
            with self.subTest(page=i):
                page.set_viewport_size.assert_not_called()


# ── Launch args ─────────────────────────────────────────────────────

class TestLaunchArgs(unittest.TestCase):
    """WEB_STEALTH-gated args, root-only --no-sandbox, --disable-gpu kept."""

    def _launch_kwargs(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        return chromium.launch.call_args.kwargs

    def test_args_steadon_non_root(self):
        with patch("agents.tools.web_browser._running_as_root",
                   return_value=False):
            args = self._launch_kwargs()["args"]
        self.assertEqual(
            args,
            ["--disable-gpu", "--disable-blink-features=AutomationControlled"])
        self.assertNotIn("--no-sandbox", args)

    def test_args_steadon_root(self):
        with patch("agents.tools.web_browser._running_as_root",
                   return_value=True):
            args = self._launch_kwargs()["args"]
        self.assertEqual(
            args,
            ["--no-sandbox", "--disable-gpu",
             "--disable-blink-features=AutomationControlled"])

    def test_args_steadoff(self):
        """WEB_STEALTH=0 -> no automation flag (pre-Phase-2 args shape)."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=False):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium",
                                       WEB_STEALTH="0"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        args = chromium.launch.call_args.kwargs["args"]
        self.assertEqual(args, ["--disable-gpu"])
        self.assertEqual(wb._cfg["stealth"], False)


# ── Channel selection ───────────────────────────────────────────────

class TestChannelSelection(unittest.TestCase):
    """auto/chrome: Chrome first, bundled fallback; chromium: pass-through."""

    def test_auto_tries_chrome_first(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=False):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="auto"), clear=True):
                wb = WebBrowser()
                result = wb.read_page("https://example.com")
        self.assertIn("Page body", result)
        chromium.launch.assert_called_once_with(
            headless=True, channel="chrome",
            args=["--disable-gpu",
                  "--disable-blink-features=AutomationControlled"])

    def test_auto_falls_back_to_bundled_on_chrome_launch_failure(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        chromium.launch.reset_mock()
        chromium.launch.side_effect = [
            RuntimeError("Executable doesn't exist at "
                         "/nonexistent/chrome (chrome not installed)"),
            browser,
        ]
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=False):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="auto"), clear=True):
                wb = WebBrowser()
                result = wb.read_page("https://example.com")
        self.assertIn("Page body", result)
        self.assertEqual(chromium.launch.call_count, 2)
        # First attempt: the installed Chrome channel.
        self.assertEqual(
            chromium.launch.call_args_list[0].kwargs["channel"], "chrome")
        # Fallback: bundled Chromium (channel=None).
        self.assertEqual(
            chromium.launch.call_args_list[1].kwargs["channel"], None)
        self.assertTrue(chromium.launch.call_args_list[1].kwargs["headless"])

    def test_explicit_chrome_falls_back_to_bundled_on_failure(self):
        """WEB_CHANNEL=chrome with no installed Chrome: warn + fall back,
        never crash (rule: bad config never crashes the agent)."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        chromium.launch.reset_mock()
        chromium.launch.side_effect = [
            RuntimeError("Executable doesn't exist (chrome not installed)"),
            browser,
        ]
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=False):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chrome"), clear=True):
                wb = WebBrowser()
                result = wb.read_page("https://example.com")
        self.assertIn("Page body", result)
        self.assertEqual(chromium.launch.call_count, 2)
        self.assertEqual(
            chromium.launch.call_args_list[0].kwargs["channel"], "chrome")
        self.assertEqual(
            chromium.launch.call_args_list[1].kwargs["channel"], None)

    def test_auto_fallback_on_persistent_path_too(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        chromium.launch_persistent_context.reset_mock()
        chromium.launch_persistent_context.side_effect = [
            RuntimeError("chrome not installed"),
            chromium.launch_persistent_context.return_value,
        ]
        with tempfile.TemporaryDirectory() as tmp:
            profile = pathlib.Path(tmp) / "profile"
            with patch("agents.tools.web_browser.sync_playwright", sync):
                with patch.dict(os.environ,
                                _clean_env(WEB_BROWSER_PROFILE=str(profile),
                                           WEB_CHANNEL="auto"),
                                clear=True):
                    wb = WebBrowser()
                    result = wb.read_page("https://example.com")
                    wb.close()
        self.assertIn("Page body", result)
        self.assertEqual(
            chromium.launch_persistent_context.call_args_list[0].kwargs[
                "channel"], "chrome")
        self.assertEqual(
            chromium.launch_persistent_context.call_args_list[1].kwargs[
                "channel"], None)

    def test_explicit_chromium_pass_through(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=False):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        chromium.launch.assert_called_once_with(
            headless=True, channel=None,
            args=["--disable-gpu",
                  "--disable-blink-features=AutomationControlled"])

    def test_explicit_chrome_pass_through(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync), \
             patch("agents.tools.web_browser._running_as_root",
                   return_value=False):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chrome"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        chromium.launch.assert_called_once_with(
            headless=True, channel="chrome",
            args=["--disable-gpu",
                  "--disable-blink-features=AutomationControlled"])

    def test_invalid_channel_env_falls_back_to_auto(self):
        """_env_config already replaced the bad value with 'auto'."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="firefox"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        self.assertEqual(wb._cfg["channel"], "auto")
        # 'auto' behavior: chrome tried first.
        self.assertEqual(chromium.launch.call_args.kwargs["channel"], "chrome")


# ── Stealth init script ─────────────────────────────────────────────

class TestStealthInitScript(unittest.TestCase):
    """add_init_script: once per page, stealth-gated, failure-safe."""

    def test_init_script_applied_once_per_page_creation(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
                self.assertEqual(len(created["pages"]), 1)
                page = created["pages"][0]
                page.add_init_script.assert_called_once()
                script = page.add_init_script.call_args[0][0]
                self.assertIn(
                    "Object.defineProperty(navigator, 'webdriver'", script)
        # Second read reuses the live page: no re-application.
        wb.read_page("https://example.com/again")
        self.assertEqual(len(created["pages"]), 1)
        created["pages"][0].add_init_script.assert_called_once()
        # Viewport is creation-level: never set on the page afterwards.
        created["pages"][0].set_viewport_size.assert_not_called()

    def test_rotated_pages_each_get_script_exactly_once(self):
        """Each rotation rebuilds context+page; every fresh page gets the
        script exactly once (two navigations -> two pages -> two adds)."""
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with tempfile.TemporaryDirectory() as tmp:
            pool = pathlib.Path(tmp) / "proxies.txt"
            pool.write_text(
                "http://proxy-a.example:3128\n"
                "http://proxy-b.example:8080\n")
            with patch("agents.tools.web_browser.sync_playwright", sync):
                with patch.dict(os.environ,
                                _clean_env(WEB_PROXY_FILE=str(pool),
                                           WEB_CHANNEL="chromium"),
                                clear=True):
                    wb = WebBrowser()
                    wb.read_page("https://example.com")
                    wb.read_page("https://example.com/other")
        self.assertEqual(len(created["pages"]), 2)
        for i, page in enumerate(created["pages"]):
            with self.subTest(page=i + 1):
                page.add_init_script.assert_called_once()
                script = page.add_init_script.call_args[0][0]
                self.assertIn(
                    "Object.defineProperty(navigator, 'webdriver'", script)

    def test_stead_off_skips_init_script(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium",
                                       WEB_STEALTH="0"), clear=True):
                wb = WebBrowser()
                wb.read_page("https://example.com")
        self.assertEqual(len(created["pages"]), 1)
        created["pages"][0].add_init_script.assert_not_called()

    def test_init_script_failure_never_crashes_navigation(self):
        sync, playwright, chromium, browser, created = _make_playwright_mock()
        browser.new_context.reset_mock()

        def _new_context(**kwargs):
            ctx = MagicMock()
            ctx.is_closed.return_value = False
            page = _make_page()
            page.add_init_script.side_effect = RuntimeError("boom")
            ctx.new_page.return_value = page
            created["pages"].append(page)
            return ctx
        browser.new_context.side_effect = _new_context
        with patch("agents.tools.web_browser.sync_playwright", sync):
            with patch.dict(os.environ,
                            _clean_env(WEB_CHANNEL="chromium"), clear=True):
                wb = WebBrowser()
                result = wb.read_page("https://example.com")
        self.assertIn("Page body", result)  # navigation still succeeded


# ── Script content (module-level) ───────────────────────────────────

class TestStealthScriptContent(unittest.TestCase):
    """Vendored _STEALTH_INIT_JS contains the plan's patch set."""

    def setUp(self):
        self.script = wbmod._stealth_init_script("pt-BR")

    def test_webdriver_undefined(self):
        self.assertIn("Object.defineProperty(navigator, 'webdriver'",
                      self.script)
        self.assertIn("get: () => undefined", self.script)

    def test_plugins_and_mime_types(self):
        self.assertIn("navigator, 'plugins'", self.script)
        self.assertIn("navigator, 'mimeTypes'", self.script)
        # length >= 4 for both arrays:
        self.assertGreaterEqual(self.script.count("filename:"), 4)
        self.assertGreaterEqual(self.script.count("type: 'application/"), 4)

    def test_no_illegal_plugin_constructors(self):
        """Chromium rejects `new Plugin()` / `new MimeType()` (Illegal
        constructor).  Inside the init script's per-patch try/catch that
        error is SWALLOWED, so a regression to constructor-based wrappers
        fails silently (plugins.length stays 0) -- guard against it."""
        self.assertNotIn("new MimeType(", self.script)
        self.assertNotIn("new Plugin(", self.script)
        # ...and the array-like surface must be provided explicitly:
        self.assertIn(".item = (i) =>", self.script)
        self.assertIn("namedItem", self.script)

    def test_window_chrome_shape(self):
        self.assertIn("window.chrome", self.script)
        self.assertIn("window.chrome.runtime", self.script)
        self.assertIn("window.chrome.app", self.script)
        self.assertIn("chrome.loadTimeData", self.script)

    def test_permissions_prompt(self):
        self.assertIn("navigator.permissions.query", self.script)
        self.assertIn("Promise.resolve({ state: 'prompt' })", self.script)

    def test_languages_pinned_to_locale(self):
        self.assertNotIn("__LOCALE_LANGUAGES__", self.script)
        self.assertIn('get: () => ["pt-BR", "pt"]', self.script)

    def test_languages_default(self):
        script = wbmod._stealth_init_script()
        self.assertIn('get: () => ["en-US", "en"]', script)

    def test_webgl_angle_swiftshader_pair(self):
        self.assertIn("UNMASKED_VENDOR_WEBGL", self.script)
        self.assertIn("UNMASKED_RENDERER_WEBGL", self.script)
        self.assertIn("Google Inc. (Google)", self.script)
        self.assertIn("ANGLE (Google, SwiftShader Device, OpenGL 4.0.0)",
                      self.script)

    def test_js_is_valid_if_node_available(self):
        """node --check on the substituted script (skipped without node)."""
        if shutil.which("node") is None:
            self.skipTest("node not available")
        with tempfile.NamedTemporaryFile("w", suffix=".js",
                                         delete=False) as f:
            f.write(self.script)
            path = f.name
        try:
            proc = subprocess.run(
                ["node", "--check", path], capture_output=True, text=True,
                timeout=30)
            self.assertEqual(
                proc.returncode, 0,
                f"node --check failed:\n{proc.stderr}")
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
