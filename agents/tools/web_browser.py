"""
Persistent Playwright browser session for web interaction.

Exposes two tiers of commands:

- **Stateless readers** (``read_page``, ``read_page_html``, ``page_links``,
  ``view_page``): each takes a URL, navigates, extracts data, and returns it.
- **Interactive session** (``browse_open``, ``browse_read``, ``browse_click``,
  ``browse_type``): stateful commands for forms, logins, and SPAs.
"""

import hashlib
import json
import os
import random
import re
import sys
import atexit
import time
import urllib.parse

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


# ── Env-config parsing (Phase 1: IP protection) ────────────────────
#
# All stealth-related settings are env-only (see the stealth plan,
# section 2.1).  They are parsed once per browser (re-)launch by
# _env_config() and stored on the WebBrowser instance.  Bad values
# warn to the TTY (stderr when /dev/tty is unavailable) and fall back
# to safe defaults -- never raise.  WEB_CHANNEL / WEB_USER_AGENT /
# WEB_LOCALE / WEB_TIMEZONE / WEB_REQUEST_DELAY / WEB_STEALTH are
# consumed by later phases; they are parsed here so the helper is final.

_VALID_PROXY_SCHEMES = ("http", "https", "socks5")
_VALID_CHANNELS = ("auto", "chrome", "chromium")
_DEFAULT_LOCALE = "en-US"
_DEFAULT_TIMEZONE = "Etc/UTC"
_DEFAULT_REQUEST_DELAY = 0.5
_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off")

# ── Fingerprint constants (Phase 2) ────────────────────────────────
#
# _CHROME_VERSION is the FALLBACK Chrome major version.  When the launched
# engine's version is readable (the default case), the UA is derived from
# THAT (see _engine_major / _get_context_options) so the UA's
# "Chrome/<major>" stays in lockstep with the engine's auto-generated
# sec-ch-ua client hints -- the "UA says X but sec-ch-ua says Y" mismatch
# is a hard, trivially-checked bot tell.  This constant is used only when
# the engine version is unreadable (tests / probe edge).  Keep it current.
_CHROME_VERSION = "149.0.0.0"

# Real Chrome UA per platform (no "HeadlessChrome" token).  The
# lower-case "windows" key doubles as the fallback for unknown
# platforms.
_CHROME_UA_TEMPLATES = {
    "linux": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/{version} Safari/537.36",
    "darwin": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} "
              "Safari/537.36",
    "windows": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} "
               "Safari/537.36",
}
_DEFAULT_UA_PLATFORM_KEY = "windows"


def _warn(message):
    """Warn about bad tool config on the TTY (stderr if unavailable).

    Malformed configuration must never crash the agent: we warn and
    continue with safe defaults.
    """
    try:
        handle = open("/dev/tty", "w")
    except OSError:
        handle = sys.stderr
    try:
        handle.write(f"web_browser: {message}\n")
        handle.flush()
    except OSError:
        pass
    finally:
        if handle is not sys.stderr:
            try:
                handle.close()
            except OSError:
                pass


def _engine_major(browser):
    """Major of the actually-launched Chrome/Chromium engine, or ``None``.

    Reads the launched ``Browser.version`` (e.g. ``"149.0.7827.53"`` ->
    ``"149"``) so the UA's ``Chrome/<major>`` matches the engine's
    auto-generated ``sec-ch-ua`` client hints.  A non-string ``version``
    (e.g. a test ``MagicMock``) yields ``None`` so the ``_CHROME_VERSION``
    fallback is used.  Never raises.
    """
    try:
        version = getattr(browser, "version", None)
    except Exception:
        return None
    if not isinstance(version, str) or not version:
        return None
    m = re.match(r"(\d+)", version)
    return m.group(1) if m else None


def _build_auto_user_agent(engine_major=None):
    """Auto-built user agent: the real Chrome UA string for this platform.

    Built from the platform templates -- never contains the
    "HeadlessChrome" token.  When *engine_major* is given (the actually
    launched engine's major, see ``_engine_major``), that version is used
    so the UA stays in lockstep with the engine's ``sec-ch-ua``; otherwise
    the ``_CHROME_VERSION`` fallback is used.  Unknown platforms fall back
    to the Windows template.
    """
    template = _CHROME_UA_TEMPLATES.get(
        sys.platform.lower(), _CHROME_UA_TEMPLATES[_DEFAULT_UA_PLATFORM_KEY])
    version = f"{engine_major}.0.0.0" if engine_major else _CHROME_VERSION
    return template.format(version=version)


def _running_as_root():
    """True when the effective uid is 0 (root/CI).  Patch seam for tests.

    ``--no-sandbox`` is added only in this case: it is normally required
    under root/CI but is an extra tell on a normal desktop box.
    """
    try:
        return os.geteuid() == 0
    except (AttributeError, OSError):
        # Non-POSIX platforms (e.g. Windows) have no uid; treat as
        # non-root -> no --no-sandbox.
        return False


def _valid_proxy_url(url):
    """Return True if *url* is a usable proxy URL (allowed scheme + netloc)."""
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return False
    return parts.scheme in _VALID_PROXY_SCHEMES and bool(parts.netloc)


def _proxy_kwargs(url):
    """Convert a proxy URL to Playwright's ``proxy=`` context kwarg dict.

    Returns ``None`` for a missing/invalid URL (caller egresses
    directly).  Credentials are included only when present.
    """
    if not url or not _valid_proxy_url(url):
        return None
    parts = urllib.parse.urlsplit(url)
    server = f"{parts.scheme}://{parts.hostname}"
    if parts.port:
        server += f":{parts.port}"
    proxy = {"server": server}
    if parts.username:
        proxy["username"] = parts.username
    if parts.password is not None:
        proxy["password"] = parts.password
    return proxy


def _env_config():
    """Parse + validate all stealth-related environment variables.

    Called once per browser (re-)launch; the result is stored on the
    WebBrowser instance (``self._cfg``).  Never raises: every bad value
    is warned (TTY) and replaced with a safe default.
    """
    env = os.environ

    # ── Proxy ────────────────────────────────────────────────────────
    raw_proxy = env.get("WEB_PROXY", "").strip()
    proxy = _proxy_kwargs(raw_proxy) if raw_proxy else None
    if raw_proxy and proxy is None:
        _warn(f"ignoring invalid WEB_PROXY {raw_proxy!r} "
              f"(need scheme http/https/socks5 + host:port); using direct egress")

    raw_proxy_file = env.get("WEB_PROXY_FILE", "").strip()
    proxy_file = os.path.expanduser(raw_proxy_file) if raw_proxy_file else None
    if proxy_file is not None and proxy is not None:
        _warn("WEB_PROXY_FILE is set together with WEB_PROXY; the proxy file wins")
        proxy = None

    # ── Persistent profile ───────────────────────────────────────────
    raw_profile = env.get("WEB_BROWSER_PROFILE", "").strip()
    profile = None
    if raw_profile:
        if raw_profile in _TRUE_VALUES:
            profile = os.path.expanduser("~/.agents/browser_profile")
        elif raw_profile not in _FALSE_VALUES:
            profile = os.path.expanduser(raw_profile)

    # Conflict: Playwright takes the proxy at launch level for persistent
    # contexts, so a rotating proxy file and a persistent profile cannot
    # coexist -- the proxy file wins and the profile is disabled.
    if profile is not None and proxy_file is not None:
        _warn("WEB_PROXY_FILE and WEB_BROWSER_PROFILE are mutually exclusive; "
              "using the proxy file, persistent profile disabled")
        profile = None

    # ── Channel (consumed by Phase 2) ────────────────────────────────
    raw_channel = env.get("WEB_CHANNEL", "").strip().lower() or "auto"
    channel = raw_channel if raw_channel in _VALID_CHANNELS else "auto"
    if channel != raw_channel:
        _warn(f"ignoring invalid WEB_CHANNEL {raw_channel!r}; using 'auto'")

    # ── Fingerprint + pacing (consumed by Phases 2-3) ────────────────
    user_agent = env.get("WEB_USER_AGENT", "").strip() or None
    locale = env.get("WEB_LOCALE", "").strip() or _DEFAULT_LOCALE
    timezone = env.get("WEB_TIMEZONE", "").strip() or _DEFAULT_TIMEZONE

    request_delay = _DEFAULT_REQUEST_DELAY
    raw_delay = env.get("WEB_REQUEST_DELAY", "").strip()
    if raw_delay:
        try:
            request_delay = max(0.0, float(raw_delay))
        except ValueError:
            _warn(f"ignoring invalid WEB_REQUEST_DELAY {raw_delay!r}; "
                  f"using {_DEFAULT_REQUEST_DELAY}")

    # "0" -> False, anything else (including unset) -> True.
    stealth = env.get("WEB_STEALTH", "1").strip() != "0"

    return {
        "proxy": proxy,
        "proxy_file": proxy_file,
        "profile": profile,
        "channel": channel,
        "user_agent": user_agent,
        "locale": locale,
        "timezone": timezone,
        "request_delay": request_delay,
        "stealth": stealth,
    }


class WebBrowser:
    """Manages a single Playwright Chromium browser and page.

    Phase 1 adds IP protection: a fixed proxy (``WEB_PROXY``), a
    rotating proxy pool (``WEB_PROXY_FILE``), and an opt-in persistent
    profile (``WEB_BROWSER_PROFILE``).

    Phase 2 adds fingerprint hardening (all env-only, see
    ``_env_config()``): a real-Chrome user agent (``WEB_USER_AGENT``,
    auto-built per platform from ``_CHROME_VERSION`` by default),
    consistent ``locale`` / ``timezone_id`` / ``viewport`` / ``screen`` /
    ``device_scale_factor`` context options, a matching
    ``Accept-Language`` header, stealth-gated launch args and the
    vendored ``_STEALTH_INIT_JS`` anti-tell init script
    (``WEB_STEALTH``), and channel selection (``WEB_CHANNEL``):
    ``auto`` drives the installed Chrome when present and falls back to
    the bundled Chromium.

    Phase 3 adds behavioral pacing: a jittered, human-plausible wait
    (``WEB_REQUEST_DELAY``) after each navigation and before interactive
    actions, so the click/type rhythm is not mechanical.  ``0`` disables
    it (pre-Phase-3 behavior).
    """

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None
        # Phase 1: env config (read once per (re)launch) + proxy state
        self._cfg = None
        self._ctx = None
        self._persistent = False
        self._proxy_pool = []
        self._proxy_idx = -1
        self._active_proxy = None
        self._engine_major = None

    # ── Lifecycle ───────────────────────────────────────────────────

    def _ensure_running(self):
        """Lazily start the browser and create a page if needed."""
        self._maybe_launch()
        if self._ctx is None and self._proxy_pool and not self._persistent:
            # Rotating state: _rotate_proxy() (called at every navigation
            # entry) owns context+page creation/rebuild -- do not create a
            # bare context here.
            return
        if self._page is None or self._page.is_closed():
            if self._ctx is None:
                # Defensive: reachable only if a launch path left no
                # context behind.
                if self._browser is None or not self._browser.is_connected():
                    return
                self._ctx = self._browser.new_context(**self._get_context_options())
            self._page = self._ctx.new_page()
            # Viewport is set at context creation (see
            # _get_context_options); no separate set_viewport_size here.
            self._apply_init_script(self._page)

    def _maybe_launch(self):
        """Start or restart the Playwright/browser handles when needed."""
        if self._playwright is None:
            self._launch()
            return
        if self._persistent:
            # BrowserContext has no is_closed() (only Page/Browser do);
            # if the context is missing the page property will observe
            # the same and _launch() below rebuilds it.
            if self._ctx is None:
                self._launch()
        elif self._browser is None or not self._browser.is_connected():
            self._launch()

    def _launch(self):
        """(Re-)launch the browser stack from the current env config.

        Config is read exactly once here and stored on ``self._cfg``.
        Three launch paths:

        1. Persistent profile (``WEB_BROWSER_PROFILE``):
           ``chromium.launch_persistent_context`` returns the context
           directly (``_persistent`` steers ``close()``); a fixed proxy
           rides along as the launch-level ``proxy=`` kwarg (Playwright
           limitation: proxy is launch-level for persistent contexts).
        2. Rotating proxy file (``WEB_PROXY_FILE``): the pool is loaded
           but no context is created -- the first navigation's
           ``_rotate_proxy`` builds it.  A persistent profile is skipped
           (mutually exclusive; the file wins, see ``_env_config``).
        3. Default / fixed proxy: one context for the whole session
           (``new_context(proxy=...)`` when ``WEB_PROXY`` is set).
        """
        self._cfg = _env_config()
        cfg = self._cfg
        self._persistent = False
        self._teardown_context()
        self._proxy_pool = []
        self._proxy_idx = -1
        self._active_proxy = None
        self._engine_major = None

        if self._playwright is None:
            self._playwright = sync_playwright().start()

        # Phase 2 -- fingerprint hardening.
        #
        # Channel (WEB_CHANNEL): "chrome" drives the installed Google
        # Chrome binary (new headless mode, far fewer tells); "chromium"
        # forces the bundled build; "auto" (default) tries chrome first
        # and falls back to bundled Chromium on launch failure (see
        # _chromium_launch / _launch_persistent for the fallback).
        # --disable-gpu is always kept: on a box without a GPU the
        # ANGLE/SwiftShader WebGL fingerprint is the honest one, and
        # spoofing a GPU we cannot rasterize is a worse sign.
        channel = cfg["channel"]
        args = self._launch_args(cfg)

        if cfg["profile"] is not None:
            # Path 1: persistent profile (cookies + storage survive the
            # process, so repeat visits look like a returning visitor).
            os.makedirs(cfg["profile"], exist_ok=True)
            # The persistent launch fuses browser + context creation, so the
            # engine version must be known BEFORE the context options (UA) are
            # built.  A throwaway headless probe launch (same channel/args)
            # reveals the real engine version so the UA stays in lockstep with
            # sec-ch-ua; if it fails we fall back to the constant.
            probe = self._chromium_launch(channel, args)
            try:
                self._engine_major = _engine_major(probe)
            finally:
                try:
                    probe.close()
                except Exception:
                    pass
            # Context options are launch-level for persistent contexts:
            # the very same fingerprint options as new_context() above.
            ctx_kwargs = self._get_context_options()
            ctx_kwargs["user_data_dir"] = cfg["profile"]
            ctx_kwargs["headless"] = True
            ctx_kwargs["args"] = args
            if cfg["proxy"]:
                ctx_kwargs["proxy"] = cfg["proxy"]
            self._ctx = self._launch_persistent(channel, ctx_kwargs)
            self._persistent = True
            return

        if cfg["proxy_file"] is not None:
            # Path 2: rotating proxies (per-navigation round-robin).
            try:
                with open(cfg["proxy_file"], "r", encoding="utf-8") as f:
                    self._proxy_pool = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.strip().startswith("#")
                    ]
            except OSError as e:
                _warn(f"cannot read WEB_PROXY_FILE {cfg['proxy_file']!r}: {e}; "
                      "using direct egress")
                self._proxy_pool = []
            if self._proxy_pool:
                # Browser handle now; contexts are (re)built by
                # _rotate_proxy on every navigation with the per-navigation
                # proxy -- the fixed-proxy kwarg therefore does not apply
                # here (ctx_opts without proxy).  No context yet.
                self._browser = self._chromium_launch(channel, args)
                self._engine_major = _engine_major(self._browser)
                return
            # Empty/unreadable pool: fall through to the fixed direct path.

        # Path 3: one fixed context for the whole session.
        #
        # The context options (user agent, locale, timezone, viewport at
        # creation, screen, device scale, Accept-Language) are applied to
        # EVERY launch path so no code path can drift back to bare
        # Playwright defaults.
        self._browser = self._chromium_launch(channel, args)
        self._engine_major = _engine_major(self._browser)
        ctx_opts = self._get_context_options()
        self._ctx = self._browser.new_context(**ctx_opts)

    def _get_context_options(self, include_fixed_proxy=True):
        """Build the fingerprint-consistent ``Browser.new_context()`` options.

        ``user_agent``, ``locale``, ``timezone_id``, ``viewport`` (applied
        AT context creation -- the separate ``set_viewport_size`` call was
        dropped), ``screen``, ``device_scale_factor`` and a matching
        ``Accept-Language`` header.  ``WEB_USER_AGENT`` overrides the
        auto-built real-Chrome UA; ``WEB_LOCALE`` / ``WEB_TIMEZONE``
        override the defaults.  The fixed-proxy kwarg is included only
        for the fixed-proxy / persistent paths (the rotation path carries
        its own per-navigation proxy) -- see *include_fixed_proxy*.
        """
        cfg = self._cfg
        # Auto UA tracks the launched engine's major (lockstep with
        # sec-ch-ua); an explicit WEB_USER_AGENT always wins.
        user_agent = cfg["user_agent"] or _build_auto_user_agent(
            getattr(self, "_engine_major", None))
        locale = cfg["locale"]
        opts = {
            "user_agent": user_agent,
            "locale": locale,
            "timezone_id": cfg["timezone"],
            "viewport": {"width": 1280, "height": 900},
            "screen": {"width": 1280, "height": 900},
            "device_scale_factor": 1,
            "extra_http_headers": {
                "Accept-Language": self._accept_language_header(locale),
            },
        }
        if include_fixed_proxy and cfg.get("proxy"):
            opts["proxy"] = cfg["proxy"]
        return opts

    @staticmethod
    def _accept_language_header(locale):
        """Accept-Language matching ``navigator.languages`` for *locale*.

        ``en-US`` -> ``en-US,en;q=0.9`` (region stripped to a ``q=0.9``
        fallback); locales without a separator (e.g. ``en``) just repeat.
        """
        if "-" in locale:
            base = locale.split("-", 1)[0]
            return f"{locale},{base};q=0.9"
        return locale

    def _launch_args(self, cfg):
        """Browser launch args (Phase 2 hardening).

        ``--disable-gpu`` is always kept: the honest ANGLE/SwiftShader WebGL
        fingerprint beats a fabricated GPU we can't rasterize.
        ``--no-sandbox`` is dropped for non-root users (still added under
        root/CI, where it is normally required).  The automation
        blink-feature flag is gated by ``WEB_STEALTH`` (``0`` reproduces
        pre-Phase-2 behavior for diffing).
        """
        args = []
        if _running_as_root():
            args.append("--no-sandbox")
        args.append("--disable-gpu")
        if cfg.get("stealth", True):
            args.append("--disable-blink-features=AutomationControlled")
        return args

    def _chromium_launch(self, channel, args):
        """``chromium.launch`` (non-persistent paths) with the
        ``WEB_CHANNEL`` fallback.

        ``auto`` (default) and explicit ``chrome`` drive the installed
        Google Chrome binary (new headless mode, fewest tells);
        ``chromium`` forces the bundled build.  When a Chrome launch
        fails (e.g. Chrome is not installed) we warn to the TTY and
        retry once with the bundled Chromium -- a missing binary must
        never crash the agent.  A *bundled* launch failure is a real
        environment problem and propagates.
        """
        primary = "chrome" if channel in ("auto", "chrome") else None
        try:
            return self._playwright.chromium.launch(
                headless=True, channel=primary, args=args)
        except Exception as e:
            if primary != "chrome":
                raise  # bundled is the floor: genuine environment failure
            _warn(f"WEB_CHANNEL={channel}: 'chrome' launch failed ({e}); "
                  "falling back to bundled Chromium")
            return self._playwright.chromium.launch(
                headless=True, channel=None, args=args)

    def _launch_persistent(self, channel, ctx_kwargs):
        """``chromium.launch_persistent_context`` with the same fallback
        as :meth:`_chromium_launch`."""
        primary = "chrome" if channel in ("auto", "chrome") else None
        try:
            return self._playwright.chromium.launch_persistent_context(
                **dict(ctx_kwargs, channel=primary))
        except Exception as e:
            if primary != "chrome":
                raise  # bundled is the floor: genuine environment failure
            _warn(f"WEB_CHANNEL={channel}: 'chrome' launch failed ({e}); "
                  "falling back to bundled Chromium")
            return self._playwright.chromium.launch_persistent_context(
                **dict(ctx_kwargs, channel=None))

    def _rotate_proxy(self):
        """Round-robin to the next proxy and rebuild context+page for it.

        Called at the entry of every navigation command (read_page,
        read_page_html, page_links, view_page, browse_open) after
        ``_ensure_running``.  Interactive commands (browse_read,
        browse_click, browse_type, browse_js) do not rotate: the page
        keeps its egress until the next navigation.  No-op when no pool
        is loaded.
        """
        if not self._proxy_pool:
            return
        next_idx = (self._proxy_idx + 1) % len(self._proxy_pool)
        url = self._proxy_pool[next_idx]
        if self._ctx is not None and self._active_proxy == url:
            # Same pool slot as the current context: keep it (a pool of
            # one would otherwise rebuild on every navigation).
            self._proxy_idx = next_idx
            return
        self._teardown_context()
        proxy = _proxy_kwargs(url)
        # Rotating path: the fixed proxy is intentionally not part of the
        # per-navigation context options (it is carried by this rotation).
        ctx_opts = self._get_context_options(include_fixed_proxy=False)
        if proxy:
            ctx_opts = dict(ctx_opts, proxy=proxy)
        self._ctx = self._browser.new_context(**ctx_opts)
        self._page = self._ctx.new_page()
        # Viewport already set at context creation; apply the stealth
        # init script exactly as for fresh context pages.
        self._apply_init_script(self._page)
        self._proxy_idx = next_idx
        self._active_proxy = url

    def _teardown_context(self):
        """Close the current page+context (rotation / relaunch).  Safe."""
        # Playwright's Page/Context close() is idempotent; note that
        # BrowserContext has NO is_closed() method (only Page does), so
        # no liveness check is applied to self._ctx.
        for obj in (self._page, self._ctx):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._page = None
        self._ctx = None

    def _apply_init_script(self, page):
        """Apply the vendored stealth init script to a fresh page.

        Called exactly once per page creation (both the context-creation
        and the proxy-rotation paths).  ``WEB_STEALTH=0`` skips it
        entirely (pre-Phase-2 behavior, useful for debugging/diffing).
        A failure here must never break navigation: warn and continue.
        """
        if not (self._cfg or {}).get("stealth", True):
            return
        try:
            page.add_init_script(
                _stealth_init_script((self._cfg or {}).get("locale")))
        except Exception as e:
            _warn(f"could not apply stealth init script: {e}")

    @property
    def page(self):
        self._ensure_running()
        return self._page

    def close(self):
        """Close the browser and clean up resources."""
        if self._persistent:
            # Path 1 lifecycle: the persistent context is the single
            # handle -- close it, then stop Playwright.  (Context
            # close() is idempotent; BrowserContext has no is_closed().)
            if self._ctx is not None:
                try:
                    self._ctx.close()
                except Exception:
                    pass
        else:
            for obj, cleanup in [
                (self._page, lambda o: not o.is_closed() and o.close()),
                (self._ctx, lambda o: o.close()),
                (self._browser, lambda o: o.is_connected() and o.close()),
            ]:
                if obj:
                    try:
                        cleanup(obj)
                    except Exception:
                        pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._playwright = None
        self._browser = None
        self._page = None
        self._ctx = None
        self._persistent = False
        self._proxy_pool = []
        self._proxy_idx = -1
        self._active_proxy = None
        self._engine_major = None
        self._cfg = None
        return "Browser closed."

    # ── Building blocks ─────────────────────────────────────────────

    def _pacing_wait(self):
        """Jittered human-plausible delay (Phase 3).

        Sleeps for ``uniform(delay*0.5, delay*1.5)`` seconds where
        ``delay`` is ``WEB_REQUEST_DELAY`` (default ``0.5``).  Returns
        immediately when the delay is ``<= 0``.  The wait lives in the
        navigation / interactive-action entry points, NOT in ``read_text``
        itself, so a double-read of an already-loaded page (e.g.
        ``browse_read`` twice) stays instant -- that's what a human does
        too.
        """
        d = (self._cfg or {}).get("request_delay", 0.5)
        if d <= 0:
            return
        time.sleep(random.uniform(max(0.0, d * 0.5), d * 1.5))

    def _navigate_then(self, url, reader, timeout=30000):
        """Navigate to *url*, return reader() result or error string."""
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except PlaywrightTimeout:
            return f"Timeout navigating to {url} after {timeout}ms."
        except Exception as e:
            return f"Navigation error: {e}"
        # Phase 3: human-plausible pause after navigation (before the
        # read), so the rhythm is not mechanical.  No wait on navigation
        # failure (nothing was loaded).
        self._pacing_wait()
        return reader()

    def read_text(self, selector=None):
        """Return visible text content of the page or a specific element."""
        try:
            if selector:
                element = self.page.query_selector(selector)
                if element is None:
                    return f"No element found matching selector: {selector}"
                text = element.inner_text()
            else:
                text = self.page.inner_text("body")
        except Exception as e:
            return f"Error reading text: {e}"
        return f"URL: {self.page.url}\nTitle: {self.page.title()}\n{'─' * 60}\n{text}"

    def read_html(self, selector=None):
        """Return the outer HTML of the page or a specific element."""
        try:
            if selector:
                element = self.page.query_selector(selector)
                if element is None:
                    return f"No element found matching selector: {selector}"
                return element.evaluate("el => el.outerHTML")
            else:
                return self.page.content()
        except Exception as e:
            return f"Error reading HTML: {e}"

    def get_links(self):
        """Return a formatted list of all links on the page."""
        try:
            links = self.page.eval_on_selector_all(
                "a[href]",
                """elements => elements.map(el => ({
                    text: (el.innerText || '').trim().substring(0, 80),
                    href: el.href
                }))""",
            )
        except Exception as e:
            return f"Error getting links: {e}"

        if not links:
            return "No links found on the page."

        lines = [f"Found {len(links)} links:\n"]
        for i, link in enumerate(links, 1):
            text = link.get("text", "").replace("\n", " ").strip()
            href = link.get("href", "")
            label = f"[{text}] -> {href}" if text else href
            lines.append(f"  {i}. {label}")
        return "\n".join(lines)

    def screenshot(self, file_path, full_page=False):
        """Take a screenshot and save it to *file_path*."""
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            self.page.screenshot(path=file_path, full_page=full_page)
        except Exception as e:
            return f"Screenshot error: {e}"
        return None

    def get_interactive_elements(self):
        """Extract interactive elements with CSS selectors for browse_click/browse_type."""
        try:
            elements = self.page.evaluate(_INTERACTIVE_ELEMENTS_JS)
        except Exception as e:
            return f"Error extracting interactive elements: {e}"

        lines = ["Interactive Elements:"]
        for category in ("links", "buttons", "inputs", "selects", "textareas"):
            items = elements.get(category)
            if items:
                lines.append(f"  {category.title()} ({len(items)}):")
                for i, el in enumerate(items, 1):
                    lines.append(f"    {i}. {_fmt_element(category, el)}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def execute_js(self, script):
        """Execute JavaScript on the page and return the result."""
        try:
            result = self.page.evaluate(script)
        except Exception as e:
            return f"JavaScript error: {e}"
        return "JavaScript executed (no return value)." if result is None else str(result)

    # ── Stateless reading commands ──────────────────────────────────

    def read_page(self, url, selector=None, timeout=30000):
        """Navigate to *url* and return visible text (optionally scoped)."""
        self._ensure_running()
        self._rotate_proxy()
        return self._navigate_then(url, lambda: self.read_text(selector), timeout)

    def read_page_html(self, url, selector=None, timeout=30000):
        """Navigate to *url* and return HTML (optionally scoped)."""
        self._ensure_running()
        self._rotate_proxy()
        return self._navigate_then(url, lambda: self.read_html(selector), timeout)

    def page_links(self, url, timeout=30000):
        """Navigate to *url* and return all links."""
        self._ensure_running()
        self._rotate_proxy()
        return self._navigate_then(url, self.get_links, timeout)

    def view_page(self, url, file_path=None, timeout=30000):
        """Navigate to *url*, screenshot, extract text + interactive elements.

        Returns ``(rich_text_response, file_path)`` tuple.
        """
        self._ensure_running()
        self._rotate_proxy()
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except PlaywrightTimeout:
            return (f"Timeout navigating to {url} after {timeout}ms.", None)
        except Exception as e:
            return (f"Navigation error: {e}", None)

        # Phase 3: pause after the load, before the screenshot/read work.
        self._pacing_wait()

        if not file_path:
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            file_path = f"/tmp/web_screenshot_{url_hash}.png"

        self.screenshot(file_path)
        text = self.read_text()
        interactive = self.get_interactive_elements()
        if interactive:
            text += f"\n{'─' * 60}\n{interactive}"
        return (text, file_path)

    # ── Interactive session commands ────────────────────────────────

    def browse_open(self, url, timeout=30000):
        """Navigate to *url* and auto-read page text."""
        self._ensure_running()
        self._rotate_proxy()
        return self._navigate_then(url, self.read_text, timeout)

    def browse_read(self, selector=None):
        """Read current page, optionally scoped by *selector* (auto-waits)."""
        if selector:
            try:
                self.page.wait_for_selector(selector, timeout=10000)
            except PlaywrightTimeout:
                return f"Timeout waiting for selector: {selector}"
            except Exception as e:
                return f"Wait error: {e}"
        return self.read_text(selector)

    def browse_click(self, selector, timeout=5000):
        """Click element, wait for navigation, then auto-read."""
        # Phase 3: a human pauses before clicking, not instantly.
        self._pacing_wait()
        try:
            self.page.click(selector, timeout=timeout)
        except PlaywrightTimeout:
            return f"Timeout clicking selector: {selector}"
        except Exception as e:
            return f"Click error: {e}"
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
        except PlaywrightTimeout:
            pass
        return self.read_text()

    def browse_type(self, selector, text, timeout=5000):
        """Type *text* into element.  Supports ``[Enter]``, ``[Tab]``, ``[Escape]`` inline.

        Auto-reads the page if *text* ends with ``[Enter]``.
        """
        tokens = re.split(r'(\[Enter\]|\[Tab\]|\[Escape\])', text)
        pending = []
        # Phase 3: pace before the FIRST real fill only -- a subsequent
        # flush (after [Tab]/[Enter]) must not re-wait.
        will_wait = True

        for token in tokens:
            if not token:
                continue
            if token in ('[Enter]', '[Tab]', '[Escape]'):
                if pending:
                    if will_wait:
                        self._pacing_wait()
                        will_wait = False
                    error = self._fill(selector, ''.join(pending), timeout)
                    if error:
                        return error
                    pending = []
                try:
                    self.page.keyboard.press(token[1:-1])
                except Exception as e:
                    return f"Key press error ({token[1:-1]}): {e}"
            else:
                pending.append(token)

        if pending:
            if will_wait:
                self._pacing_wait()
                will_wait = False
            error = self._fill(selector, ''.join(pending), timeout)
            if error:
                return error

        if text.rstrip().endswith('[Enter]'):
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            except PlaywrightTimeout:
                pass
            return self.read_text()

        return f"Typed into: {selector}"

    def _fill(self, selector, text, timeout):
        """Fill *selector* with *text*.  Returns error string or ``None``."""
        try:
            self.page.fill(selector, text, timeout=timeout)
        except PlaywrightTimeout:
            return f"Timeout typing into selector: {selector}"
        except Exception as e:
            return f"Type error: {e}"
        return None


# ── JavaScript for interactive element extraction ───────────────────

_INTERACTIVE_ELEMENTS_JS = """() => {
    function bestSelector(el) {
        if (el.id) return '#' + CSS.escape(el.id);
        if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
        if (el.className && typeof el.className === 'string') {
            const cls = el.className.trim().split(/\\s+/).filter(c => c.length > 0);
            if (cls.length > 0) {
                const sel = el.tagName.toLowerCase() + '.' + cls.join('.');
                if (document.querySelectorAll(sel).length === 1) return sel;
            }
        }
        const parent = el.parentElement;
        if (parent) {
            const siblings = Array.from(parent.children).filter(c => c.tagName === el.tagName);
            const idx = siblings.indexOf(el) + 1;
            return bestSelector(parent) + ' > ' + el.tagName.toLowerCase() + ':nth-child(' + idx + ')';
        }
        return el.tagName.toLowerCase();
    }

    const MAX = 100;
    const result = { links: [], buttons: [], inputs: [], selects: [], textareas: [] };

    for (const el of Array.from(document.querySelectorAll('a[href]')).slice(0, MAX))
        result.links.push({
            text: (el.innerText || '').trim().substring(0, 60),
            href: el.href,
            selector: bestSelector(el)
        });

    for (const el of Array.from(document.querySelectorAll(
            'button, input[type="button"], input[type="submit"]')).slice(0, MAX))
        result.buttons.push({
            text: (el.innerText || el.value || '').trim().substring(0, 60),
            selector: bestSelector(el)
        });

    for (const el of Array.from(document.querySelectorAll(
            'input:not([type="button"]):not([type="submit"]):not([type="hidden"])')).slice(0, MAX))
        result.inputs.push({
            type: el.type || 'text',
            name: el.name || '',
            placeholder: el.placeholder || '',
            value: el.value || '',
            selector: bestSelector(el)
        });

    for (const el of Array.from(document.querySelectorAll('select')).slice(0, MAX))
        result.selects.push({
            name: el.name || '',
            value: el.value || '',
            optionCount: el.options.length,
            selector: bestSelector(el)
        });

    for (const el of Array.from(document.querySelectorAll('textarea')).slice(0, MAX))
        result.textareas.push({
            name: el.name || '',
            placeholder: el.placeholder || '',
            selector: bestSelector(el)
        });

    return result;
}"""


# ── Stealth init script (Phase 2) ───────────────────────────────────
#
# Vendored, dependency-free anti-tell patch set, applied via
# page.add_init_script so it runs before page scripts in every frame
# and on every navigation.  The __LOCALE_LANGUAGES__ sentinel is
# replaced with a JSON array at first use (see _stealth_init_script()).
#
# Deliberately out of scope (stealth plan, section 6): CDP
# Runtime.enable leaks, font-enumeration parity, canvas noise tuning.

_STEALTH_INIT_JS = """
(() => {
  "use strict";

  // 1. navigator.webdriver: real Chrome reports `undefined`, not `true`.
  try {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  } catch (e) {}

  // 2. Plausible navigator.plugins / navigator.mimeTypes sets.  Real
  //    desktop Chrome exposes several plugins; a length of 0 is a
  //    strong headless tell.  Keep the two lists consistent with each
  //    other (n >= 4 each).
  //
  //    NOTE: the built-in Plugin / MimeType constructors reject "new"
  //    (Illegal constructor in Chromium), so those wrappers cannot be
  //    built from page JS -- a try/catch around them would fail
  //    SILENTLY.  Plain arrays of plain objects are used instead; they
  //    expose the same length / index / item() / namedItem() surface.
  try {
    const MIME_KINDS = [
      { type: 'application/pdf', suffixes: 'pdf',
        description: 'Portable Document Format' },
      { type: 'application/x-google-chrome-pdf', suffixes: 'pdf',
        description: 'Chrome PDF Viewer' },
      { type: 'application/x-nacl', suffixes: 'nexe',
        description: 'Native Client Executable' },
      { type: 'application/x-pnacl', suffixes: 'nexe',
        description: 'Portable Native Client Executable' },
    ];
    const mimeArray = MIME_KINDS.map((m) =>
      ({ type: m.type, suffixes: m.suffixes, description: m.description }));
    mimeArray.item = (i) => mimeArray[i] || null;
    mimeArray.namedItem = (n) =>
      mimeArray.find((m) => m.type === n) || null;

    const PLUGIN_SPECS = [
      { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', mime: 0 },
      { name: 'Chrome PDF Viewer',
        filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', mime: 1 },
      { name: 'Chromium PDF Viewer',
        filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', mime: 0 },
      { name: 'PDF Viewer',
        filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', mime: 1 },
      { name: 'Chromium PDF Plugin', filename: 'internal-pdf-viewer', mime: 0 },
      { name: 'Native Client', filename: 'internal-nacl-plugin', mime: 2 },
    ];
    const pluginArray = PLUGIN_SPECS.map((spec) => {
      const mime = mimeArray[spec.mime];
      return {
        name: spec.name,
        filename: spec.filename,
        description: 'Portable Document Format',
        length: 1,
        0: mime,
        item: (i) => (i === 0 ? mime : null),
      };
    });
    pluginArray.item = (i) => pluginArray[i] || null;
    pluginArray.namedItem = (n) =>
      pluginArray.find((p) => p.name === n) || null;

    Object.defineProperty(navigator, 'plugins', { get: () => pluginArray });
    Object.defineProperty(navigator, 'mimeTypes', { get: () => mimeArray });
  } catch (e) {}

  // 3. window.chrome: present with the minimal real-Chrome shape,
  //    including the chrome.loadTimeData internal-API stub.
  try {
    if (!window.chrome) {
      window.chrome = {};
    }
    if (!window.chrome.runtime) {
      window.chrome.runtime = {
        OnInstalledReason: { CHROME_UPDATE: 'chrome_update',
                             INSTALL: 'install',
                             SHARED_MODULE_UPDATE: 'shared_module_update',
                             UPDATE: 'update' },
        OnRestartRequiredReason: { APP_UPDATE: 'app_update',
                                   OS_UPDATE: 'os_update',
                                   PERIODIC: 'periodic' },
        PlatformArch: { ARM: 'arm', MIPS: 'mips', MIPS64: 'mips64',
                        X86_32: 'x86-32', X86_64: 'x86-64' },
        PlatformOs: { ANDROID: 'android', CROS: 'cros', LINUX: 'linux',
                      MAC: 'mac', OPENBSD: 'openbsd', WIN: 'win' },
        RequestUpdateCheckStatus: { NO_UPDATE: 'no_update',
                                    THROTTLED: 'throttled',
                                    UPDATE_AVAILABLE: 'update_available' },
      };
    }
    if (!window.chrome.app) {
      window.chrome.app = {
        isInstalled: false,
        InstallState: { DISABLED: 'disabled', INSTALLED: 'installed',
                        NOT_INSTALLED: 'not_installed' },
        RunningState: { CANNOT_RUN: 'cannot_run',
                        READY_TO_RUN: 'ready_to_run',
                        RUNNING: 'running' },
      };
    }
    window.chrome.loadTimeData = {
      getString: () => '',
      getInt: () => 0,
      getBoolean: () => false,
      getComputedInt: () => 0,
      deleteValue: () => {},
    };
  } catch (e) {}

  // 4. navigator.languages: aligned with the pinned context locale
  //    (matches the Accept-Language header).
  try {
    Object.defineProperty(navigator, 'languages', {
      get: () => __LOCALE_LANGUAGES__,
    });
  } catch (e) {}

  // 5. Permissions: headless Chrome reports the notifications
  //    permission as 'denied'; a fresh real browser says 'prompt'.
  try {
    const originalQuery =
      navigator.permissions.query.bind(navigator.permissions);
    navigator.permissions.query = (parameters) => (
      parameters.name === 'notifications'
        ? Promise.resolve({ state: 'prompt' })
        : originalQuery(parameters)
    );
  } catch (e) {}

  // 6. WebGL unmasked vendor/renderer: a software-rasterizing box
  //    honestly reports ANGLE + SwiftShader.  Keep the pair consistent
  //    -- do not fabricate a GPU this box cannot actually rasterize.
  try {
    const UNMASKED_VENDOR_WEBGL = 0x9245;
    const UNMASKED_RENDERER_WEBGL = 0x9246;
    const originalGetParameter =
      WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function (parameter) {
      if (parameter === UNMASKED_VENDOR_WEBGL) {
        return 'Google Inc. (Google)';
      }
      if (parameter === UNMASKED_RENDERER_WEBGL) {
        return 'ANGLE (Google, SwiftShader Device, OpenGL 4.0.0)';
      }
      return originalGetParameter.call(this, parameter);
    };
  } catch (e) {}
})();
"""


def _stealth_init_script(locale=None):
    """Return ``_STEALTH_INIT_JS`` with ``navigator.languages`` pinned.

    ``en-US`` -> ``["en-US", "en"]`` (region-stripped fallback),
    ``pt`` -> ``["pt"]``, unset -> the default ``["en-US", "en"]``.
    """
    if not locale:
        languages = ["en-US", "en"]
    elif "-" in locale:
        languages = [locale, locale.split("-", 1)[0]]
    else:
        languages = [locale]
    return _STEALTH_INIT_JS.replace(
        "__LOCALE_LANGUAGES__", json.dumps(languages))


# ── Element formatters ──────────────────────────────────────────────

def _fmt_element(category, el):
    """Format a single interactive element for display."""
    sel = el.get("selector", "")
    if category == "links":
        text, href = el.get("text", ""), el.get("href", "")
        return f"[{text}] -> {href} -- {sel}" if text else f"{href} -- {sel}"
    if category == "buttons":
        return f"[{el.get('text', '')}] -- {sel}"
    if category == "inputs":
        parts = [el.get("type", "text")]
        if el.get("name"):
            parts.append(f'name="{el["name"]}"')
        if el.get("placeholder"):
            parts.append(f'placeholder="{el["placeholder"]}"')
        return f"{' '.join(parts)} -- {sel}"
    if category == "selects":
        return (f'name="{el.get("name", "")}" value="{el.get("value", "")}" '
                f'({el.get("optionCount", 0)} options) -- {sel}')
    # textareas
    parts = []
    if el.get("name"):
        parts.append(f'name="{el["name"]}"')
    if el.get("placeholder"):
        parts.append(f'placeholder="{el["placeholder"]}"')
    return f"{' '.join(parts)} -- {sel}"


# ── Module-level singleton ──────────────────────────────────────────

_browser_instance = None


def get_browser():
    """Return the singleton WebBrowser instance."""
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = WebBrowser()
    return _browser_instance


def close_browser():
    """Close the singleton browser if it exists."""
    global _browser_instance
    if _browser_instance is not None:
        _browser_instance.close()
        _browser_instance = None


atexit.register(close_browser)
