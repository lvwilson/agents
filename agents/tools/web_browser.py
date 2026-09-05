"""
Persistent Playwright browser session for web interaction.

Exposes two tiers of commands:

- **Stateless readers** (``read_page``, ``read_page_html``, ``page_links``,
  ``view_page``): each takes a URL, navigates, extracts data, and returns it.
- **Interactive session** (``browse_open``, ``browse_read``, ``browse_click``,
  ``browse_type``): stateful commands for forms, logins, and SPAs.
"""

import hashlib
import os
import re
import sys
import atexit
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
    profile (``WEB_BROWSER_PROFILE``).  See ``_env_config()``.
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
                self._ctx = self._browser.new_context()
            self._page = self._ctx.new_page()
            self._page.set_viewport_size({"width": 1280, "height": 900})

    def _maybe_launch(self):
        """Start or restart the Playwright/browser handles when needed."""
        if self._playwright is None:
            self._launch()
            return
        if self._persistent:
            if self._ctx is None or self._ctx.is_closed():
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

        if self._playwright is None:
            self._playwright = sync_playwright().start()

        # Phase 2 seam: WEB_CHANNEL resolution + args hardening
        # (--disable-blink-features=AutomationControlled, conditional
        # --no-sandbox) land here in a later commit; channel/args stay
        # exactly today's values for this commit.
        channel = None
        args = ["--no-sandbox", "--disable-gpu"]

        if cfg["profile"] is not None:
            # Path 1: persistent profile (cookies + storage survive the
            # process, so repeat visits look like a returning visitor).
            os.makedirs(cfg["profile"], exist_ok=True)
            ctx_kwargs = {
                "user_data_dir": cfg["profile"],
                "headless": True,
                "channel": channel,
                "args": args,
            }
            if cfg["proxy"]:
                ctx_kwargs["proxy"] = cfg["proxy"]
            self._ctx = self._playwright.chromium.launch_persistent_context(**ctx_kwargs)
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
                # _rotate_proxy on every navigation.  No context yet.
                self._browser = self._playwright.chromium.launch(
                    headless=True, channel=channel, args=args)
                return
            # Empty/unreadable pool: fall through to the fixed direct path.

        # Path 3: one fixed context for the whole session.
        self._browser = self._playwright.chromium.launch(
            headless=True, channel=channel, args=args)
        if cfg["proxy"]:
            self._ctx = self._browser.new_context(proxy=cfg["proxy"])
        else:
            self._ctx = self._browser.new_context()

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
        if proxy:
            self._ctx = self._browser.new_context(proxy=proxy)
        else:
            # Malformed pool entry: rotate past it via direct egress.
            self._ctx = self._browser.new_context()
        self._page = self._ctx.new_page()
        self._page.set_viewport_size({"width": 1280, "height": 900})
        self._proxy_idx = next_idx
        self._active_proxy = url

    def _teardown_context(self):
        """Close the current page+context (rotation / relaunch).  Safe."""
        for obj in (self._page, self._ctx):
            if obj is not None:
                try:
                    if not obj.is_closed():
                        obj.close()
                except Exception:
                    pass
        self._page = None
        self._ctx = None

    @property
    def page(self):
        self._ensure_running()
        return self._page

    def close(self):
        """Close the browser and clean up resources."""
        if self._persistent:
            # Path 1 lifecycle: the persistent context is the single
            # handle -- close it once, then stop Playwright.
            if self._ctx is not None:
                try:
                    if not self._ctx.is_closed():
                        self._ctx.close()
                except Exception:
                    pass
        else:
            for obj, cleanup in [
                (self._page, lambda o: not o.is_closed() and o.close()),
                (self._ctx, lambda o: not o.is_closed() and o.close()),
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
        self._cfg = None
        return "Browser closed."

    # ── Building blocks ─────────────────────────────────────────────

    def _navigate_then(self, url, reader, timeout=30000):
        """Navigate to *url*, return reader() result or error string."""
        try:
            self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        except PlaywrightTimeout:
            return f"Timeout navigating to {url} after {timeout}ms."
        except Exception as e:
            return f"Navigation error: {e}"
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

        for token in tokens:
            if not token:
                continue
            if token in ('[Enter]', '[Tab]', '[Escape]'):
                if pending:
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
