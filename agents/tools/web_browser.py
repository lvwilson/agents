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
import atexit

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


class WebBrowser:
    """Manages a single Playwright Chromium browser and page."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None

    # ── Lifecycle ───────────────────────────────────────────────────

    def _ensure_running(self):
        """Lazily start the browser and create a page if needed."""
        if self._playwright is None:
            self._playwright = sync_playwright().start()
        if self._browser is None or not self._browser.is_connected():
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-gpu"],
            )
        if self._page is None or self._page.is_closed():
            self._page = self._browser.new_page()
            self._page.set_viewport_size({"width": 1280, "height": 900})

    @property
    def page(self):
        self._ensure_running()
        return self._page

    def close(self):
        """Close the browser and clean up resources."""
        for obj, cleanup in [
            (self._page, lambda o: not o.is_closed() and o.close()),
            (self._browser, lambda o: o.is_connected() and o.close()),
            (self._playwright, lambda o: o.stop()),
        ]:
            if obj:
                try:
                    cleanup(obj)
                except Exception:
                    pass
        self._page = self._browser = self._playwright = None
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
        return self._navigate_then(url, lambda: self.read_text(selector), timeout)

    def read_page_html(self, url, selector=None, timeout=30000):
        """Navigate to *url* and return HTML (optionally scoped)."""
        return self._navigate_then(url, lambda: self.read_html(selector), timeout)

    def page_links(self, url, timeout=30000):
        """Navigate to *url* and return all links."""
        return self._navigate_then(url, self.get_links, timeout)

    def view_page(self, url, file_path=None, timeout=30000):
        """Navigate to *url*, screenshot, extract text + interactive elements.

        Returns ``(rich_text_response, file_path)`` tuple.
        """
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
