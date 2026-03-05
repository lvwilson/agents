"""
Persistent Playwright browser session for web interaction.

Provides a lazily-initialized singleton browser that persists across
command invocations, allowing the LLM agent to navigate, read, click,
type, screenshot, and execute JavaScript on web pages.
"""

import os
import atexit
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


class WebBrowser:
    """Manages a single Playwright Chromium browser and page."""

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None

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

    # ── Navigation ──────────────────────────────────────────────────

    def navigate(self, url, timeout=30000):
        """Navigate to *url* and return a status summary."""
        self._ensure_running()
        try:
            response = self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            status = response.status if response else "unknown"
        except PlaywrightTimeout:
            return f"Timeout navigating to {url} after {timeout}ms."
        except Exception as e:
            return f"Navigation error: {e}"

        title = self.page.title()
        current_url = self.page.url
        return f"Navigated to: {current_url}\nStatus: {status}\nTitle: {title}"

    def back(self):
        """Go back one page in history."""
        self._ensure_running()
        try:
            self.page.go_back(wait_until="domcontentloaded", timeout=10000)
        except PlaywrightTimeout:
            return "Timeout going back."
        title = self.page.title()
        return f"Went back to: {self.page.url}\nTitle: {title}"

    def forward(self):
        """Go forward one page in history."""
        self._ensure_running()
        try:
            self.page.go_forward(wait_until="domcontentloaded", timeout=10000)
        except PlaywrightTimeout:
            return "Timeout going forward."
        title = self.page.title()
        return f"Went forward to: {self.page.url}\nTitle: {title}"

    # ── Reading ─────────────────────────────────────────────────────

    def read_text(self, selector=None):
        """Return visible text content of the page or a specific element."""
        self._ensure_running()
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

        url = self.page.url
        title = self.page.title()
        header = f"URL: {url}\nTitle: {title}\n{'─' * 60}\n"
        return header + text

    def read_html(self, selector=None):
        """Return the outer HTML of the page or a specific element."""
        self._ensure_running()
        try:
            if selector:
                element = self.page.query_selector(selector)
                if element is None:
                    return f"No element found matching selector: {selector}"
                html = element.evaluate("el => el.outerHTML")
            else:
                html = self.page.content()
        except Exception as e:
            return f"Error reading HTML: {e}"
        return html

    def get_links(self):
        """Return a formatted list of all links on the page."""
        self._ensure_running()
        try:
            links = self.page.eval_on_selector_all(
                "a[href]",
                """elements => elements.map(el => ({
                    text: (el.innerText || '').trim().substring(0, 80),
                    href: el.href
                }))"""
            )
        except Exception as e:
            return f"Error getting links: {e}"

        if not links:
            return "No links found on the page."

        lines = [f"Found {len(links)} links:\n"]
        for i, link in enumerate(links, 1):
            text = link.get("text", "").replace("\n", " ").strip()
            href = link.get("href", "")
            if text:
                lines.append(f"  {i}. [{text}] -> {href}")
            else:
                lines.append(f"  {i}. {href}")
        return "\n".join(lines)

    # ── Interaction ─────────────────────────────────────────────────

    def click(self, selector, timeout=5000):
        """Click an element matching *selector*."""
        self._ensure_running()
        try:
            self.page.click(selector, timeout=timeout)
        except PlaywrightTimeout:
            return f"Timeout clicking selector: {selector}"
        except Exception as e:
            return f"Click error: {e}"
        self.page.wait_for_load_state("domcontentloaded", timeout=10000)
        return f"Clicked: {selector}\nCurrent URL: {self.page.url}"

    def type_text(self, selector, text, timeout=5000):
        """Type *text* into the element matching *selector*."""
        self._ensure_running()
        try:
            self.page.fill(selector, text, timeout=timeout)
        except PlaywrightTimeout:
            return f"Timeout typing into selector: {selector}"
        except Exception as e:
            return f"Type error: {e}"
        return f"Typed into: {selector}"

    def press_key(self, key):
        """Press a keyboard key (e.g. 'Enter', 'Tab', 'Escape')."""
        self._ensure_running()
        try:
            self.page.keyboard.press(key)
        except Exception as e:
            return f"Key press error: {e}"
        return f"Pressed key: {key}"

    def select_option(self, selector, value, timeout=5000):
        """Select an option from a <select> element."""
        self._ensure_running()
        try:
            self.page.select_option(selector, value, timeout=timeout)
        except Exception as e:
            return f"Select error: {e}"
        return f"Selected '{value}' in: {selector}"

    # ── Screenshots ─────────────────────────────────────────────────

    def screenshot(self, file_path, selector=None, full_page=False):
        """Take a screenshot and save it to *file_path*."""
        self._ensure_running()
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        try:
            if selector:
                element = self.page.query_selector(selector)
                if element is None:
                    return f"No element found matching selector: {selector}"
                element.screenshot(path=file_path)
            else:
                self.page.screenshot(path=file_path, full_page=full_page)
        except Exception as e:
            return f"Screenshot error: {e}"

        return f"Screenshot saved to: {file_path} ({self.page.url})"

    # ── JavaScript ──────────────────────────────────────────────────

    def execute_js(self, script):
        """Execute JavaScript on the page and return the result."""
        self._ensure_running()
        try:
            result = self.page.evaluate(script)
        except Exception as e:
            return f"JavaScript error: {e}"
        if result is None:
            return "JavaScript executed (no return value)."
        return str(result)

    # ── Waiting ─────────────────────────────────────────────────────

    def wait_for_selector(self, selector, timeout=10000):
        """Wait for an element matching *selector* to appear."""
        self._ensure_running()
        try:
            self.page.wait_for_selector(selector, timeout=timeout)
        except PlaywrightTimeout:
            return f"Timeout waiting for selector: {selector} ({timeout}ms)"
        except Exception as e:
            return f"Wait error: {e}"
        return f"Element found: {selector}"

    # ── Page info ───────────────────────────────────────────────────

    def get_page_info(self):
        """Return current URL, title, and viewport size."""
        self._ensure_running()
        url = self.page.url
        title = self.page.title()
        viewport = self.page.viewport_size
        return (
            f"URL: {url}\n"
            f"Title: {title}\n"
            f"Viewport: {viewport['width']}x{viewport['height']}"
        )

    # ── Lifecycle ───────────────────────────────────────────────────

    def close(self):
        """Close the browser and clean up resources."""
        try:
            if self._page and not self._page.is_closed():
                self._page.close()
        except Exception:
            pass
        self._page = None

        try:
            if self._browser and self._browser.is_connected():
                self._browser.close()
        except Exception:
            pass
        self._browser = None

        try:
            if self._playwright:
                self._playwright.stop()
        except Exception:
            pass
        self._playwright = None

        return "Browser closed."


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


# Clean up on process exit
atexit.register(close_browser)
