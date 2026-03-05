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

    # ── Core tools (stateless reading) ──────────────────────────────

    def read_page(self, url, selector=None, timeout=30000):
        """Navigate to *url* and return visible text (optionally scoped by *selector*)."""
        nav_result = self.navigate(url, timeout)
        if "Timeout" in nav_result or "error" in nav_result.lower():
            return nav_result
        return self.read_text(selector)

    def read_page_html(self, url, selector=None, timeout=30000):
        """Navigate to *url* and return HTML (optionally scoped by *selector*)."""
        nav_result = self.navigate(url, timeout)
        if "Timeout" in nav_result or "error" in nav_result.lower():
            return nav_result
        return self.read_html(selector)

    def page_links(self, url, timeout=30000):
        """Navigate to *url* and return all links."""
        nav_result = self.navigate(url, timeout)
        if "Timeout" in nav_result or "error" in nav_result.lower():
            return nav_result
        return self.get_links()

    def view_page(self, url, file_path=None, timeout=30000):
        """Navigate to *url*, screenshot, extract text + interactive elements.

        Returns ``(rich_text_response, file_path)`` tuple.
        """
        nav_result = self.navigate(url, timeout)
        if "Timeout" in nav_result or "error" in nav_result.lower():
            return (nav_result, None)

        # Generate file path if not provided
        if not file_path:
            import hashlib
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            file_path = f"/tmp/web_screenshot_{url_hash}.png"

        self.screenshot(file_path)
        text_content = self.read_text()
        interactive = self.get_interactive_elements()

        rich_text = text_content
        if interactive:
            rich_text += f"\n{'─' * 60}\n{interactive}"

        return (rich_text, file_path)

    def get_interactive_elements(self):
        """Extract all interactive elements with their selectors and metadata."""
        self._ensure_running()
        try:
            elements = self.page.evaluate("""() => {
                function bestSelector(el) {
                    if (el.id) return '#' + CSS.escape(el.id);
                    if (el.name) return el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                    // Try class-based selector
                    if (el.className && typeof el.className === 'string') {
                        const cls = el.className.trim().split(/\\s+/).filter(c => c.length > 0);
                        if (cls.length > 0) {
                            const sel = el.tagName.toLowerCase() + '.' + cls.join('.');
                            if (document.querySelectorAll(sel).length === 1) return sel;
                        }
                    }
                    // Fallback: nth-child
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

                // Links
                const links = Array.from(document.querySelectorAll('a[href]')).slice(0, MAX);
                for (const el of links) {
                    result.links.push({
                        text: (el.innerText || '').trim().substring(0, 60),
                        href: el.href,
                        selector: bestSelector(el)
                    });
                }

                // Buttons
                const buttons = Array.from(document.querySelectorAll('button, input[type="button"], input[type="submit"]')).slice(0, MAX);
                for (const el of buttons) {
                    result.buttons.push({
                        text: (el.innerText || el.value || '').trim().substring(0, 60),
                        selector: bestSelector(el)
                    });
                }

                // Inputs (not buttons)
                const inputs = Array.from(document.querySelectorAll('input:not([type="button"]):not([type="submit"]):not([type="hidden"])')).slice(0, MAX);
                for (const el of inputs) {
                    result.inputs.push({
                        type: el.type || 'text',
                        name: el.name || '',
                        placeholder: el.placeholder || '',
                        value: el.value || '',
                        selector: bestSelector(el)
                    });
                }

                // Selects
                const selects = Array.from(document.querySelectorAll('select')).slice(0, MAX);
                for (const el of selects) {
                    result.selects.push({
                        name: el.name || '',
                        value: el.value || '',
                        optionCount: el.options.length,
                        selector: bestSelector(el)
                    });
                }

                // Textareas
                const textareas = Array.from(document.querySelectorAll('textarea')).slice(0, MAX);
                for (const el of textareas) {
                    result.textareas.push({
                        name: el.name || '',
                        placeholder: el.placeholder || '',
                        selector: bestSelector(el)
                    });
                }

                return result;
            }""")
        except Exception as e:
            return f"Error extracting interactive elements: {e}"

        lines = ["Interactive Elements:"]

        if elements.get("links"):
            lines.append(f"  Links ({len(elements['links'])}):")
            for i, el in enumerate(elements["links"], 1):
                text = el.get("text", "")
                href = el.get("href", "")
                sel = el.get("selector", "")
                lines.append(f"    {i}. [{text}] -> {href} -- {sel}")

        if elements.get("buttons"):
            lines.append(f"  Buttons ({len(elements['buttons'])}):")
            for i, el in enumerate(elements["buttons"], 1):
                text = el.get("text", "")
                sel = el.get("selector", "")
                lines.append(f"    {i}. [{text}] -- {sel}")

        if elements.get("inputs"):
            lines.append(f"  Inputs ({len(elements['inputs'])}):")
            for i, el in enumerate(elements["inputs"], 1):
                typ = el.get("type", "text")
                name = el.get("name", "")
                placeholder = el.get("placeholder", "")
                sel = el.get("selector", "")
                parts = [typ]
                if name:
                    parts.append(f'name="{name}"')
                if placeholder:
                    parts.append(f'placeholder="{placeholder}"')
                lines.append(f"    {i}. {' '.join(parts)} -- {sel}")

        if elements.get("selects"):
            lines.append(f"  Selects ({len(elements['selects'])}):")
            for i, el in enumerate(elements["selects"], 1):
                name = el.get("name", "")
                value = el.get("value", "")
                count = el.get("optionCount", 0)
                sel = el.get("selector", "")
                lines.append(f"    {i}. name=\"{name}\" value=\"{value}\" ({count} options) -- {sel}")

        if elements.get("textareas"):
            lines.append(f"  Textareas ({len(elements['textareas'])}):")
            for i, el in enumerate(elements["textareas"], 1):
                name = el.get("name", "")
                placeholder = el.get("placeholder", "")
                sel = el.get("selector", "")
                parts = []
                if name:
                    parts.append(f'name="{name}"')
                if placeholder:
                    parts.append(f'placeholder="{placeholder}"')
                lines.append(f"    {i}. {' '.join(parts)} -- {sel}")

        # If no interactive elements found at all
        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    # ── Interactive tools (stateful browsing) ───────────────────────

    def browse_open(self, url, timeout=30000):
        """Navigate to *url* and auto-read: return page text."""
        nav_result = self.navigate(url, timeout)
        if "Timeout" in nav_result or "error" in nav_result.lower():
            return nav_result
        return self.read_text()

    def browse_read(self, selector=None):
        """Read current page, optionally scoped by *selector*. Auto-waits for selector."""
        if selector:
            wait_result = self.wait_for_selector(selector, timeout=10000)
            if "Timeout" in wait_result or "error" in wait_result.lower():
                return wait_result
        return self.read_text(selector)

    def browse_click(self, selector, timeout=5000):
        """Click element, auto-wait, then auto-read resulting page."""
        self._ensure_running()
        try:
            self.page.click(selector, timeout=timeout)
        except PlaywrightTimeout:
            return f"Timeout clicking selector: {selector}"
        except Exception as e:
            return f"Click error: {e}"
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=10000)
        except PlaywrightTimeout:
            pass  # Page may not navigate; still read what we have
        return self.read_text()

    def browse_type(self, selector, text, timeout=5000):
        """Type text into element. Supports [Enter], [Tab], [Escape] inline.

        Auto-reads if text ends with [Enter].
        """
        import re as _re
        self._ensure_running()

        # Split text on inline key tokens
        tokens = _re.split(r'(\[Enter\]|\[Tab\]|\[Escape\])', text)
        ends_with_enter = text.rstrip().endswith('[Enter]')

        # Collect the pure text portion (everything before the first key token)
        pure_text_parts = []
        key_sequence_started = False

        for token in tokens:
            if not token:
                continue
            if token in ('[Enter]', '[Tab]', '[Escape]'):
                key_sequence_started = True
                # First fill any accumulated text
                if pure_text_parts:
                    fill_text = ''.join(pure_text_parts)
                    try:
                        self.page.fill(selector, fill_text, timeout=timeout)
                    except PlaywrightTimeout:
                        return f"Timeout typing into selector: {selector}"
                    except Exception as e:
                        return f"Type error: {e}"
                    pure_text_parts = []

                # Press the key
                key_name = token[1:-1]  # Strip [ and ]
                try:
                    self.page.keyboard.press(key_name)
                except Exception as e:
                    return f"Key press error ({key_name}): {e}"
            else:
                pure_text_parts.append(token)

        # Fill any remaining text that wasn't followed by a key
        if pure_text_parts:
            fill_text = ''.join(pure_text_parts)
            try:
                self.page.fill(selector, fill_text, timeout=timeout)
            except PlaywrightTimeout:
                return f"Timeout typing into selector: {selector}"
            except Exception as e:
                return f"Type error: {e}"

        if ends_with_enter:
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=10000)
            except PlaywrightTimeout:
                pass  # May not navigate
            return self.read_text()

        return f"Typed into: {selector}"

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
