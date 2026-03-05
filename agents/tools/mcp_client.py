"""
MCP client — generic bridge to Model Context Protocol servers.

Manages stdio connections to MCP servers, provides tool discovery
and invocation.  Servers are configured in ~/.config/agents/mcp_servers.json:

    {
        "codebase-memory": {
            "command": "codebase-memory-mcp",
            "args": []
        },
        "my-server": {
            "command": "python",
            "args": ["-m", "my_mcp_server"],
            "env": {"SOME_VAR": "value"}
        }
    }
"""

import asyncio
import atexit
import json
import os
import threading
from pathlib import Path

# MCP SDK imports — optional dependency
try:
    from mcp.client.stdio import stdio_client, StdioServerParameters
    from mcp.client.session import ClientSession
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

CONFIG_PATH = Path.home() / ".config" / "agents" / "mcp_servers.json"


def _load_config():
    """Load MCP server configurations from disk."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH) as f:
        return json.load(f)


# ── Async core ──────────────────────────────────────────────────────
#
# Each MCP server gets a persistent background event loop + connection.
# We keep sessions alive so repeated calls don't pay reconnection cost.

class _ServerHandle:
    """Holds a live MCP session for one server."""
    __slots__ = ("name", "config", "session", "_read", "_write",
                 "_ctx_stdio", "_ctx_session", "_loop", "_thread")

    def __init__(self, name, config):
        self.name = name
        self.config = config
        self.session = None
        self._loop = None
        self._thread = None

    async def _connect(self):
        params = StdioServerParameters(
            command=self.config["command"],
            args=self.config.get("args", []),
            env=self.config.get("env"),
        )
        self._ctx_stdio = stdio_client(params)
        self._read, self._write = await self._ctx_stdio.__aenter__()
        self._ctx_session = ClientSession(self._read, self._write)
        self.session = await self._ctx_session.__aenter__()
        await self.session.initialize()

    async def _disconnect(self):
        try:
            if hasattr(self, '_ctx_session'):
                await self._ctx_session.__aexit__(None, None, None)
        except Exception:
            pass
        try:
            if hasattr(self, '_ctx_stdio'):
                await self._ctx_stdio.__aexit__(None, None, None)
        except Exception:
            pass
        self.session = None


class MCPManager:
    """Manages connections to all configured MCP servers.

    Thread-safe: each server runs on its own asyncio event loop in a
    background daemon thread.  Public methods are synchronous.
    """

    def __init__(self):
        self._handles = {}  # name -> _ServerHandle
        self._config = None
        self._lock = threading.Lock()

    def _ensure_config(self):
        if self._config is None:
            self._config = _load_config()

    def _get_handle(self, server_name):
        """Get or create a connected handle for *server_name*."""
        with self._lock:
            if server_name in self._handles:
                h = self._handles[server_name]
                if h.session is not None:
                    return h

            self._ensure_config()
            if server_name not in self._config:
                raise ValueError(f"Unknown MCP server: {server_name!r}. "
                                 f"Available: {list(self._config.keys())}")

            h = _ServerHandle(server_name, self._config[server_name])

            # Each handle gets its own event loop in a daemon thread
            loop = asyncio.new_event_loop()
            h._loop = loop

            def _run_loop():
                asyncio.set_event_loop(loop)
                loop.run_forever()

            t = threading.Thread(target=_run_loop, daemon=True)
            t.start()
            h._thread = t

            # Connect synchronously from the caller's thread
            future = asyncio.run_coroutine_threadsafe(h._connect(), loop)
            future.result(timeout=30)

            self._handles[server_name] = h
            return h

    def _run_on(self, handle, coro):
        """Schedule *coro* on the handle's loop and block for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, handle._loop)
        return future.result(timeout=60)

    # ── Public API (synchronous) ────────────────────────────────────

    def list_servers(self):
        """Return list of configured server names."""
        self._ensure_config()
        return list(self._config.keys())

    def list_tools(self, server_name=None):
        """List tools from one server, or all servers if name is None.

        Returns a list of dicts: {server, name, description, schema}.
        """
        self._ensure_config()
        servers = [server_name] if server_name else list(self._config.keys())
        results = []
        for name in servers:
            try:
                h = self._get_handle(name)
                tools_result = self._run_on(h, h.session.list_tools())
                for tool in tools_result.tools:
                    results.append({
                        "server": name,
                        "name": tool.name,
                        "description": tool.description or "",
                        "schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
                    })
            except Exception as e:
                results.append({
                    "server": name,
                    "name": "(error)",
                    "description": f"Failed to connect: {e}",
                    "schema": {},
                })
        return results

    def call_tool(self, server_name, tool_name, arguments=None):
        """Call a tool on a server. *arguments* is a dict.

        Returns the text content from the MCP response.
        """
        h = self._get_handle(server_name)
        result = self._run_on(h, h.session.call_tool(tool_name, arguments or {}))
        parts = []
        for item in result.content:
            if hasattr(item, 'text'):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else "ok"

    def shutdown(self):
        """Disconnect all servers and stop their event loops."""
        with self._lock:
            for h in self._handles.values():
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        h._disconnect(), h._loop)
                    future.result(timeout=5)
                except Exception:
                    pass
                try:
                    h._loop.call_soon_threadsafe(h._loop.stop)
                except Exception:
                    pass
            self._handles.clear()


# ── Module-level singleton ──────────────────────────────────────────

_manager = None


def get_manager():
    """Return the global MCPManager, creating it on first call."""
    global _manager
    if _manager is None:
        if not MCP_AVAILABLE:
            raise ImportError(
                "MCP SDK not installed. Run: pip install mcp"
            )
        _manager = MCPManager()
        atexit.register(_manager.shutdown)
    return _manager
