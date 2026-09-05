"""Tests for backend routing and retry behaviour.

Covers:
* ``openai`` + a custom ``base_url`` routes to the shared
  OpenAI-compatible *chat-completions* backend (local servers implement
  chat completions, not the Responses API), while hosted ``openai`` keeps
  the Responses-API backend.
* HTTP 401 authentication errors fail fast (no retries) with a clear
  message, instead of being retried as transient errors.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.backends import create_backend  # noqa: E402
from agents.backends.openai_compat_backend import OpenAICompatBackend  # noqa: E402
from agents.backends.openai_backend import OpenAIBackend  # noqa: E402
from agents.backends.cerebras_backend import CerebrasBackend  # noqa: E402


class TestBaseUrlRouting(unittest.TestCase):
    """openai + base_url → chat-completions backend; hosted → Responses."""

    def test_openai_with_base_url_uses_chat_completions(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}):
            backend = create_backend("openai", model="local-model",
                                     base_url="http://localhost:8000/v1")
        self.assertIsInstance(backend, OpenAICompatBackend)
        self.assertNotIsInstance(backend, OpenAIBackend)
        self.assertTrue(backend.is_local)

    def test_openai_hosted_uses_responses_api(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}):
            backend = create_backend("openai", model="gpt-5.3-codex")
        self.assertIsInstance(backend, OpenAIBackend)
        self.assertFalse(backend.is_local)

    def test_cerebras_hosted_is_cerebras(self):
        with mock.patch.dict(os.environ, {"CEREBRAS_API_KEY": "sk"}):
            backend = create_backend("cerebras", model="qwen-3.8-27b")
        self.assertIsInstance(backend, CerebrasBackend)
        self.assertFalse(backend.is_local)

    def test_cerebras_with_base_url_still_cerebras(self):
        # A proxy base_url keeps the Cerebras backend (its own SDK).
        with mock.patch.dict(os.environ, {"CEREBRAS_API_KEY": "sk"}):
            backend = create_backend("cerebras", model="qwen-3.8-27b",
                                     base_url="http://proxy:9000/v1")
        self.assertIsInstance(backend, CerebrasBackend)
        self.assertTrue(backend.is_local)


class _FakeError(Exception):
    """Stand-in for an SDK HTTP error carrying a status code."""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class TestAuthFailFast(unittest.TestCase):
    """HTTP 401 must fail immediately, not be retried as transient."""

    def _make_backend(self):
        with mock.patch.dict(os.environ, {"CEREBRAS_API_KEY": "sk"}):
            with mock.patch("cerebras.cloud.sdk.Cerebras"):
                return CerebrasBackend(model="qwen-3.8-27b")

    def test_401_not_retried(self):
        backend = self._make_backend()
        # Make the streaming call raise an auth error.
        backend._client = mock.MagicMock()
        backend._client.chat.completions.create.side_effect = _FakeError(401)

        with self.assertRaises(Exception) as ctx:
            backend.generate_response(
                "sys", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
            )
        self.assertIn("401", str(ctx.exception))
        self.assertIn("Authentication", str(ctx.exception))
        # Only ONE attempt was made (no retries).
        self.assertEqual(
            backend._client.chat.completions.create.call_count, 1)

    def test_non_401_is_retried(self):
        backend = self._make_backend()
        backend._client = mock.MagicMock()
        backend._client.chat.completions.create.side_effect = _FakeError(500)
        # Shorten the transient delay so the test is fast.
        backend.TRANSIENT_RETRY_DELAY = 0
        with self.assertRaises(Exception):
            backend.generate_response(
                "sys", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
            )
        # 500 is transient → retried up to MAX_ERROR_RETRIES times.
        self.assertEqual(
            backend._client.chat.completions.create.call_count,
            backend.MAX_ERROR_RETRIES)

    def test_is_auth_error_detects_response_status(self):
        backend = self._make_backend()

        class RespError(Exception):
            def __init__(self):
                super().__init__("auth")
                self.response = mock.MagicMock(status_code=401)

        self.assertTrue(backend._is_auth_error(RespError()))
        self.assertFalse(backend._is_auth_error(_FakeError(403)))
        self.assertFalse(backend._is_auth_error(RuntimeError("boom")))


if __name__ == "__main__":
    unittest.main()
