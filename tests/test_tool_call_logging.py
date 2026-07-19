"""Tests for native API tool-call logging.

This harness executes textual ``Command:`` lines — it never sends tool
definitions to the model and never executes native API tool calls.  If
a model (especially OpenAI-compatible local servers) emits native
tool_calls anyway, they must not be silently dropped: the backend
buffers them in ``_pending_tool_calls`` during the stream and
``generate_response`` drains them through the ``on_tool_call`` stream
hook, which the UI renders in yellow.
"""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_backend import LLMBackend, StreamHandler  # noqa: E402
from agents.backends.kimi_backend import KimiBackend  # noqa: E402
from agents.backends.openai_backend import OpenAIBackend  # noqa: E402
from agents.backends.gemini_backend import GeminiBackend  # noqa: E402
from agents.backends.anthropic_backend import AnthropicBackend  # noqa: E402


class _RecordingHandler(StreamHandler):
    """StreamHandler that records on_tool_call invocations."""

    def __init__(self):
        super().__init__()
        self.tool_calls = []

    def on_tool_call(self, name, arguments=""):
        self.tool_calls.append((name, arguments))


def _bare(cls, **overrides):
    """Construct a backend with __init__ bypassed and defaults set."""
    backend = cls.__new__(cls)
    backend.model = "test-model"
    backend.base_url = None
    backend.is_local = False
    backend.stream_handler = _RecordingHandler()
    backend.temperature = 1.0
    backend.cost = 0.0
    backend.cost_without_cache = 0.0
    backend.call_count = 0
    backend.last_input_tokens = 0
    backend.last_output_tokens = 0
    backend.last_total_context_tokens = 0
    backend.peak_context_tokens = 0
    backend._pending_tool_calls = []
    for key, value in overrides.items():
        setattr(backend, key, value)
    return backend


class _ConcreteBackend(LLMBackend):
    """Minimal concrete backend for testing base-class behaviour."""

    def generate_response(self, system_prompt, context):
        return ""


class TestEmitToolCalls(unittest.TestCase):
    """The base-class drain helper flushes buffered calls through the
    stream handler exactly once and leaves the buffer empty."""

    def test_emit_flushes_pending_calls(self):
        backend = _bare(_ConcreteBackend)
        backend._pending_tool_calls = [("read_file", "a.py"), ("stdout", "hi")]
        backend._emit_tool_calls()
        self.assertEqual(
            backend.stream_handler.tool_calls,
            [("read_file", "a.py"), ("stdout", "hi")],
        )
        self.assertEqual(backend._pending_tool_calls, [])

    def test_emit_with_no_pending_calls_is_noop(self):
        backend = _bare(_ConcreteBackend)
        backend._emit_tool_calls()
        self.assertEqual(backend.stream_handler.tool_calls, [])

    def test_base_handler_on_tool_call_is_noop(self):
        handler = StreamHandler()
        handler.on_tool_call("anything", "args")  # must not raise


# ── Kimi (chat-completions style incremental tool_calls) ─────────────

def _kimi_tool_call_event(index, name=None, arguments=None):
    func = SimpleNamespace(name=name, arguments=arguments)
    tc = SimpleNamespace(index=index, function=func)
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            delta=SimpleNamespace(
                reasoning_content=None, content=None, tool_calls=[tc]))],
    )


def _kimi_content_event(text):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(
            delta=SimpleNamespace(
                reasoning_content=None, content=text, tool_calls=None))],
    )


def _kimi_usage_event():
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                              prompt_tokens_details=None),
        choices=[],
    )


def _make_kimi(events):
    backend = _bare(KimiBackend)
    backend._client = mock.Mock()
    backend._client.chat.completions.create.return_value = iter(events)
    return backend


class TestKimiToolCallLogging(unittest.TestCase):
    """Kimi streams tool calls as per-index fragments; the fragments
    must be reassembled and reported after the response is processed."""

    def test_fragmented_tool_call_reassembled(self):
        backend = _make_kimi([
            _kimi_tool_call_event(0, name="read_file"),
            _kimi_tool_call_event(0, arguments='{"path":'),
            _kimi_tool_call_event(0, arguments='"a.py"}'),
            _kimi_content_event("reading the file"),
            _kimi_usage_event(),
        ])
        text = backend.generate_response("sys", [])
        self.assertEqual(text, "reading the file")
        self.assertEqual(
            backend.stream_handler.tool_calls,
            [("read_file", '{"path":"a.py"}')],
        )

    def test_multiple_tool_calls_by_index(self):
        backend = _make_kimi([
            _kimi_tool_call_event(0, name="read_file", arguments="a.py"),
            _kimi_tool_call_event(1, name="stdout", arguments="hello"),
            _kimi_content_event("done"),
            _kimi_usage_event(),
        ])
        backend.generate_response("sys", [])
        self.assertEqual(
            backend.stream_handler.tool_calls,
            [("read_file", "a.py"), ("stdout", "hello")],
        )

    def test_no_tool_calls_no_logging(self):
        backend = _make_kimi([_kimi_content_event("plain"), _kimi_usage_event()])
        backend.generate_response("sys", [])
        self.assertEqual(backend.stream_handler.tool_calls, [])


# ── OpenAI (Responses API function_call output items) ────────────────

def _make_openai(events):
    backend = _bare(OpenAIBackend)
    backend._client = mock.Mock()
    backend._client.responses.create.return_value = iter(events)
    return backend


class TestOpenAIToolCallLogging(unittest.TestCase):
    """Responses API ``function_call`` output items must be logged."""

    def test_function_call_item_logged(self):
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="calling tool"),
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(
                    type="function_call", name="run_console_command",
                    arguments='{"command": "ls"}'),
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=10, output_tokens=5,
                        input_tokens_details=None))),
        ]
        backend = _make_openai(events)
        text = backend.generate_response("sys", [])
        self.assertEqual(text, "calling tool")
        self.assertEqual(
            backend.stream_handler.tool_calls,
            [("run_console_command", '{"command": "ls"}')],
        )

    def test_non_function_items_ignored(self):
        events = [
            SimpleNamespace(
                type="response.output_item.done",
                item=SimpleNamespace(type="reasoning", summary=[]),
            ),
            SimpleNamespace(type="response.output_text.delta", delta="hi"),
            SimpleNamespace(type="response.completed", response=None),
        ]
        backend = _make_openai(events)
        backend.generate_response("sys", [])
        self.assertEqual(backend.stream_handler.tool_calls, [])


# ── Gemini (chunk.function_calls convenience property) ───────────────

def _make_gemini(chunks):
    backend = _bare(GeminiBackend)
    backend.cache_step = 2
    backend._cache_name = None
    backend._cached_msg_count = 0
    backend._cached_system_prompt = None
    # _get_response references self._types for GenerateContentConfig;
    # with an empty context no Part/Content constructors are needed.
    backend._types = SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs,
    )
    backend._client = mock.Mock()
    backend._client.models.generate_content_stream.return_value = iter(chunks)
    return backend


class TestGeminiToolCallLogging(unittest.TestCase):
    """Gemini function calls attached to chunks must be logged."""

    def test_chunk_function_calls_logged(self):
        fc = SimpleNamespace(name="web_search", args={"query": "news"})
        chunk = SimpleNamespace(
            usage_metadata=None,
            function_calls=[fc],
            text="searching…",
        )
        backend = _make_gemini([chunk])
        text = backend.generate_response("sys", [])
        self.assertEqual(text, "searching…")
        self.assertEqual(
            backend.stream_handler.tool_calls,
            [("web_search", str({"query": "news"}))],
        )

    def test_chunk_without_function_calls_no_logging(self):
        chunk = SimpleNamespace(
            usage_metadata=None, function_calls=None, text="plain")
        backend = _make_gemini([chunk])
        backend.generate_response("sys", [])
        self.assertEqual(backend.stream_handler.tool_calls, [])


# ── Anthropic (tool_use blocks in the final response) ────────────────

def _make_anthropic(blocks):
    backend = _bare(AnthropicBackend)
    response = SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=5,
            cache_creation_input_tokens=0, cache_read_input_tokens=0),
    )
    backend._get_response = mock.Mock(return_value=response)
    return backend


class TestAnthropicToolCallLogging(unittest.TestCase):
    """tool_use blocks in the final message must be logged, not dropped."""

    def test_tool_use_block_logged(self):
        blocks = [
            SimpleNamespace(type="tool_use", name="read_file",
                            input={"file_path": "a.py"}),
            SimpleNamespace(type="text", text="reading"),
        ]
        backend = _make_anthropic(blocks)
        text = backend.generate_response("sys", [])
        self.assertEqual(text, "reading")
        self.assertEqual(
            backend.stream_handler.tool_calls,
            [("read_file", str({"file_path": "a.py"}))],
        )

    def test_text_only_response_no_logging(self):
        blocks = [SimpleNamespace(type="text", text="just text")]
        backend = _make_anthropic(blocks)
        backend.generate_response("sys", [])
        self.assertEqual(backend.stream_handler.tool_calls, [])


# ── UI rendering ─────────────────────────────────────────────────────

class TestRichStreamHandlerToolCall(unittest.TestCase):
    """The interactive handler renders tool calls in yellow without
    touching the spinner (logging happens post-stream)."""

    def test_renders_yellow_warning(self):
        from agents import ui
        handler = ui.RichStreamHandler()
        with mock.patch.object(ui, "safe_console_print") as mock_print:
            handler.on_tool_call("read_file", "a.py")
        mock_print.assert_called_once()
        message, kwargs = mock_print.call_args[0][0], mock_print.call_args[1]
        self.assertIn("read_file", message)
        self.assertIn("a.py", message)
        self.assertEqual(kwargs.get("style"), "warning")

    def test_long_arguments_truncated(self):
        from agents import ui
        handler = ui.RichStreamHandler()
        with mock.patch.object(ui, "safe_console_print") as mock_print:
            handler.on_tool_call("write_file", "x" * 500)
        message = mock_print.call_args[0][0]
        self.assertIn("…", message)
        self.assertLess(len(message), 250)


if __name__ == "__main__":
    unittest.main()
