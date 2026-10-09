"""Native reasoning replay: fake network clients, genuine SDK payload models."""

import copy
import json
from types import SimpleNamespace as NS
from unittest import mock

import pytest
from anthropic.types import RedactedThinkingBlock, TextBlock, ThinkingBlock
from google.genai import types
from openai.types.responses import ResponseReasoningItem

from agents.backends.anthropic_backend import AnthropicBackend
from agents.backends.deepseek_backend import DeepSeekBackend
from agents.backends.gemini_backend import GeminiBackend
from agents.backends.minimax_backend import MinimaxBackend
from agents.backends.openai_backend import OpenAIBackend
from agents.llm_backend import EmptyResponseError, LLMBackend


def bare(cls, model):
    backend = cls.__new__(cls)
    LLMBackend.__init__(backend, model=model, base_url="http://localhost:8000")
    backend.last_reasoning_context = {}
    backend._client = mock.Mock()
    backend.cache_step = 2
    backend.MAX_ERROR_RETRIES = 2
    backend.TRANSIENT_RETRY_DELAY = 0
    if issubclass(cls, AnthropicBackend):
        backend._anthropic = NS(RateLimitError=type("RateLimitError", (Exception,), {}))
        backend._use_thinking_stream = True
        backend._supports_thinking_api = False
        backend._thinking_enabled = True
    if cls is OpenAIBackend:
        backend._openai = NS(RateLimitError=type("RateLimitError", (Exception,), {}))
        backend.MAX_COMPLETION_TOKENS = 16384
    if cls is GeminiBackend:
        backend._types = types
        backend._cache_name = None
        backend._cached_msg_count = 0
        backend._cached_system_prompt = None
        backend._cached_prefix_digest = None
    return backend


def assistant(native, text="answer"):
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "reasoning_context": copy.deepcopy(native),
    }


def event(kind, **kwargs):
    return NS(type=kind, **kwargs)


def failing_stream(events, error=RuntimeError("stream broke")):
    yield from events
    raise error


def anthropic_response(blocks):
    return NS(
        content=blocks,
        usage=NS(input_tokens=10, output_tokens=7),
    )


@pytest.mark.parametrize(
    "cls,model",
    [
        (AnthropicBackend, "claude-opus-4-6"),
        (DeepSeekBackend, "deepseek-v4-pro"),
        (MinimaxBackend, "MiniMax-M2.5"),
    ],
)
def test_anthropic_signed_and_redacted_roundtrip(cls, model):
    backend = bare(cls, model)
    blocks = [
        ThinkingBlock(type="thinking", thinking="private", signature="vendor-signature"),
        RedactedThinkingBlock(type="redacted_thinking", data="opaque"),
        TextBlock(type="text", text="answer"),
    ]
    backend._get_response = mock.Mock(return_value=anthropic_response(blocks))
    assert backend.generate_response("sys", []) == "answer"
    native = json.loads(json.dumps(backend.last_reasoning_context))
    expected_blocks = blocks[:2] if cls is AnthropicBackend else blocks[:1]
    assert native == {
        "format": backend.REASONING_FORMAT,
        "model": model,
        "data": [b.model_dump(mode="json", exclude_none=True) for b in expected_blocks],
    }
    context = [assistant(native)]
    before = copy.deepcopy(context)
    wire = backend._format_messages(context, backend.model)
    assert wire[0]["content"] == native["data"] + [{"type": "text", "text": "answer"}]
    assert context == before
    assert "reasoning_context" not in wire[0]


@pytest.mark.parametrize("changed", [{"format": "gemini"}, {"model": "different"}])
def test_anthropic_replay_requires_same_format_and_model(changed):
    backend = bare(AnthropicBackend, "claude-opus-4-6")
    native = {
        "format": "anthropic", "model": backend.model,
        "data": [{"type": "thinking", "thinking": "secret", "signature": "sig"}],
        **changed,
    }
    assert backend._format_messages([assistant(native)], backend.model)[0]["content"] == [
        {"type": "text", "text": "answer"}
    ]


def test_anthropic_unsigned_text_is_not_a_signature_and_plain_turn_clears_state():
    backend = bare(AnthropicBackend, "claude-opus-4-6")
    backend.last_reasoning_context = {"stale": True}
    backend._get_response = mock.Mock(return_value=anthropic_response([
        ThinkingBlock(type="thinking", thinking="unsigned", signature=""),
        TextBlock(type="text", text="answer"),
    ]))
    assert backend.generate_response("sys", []) == "answer"
    assert backend.last_reasoning_context == {}


def test_anthropic_failed_final_message_does_not_capture_partial_blocks():
    backend = bare(AnthropicBackend, "claude-opus-4-6")
    stream = mock.MagicMock()
    stream.__iter__.return_value = iter([
        event("content_block_start", content_block=NS(type="thinking")),
        event("content_block_delta", delta=NS(thinking="partial")),
    ])
    stream.get_final_message.side_effect = RuntimeError("not complete")
    backend._client.messages.stream.return_value = mock.MagicMock()
    backend._client.messages.stream.return_value.__enter__.return_value = stream
    with pytest.raises(Exception, match="Maximum retries"):
        backend.generate_response("sys", [])
    assert backend.last_reasoning_context == {}


def reasoning_item(identifier="rs_1"):
    return ResponseReasoningItem(
        type="reasoning", id=identifier, encrypted_content="opaque-ciphertext",
        summary=[{"type": "summary_text", "text": "summary"}],
    )


def openai_events(item=None, completed=True):
    events = [
        event("response.reasoning_summary_text.delta", delta="thinking summary"),
        event("response.reasoning_summary_text.done"),
        event("response.output_item.done", item=item or reasoning_item()),
        event("response.output_text.delta", delta="answer"),
    ]
    if completed:
        events.append(event(
            "response.completed",
            response=NS(
                output=[item or reasoning_item()],
                usage=NS(input_tokens=20, output_tokens=9),
            ),
        ))
    return events


def test_openai_encrypted_reasoning_roundtrip_and_summary_ui():
    backend = bare(OpenAIBackend, "gpt-5.3")
    backend._client.responses.create.return_value = iter(openai_events())
    assert backend.generate_response("sys", []) == "answer"
    assert backend.stream_handler.get_buffered_text() == "answer"
    assert backend.stream_handler.get_buffered_reasoning() == "thinking summary"
    native = json.loads(json.dumps(backend.last_reasoning_context))
    assert native["format"] == "openai_responses"
    assert native["data"][0]["encrypted_content"] == "opaque-ciphertext"
    assert backend._client.responses.create.call_args.kwargs["include"] == [
        "reasoning.encrypted_content"
    ]
    wire = backend._format_messages("", [assistant(native)], backend.model)
    assert wire == native["data"] + [{
        "role": "assistant", "content": [{"type": "output_text", "text": "answer"}],
    }]
    backend._validate_responses_input(wire)


@pytest.mark.parametrize("changed", [{"format": "anthropic"}, {"model": "different"}])
def test_openai_model_and_format_gating(changed):
    backend = bare(OpenAIBackend, "gpt-5.3")
    native = {
        "format": "openai_responses", "model": backend.model,
        "data": [reasoning_item().model_dump(mode="json", exclude_none=True)],
        **changed,
    }
    wire = backend._format_messages("", [assistant(native)], backend.model)
    assert len(wire) == 1
    assert wire[0]["role"] == "assistant"


def test_openai_uncompleted_stream_does_not_publish_reasoning():
    backend = bare(OpenAIBackend, "gpt-5.3")
    backend.last_reasoning_context = {"stale": True}
    backend._client.responses.create.return_value = iter(openai_events(completed=False))
    assert backend.generate_response("sys", []) == "answer"
    assert backend.last_reasoning_context == {}


def test_openai_retry_keeps_only_successful_reasoning():
    backend = bare(OpenAIBackend, "gpt-5.3")
    backend._client.responses.create.side_effect = [
        failing_stream(openai_events(reasoning_item("partial"))),
        iter(openai_events(reasoning_item("successful"))),
    ]
    assert backend.generate_response("sys", []) == "answer"
    assert [i["id"] for i in backend.last_reasoning_context["data"]] == ["successful"]


@pytest.mark.parametrize("kind", ["response.failed", "response.incomplete", "error"])
def test_openai_explicit_failure_never_publishes_native_items(kind):
    backend = bare(OpenAIBackend, "gpt-5.3")
    backend._client.responses.create.side_effect = lambda **kw: iter(
        openai_events(completed=False) + [event(kind)]
    )
    with pytest.raises(Exception, match="Maximum retries"):
        backend.generate_response("sys", [])
    assert backend.last_reasoning_context == {}


def gemini_chunk(parts, finish=None, usage=None):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            index=0, content=types.Content(role="model", parts=parts),
            finish_reason=finish,
        )],
        usage_metadata=usage,
    )


def gemini_chunks(signature=b"\x00\xff", finished=True):
    return [
        gemini_chunk([types.Part(text="private thought", thought=True)]),
        gemini_chunk([types.Part(text="answer", thought_signature=signature)]),
        gemini_chunk(
            [types.Part(text=" tail")],
            finish="STOP" if finished else None,
            usage=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100, candidates_token_count=5,
                thoughts_token_count=11, cached_content_token_count=20,
            ),
        ),
    ]


def test_gemini_ordered_signature_roundtrip_thought_ui_and_usage():
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    backend._client.models.generate_content_stream.return_value = iter(gemini_chunks())
    assert backend.generate_response("sys", []) == "answer tail"
    assert backend.stream_handler.get_buffered_text() == "answer tail"
    assert backend.stream_handler.get_buffered_reasoning() == "private thought"
    assert backend.last_output_tokens == 16
    assert backend.last_total_context_tokens == 116
    assert backend.cost == pytest.approx(backend.calculate_cost(80, 16, cache_read_tokens=20))
    assert backend.cost_without_cache == pytest.approx(backend.calculate_cost(100, 16))
    native = json.loads(json.dumps(backend.last_reasoning_context))
    assert native["format"] == "gemini"
    assert native["data"] == [
        {"text": "private thought", "thought": True},
        {"text": "answer", "thought_signature": "AP8="},
        {"text": " tail"},
    ]
    replay = backend._translate_messages([assistant(native, text="rewritten answer")])
    assert [p.text for p in replay[0].parts] == ["private thought", "answer", " tail"]
    assert replay[0].parts[1].thought_signature == b"\x00\xff"
    assert replay[0].parts[0].thought is True


@pytest.mark.parametrize("changed", [{"format": "anthropic"}, {"model": "different"}])
def test_gemini_replay_gated_by_format_and_model(changed):
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    native = {
        "format": "gemini", "model": backend.model,
        "data": [{"text": "secret", "thought": True, "thought_signature": "AP8="}],
        **changed,
    }
    replay = backend._translate_messages([assistant(native)])
    assert len(replay[0].parts) == 1
    assert replay[0].parts[0].text == "answer"
    assert replay[0].parts[0].thought_signature is None


def test_gemini_signature_only_part_preserves_position_and_bytes():
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    backend._client.models.generate_content_stream.return_value = iter([
        gemini_chunk([
            types.Part(text="reasoning", thought=True),
            types.Part(thought_signature=b"\x80\x00"),
            types.Part(text="answer"),
        ], finish="STOP"),
    ])
    assert backend.generate_response("sys", []) == "answer"
    wire = backend._translate_messages([assistant(backend.last_reasoning_context)])
    assert wire[0].parts[1].thought_signature == b"\x80\x00"
    assert wire[0].parts[1].text is None
    assert wire[0].parts[2].text == "answer"


@pytest.mark.parametrize("signature,finished", [(None, True), (b"sig", False)])
def test_gemini_requires_native_signature_and_completed_candidate(signature, finished):
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    backend.last_reasoning_context = {"stale": True}
    backend._client.models.generate_content_stream.return_value = iter(
        gemini_chunks(signature=signature, finished=finished)
    )
    assert backend.generate_response("sys", []) == "answer tail"
    assert backend.last_reasoning_context == {}


def test_gemini_retry_does_not_keep_partial_parts():
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    backend._client.models.generate_content_stream.side_effect = [
        failing_stream(gemini_chunks(signature=b"failed")),
        iter(gemini_chunks(signature=b"success")),
    ]
    assert backend.generate_response("sys", []) == "answer tail"
    replay = backend._translate_messages([assistant(backend.last_reasoning_context)])
    assert len(replay[0].parts) == 3
    assert replay[0].parts[1].thought_signature == b"success"


def test_gemini_reasoning_only_turn_is_empty_not_command_output():
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    backend._client.models.generate_content_stream.return_value = iter([
        gemini_chunk([
            types.Part(text="Command: must_not_execute", thought=True,
                       thought_signature=b"sig"),
        ], finish="STOP"),
    ])
    with pytest.raises(EmptyResponseError):
        backend.generate_response("sys", [])
    assert backend.stream_handler.get_buffered_text() == ""
    assert backend.last_reasoning_context == {}


def test_gemini_function_calls_use_candidate_parts_not_convenience_accessor():
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    backend.stream_handler.on_tool_call = mock.Mock()
    backend._client.models.generate_content_stream.return_value = iter([
        gemini_chunk([
            types.Part(function_call=types.FunctionCall(name="read_file", args={"p": "x"})),
            types.Part(text="answer"),
        ], finish="STOP"),
    ])
    assert backend.generate_response("sys", []) == "answer"
    backend.stream_handler.on_tool_call.assert_called_once_with("read_file", "{'p': 'x'}")


def test_gemini_cache_checks_content_and_signature_not_just_prefix_length():
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    native = {
        "format": "gemini", "model": backend.model,
        "data": [{"text": "answer", "thought_signature": "AP8="}],
    }
    context = [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        assistant(native),
        {"role": "user", "content": [{"type": "text", "text": "next"}]},
    ]
    backend._client.caches.create.return_value = NS(
        name="cached/prefix", usage_metadata=NS(total_token_count=20),
    )
    assert backend._create_cache("sys", context)
    assert backend._is_cache_valid("sys", context)
    changed = copy.deepcopy(context)
    changed[0]["content"][0]["text"] = "changed"
    assert not backend._is_cache_valid("sys", changed)
    changed = copy.deepcopy(context)
    changed[1]["reasoning_context"]["data"][0]["thought_signature"] = "AAAA"
    assert not backend._is_cache_valid("sys", changed)
    assert not backend._is_cache_valid("changed system", context)
    # Tail changes are safe: this portion isn't in the cache.
    context[-1]["content"][0]["text"] = "new tail"
    assert backend._is_cache_valid("sys", context)
    backend._delete_cache()
    assert backend._cached_prefix_digest is None


def test_anthropic_replay_does_not_promote_unsigned_or_plain_reasoning():
    backend = bare(AnthropicBackend, "claude-opus-4-6")
    native = {
        "format": "anthropic", "model": backend.model,
        "data": [
            {"type": "thinking", "thinking": "unsigned"},
            {"type": "text", "text": "plain reasoning"},
        ],
    }
    context = [assistant(native)]
    context[0]["reasoning"] = "legacy plain reasoning"
    assert backend._format_messages(context, backend.model)[0]["content"] == [
        {"type": "text", "text": "answer"}
    ]


def test_gemini_does_not_mix_alternate_candidate_signatures_or_thoughts():
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    chunk = gemini_chunk([
        types.Part(text="answer", thought_signature=b"first"),
    ], finish="STOP")
    chunk.candidates.append(types.Candidate(
        index=1, finish_reason="STOP",
        content=types.Content(role="model", parts=[
            types.Part(text="alternate", thought=True, thought_signature=b"second"),
        ]),
    ))
    backend._client.models.generate_content_stream.return_value = iter([chunk])
    assert backend.generate_response("sys", []) == "answer"
    assert backend.stream_handler.get_buffered_reasoning() == ""
    assert len(backend.last_reasoning_context["data"]) == 1


def test_openai_plain_success_after_native_success_clears_metadata():
    backend = bare(OpenAIBackend, "gpt-5.3")
    backend._client.responses.create.side_effect = [
        iter(openai_events()),
        iter([
            event("response.output_text.delta", delta="plain"),
            event("response.completed", response=NS(output=[], usage=None)),
        ]),
    ]
    assert backend.generate_response("sys", []) == "answer"
    assert backend.last_reasoning_context
    assert backend.generate_response("sys", []) == "plain"
    assert backend.last_reasoning_context == {}


def test_gemini_cache_fallback_does_not_publish_failed_stream_signatures():
    backend = bare(GeminiBackend, "gemini-3-flash-preview")
    backend.is_local = False
    backend._cache_name = "cached/old"
    backend._is_cache_valid = mock.Mock(return_value=True)
    backend._client.models.generate_content_stream.side_effect = [
        failing_stream(gemini_chunks(signature=b"cached"), RuntimeError("cache expired")),
        iter(gemini_chunks(signature=b"uncached")),
    ]
    assert backend.generate_response("sys", []) == "answer tail"
    replay = backend._translate_messages([assistant(backend.last_reasoning_context)])
    assert replay[0].parts[1].thought_signature == b"uncached"
    assert backend._cache_name is None



@pytest.mark.parametrize("cls,model", [
    (DeepSeekBackend, "deepseek-v4-pro"), (MinimaxBackend, "MiniMax-M2.5"),
])
def test_compatible_anthropic_preserves_unsigned_native_thinking(cls, model):
    backend = bare(cls, model)
    block = ThinkingBlock(type="thinking", thinking="native unsigned", signature="")
    backend._get_response = mock.Mock(return_value=anthropic_response([
        block, TextBlock(type="text", text="answer"),
    ]))
    backend.generate_response("sys", [])
    native = json.loads(json.dumps(backend.last_reasoning_context))
    assert native["data"] == [block.model_dump(mode="json", exclude_none=True)]
    assert backend._format_messages([assistant(native)], model)[0]["content"][0] == native["data"][0]
    # A Claude client must never accept these unsigned native blocks.
    assert AnthropicBackend._format_messages([assistant(native)], model)[0]["content"] == [
        {"type": "text", "text": "answer"}
    ]
