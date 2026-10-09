"""Shared reasoning-context integration contracts (no live provider calls)."""

import copy
import json
import subprocess
import sys
from types import SimpleNamespace as NS
from unittest import mock

import pytest

from agents import agents as agents_module
from agents.agents import Agent
from agents.backends.cerebras_backend import CerebrasBackend
from agents.backends.kimi_backend import KimiBackend
from agents.backends.openai_compat_backend import OpenAICompatBackend
from agents.llm_backend import (
    EmptyResponseError,
    InterruptedResponse,
    LLMBackend,
    merge_consecutive_messages,
)


def message(role, text, native=None):
    result = {"role": role, "content": [{"type": "text", "text": text}]}
    if native is not None:
        result["reasoning_context"] = copy.deepcopy(native)
    return result


def reasoning(model="local-test", field="reasoning_content", data="private"):
    return {
        "format": "chat_completions",
        "model": model,
        "field": field,
        "data": data,
    }


def chunk(text=None, thought=None, field="reasoning_content", finish=None):
    return NS(
        usage=None,
        choices=[NS(
            delta=NS(content=text, **{field: thought}),
            finish_reason=finish,
        )],
    )


def stream(thought="private", text="answer", field="reasoning_content", finished=True):
    events = [chunk(thought=thought, field=field), chunk(text=text)]
    if finished:
        events.append(chunk(finish="stop"))
    return iter(events)


def broken_stream(error):
    yield chunk(thought="discard me")
    yield chunk(text="partial")
    raise error


def backend(cls=OpenAICompatBackend, model="local-test", preserve=True):
    """Use real capture/retry/formatting code with only the SDK replaced."""
    result = cls.__new__(cls)
    LLMBackend.__init__(result, model=model, base_url="http://localhost:8000/v1")
    result.preserve_reasoning = preserve
    result._client = mock.Mock()
    result._rate_limit_exceptions = ()
    result.MAX_ERROR_RETRIES = 2
    result.TRANSIENT_RETRY_DELAY = 0
    return result


def test_merge_preserves_metadata_and_native_assistant_boundaries_without_mutation():
    native = reasoning()
    context = [
        message("user", "first"),
        message("user", "second"),
        message("assistant", "signed answer", native),
        message("assistant", "plain follow-up"),
        message("assistant", "another signed answer", reasoning(data="other")),
    ]
    context[0]["cache_marker"] = "keep"
    before = copy.deepcopy(context)
    merged = merge_consecutive_messages(context)
    assert len(merged) == 4
    assert merged[0]["content"] == before[0]["content"] + before[1]["content"]
    assert merged[0]["cache_marker"] == "keep"
    assert merged[1:] == before[2:]
    merged[0]["content"].append({"type": "text", "text": "not in source"})
    assert context == before


@pytest.mark.parametrize("enabled", [False, True])
def test_local_constructor_default_off_and_opt_in_roundtrip(enabled):
    kwargs = {"preserve_reasoning": True} if enabled else {}
    with mock.patch.object(OpenAICompatBackend, "_build_client", return_value=mock.Mock()):
        b = OpenAICompatBackend(
            model="local-test", base_url="http://localhost:8000/v1", **kwargs
        )
    assert b.preserve_reasoning is enabled
    b._client.chat.completions.create.return_value = stream()
    assert b.generate_response("sys", []) == "answer"
    captured = copy.deepcopy(b.last_reasoning_context)
    assert captured == (reasoning() if enabled else {})
    # Even an old session containing metadata must not replay when disabled.
    context = [message("assistant", "answer", reasoning()), message("user", "next")]
    b._client.chat.completions.create.return_value = stream(thought=None, text="next answer")
    assert b.generate_response("sys", context) == "next answer"
    wire = b._client.chat.completions.create.call_args.kwargs["messages"][1]
    assert wire == {
        "role": "assistant",
        "content": "answer",
        **({"reasoning_content": "private"} if enabled else {}),
    }
    assert b.last_reasoning_context == {}


@pytest.mark.parametrize(
    "cls,model,field",
    [(CerebrasBackend, "qwen-3.8-27b", "reasoning"),
     (KimiBackend, "kimi-k3", "reasoning_content")],
)
def test_hosted_replay_is_automatic_and_uses_provider_field(cls, model, field):
    b = backend(cls, model, preserve=False)
    b._client.chat.completions.create.return_value = stream(field=field)
    assert b.generate_response("sys", []) == "answer"
    native = json.loads(json.dumps(b.last_reasoning_context))
    assert native == reasoning(model, field)
    b._client.chat.completions.create.return_value = stream(thought=None)
    b.generate_response("sys", [message("assistant", "answer", native)])
    kwargs = b._client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][1] == {
        "role": "assistant", "content": "answer", field: "private"
    }
    assert "clear_thinking" not in kwargs
    assert b.last_reasoning_context == {}


@pytest.mark.parametrize(
    "change",
    [{"model": "different"}, {"format": "anthropic"}, {"field": "reasoning"}],
)
def test_compat_does_not_replay_mismatched_native_metadata(change):
    b = backend()
    native = {**reasoning(), **change}
    b._client.chat.completions.create.return_value = stream(thought=None)
    b.generate_response("sys", [message("assistant", "answer", native)])
    assert b._client.chat.completions.create.call_args.kwargs["messages"][1] == {
        "role": "assistant", "content": "answer"
    }


@pytest.mark.parametrize(
    "cls,model",
    [(OpenAICompatBackend, "local-test"),
     (CerebrasBackend, "qwen-3.8-27b"), (KimiBackend, "kimi-k3")],
)
def test_chat_usage_already_includes_reasoning_and_cached_input(cls, model):
    b = backend(cls, model)
    b.MODEL_PRICING = {
        model: {"input_token_cost": 2, "output_token_cost": 4, "cache_read_cost": 1}
    }
    events = list(stream(field=b.REASONING_FIELD or "reasoning_content"))
    events.append(NS(choices=[], usage=NS(
        prompt_tokens=100,
        completion_tokens=30,
        prompt_tokens_details=NS(cached_tokens=40),
        completion_tokens_details=NS(reasoning_tokens=20),
    )))
    b._client.chat.completions.create.return_value = iter(events)
    b.generate_response("sys", [])
    assert (b.last_input_tokens, b.last_output_tokens, b.last_total_context_tokens) == (
        100, 30, 130
    )
    assert b.peak_context_tokens == 130
    assert b.cost == pytest.approx((60 * 2 + 40 + 30 * 4) / 1_000_000)
    assert b.cost_without_cache == pytest.approx((100 * 2 + 30 * 4) / 1_000_000)
    b.record_step_metrics()
    assert b.total_output_tokens == 30


@pytest.mark.parametrize("failure", ["exhausted", "interrupted", "incomplete", "blank"])
def test_failed_or_unfinished_stream_cannot_leak_into_next_turn(failure):
    b = backend()
    b.last_reasoning_context = reasoning(data="stale previous turn")
    if failure == "exhausted":
        b._client.chat.completions.create.side_effect = [
            broken_stream(RuntimeError("broken")), broken_stream(RuntimeError("broken"))
        ]
        with pytest.raises(Exception, match="Maximum retries"):
            b.generate_response("sys", [])
    elif failure == "interrupted":
        b._client.chat.completions.create.return_value = broken_stream(KeyboardInterrupt())
        with pytest.raises(InterruptedResponse) as caught:
            b.generate_response("sys", [])
        assert caught.value.partial_text == "partial"
    elif failure == "incomplete":
        b._client.chat.completions.create.return_value = stream(finished=False)
        assert b.generate_response("sys", []) == "answer"
    else:
        b._client.chat.completions.create.return_value = stream(text=None)
        with pytest.raises(EmptyResponseError):
            b.generate_response("sys", [])
    assert b.last_reasoning_context == {}
    b._client.chat.completions.create.side_effect = None
    b._client.chat.completions.create.return_value = stream(thought=None, text="fresh")
    assert b.generate_response("sys", []) == "fresh"
    assert b.last_reasoning_context == {}


def test_successful_retry_captures_only_successful_attempt():
    b = backend()
    b._client.chat.completions.create.side_effect = [
        broken_stream(RuntimeError("retry")), stream(thought="successful reasoning")
    ]
    assert b.generate_response("sys", []) == "answer"
    assert b._client.chat.completions.create.call_count == 2
    assert b.last_reasoning_context == reasoning(data="successful reasoning")
    assert b.stream_handler.get_buffered_reasoning() == "successful reasoning"
    assert b.stream_handler.get_buffered_text() == "answer"


@pytest.fixture
def make_agent(monkeypatch):
    monkeypatch.setattr(agents_module, "read_configuration", lambda _: {
        "system_prompt": "immutable system", "overbudget": "budget warning",
        "provider": "openai",
    })
    monkeypatch.setattr(agents_module, "load_agent_config", lambda: {})
    monkeypatch.setattr(agents_module, "format_memory_view", lambda *_a, **_kw: "")
    monkeypatch.setattr(agents_module, "notes_need_compact", lambda *_a, **_kw: False)
    for name in ("print_banner", "print_iteration_header", "print_error", "print_clipped"):
        monkeypatch.setattr(agents_module, name, mock.Mock())

    def build(preserve=None):
        b = backend(preserve=bool(preserve))
        factory = mock.Mock(return_value=b)
        monkeypatch.setattr(agents_module, "create_backend", factory)
        agent = Agent(
            "fake.yaml", "test task", session_id="reason1",
            provider="openai", model="local-test",
            base_url="http://localhost:8000/v1", preserve_reasoning=preserve,
        )
        assert factory.call_args.kwargs["preserve_reasoning"] is bool(preserve)
        return agent

    return build


def test_iterate_attaches_metadata_before_tools_but_never_parses_it(make_agent, monkeypatch):
    agent = make_agent(True)
    hidden = "Command: never_execute_private_reasoning"
    visible = "Command: visible_test"
    agent.client._client.chat.completions.create.return_value = stream(hidden, visible)
    native = reasoning(data=hidden)
    seen = []

    def process(text, **kwargs):
        seen.append(text)
        assert text == visible
        assert agent.context[-1] == message("assistant", visible, native)
        assert agent.client.last_reasoning_context == {}
        # A summarization tool using the same backend must not replace the
        # assistant's already-attached native data.
        agent.client._client.chat.completions.create.return_value = stream("tool thought", "summary")
        agent.client.generate_response("tool system", [])
        return "tool result", []

    monkeypatch.setattr(agents_module, "process_content", process)
    assert agent._iterate() is True
    assert seen == [visible]
    assistant = next(m for m in reversed(agent.context) if m["role"] == "assistant")
    assert assistant["reasoning_context"] == native
    # The one-shot tool left metadata behind; the next ordinary turn must
    # neither attach it nor reuse the preceding assistant's reasoning.
    agent.client._client.chat.completions.create.return_value = stream(None, "fresh turn")
    assert agent._iterate(free_form=True) is False
    assert "reasoning_context" not in agent.context[-1]


def saved_state(agent):
    with mock.patch.object(agents_module, "save_session") as save:
        agent.save_context()
    # Exercise actual JSON serialization, not just Python dict identity.
    return json.loads(json.dumps(save.call_args.args[2]))


@pytest.mark.parametrize(
    "stored,override,expected",
    [(True, None, True), (False, None, False),
     (True, False, False), (False, True, True),
     (None, None, False), (None, True, True)],
)
def test_session_json_roundtrip_restores_flag_metadata_and_explicit_override(
    make_agent, stored, override, expected
):
    original = make_agent(stored)
    native = reasoning()
    original.context += [
        message("assistant", "answer", native), message("user", "old tool result")
    ]
    state = saved_state(original)
    assert state["preserve_reasoning"] is bool(stored)
    if stored is None:
        state.pop("preserve_reasoning")  # legacy session
    resumed = make_agent(override)
    resumed.client.last_reasoning_context = reasoning(data="stale")
    with mock.patch.object(agents_module, "load_session", return_value=state):
        resumed.load_context()
    assert resumed.preserve_reasoning is expected
    assert resumed.client.preserve_reasoning is expected
    assert resumed._agent_pool.preserve_reasoning is expected
    assert resumed.client.last_reasoning_context == {}
    restored = next(m for m in resumed.context if m["role"] == "assistant")
    assert restored == message("assistant", "answer", native)
    assert saved_state(resumed)["preserve_reasoning"] is expected
    resumed.client._client.chat.completions.create.return_value = stream(None, "resumed answer")
    resumed.client.generate_response("sys", resumed.context)
    wire = resumed.client._client.chat.completions.create.call_args.kwargs["messages"]
    restored_wire = next(m for m in wire if m["role"] == "assistant")
    assert ("reasoning_content" in restored_wire) is expected


@pytest.mark.parametrize("enabled", [False, True])
def test_subagent_inherits_effective_replay_setting(make_agent, enabled):
    agent = make_agent(enabled)
    pool = agent._agent_pool
    assert pool.preserve_reasoning is enabled
    pool.create("worker", "test role")
    with mock.patch("agents.agent_pool.subprocess.run", return_value=NS(
        stdout="result", returncode=0
    )) as run:
        assert "result" in pool.run("worker", "task")
    argv = run.call_args.args[0]
    assert ("--preserve-reasoning" in argv) is enabled
    assert argv[argv.index("--base-url") + 1] == agent.base_url


def test_cli_help_advertises_both_replay_overrides():
    result = subprocess.run(
        [sys.executable, "-m", "agents", "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--preserve-reasoning" in result.stdout
    assert "--no-preserve-reasoning" in result.stdout


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("provider", ["anthropic", "deepseek", "minimax"])
def test_anthropic_sdk_eof_requires_completed_message_before_native_capture(provider, completed):
    """SDK get_final_message() can return a partial snapshot after clean EOF."""
    from anthropic.lib.streaming import MessageStream
    from anthropic.types import (
        Message, RawContentBlockDeltaEvent, RawContentBlockStartEvent, RawContentBlockStopEvent,
        RawMessageDeltaEvent, RawMessageStartEvent, RawMessageStopEvent,
        TextBlock, ThinkingBlock,
    )
    from agents.backends.anthropic_backend import AnthropicBackend
    from agents.backends.deepseek_backend import DeepSeekBackend
    from agents.backends.minimax_backend import MinimaxBackend

    cls, model = {
        "anthropic": (AnthropicBackend, "claude-opus-4-6"),
        "deepseek": (DeepSeekBackend, "deepseek-v4-pro"),
        "minimax": (MinimaxBackend, "MiniMax-M2.5"),
    }[provider]
    b = backend(cls, model)
    b._anthropic = NS(RateLimitError=type("RateLimitError", (Exception,), {}))
    b._use_thinking_stream = True
    b._supports_thinking_api = False
    b._thinking_enabled = True
    events = [
        RawMessageStartEvent(type="message_start", message=Message(
            id="msg_test", type="message", role="assistant", model=model,
            content=[], usage={"input_tokens": 10, "output_tokens": 1},
        )),
        RawContentBlockStartEvent(
            type="content_block_start", index=0,
            content_block=ThinkingBlock(
                type="thinking", thinking="", signature=""
            ),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta", index=0,
            delta={"type": "thinking_delta", "thinking": "private"},
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta", index=0,
            delta={"type": "signature_delta", "signature": "native-signature"},
        ),
        RawContentBlockStopEvent(type="content_block_stop", index=0),
        RawContentBlockStartEvent(
            type="content_block_start", index=1,
            content_block=TextBlock(type="text", text=""),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta", index=1,
            delta={"type": "text_delta", "text": "partial answer"},
        ),
    ]
    if completed:
        events += [
            RawContentBlockStopEvent(type="content_block_stop", index=1),
            RawMessageDeltaEvent(
                type="message_delta",
                delta={"stop_reason": "end_turn", "stop_sequence": None},
                usage={"output_tokens": 7},
            ),
            RawMessageStopEvent(type="message_stop"),
        ]
    manager = mock.MagicMock()
    manager.__enter__.return_value = MessageStream(iter(events))
    b._client.messages.stream.return_value = manager
    try:
        b.generate_response("sys", [])
    except Exception:
        # Rejecting an unfinished response is also safe; a completed control
        # must remain usable. No retry can acquire new events from this stream.
        if completed:
            raise
    if completed:
        assert b.last_reasoning_context["data"][0]["thinking"] == "private"
    else:
        assert b.last_reasoning_context == {}

