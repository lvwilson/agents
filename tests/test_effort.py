"""Tests for the ``--effort`` / reasoning-effort feature.

The canonical effort scale (``none, low, medium, high, xhigh, max``)
lives in :mod:`agents.llm_backend`; every backend clamps a user request
to the levels its models actually accept — so one flag works across
providers without 400s:

* Cerebras ``qwen-3.8-27b``: none / low / medium / high
* Cerebras ``gpt-oss-120b``: low / medium / high (cannot disable)
* Kimi K3: max only ("more levels are coming soon")
* DeepSeek ``output_config.effort``: low / high / max

Without a flag the backends emit exactly what they always have (the
per-model defaults), so no request shape changes for existing users.
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import agents as agents_module  # noqa: E402
from agents.agent_pool import AgentPool  # noqa: E402
from agents.backends.cerebras_backend import CerebrasBackend  # noqa: E402
from agents.backends.deepseek_backend import DeepSeekBackend  # noqa: E402
from agents.backends.kimi_backend import KimiBackend  # noqa: E402
from agents.llm_backend import StreamHandler, map_effort, parse_effort  # noqa: E402

# ── Construction helpers (SDK clients mocked; no network) ─────────────

def _make_cerebras(model="qwen-3.8-27b", **kwargs):
    with mock.patch.dict(os.environ, {"CEREBRAS_API_KEY": "sk-test"}):
        with mock.patch("cerebras.cloud.sdk.Cerebras"):
            return CerebrasBackend(model=model, **kwargs)


def _make_kimi(**kwargs):
    with mock.patch.dict(os.environ, {"MOONSHOT_API_KEY": "sk-test"}):
        with mock.patch("openai.OpenAI"):
            return KimiBackend(**kwargs)


def _make_deepseek(**kwargs):
    with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-test"}):
        with mock.patch("anthropic.Anthropic"):
            return DeepSeekBackend(**kwargs)


# ── Canonical scale ──────────────────────────────────────────────────

class TestParseEffort(unittest.TestCase):
    """parse_effort normalises user tokens onto the canonical scale."""

    def test_canonical_values_keep_rank(self):
        self.assertEqual(parse_effort("none"), 0)
        self.assertEqual(parse_effort("low"), 1)
        self.assertEqual(parse_effort("medium"), 2)
        self.assertEqual(parse_effort("high"), 3)
        self.assertEqual(parse_effort("xhigh"), 4)
        self.assertEqual(parse_effort("max"), 5)

    def test_aliases(self):
        for token in ("off", "disabled", "no"):
            self.assertEqual(parse_effort(token), 0, token)
        self.assertEqual(parse_effort("mid"), 2)
        for token in ("extra", "ultra", "veryhigh", "x-high"):
            self.assertEqual(parse_effort(token), 4, token)
        for token in ("maximum", "highest"):
            self.assertEqual(parse_effort(token), 5, token)

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(parse_effort("  High "), 3)

    def test_none_and_blank_mean_backend_default(self):
        self.assertIsNone(parse_effort(None))
        self.assertIsNone(parse_effort(""))
        self.assertIsNone(parse_effort("   "))

    def test_unrecognised_raises_with_valid_list(self):
        with self.assertRaises(ValueError) as ctx:
            parse_effort("ultramax")
        for valid in ("none", "low", "medium", "high", "xhigh", "max"):
            self.assertIn(valid, str(ctx.exception))


class TestMapEffort(unittest.TestCase):
    """map_effort clamps a request to the supported subset, tie → lower."""

    def test_none_uses_default(self):
        self.assertEqual(
            map_effort(None, ("none", "low", "medium", "high"), "high"),
            "high",
        )

    def test_exact_match_passes_through(self):
        self.assertEqual(
            map_effort("low", ("low", "high", "max"), "max"), "low"
        )

    def test_clamps_to_nearest_supported(self):
        # medium between low(1) and high(3) → tie → lower (low)? No:
        # nearest of {1,3} to 2 is a tie → lower rank wins → low.
        self.assertEqual(
            map_effort("medium", ("low", "high", "max"), "max"), "low"
        )
        # xhigh(4) between high(3) and max(5) → tie → high.
        self.assertEqual(
            map_effort("xhigh", ("low", "high", "max"), "max"), "high"
        )
        # max(5) → max (exact).
        self.assertEqual(
            map_effort("max", ("low", "high", "max"), "max"), "max"
        )
        # none(0) on a set without none → low (shallowest).
        self.assertEqual(
            map_effort("none", ("low", "high", "max"), "max"), "low"
        )

    def test_higher_rank_maps_down_without_disabling(self):
        # gpt-oss (low/medium/high, no none): max → high, none → low.
        levels = ("low", "medium", "high")
        self.assertEqual(map_effort("max", levels, "medium"), "high")
        self.assertEqual(map_effort("none", levels, "medium"), "low")

    def test_empty_supported_set_means_unexpressible(self):
        self.assertIsNone(map_effort("high", (), "high"))


# ── Per-backend emission ─────────────────────────────────────────────

class TestCerebrasEffortEmission(unittest.TestCase):
    """Cerebras: per-model levels drive the reasoning_effort param."""

    def test_default_emission_unchanged_without_flag(self):
        self.assertEqual(
            _make_cerebras()._extra_create_kwargs(),
            {"reasoning_effort": "high"},
        )
        self.assertEqual(
            _make_cerebras(model="gpt-oss-120b")._extra_create_kwargs(),
            {"reasoning_effort": "medium"},
        )

    def test_flag_overrides_model_default(self):
        self.assertEqual(
            _make_cerebras(reasoning_effort="low")._extra_create_kwargs(),
            {"reasoning_effort": "low"},
        )
        self.assertEqual(
            _make_cerebras(model="gpt-oss-120b",
                           reasoning_effort="high")._extra_create_kwargs(),
            {"reasoning_effort": "high"},
        )

    def test_none_disables_qwen_but_not_gpt_oss(self):
        # qwen: none is a real level → sent explicitly.  Cerebras applies
        # its own default (high) when the parameter is omitted, so an
        # omission would leave reasoning ON.
        self.assertEqual(
            _make_cerebras(reasoning_effort="none")._extra_create_kwargs(),
            {"reasoning_effort": "none"},
        )
        # gpt-oss cannot disable: none clamps to the shallowest (low).
        self.assertEqual(
            _make_cerebras(model="gpt-oss-120b",
                           reasoning_effort="none")._extra_create_kwargs(),
            {"reasoning_effort": "low"},
        )

    def test_beyond_high_clamps_to_high(self):
        self.assertEqual(
            _make_cerebras(reasoning_effort="xhigh")._extra_create_kwargs(),
            {"reasoning_effort": "high"},
        )
        self.assertEqual(
            _make_cerebras(model="gpt-oss-120b",
                           reasoning_effort="max")._extra_create_kwargs(),
            {"reasoning_effort": "high"},
        )

    def test_unknown_model_keeps_server_default(self):
        # No configured levels (and no configured default) → whatever the
        # server does, unchanged by a flag.
        self.assertEqual(
            _make_cerebras(model="some-future-model",
                           reasoning_effort="max")._extra_create_kwargs(),
            {},
        )


class TestKimiEffortEmission(unittest.TestCase):
    """Kimi K3 only accepts max, so every request clamps to max."""

    def test_default_max(self):
        self.assertEqual(
            _make_kimi()._extra_create_kwargs(), {"reasoning_effort": "max"}
        )

    def test_any_effort_clamps_to_max(self):
        for token in ("none", "low", "medium", "high", "xhigh", "max"):
            self.assertEqual(
                _make_kimi(reasoning_effort=token)._extra_create_kwargs(),
                {"reasoning_effort": "max"},
                token,
            )


class TestDeepSeekEffortEmission(unittest.TestCase):
    """DeepSeek: output_config.effort via extra_body, low/high/max."""

    def _effort(self, backend):
        return backend._extra_stream_kwargs()["extra_body"][
            "output_config"
        ]["effort"]

    def test_default_max(self):
        self.assertEqual(self._effort(_make_deepseek()), "max")

    def test_low_passthrough(self):
        self.assertEqual(
            self._effort(_make_deepseek(reasoning_effort="low")), "low"
        )

    def test_medium_clamps_down_to_low(self):
        # medium(2) is a tie between low(1) and high(3) → lower wins.
        self.assertEqual(
            self._effort(_make_deepseek(reasoning_effort="medium")), "low"
        )

    def test_beyond_high_clamps_to_high(self):
        self.assertEqual(
            self._effort(_make_deepseek(reasoning_effort="xhigh")), "high"
        )

    def test_none_clamps_to_low(self):
        self.assertEqual(
            self._effort(_make_deepseek(reasoning_effort="none")), "low"
        )


# ── Thinking request shapes (de-messed budget mechanism) ─────────────
#
# The Anthropic budget machinery (DEFAULT_THINKING_BUDGET = 8192, the
# CLAUDE_THINKING_BUDGET env var, the context/max_output clamp) only
# ever reaches the wire on "enabled"-mode requests.  Every current
# Claude model is adaptive (the model decides how much to think — no
# budget sent), and the Anthropic-compatible subclasses (DeepSeek,
# MiniMax) request the bare enabled field because their endpoints
# ignore budget_tokens.  These tests pin that contract so the dead
# paths don't creep back.

class TestThinkingConfigShapes(unittest.TestCase):
    """_thinking_config() sends exactly what each endpoint honours."""

    def test_all_current_models_are_adaptive(self):
        from agents.backends.anthropic_backend import AnthropicBackend
        self.assertEqual(
            AnthropicBackend.THINKING_MODELS,
            AnthropicBackend.ADAPTIVE_THINKING_MODELS,
        )

    def test_anthropic_adaptive_sends_no_budget(self):
        from agents.backends.anthropic_backend import AnthropicBackend
        with mock.patch.dict(
                os.environ, {"CLAUDE_API_KEY": "sk-test",
                             "CLAUDE_THINKING_ENABLED": "true",
                             "CLAUDE_THINKING_BUDGET": "16384"}):
            with mock.patch("anthropic.Anthropic"):
                backend = AnthropicBackend(model="claude-opus-4-6")
        self.assertEqual(backend._thinking_config(), {"type": "adaptive"})
        # The budget is resolved/clamped in __init__ but never shipped
        # for adaptive models.
        self.assertEqual(backend._thinking_budget, 16384)

    def test_anthropic_enabled_mode_sends_clamped_budget(self):
        # Hypothetical non-adaptive thinking model → budget is sent,
        # clamped below max_output.
        from agents.backends.anthropic_backend import AnthropicBackend
        with mock.patch.dict(os.environ, {"CLAUDE_API_KEY": "sk-test",
                                          "CLAUDE_THINKING_ENABLED": "true"}):
            with mock.patch("anthropic.Anthropic"):
                backend = AnthropicBackend(model="claude-fable-5")
        cfg = backend._thinking_config()
        self.assertEqual(cfg["type"], "enabled")
        self.assertLess(cfg["budget_tokens"],
                        AnthropicBackend.MODEL_MAX_OUTPUT["claude-fable-5"])

    def test_deepseek_sends_bare_enabled_no_budget(self):
        backend = _make_deepseek()
        self.assertEqual(backend._thinking_config(), {"type": "enabled"})

    def test_minimax_sends_bare_enabled_no_budget(self):
        from agents.backends.minimax_backend import MinimaxBackend
        with mock.patch.dict(os.environ, {"MINIMAX_API_KEY": "sk-api-kt-test"}):
            with mock.patch("anthropic.Anthropic"):
                backend = MinimaxBackend()
        self.assertEqual(backend._thinking_config(), {"type": "enabled"})

    def test_anthropic_get_response_omits_budget_for_adaptive(self):
        from agents.backends.anthropic_backend import AnthropicBackend
        with mock.patch.dict(os.environ, {"CLAUDE_API_KEY": "sk-test",
                                          "CLAUDE_THINKING_ENABLED": "true"}):
            with mock.patch("anthropic.Anthropic") as client_cls:
                backend = AnthropicBackend(model="claude-opus-4-6")
        backend.stream_handler = StreamHandler()
        stream = mock.MagicMock()
        stream.__iter__ = lambda self: iter([])
        stream.get_final_message.return_value = _fake_message()
        ctx = backend._client.messages.stream.return_value
        ctx.__enter__.return_value = stream
        backend._get_response("s", [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ])
        kwargs = backend._client.messages.stream.call_args.kwargs
        self.assertEqual(kwargs["thinking"], {"type": "adaptive"})
        self.assertNotIn("budget_tokens", kwargs["thinking"])

    def test_cerebras_explicit_none_is_sent(self):
        # Regression: --effort none must reach the wire (Cerebras
        # defaults to high when the parameter is omitted).
        self.assertEqual(
            _make_cerebras(reasoning_effort="none")._extra_create_kwargs(),
            {"reasoning_effort": "none"},
        )
        self.assertEqual(
            _make_cerebras()._extra_create_kwargs(),
            {"reasoning_effort": "high"},
        )


def _fake_message():
    """Minimal fake stream final message for AnthropicBackend tests."""
    from types import SimpleNamespace
    return SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=1, output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        content=[],
    )


# ── Agent / CLI plumbing ─────────────────────────────────────────────

_FAKE_CONFIG = {
    "system_prompt": "IMMUTABLE SYSTEM PROMPT",
    "overbudget": "over budget",
    "provider": "anthropic",
}


def _make_agent(**kwargs):
    """Construct a minimal Agent with the backend mocked out."""
    with mock.patch.object(agents_module, "read_configuration",
                           return_value=_FAKE_CONFIG), \
         mock.patch.object(agents_module, "format_memory_view",
                           return_value=""), \
         mock.patch.object(agents_module, "notes_need_compact",
                           return_value=False), \
         mock.patch.object(agents_module, "create_backend") as mock_backend, \
         mock.patch.object(agents_module, "print_banner"):
        client = mock_backend.return_value
        client.display_name = "MockModel"
        client.context_window_size = 200_000
        client.cost = 0.0
        client.cost_without_cache = 0.0
        client.peak_context_tokens = 0
        client.last_input_tokens = 0
        client.last_output_tokens = 0
        client.last_total_context_tokens = 0
        return agents_module.Agent("fake.yaml", "do the thing",
                                   session_id="tst1", **kwargs), mock_backend


class TestAgentEffortPlumbing(unittest.TestCase):
    """Agent resolves effort and forwards it to the backend + pool."""

    def test_default_keeps_backend_defaults(self):
        agent, mock_backend = _make_agent()
        self.assertIsNone(agent.reasoning_effort)
        kwargs = mock_backend.call_args.kwargs
        self.assertNotIn("reasoning_effort", kwargs)
        self.assertIsNone(agent._agent_pool.effort)

    def test_flag_effort_forwarded_to_backend_and_pool(self):
        agent, mock_backend = _make_agent(effort="low")
        self.assertEqual(agent.reasoning_effort, 1)  # canonical rank
        self.assertEqual(
            mock_backend.call_args.kwargs.get("reasoning_effort"), 1
        )
        self.assertEqual(agent._agent_pool.effort, 1)

    def test_config_file_effort_used_when_no_flag(self):
        cfg = dict(_FAKE_CONFIG, effort="high")
        with mock.patch.object(agents_module, "read_configuration",
                               return_value=cfg), \
             mock.patch.object(agents_module, "format_memory_view",
                               return_value=""), \
             mock.patch.object(agents_module, "notes_need_compact",
                               return_value=False), \
             mock.patch.object(agents_module, "create_backend") as mb, \
             mock.patch.object(agents_module, "print_banner"):
            mb.return_value.display_name = "M"
            mb.return_value.context_window_size = 1
            mb.return_value.cost = 0.0
            mb.return_value.cost_without_cache = 0.0
            mb.return_value.peak_context_tokens = 0
            mb.return_value.last_input_tokens = 0
            mb.return_value.last_output_tokens = 0
            mb.return_value.last_total_context_tokens = 0
            agent = agents_module.Agent(
                "fake.yaml", "t", session_id="tst2",
                model="claude-opus-4-6",
            )
            # Agent __init__ reads load_agent_config() for the config
            # layer; with no .agent/global file and no AGENT_EFFORT the
            # effort comes from the agent YAML ("high" → rank 3).
            self.assertEqual(agent.reasoning_effort, 3)
            self.assertEqual(mb.call_args.kwargs.get("reasoning_effort"), 3)

    def test_flag_beats_config(self):
        cfg = dict(_FAKE_CONFIG, effort="max")
        with mock.patch.object(agents_module, "read_configuration",
                               return_value=cfg), \
             mock.patch.object(agents_module, "create_backend") as mb, \
             mock.patch.object(agents_module, "print_banner"):
            mb.return_value.display_name = "M"
            mb.return_value.context_window_size = 1
            agent = agents_module.Agent(
                "fake.yaml", "t", session_id="tst3",
                model="claude-opus-4-6",
                provider="anthropic",
                effort="low",
            )
            self.assertEqual(agent.reasoning_effort, 1)


class TestCliEffort(unittest.TestCase):
    """The -e/--effort CLI flag validates and propagates."""

    def _run_main(self, argv, expect_exit=None):
        with mock.patch.object(sys, "argv", ["agents"] + argv), \
             mock.patch.object(sys.stdin, "isatty", return_value=True), \
             mock.patch.object(agents_module, "run_agent",
                               return_value=("done", True, "sid1")) as run, \
             mock.patch.object(agents_module, "report_home_config"), \
             mock.patch.object(agents_module, "safe_console_print"), \
             mock.patch.object(agents_module, "print_completion_result"):
            try:
                agents_module.main()
                return run
            except SystemExit as e:
                self.assertEqual(e.code, expect_exit)
                self.assertFalse(run.called)
                return None

    def test_valid_effort_propagates(self):
        run = self._run_main(["task", "--nogit", "-e", "mid"])
        self.assertEqual(run.call_args.kwargs.get("effort"), 2)

    def test_no_effort_flag_passes_none(self):
        run = self._run_main(["task", "--nogit"])
        self.assertIsNone(run.call_args.kwargs.get("effort"))

    def test_invalid_effort_is_cli_error(self):
        self._run_main(["task", "--nogit", "--effort", "ultramax"],
                       expect_exit=2)


class TestAgentPoolEffortPropagation(unittest.TestCase):
    """run_agent command lines carry -e when the parent set an effort."""

    def _cmd(self, effort, model="qwen-3.8-27b"):
        pool = AgentPool()
        pool.create("helper", "You help.")
        pool.model = model
        pool.effort = effort
        with mock.patch("subprocess.run") as run_proc:
            run_proc.return_value = mock.Mock(
                stdout="ok\n", returncode=0)
            pool.run("helper", "do a thing", budget=1.0, timeout=5)
        return run_proc.call_args.args[0]

    def test_effort_appended_when_set(self):
        cmd = self._cmd(1)  # rank 1 → low
        self.assertIn("-e", cmd)
        self.assertEqual(cmd[cmd.index("-e") + 1], "low")

    def test_no_effort_flag_when_none(self):
        cmd = self._cmd(None)
        self.assertNotIn("-e", cmd)


if __name__ == "__main__":
    unittest.main()
