"""Regression tests for per-step and final-session metrics.

Covers:
* ``LLMBackend.record_step_metrics`` — accumulating output-token totals,
  generation time, and the derived tokens/second + cost-per-hour rates.
* ``LLMBackend._run_with_retries`` recording ``last_call_duration`` only
  on a *successful* generation (a failed / interrupted attempt must not
  pollute the per-call timing).
* the UI: ``format_rate`` / ``format_duration`` helpers, the per-step
  ``rate:`` field in the iteration header, and the final metrics panel
  (``build_final_metrics``) — including the estimated cost per hour.
* Agent wiring: a successful ``_iterate`` fold step into the metrics via
  ``client.record_step_metrics`` (free-form turns included).
"""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console  # noqa: E402

from agents import agents as agents_module  # noqa: E402
from agents import ui as agents_ui  # noqa: E402
from agents.agents import Agent  # noqa: E402
from agents.llm_backend import LLMBackend, InterruptedResponse  # noqa: E402


# ── Fake backend for metric-math tests ─────────────────────────────────

class _FakeBackend(LLMBackend):
    """Minimal concrete backend to exercise the shared metric logic."""
    MODEL_DISPLAY_NAMES = {"m1": "M1"}

    def generate_response(self, system_prompt, context):
        return "ok"


def _cap_console(width=200):
    """Return an in-memory Rich Console that records what is printed."""
    return Console(
        record=True, force_terminal=True,
        width=width, color_system=None, legacy_windows=False,
    )


# ── record_step_metrics ────────────────────────────────────────────────

class TestRecordStepMetrics(unittest.TestCase):
    def _backend(self):
        return _FakeBackend(model="m1")

    def test_single_step_derives_rates(self):
        b = self._backend()
        b.cost = 1.0
        b.last_call_duration = 10.0
        b.last_output_tokens = 100
        b.record_step_metrics()

        self.assertEqual(b.total_output_tokens, 100)
        self.assertAlmostEqual(b.total_call_duration, 10.0)
        self.assertAlmostEqual(b.step_rate_tokens_per_sec, 10.0)
        self.assertAlmostEqual(b.output_rate_tokens_per_sec, 10.0)
        # $1.00 spent over 10s => $360.00/hour.
        self.assertAlmostEqual(b.cost_per_hour, 360.0)

    def test_accumulates_across_steps(self):
        b = self._backend()
        # Step 1
        b.cost = 0.5
        b.last_call_duration = 10.0
        b.last_output_tokens = 100
        b.record_step_metrics()
        # Step 2 (running cost total now includes step 2 spend)
        b.cost = 1.0
        b.last_call_duration = 10.0
        b.last_output_tokens = 100
        b.record_step_metrics()

        self.assertEqual(b.total_output_tokens, 200)
        self.assertAlmostEqual(b.total_call_duration, 20.0)
        self.assertAlmostEqual(b.output_rate_tokens_per_sec, 10.0)
        # $1.00 total over 20s => $180.00/hour (1 * 3600 / 20).
        self.assertAlmostEqual(b.cost_per_hour, 180.0)

    def test_zero_duration_is_ignored(self):
        """No completed generation (duration 0) must leave metrics alone."""
        b = self._backend()
        b.cost = 0.25
        b.last_call_duration = 0.0
        b.last_output_tokens = 500
        b.record_step_metrics()

        self.assertEqual(b.total_output_tokens, 0)
        self.assertEqual(b.total_call_duration, 0.0)
        self.assertIsNone(b.output_rate_tokens_per_sec)
        self.assertIn(b.cost_per_hour, (None, 0.0))

    def test_zero_output_tokens_still_tracks_time_and_cost(self):
        """A thinking-only step has no output tokens but real time/cost."""
        b = self._backend()
        b.cost = 1.0
        b.last_call_duration = 120.0
        b.last_output_tokens = 0
        b.record_step_metrics()

        self.assertEqual(b.total_output_tokens, 0)
        self.assertAlmostEqual(b.total_call_duration, 120.0)
        self.assertIsNone(b.step_rate_tokens_per_sec)
        # $1.00 over 120s => $30.00/hour (1 * 3600 / 120).
        self.assertAlmostEqual(b.cost_per_hour, 30.0)


# ── _run_with_retries timing ───────────────────────────────────────────

class TestRetryLoopRecordsDuration(unittest.TestCase):
    def test_success_records_positive_duration(self):
        b = _FakeBackend(model="m1")
        b.last_call_duration = 0.0

        def attempt():
            time.sleep(0.05)
            return "done"

        result = b._run_with_retries(attempt)

        self.assertEqual(result, "done")
        self.assertGreaterEqual(b.last_call_duration, 0.05)

    def test_interrupted_attempt_does_not_record(self):
        b = _FakeBackend(model="m1")
        b.last_call_duration = 0.75  # sentinel: must be left untouched

        def attempt():
            raise KeyboardInterrupt()

        with self.assertRaises(InterruptedResponse):
            b._run_with_retries(attempt)

        self.assertEqual(b.last_call_duration, 0.75)


# ── UI helpers ─────────────────────────────────────────────────────────

class TestFormatHelpers(unittest.TestCase):
    def test_format_rate(self):
        self.assertEqual(agents_ui.format_rate(None), "—")
        self.assertEqual(agents_ui.format_rate(85.5), "85.5 tok/s")
        self.assertEqual(agents_ui.format_rate(123), "123.0 tok/s")
        self.assertTrue(agents_ui.format_rate(0.04).endswith(" tok/s"))

    def test_format_duration(self):
        self.assertEqual(agents_ui.format_duration(42), "42s")
        self.assertEqual(agents_ui.format_duration(102), "1m 42s")
        self.assertEqual(agents_ui.format_duration(0), "0s")
        self.assertEqual(agents_ui.format_duration(None), "0s")


class TestIterationHeaderRate(unittest.TestCase):
    def test_rate_shown_when_measured(self):
        c = _cap_console()
        with mock.patch.object(agents_ui, "_console", c):
            agents_ui.print_iteration_header(
                2, 0.1, 2.0,
                last_input_tokens=1000, last_output_tokens=500,
                last_total_context_tokens=5000,
                context_window_tokens=200_000,
                step_tokens_per_sec=85.5,
            )
        self.assertIn("85.5 tok/s", c.export_text())

    def test_rate_hidden_before_measured(self):
        c = _cap_console()
        with mock.patch.object(agents_ui, "_console", c):
            agents_ui.print_iteration_header(
                1, 0.0, 2.0,
                last_input_tokens=1000, last_output_tokens=0,
                last_total_context_tokens=2000,
                context_window_tokens=200_000,
            )
        self.assertNotIn("tok/s", c.export_text())


# ── final metrics panel ────────────────────────────────────────────────

class TestFinalMetricsPanel(unittest.TestCase):
    def test_full_metrics_rendered(self):
        c = _cap_console()
        agents_ui.build_final_metrics(
            c, cost=0.5, steps=3, elapsed=75, compute_budget=2.0,
            peak_context_tokens=100_000, cost_without_cache=0.6,
            context_window_tokens=200_000,
            total_output_tokens=4000,
            output_rate_tokens_per_sec=85.5,
            cost_per_hour=10.0,
        )
        text = c.export_text()
        self.assertIn("Session Complete", text)
        self.assertIn("Cost:", text)
        self.assertIn("Steps:", text)
        self.assertIn("Duration:", text)
        self.assertIn("1m 15s", text)          # 75s
        self.assertIn("Peak context:", text)
        self.assertIn("Output tokens:", text)
        self.assertIn("4.0K", text)
        self.assertIn("Output rate:", text)
        self.assertIn("85.5 tok/s", text)
        self.assertIn("Est. cost/hour:", text)
        self.assertIn("$10.00", text)
        self.assertIn("17% saved", text)      # (0.6-0.5)/0.6 ≈ 16.6%

    def test_optional_metrics_hidden_when_unmeasured(self):
        c = _cap_console()
        agents_ui.build_final_metrics(
            c, cost=0.5, steps=2, elapsed=30, compute_budget=2.0,
            peak_context_tokens=0, cost_without_cache=0.0,
            context_window_tokens=200_000,
            total_output_tokens=0,
            output_rate_tokens_per_sec=None,
            cost_per_hour=None,
        )
        text = c.export_text()
        self.assertIn("Session Complete", text)
        self.assertNotIn("Est. cost/hour:", text)
        self.assertNotIn("Output tokens:", text)
        self.assertNotIn("Peak context:", text)
        # Rate line is still shown, rendered as an unmeasured "—".
        self.assertIn("Output rate:", text)
        self.assertIn("—", text)

    def test_print_summary_delegates(self):
        """print_summary forwards the new aggregate fields unchanged."""
        captured = {}

        def fake_build(context, **kwargs):
            captured.update(kwargs)

        with mock.patch.object(agents_ui, "build_final_metrics", fake_build):
            agents_ui.print_summary(
                0.5, 3, 75, 2.0, peak_context_tokens=999,
                cost_without_cache=0.6, context_window_tokens=200_000,
                total_output_tokens=4000,
                output_rate_tokens_per_sec=85.5,
                cost_per_hour=10.0,
            )
        self.assertEqual(captured.get("total_output_tokens"), 4000)
        self.assertEqual(captured.get("output_rate_tokens_per_sec"), 85.5)
        self.assertEqual(captured.get("cost_per_hour"), 10.0)
        self.assertEqual(captured.get("peak_context_tokens"), 999)


# ── Agent wiring ───────────────────────────────────────────────────────

_FAKE_CONFIG = {
    "system_prompt": "IMMUTABLE SYSTEM PROMPT",
    "overbudget": "over budget",
    "provider": "kimi",
}


def _make_agent(task="do the thing"):
    """Build a minimal Agent with config/memory/backend mocked out."""
    with mock.patch.object(agents_module, "read_configuration", return_value=_FAKE_CONFIG), \
         mock.patch.object(agents_module, "format_memory_view", return_value=""), \
         mock.patch.object(agents_module, "notes_need_compact", return_value=False), \
         mock.patch.object(agents_module, "create_backend") as mock_backend, \
         mock.patch.object(agents_module, "print_banner"), \
         mock.patch.object(agents_module, "print_iteration_header"), \
         mock.patch.object(agents_module, "print_error"):
        client = mock_backend.return_value
        client.display_name = "MockModel"
        client.context_window_size = 200_000
        client.cost = 0.0
        client.cost_without_cache = 0.0
        client.peak_context_tokens = 0
        client.last_input_tokens = 0
        client.last_output_tokens = 0
        client.last_total_context_tokens = 0
        return Agent("fake.yaml", task, session_id="tst1")


class TestIterateFoldsMetrics(unittest.TestCase):
    def test_successful_iterate_calls_record_step_metrics(self):
        agent = _make_agent()
        agent.client.generate_response = mock.Mock(return_value="Done.")
        agent.client.step_rate_tokens_per_sec = None
        # Deterministic filtering / clipping so the free-form path is
        # exercised without depending on the real filter content.
        with mock.patch.object(agents_module, "filter_content",
                               side_effect=lambda s: s), \
             mock.patch.object(agents_module, "print_clipped"):
            result = agent._iterate(free_form=True)

        # free-form returns False (no command to keep looping), but the
        # generation was folded into the session metrics exactly once.
        self.assertFalse(result)
        agent.client.generate_response.assert_called_once()
        agent.client.record_step_metrics.assert_called_once()


if __name__ == "__main__":
    unittest.main()
