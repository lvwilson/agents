"""Tests for the streaming loop detector (agents/loop_detector.py).

A looping model never finishes its output, so the detector runs *during*
streaming and must be cheap.  These tests pin down:

* the repetition **score** — near 0 for original content, tending to 1.0
  for degenerate loops, with a wide safety margin between the two;
* **incremental == batch** equivalence (the O(1) streaming score must
  exactly match the reference implementation);
* **streaming detection** — an unbounded loop is guaranteed to trip the
  threshold, while real (non-looping) output never does;
* **chunking invariance** — the result is independent of how the output
  is split into tokens;
* **timing** — well under the 10 ms/token CPU budget;
* the **stream-handler wrapper** that aborts a generation mid-stream;
* the **agent integration** — a terminated (looping) generation is
  discarded (never saved to context), logged as an error, and redone,
  with a bounded number of retries.

The two real examples captured from production live in ``tests/data/``:
``loop_example.txt`` (a stuck re-emission) and ``not_loop_example.txt``
(a legitimate multi-command turn).
"""

import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.loop_detector import (  # noqa: E402
    LoopDetector,
    LoopDetection,
    LoopDetectedError,
    LoopGuardedStreamHandler,
)
from agents.llm_backend import StreamHandler  # noqa: E402
from agents.agents import Agent  # noqa: E402

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _load(name):
    with open(os.path.join(_DATA, name)) as f:
        return f.read()


def _build_loop(n_blocks):
    """Reconstruct the loop example with *n_blocks* repetitions.

    The example is a unique preamble followed by the same empty
    ``tool_call`` block repeated.  Repeating the block an arbitrary number
    of times models the *unbounded* loop the example is a sample of.
    Returns ``(text, unit)``.
    """
    text = _load("loop_example.txt")
    marker = "I'll apply both fixes now."
    idx = text.find(marker)
    if idx < 0:
        raise AssertionError("marker not found in loop_example.txt")
    preamble = text[: idx + len(marker)]
    tail = text[idx + len(marker):].strip()
    count = tail.count("tool_call")
    if count <= 0:
        raise AssertionError("no tool_call blocks found in loop_example.txt")
    unit = tail[: len(tail) // count]
    return preamble + "\n\n" + unit * n_blocks, unit


def _feed_all(detector, text, chunk=1):
    """Feed *text* to *detector* in *chunk*-sized pieces; return last result."""
    result = LoopDetection(False)
    for i in range(0, len(text), chunk):
        result = detector.feed(text[i:i + chunk])
    return result


class TestScoreBasics(unittest.TestCase):
    """The repetition score separates loops from real output."""

    def test_random_text_scores_zero(self):
        import random
        import string
        rng = random.Random(42)
        text = "".join(rng.choices(string.ascii_letters + " ", k=20000))
        self.assertLess(LoopDetector.repetition_score(text), 0.05)

    def test_loop_scores_high(self):
        loop, _ = _build_loop(200)
        self.assertGreater(LoopDetector.repetition_score(loop), 0.90)

    def test_not_loop_scores_low(self):
        not_loop = _load("not_loop_example.txt")
        self.assertLess(LoopDetector.repetition_score(not_loop), 0.50)

    def test_real_code_scores_low(self):
        code = open(os.path.join(_DATA, "..", "..", "agents", "agents.py")).read()
        self.assertLess(LoopDetector.repetition_score(code), 0.50)

    def test_separation_margin(self):
        loop, _ = _build_loop(200)
        not_loop = _load("not_loop_example.txt")
        s_loop = LoopDetector.repetition_score(loop)
        s_not = LoopDetector.repetition_score(not_loop)
        # A wide, comfortable margin between a loop and real output.
        self.assertGreater(s_loop - s_not, 0.40)

    def test_identical_lines_score_high(self):
        # Pathological but legitimate (a log of identical lines) is
        # indistinguishable from a loop by design — documented behaviour.
        line = "2026-08-30 10:44:57 INFO worker-3 processed batch id=88231 status=ok\n"
        self.assertGreater(LoopDetector.repetition_score(line * 500), 0.90)

    def test_score_is_bounded(self):
        for text in ("", "a", "ab", "abcd", "a b c d e f g h i j"):
            s = LoopDetector.repetition_score(text)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class TestIncrementalEquivalence(unittest.TestCase):
    """The O(1) streaming score must exactly match the batch oracle."""

    def _texts(self):
        loop, _ = _build_loop(120)
        return {
            "not_loop": _load("not_loop_example.txt"),
            "loop12": _build_loop(12)[0],
            "loop120": loop,
            "big_code": open(os.path.join(_DATA, "..", "..", "agents", "agents.py")).read(),
            "log": ("line id=1 status=ok latency=12\n" * 3000),
            "random": ("xQ7 mN2 pL9 vB4 " * 1500),
        }

    def test_incremental_matches_oracle(self):
        for name, text in self._texts().items():
            d = LoopDetector()
            _feed_all(d, text)
            oracle = LoopDetector.repetition_score(text)
            self.assertAlmostEqual(
                d.current_score(), oracle, places=9,
                msg=f"{name}: incremental {d.current_score()} != oracle {oracle}",
            )

    def test_chunking_invariance(self):
        text = _build_loop(80)[0] + "\n" + _load("not_loop_example.txt")
        scores = []
        for chunk in (1, 3, 7, 64, 1000):
            d = LoopDetector()
            _feed_all(d, text, chunk=chunk)
            scores.append(d.current_score())
        for s in scores[1:]:
            self.assertAlmostEqual(s, scores[0], places=9)


class TestStreamingDetection(unittest.TestCase):
    """An unbounded loop is guaranteed to trip the threshold; real output
    never does."""

    def test_loop_fires_mid_stream(self):
        loop, _ = _build_loop(300)
        d = LoopDetector()
        fired_at = None
        for i, ch in enumerate(loop):
            if d.feed(ch).is_loop and fired_at is None:
                fired_at = i + 1
                break
        self.assertIsNotNone(fired_at, "the loop must be detected")
        # Detected well before the (unbounded) loop ends — we save output.
        self.assertLess(fired_at, len(loop))
        self.assertGreaterEqual(d.score, d.threshold)

    def test_not_loop_never_fires(self):
        not_loop = _load("not_loop_example.txt")
        d = LoopDetector()
        for ch in not_loop:
            self.assertFalse(d.feed(ch).is_loop)
        self.assertLess(d.score, d.threshold)

    def test_guaranteed_termination_of_unbounded_loop(self):
        # A loop that keeps going must eventually cross the threshold.
        loop, _ = _build_loop(100000)
        d = LoopDetector()
        fired = False
        for i in range(0, len(loop), 16):
            if d.feed(loop[i:i + 16]).is_loop:
                fired = True
                break
        self.assertTrue(fired, "an unbounded loop must be terminated")

    def test_reset_clears_state(self):
        loop, _ = _build_loop(300)
        d = LoopDetector()
        # Drive a loop until it fires.
        for ch in loop:
            if d.feed(ch).is_loop:
                break
        self.assertGreaterEqual(d.score, d.threshold)
        # A fresh generation (reset) is judged on its own output.
        d.reset()
        self.assertEqual(d.score, 0.0)
        self.assertEqual(d.current_score(), 0.0)
        for ch in _load("not_loop_example.txt"):
            self.assertFalse(d.feed(ch).is_loop)
        self.assertLess(d.score, d.threshold)

    def test_disabled_detector_never_fires(self):
        loop, _ = _build_loop(300)
        d = LoopDetector()
        d.enabled = False
        for ch in loop:
            self.assertFalse(d.feed(ch).is_loop)


class TestTiming(unittest.TestCase):
    """The detector must stay far under the 10 ms/token CPU budget."""

    def test_per_token_under_budget(self):
        loop, _ = _build_loop(2000)  # ~40k chars
        d = LoopDetector()
        tokens = 0
        t0 = time.perf_counter()
        for i in range(0, len(loop), 16):
            d.feed(loop[i:i + 16])
            tokens += 1
        dt = time.perf_counter() - t0
        per_token_ms = dt / tokens * 1000
        # Comfortably under the 10 ms/token budget (typically ~0.001 ms).
        self.assertLess(per_token_ms, 10.0,
                        f"per-token cost {per_token_ms:.4f} ms exceeds budget")

    def test_worst_case_every_char_evaluated(self):
        # Even evaluating the score on every single character (the most
        # expensive configuration) stays well under budget.
        loop, _ = _build_loop(1000)
        d = LoopDetector(rescan_every=1)
        t0 = time.perf_counter()
        for ch in loop:
            d.feed(ch)
        dt = time.perf_counter() - t0
        per_token_ms = dt / len(loop) * 1000
        self.assertLess(per_token_ms, 10.0)


class _RecordingHandler(StreamHandler):
    """StreamHandler that records every hook invocation."""

    def __init__(self):
        super().__init__()
        self.events = []

    def on_stream_start(self):
        super().on_stream_start()
        self.events.append("start")

    def on_stream_token(self, token):
        super().on_stream_token(token)
        self.events.append(("token", token))

    def on_stream_end(self):
        super().on_stream_end()
        self.events.append("end")

    def on_tool_call(self, name, arguments=""):
        super().on_tool_call(name, arguments)
        self.events.append(("tool_call", name))


class TestGuardedStreamHandler(unittest.TestCase):
    """The wrapper aborts a looping generation and delegates the rest."""

    def test_raises_on_loop(self):
        inner = _RecordingHandler()
        d = LoopDetector()
        wrapper = LoopGuardedStreamHandler(inner, d)
        wrapper.on_stream_start()
        loop, _ = _build_loop(300)
        with self.assertRaises(LoopDetectedError) as ctx:
            for ch in loop:
                wrapper.on_stream_token(ch)
        self.assertGreaterEqual(ctx.exception.detection.score, d.threshold)
        # The partial (looping) output was still buffered by the inner
        # handler, so it can be reported — but it is never added to context.
        self.assertTrue(inner.get_buffered_text())

    def test_no_raise_on_clean_output(self):
        inner = _RecordingHandler()
        wrapper = LoopGuardedStreamHandler(inner, LoopDetector())
        wrapper.on_stream_start()
        for ch in _load("not_loop_example.txt"):
            wrapper.on_stream_token(ch)  # must not raise
        self.assertFalse(wrapper._detector.score >= wrapper._detector.threshold)

    def test_delegates_other_hooks(self):
        inner = _RecordingHandler()
        wrapper = LoopGuardedStreamHandler(inner, LoopDetector())
        wrapper.on_stream_start()
        wrapper.on_stream_token("hi ")
        wrapper.on_tool_call("some_tool", "{}")
        wrapper.on_stream_end()
        self.assertIn("start", inner.events)
        self.assertIn(("token", "hi "), inner.events)
        self.assertIn(("tool_call", "some_tool"), inner.events)
        self.assertIn("end", inner.events)
        # Buffer is delegated, not shadowed by the wrapper.
        self.assertEqual(wrapper.get_buffered_text(), "hi ")

    def test_on_stream_start_resets_detector(self):
        d = LoopDetector()
        wrapper = LoopGuardedStreamHandler(_RecordingHandler(), d)
        wrapper.on_stream_start()
        loop, _ = _build_loop(300)
        for ch in loop:
            try:
                wrapper.on_stream_token(ch)
            except LoopDetectedError:
                break
        self.assertGreaterEqual(d.score, d.threshold)
        # A new generation resets the moving window.
        wrapper.on_stream_start()
        self.assertEqual(d.score, 0.0)
        self.assertEqual(d.current_score(), 0.0)


class TestBackendNoRetry(unittest.TestCase):
    """A detected loop must propagate out of the backend without retrying.

    Re-issuing the same request would just stream the same loop again, so
    the retry template method must re-raise ``LoopDetectedError`` on the
    first occurrence.
    """

    def test_loop_error_not_retried(self):
        from agents.llm_backend import LLMBackend
        attempts = []

        class Fake(LLMBackend):
            def generate_response(self, system_prompt, context):
                def attempt():
                    attempts.append(1)
                    loop, _ = _build_loop(300)
                    for i in range(0, len(loop), 8):
                        # Stream through the guarded handler; it raises
                        # mid-stream once the score crosses the threshold.
                        self.stream_handler.on_stream_token(loop[i:i + 8])
                    return "done"
                return self._run_with_retries(attempt)

        inner = _RecordingHandler()
        wrapper = LoopGuardedStreamHandler(inner, LoopDetector())
        backend = Fake.__new__(Fake)
        LLMBackend.__init__(backend, model="fake", stream_handler=wrapper)

        with self.assertRaises(LoopDetectedError):
            backend.generate_response("sys", [])
        self.assertEqual(len(attempts), 1,
                         "a looping generation must not be retried")
        # The stream was ended (so the UI spinner stops) before re-raising.
        self.assertIn("end", inner.events)

    def test_transient_error_still_retries(self):
        # Sanity: a *transient* (non-loop) error must still be retried,
        # confirming the loop path is distinct from the retry path.
        from agents.llm_backend import LLMBackend
        attempts = []

        class Fake(LLMBackend):
            def generate_response(self, system_prompt, context):
                def attempt():
                    attempts.append(1)
                    raise ConnectionError("transient failure")
                return self._run_with_retries(attempt)

        backend = Fake.__new__(Fake)
        LLMBackend.__init__(backend, model="fake",
                            stream_handler=_RecordingHandler())
        with mock.patch("time.sleep"):
            with self.assertRaises(Exception):
                backend.generate_response("sys", [])
        self.assertGreater(len(attempts), 1,
                           "transient errors must still be retried")


class TestAgentIntegration(unittest.TestCase):
    """A terminated (looping) generation is discarded, logged, and redone."""

    _FAKE_CONFIG = {
        "system_prompt": "IMMUTABLE SYSTEM PROMPT",
        "overbudget": "over budget",
        "provider": "kimi",
    }

    def _make_agent(self, task="do the thing"):
        from agents import agents as agents_module
        with mock.patch.object(agents_module, "read_configuration",
                               return_value=self._FAKE_CONFIG), \
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
            agent = Agent("fake.yaml", task, session_id="tst1")
            # Expose the create_backend mock so tests can inspect the
            # stream handler the agent wired up.
            agent._create_backend_mock = mock_backend
            return agent

    def _loop_error(self):
        return LoopDetectedError(
            LoopDetection(True, "intra", 0.95, "last output: '…'")
        )

    def test_loop_error_discarded_and_redone(self):
        from agents.agents import LOOP_BREAK_FEEDBACK
        agent = self._make_agent()
        agent.client.generate_response = mock.Mock(side_effect=self._loop_error())
        running = agent._iterate()
        self.assertTrue(running, "a looped turn must be redone, not ended")
        self.assertEqual(agent._loop_terminations, 1)
        # The looping (partial) response was NOT saved to context.
        self.assertFalse(any(m["role"] == "assistant" for m in agent.context))
        # A break-the-loop nudge was injected as a user message.
        self.assertEqual(agent.context[-1]["role"], "user")
        self.assertEqual(agent.context[-1]["content"][0]["text"], LOOP_BREAK_FEEDBACK)

    def test_loop_error_is_logged_as_error(self):
        from agents import agents as agents_module
        agent = self._make_agent()
        agent.client.generate_response = mock.Mock(side_effect=self._loop_error())
        with mock.patch.object(agents_module, "print_error") as pe:
            agent._iterate()
        pe.assert_called_once()
        msg = pe.call_args[0][0]
        self.assertIn("Loop detected", msg)

    def test_max_retries_then_raises(self):
        from agents.agents import MAX_LOOP_RETRIES
        agent = self._make_agent()
        agent.client.generate_response = mock.Mock(side_effect=self._loop_error())
        for _ in range(MAX_LOOP_RETRIES - 1):
            self.assertTrue(agent._iterate())
        with self.assertRaises(RuntimeError) as ctx:
            agent._iterate()
        self.assertIn("kept producing looping output", str(ctx.exception))
        self.assertEqual(agent._loop_terminations, MAX_LOOP_RETRIES)

    def test_clean_response_resets_terminations(self):
        agent = self._make_agent()
        # First turn loops; second turn is clean (no command → the
        # no-output reminder path, but the termination counter resets).
        agent.client.generate_response = mock.Mock(
            side_effect=[self._loop_error(), "I will now read the file."]
        )
        self.assertTrue(agent._iterate())
        self.assertEqual(agent._loop_terminations, 1)
        agent._iterate()
        self.assertEqual(agent._loop_terminations, 0,
                         "a clean generation must re-arm the loop budget")

    def test_agent_uses_guarded_stream_handler(self):
        agent = self._make_agent()
        # The agent must hand the backend a loop-guarded stream handler
        # sharing the agent's detector.
        args, kwargs = agent._create_backend_mock.call_args
        handler = kwargs["stream_handler"]
        self.assertIsInstance(handler, LoopGuardedStreamHandler)
        self.assertIs(handler._detector, agent.loop_detector)
        # The inner handler is the interactive Rich handler.
        from agents.ui import RichStreamHandler
        self.assertIsInstance(handler._inner, RichStreamHandler)


if __name__ == "__main__":
    unittest.main()
