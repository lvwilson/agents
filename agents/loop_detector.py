"""Incremental loop detection for streaming LLM output.

A looping model never finishes its output — it re-emits the same content
forever — so this detector runs *during* streaming and must be cheap: the
target is well under 10 ms of CPU per output token.  It therefore computes
its signal **incrementally** in O(1) amortised time per character, with no
third-party dependency.

How it works
------------
The detector keeps a *moving window* of the last ``window`` characters of
the current generation and maintains, incrementally, how many distinct
character n-grams that window contains.  From that it derives a
**repetition score** in [0, 1]:

    score = 1 - (distinct n-grams in window) / (total n-grams in window)

* fully original content  → score ≈ 0.0
* the same material re-emitted again and again → score → 1.0

The score thus *tends towards 1.0* as the model falls into a loop, and a
generation is terminated the moment the score crosses ``threshold`` (and
the response is long enough to score reliably).  Because a genuine loop
keeps appending the same material, its score climbs monotonically toward
1.0 and is *guaranteed* to cross the threshold — which is what lets us
terminate a generation that would otherwise never end.  Short preambles,
repeated command names, similar-but-different payloads, and even fairly
repetitive code keep the score well below the threshold, so ordinary work
is not disturbed.

Why an n-gram counter (and not a compression ratio)?
----------------------------------------------------
A zlib/entropy ratio is a reasonable "fit-for-purpose" library choice, but
its baseline for normal prose/code is ~0.5, leaving only a small gap to a
loop.  The n-gram self-repetition rate is sharply bimodal — degenerate
loops sit at ~0.85-0.97 while normal output stays below ~0.4 — so a single
threshold separates them with a wide safety margin.  It is also trivially
incremental and dependency-free.

Integration
-----------
Wrap the backend's stream handler in :class:`LoopGuardedStreamHandler`.
When the score crosses the threshold the wrapper raises
:class:`LoopDetectedError` from ``on_stream_token``; the backend's retry
loop re-raises it (it is not a transient error), and the agent catches it
to discard the partial response and redo the generation.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .llm_backend import StreamHandler


__all__ = [
    "LoopDetection",
    "LoopDetector",
    "LoopDetectedError",
    "LoopGuardedStreamHandler",
]


def _normalize_stream(text: str) -> str:
    """Collapse whitespace runs to single spaces, dropping leading runs.

    This is the *streaming-friendly* form of :func:`_normalize`:
    :meth:`LoopDetector.feed` reproduces it exactly by carrying the same
    two-bit state (``started``, ``last_was_space``) across chunks.  It
    differs from :func:`_normalize` only in that it keeps trailing
    whitespace — a pure stream cannot strip it — which is irrelevant to
    the repetition score.
    """
    out = []
    last_space = False
    started = False
    for ch in text:
        if ch.isspace():
            if started and not last_space:
                out.append(" ")
                last_space = True
        else:
            out.append(ch)
            last_space = False
            started = True
    return "".join(out)


@dataclass
class LoopDetection:
    """The outcome of a loop check."""

    is_loop: bool
    kind: str = ""        # "intra" (looping generation) or "inter" (repeated turn)
    score: float = 0.0    # repetition score in [0, 1]
    detail: str = ""      # short human-readable context for logs / UI

    @property
    def description(self) -> str:
        """A short, human-readable description for logs and the UI."""
        if not self.is_loop:
            return ""
        return f"{self.kind} loop detected (repetition score {self.score:.2f})"


class LoopDetectedError(Exception):
    """Raised mid-stream when the repetition score crosses the threshold.

    Carries the :class:`LoopDetection` so the caller can log exactly how
    repetitive the output was.  The backend re-raises this without
    retrying; the agent catches it to discard the partial response and
    redo the generation.
    """

    def __init__(self, detection: LoopDetection):
        self.detection = detection
        super().__init__(f"Loop detected: {detection.description}")


class LoopDetector:
    """Incremental loop detector driven by a moving-window repetition score.

    Feed it the visible chunks of the current response with :meth:`feed`;
    it returns a :class:`LoopDetection` (usually ``is_loop=False``) and
    never raises.  The current score is kept in :attr:`score`.

    The score is maintained *incrementally* over a sliding window: each
    appended character updates the n-gram frequency counter in O(n) and
    each evicted character rolls it back, so the per-token cost is a few
    microseconds — far below the 10 ms/token budget even when checked on
    every token.

    :meth:`reset` is called at the start of every generation (the stream
    wrapper does this via ``on_stream_start``), so a runaway generation is
    judged on its own output and a redo is evaluated fresh.
    """

    def __init__(
        self,
        n: int = 8,
        window: int = 8000,
        threshold: float = 0.90,
        min_score_len: int = 256,
        rescan_every: int = 64,
    ):
        if window < n:
            raise ValueError("window must be >= n")
        # Moving-window score tuning.
        self.n = n                      # n-gram size (characters)
        self.window = window            # window length (normalised chars)
        self.threshold = threshold      # score that triggers termination
        self.min_score_len = min_score_len  # min output before scoring counts
        self.rescan_every = rescan_every    # re-evaluate every N new chars

        self.enabled = True

        # Incremental window state (the current generation).
        self._chars: deque[str] = deque(maxlen=window)
        self._freq: dict[str, int] = {}
        self._unique = 0
        self._len = 0        # total normalised chars fed this generation
        self._checked = 0    # normalised chars fed at the last evaluation
        self.score = 0.0

        # Streaming-normalization state (see _normalize_stream).
        self._started = False
        self._last_space = False

    # ── Lifecycle ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear the in-progress response state (start of a generation)."""
        self._chars.clear()
        self._freq.clear()
        self._unique = 0
        self._len = 0
        self._checked = 0
        self.score = 0.0
        self._started = False
        self._last_space = False

    # ── Incremental window maintenance ───────────────────────────────

    def _add_gram(self, gram: str) -> None:
        c = self._freq.get(gram, 0)
        if c == 0:
            self._unique += 1
        self._freq[gram] = c + 1

    def _remove_gram(self, gram: str) -> None:
        c = self._freq[gram]
        if c == 1:
            del self._freq[gram]
            self._unique -= 1
        else:
            self._freq[gram] = c - 1

    def _first_n(self, n: int) -> str:
        """Return the first *n* characters of the window (O(n)).

        ``deque`` does not support slicing, so the head is read by
        iteration (O(1) per element from either end).
        """
        it = iter(self._chars)
        return "".join(next(it) for _ in range(n))

    def _last_n(self, n: int) -> str:
        """Return the last *n* characters of the window (O(n))."""
        return "".join(self._chars[-i] for i in range(n, 0, -1))

    def _append(self, text: str) -> None:
        """Fold *text* into the sliding window, updating the n-gram counts."""
        append = self._chars.append
        n = self.n
        window = self.window
        for ch in text:
            was_full = len(self._chars) == window
            # The n-gram that leaves the window is the one that started at
            # the about-to-be-evicted first character; capture it before the
            # append evicts that character.
            left_gram = self._first_n(n) if was_full else None
            append(ch)
            self._len += 1
            if len(self._chars) >= n:
                # The n-gram that enters the window ends at the new char.
                self._add_gram(self._last_n(n))
            if left_gram is not None:
                self._remove_gram(left_gram)

    def _total_grams(self) -> int:
        L = len(self._chars)
        return L - self.n + 1 if L >= self.n else 0

    def current_score(self) -> float:
        """Return the current repetition score in [0, 1] (O(1))."""
        total = self._total_grams()
        if total <= 0:
            return 0.0
        return 1.0 - (self._unique / total)

    # ── Batch score (oracle / testing) ───────────────────────────────

    @staticmethod
    def repetition_score(text: str, n: int = 8, window: int = 8000) -> float:
        """Return the repetition score of *text* in [0, 1].

        This is the non-incremental reference implementation: it scores
        the trailing ``window`` characters of *text* directly.  It is used
        to validate the incremental :meth:`current_score` and in tests.
        It applies the same streaming-safe normalization the incremental
        path uses, so the two scores are exactly equal.
        """
        w = _normalize_stream(text)[-window:]
        total = len(w) - n + 1
        if total <= 0:
            return 0.0
        seen = set()
        add = seen.add
        for i in range(total):
            add(w[i:i + n])
        return 1.0 - (len(seen) / total)

    # ── Streaming (moving-window score) ──────────────────────────────

    def _normalize_chunk(self, chunk: str) -> str:
        """Normalize *chunk* for the stream, carrying state across calls.

        Carries the two-bit state (``started``, ``last_space``) so that the
        concatenation of every returned piece is exactly
        ``_normalize_stream`` of the full input — i.e. whitespace runs are
        collapsed to single spaces with leading runs dropped.  Unlike
        per-chunk :func:`_normalize`, this never drops the single spaces
        that separate words, so the incremental score matches the batch
        oracle exactly.
        """
        out = []
        started = self._started
        last_space = self._last_space
        for ch in chunk:
            if ch.isspace():
                if started and not last_space:
                    out.append(" ")
                    last_space = True
            else:
                out.append(ch)
                last_space = False
                started = True
        self._started = started
        self._last_space = last_space
        return "".join(out)

    def feed(self, chunk: str) -> LoopDetection:
        """Consume a chunk of the current response and check for a loop.

        Whitespace is normalized in a streaming-safe way (see
        :meth:`_normalize_chunk`) and the window is updated for every
        character (cheap).  The score is only *evaluated* against the
        threshold once at least ``rescan_every`` new characters have
        arrived, keeping the hot path minimal.  A loop is reported when
        the score crosses ``threshold`` and the response is at least
        ``min_score_len`` characters long.
        """
        norm = self._normalize_chunk(chunk)
        if norm:
            self._append(norm)
        if not self.enabled:
            return LoopDetection(False, score=self.score)
        if self._len - self._checked < self.rescan_every:
            return LoopDetection(False, score=self.score)
        self._checked = self._len
        self.score = self.current_score()
        if self.score >= self.threshold and self._len >= self.min_score_len:
            tail = "".join(self._chars)[-80:]
            return LoopDetection(
                True, "intra", self.score, f"last output: {tail!r}",
            )
        return LoopDetection(False, score=self.score)


class LoopGuardedStreamHandler:
    """Stream-handler wrapper that aborts a generation on a detected loop.

    It feeds every visible token to a shared :class:`LoopDetector` and
    raises :class:`LoopDetectedError` as soon as the repetition score
    crosses the threshold, so the backend stops generating.  All other
    hooks (reasoning, tool calls, retries, the token buffer) are delegated
    to the wrapped handler unchanged, so the UI behaves exactly as before
    until a loop occurs.

    The wrapper is deliberately *not* a ``StreamHandler`` subclass: it has
    no buffer of its own, and ``__getattr__`` delegation keeps
    ``get_buffered_text`` and friends pointing at the inner handler.
    """

    def __init__(self, inner: "StreamHandler", detector: LoopDetector):
        self._inner = inner
        self._detector = detector

    def on_stream_start(self) -> None:
        # A fresh generation → fresh moving-window state, so a redo is
        # evaluated on its own output rather than the discarded loop.
        self._detector.reset()
        self._inner.on_stream_start()

    def on_stream_token(self, token: str) -> None:
        if self._detector.enabled:
            detection = self._detector.feed(token)
            if detection.is_loop:
                raise LoopDetectedError(detection)
        self._inner.on_stream_token(token)

    def __getattr__(self, name):
        # Delegate everything we don't override (on_stream_end, the
        # reasoning hooks, on_tool_call, on_retry, on_error,
        # get_buffered_text, …) to the wrapped handler.
        return getattr(self._inner, name)
