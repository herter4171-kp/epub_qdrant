"""Tests for the judge streaming abort + flat retry logic.

Run with pytest (preferred):
    .venv/bin/pytest tests/test_judge_streaming.py -v

A subset can also run directly without pytest:
    python3 tests/test_judge_streaming.py
The standalone runner skips tests that need pytest fixtures (monkeypatch).

The eval suite lives under ``scripts/eval_suite``. We import via
``scripts.eval_suite.critique`` after putting the project root on
``sys.path``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Iterator, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_suite.critique import (
    StreamAbort,
    _drain_stream,
    _is_line_loop,
    _is_thinking_loop,
    _run_one_judgement,
    _strip_wrappers,
    _validate_parsed,
)


# ── Fakes ─────────────────────────────────────────────────────────────────────


class _FakeDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeStream:
    """Minimal iterable stream stand-in with a close() hook."""

    def __init__(self, deltas: Iterator[str], inter_chunk_delay: float = 0.0) -> None:
        self._deltas = deltas
        self._delay = inter_chunk_delay
        self.closed = False

    def __iter__(self):
        for d in self._deltas:
            if self._delay:
                time.sleep(self._delay)
            yield _FakeChunk(d)

    def close(self) -> None:
        self.closed = True


def _looping_stream(line: str, count: int = 1000) -> _FakeStream:
    return _FakeStream(iter([line] * count))


# ── _is_thinking_loop ────────────────────────────────────────────────────────


def test_loop_detection_positive_repeats():
    line = "Step 1: re-examine the evidence carefully and proceed.  "
    # Need payload >= 1200 chars to clear the lookback floor.
    payload = "intro " * 10 + line * 30  # 60 + 56*30 = 1740 chars
    assert _is_thinking_loop(payload), "should flag obvious tail-repeat loop"


def test_loop_detection_short_input_safe():
    assert not _is_thinking_loop("hello world"), "short input must not flag"


def test_loop_detection_varied_content():
    # 1500+ chars, no near-verbatim tail repetition
    base = "Alpha beta gamma delta. " "Epsilon zeta eta theta. " "Iota kappa lambda mu. "
    payload = (base * 25)[: 1500]
    # Tail of 200 chars within base*25 *will* repeat — so vary slightly:
    payload += "UNIQUE-TAIL-XYZ-123 final words here that do not echo."
    assert not _is_thinking_loop(payload), "varied tail must not flag"


def test_loop_detection_whitespace_tail_skipped():
    payload = "real content " * 100 + (" " * 250)
    assert not _is_thinking_loop(payload), "whitespace-only tail must not flag"


# ── _drain_stream ────────────────────────────────────────────────────────────


def test_drain_normal_completion():
    parts = ['{"chunks":', '[],', '"reply":""', ',"satisfaction":5}']
    stream = _FakeStream(iter(parts))
    out = _drain_stream(stream, label="t", use_tty=False)
    assert out == '{"chunks":[],"reply":"","satisfaction":5}'


def test_drain_aborts_on_thinking_loop():
    line = "I should reconsider my reasoning step by step very carefully now.  "
    stream = _looping_stream(line, count=2000)
    start = time.monotonic()
    raised = None
    try:
        _drain_stream(stream, label="t", use_tty=False)
    except StreamAbort as exc:
        raised = exc
    elapsed = time.monotonic() - start
    assert raised is not None, "looping stream must raise StreamAbort"
    assert "loop" in str(raised).lower()
    assert elapsed < 5.0, f"loop detection took too long: {elapsed:.2f}s"
    assert stream.closed, "stream.close() should be called on abort"


def test_drain_aborts_on_per_chunk_inactivity():
    """Stream that never yields a chunk: must abort on per-chunk timeout."""

    def slow():
        time.sleep(5)  # longer than per_chunk_timeout below
        yield "late"

    stream = _FakeStream(slow())
    start = time.monotonic()
    raised = None
    try:
        _drain_stream(
            stream, label="t", use_tty=False,
            total_timeout=10.0, per_chunk_timeout=1.0,
        )
    except StreamAbort as exc:
        raised = exc
    elapsed = time.monotonic() - start
    assert raised is not None, "silent stream must raise StreamAbort"
    assert "stuck" in str(raised).lower() or "chunk" in str(raised).lower()
    assert elapsed < 3.0, f"per-chunk abort fired too late: {elapsed:.2f}s"


def test_drain_aborts_on_total_wallclock():
    """Stream emitting tiny non-loopy deltas indefinitely: total cap fires."""
    counter = [0]

    def steady():
        while True:
            counter[0] += 1
            yield f"{counter[0]:08d} unique fragment of varied content with id {counter[0]} | "
            time.sleep(0.01)

    stream = _FakeStream(steady())
    start = time.monotonic()
    raised = None
    try:
        _drain_stream(
            stream, label="t", use_tty=False,
            total_timeout=1.0, per_chunk_timeout=10.0,
        )
    except StreamAbort as exc:
        raised = exc
    elapsed = time.monotonic() - start
    assert raised is not None, "long stream must raise StreamAbort on wall-clock"
    assert "wall" in str(raised).lower() or "exceed" in str(raised).lower()
    assert 0.5 <= elapsed < 3.0, f"wall-clock abort fired off-window: {elapsed:.2f}s"


def test_drain_propagates_reader_exception():
    """Reader-thread exceptions surface in the main thread."""

    class _Boom(Exception):
        pass

    def angry():
        yield "ok-prefix "
        raise _Boom("kaboom")

    stream = _FakeStream(angry())
    raised = None
    try:
        _drain_stream(stream, label="t", use_tty=False)
    except _Boom as exc:
        raised = exc
    assert raised is not None, "reader-thread exception must propagate"


# ── line-loop detection ──────────────────────────────────────────────────────


def test_line_loop_two_line_block_repeat():
    block = "Step 1: analyze the data carefully.\nStep 2: report the result.\n"
    payload = "intro line\n" + block * 6
    assert _is_line_loop(payload), "two-line block repeated 6× must flag"


def test_line_loop_single_line_repeat_with_min_block_2():
    payload = "alpha\n" * 10
    # Single-line repeats meet min_block=2 if last 2 lines == prior 2 lines
    assert _is_line_loop(payload), "uniform single-line repeats also satisfy block test"


def test_line_loop_varied_content_no_flag():
    payload = "\n".join(f"unique line {i}: this content varies in id {i*7}" for i in range(50))
    assert not _is_line_loop(payload), "varied lines must not flag"


def test_line_loop_blank_lines_ignored():
    payload = "real line\n\n\n\nreal line\n\n\n\n" * 3
    # Stripping blanks leaves "real line" repeated; should flag as block-repeat
    assert _is_line_loop(payload), "blank-line padding shouldn't mask the repeat"


def test_drain_aborts_on_line_loop():
    """Stream emitting a 3-line repeating block should be killed by line detector."""
    block_lines = [
        "Reasoning step A: I should think about this.\n",
        "Reasoning step B: Let me reconsider.\n",
        "Reasoning step C: Going to try again.\n",
    ]

    def looper():
        for _ in range(50):
            for ln in block_lines:
                yield ln

    stream = _FakeStream(looper())
    raised = None
    try:
        _drain_stream(stream, label="t", use_tty=False, total_timeout=10.0)
    except StreamAbort as exc:
        raised = exc
    assert raised is not None, "block-loop stream must raise StreamAbort"
    assert "loop" in str(raised).lower()
    assert stream.closed, "stream.close() should be called on abort"


# ── helpers ──────────────────────────────────────────────────────────────────


def test_strip_wrappers_think_and_fence():
    raw = '<think>reasoning here</think>\n```json\n{"x": 1}\n```'
    assert _strip_wrappers(raw) == '{"x": 1}'


def test_strip_wrappers_open_think_only():
    raw = "stray reasoning </think>\n{\"x\": 1}"
    assert _strip_wrappers(raw) == '{"x": 1}'


def test_validate_parsed_happy_path():
    parsed = {
        "chunks": [
            {"id": "a", "relevance": 5},
            {"id": "b", "relevance": 8},
        ],
        "reply": "some reply",
        "satisfaction": 7,
    }
    ok, err = _validate_parsed(parsed, {"a", "b"}, expected_chunks=2)
    assert ok and err is None


def test_validate_parsed_unknown_id():
    parsed = {
        "chunks": [{"id": "z", "relevance": 5}],
        "reply": "x",
        "satisfaction": 5,
    }
    ok, err = _validate_parsed(parsed, {"a"}, expected_chunks=1)
    assert not ok and "unknown id" in (err or "")


# ── flat-retry semantics in _run_one_judgement ──────────────────────────────


class _FakeOpenAI:
    """Stand-in for the OpenAI client. _stream_judge is patched per test."""


from scripts.eval_suite import critique as _critique_mod  # noqa: E402

_GOOD_JSON = '{"chunks":[{"id":"a","relevance":5}],"reply":"r","satisfaction":7}'


def _patch_stream(monkeypatch, items):
    """Replace _stream_judge with one that pops queued payloads.

    Each item is either a string (returned) or an Exception (raised).
    Returns a counter dict so tests can assert call count.
    """
    queue_local = list(items)
    state = {"calls": 0}

    def _impl(*args, **kwargs):
        state["calls"] += 1
        if not queue_local:
            raise RuntimeError("ran out of fake stream payloads")
        item = queue_local.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(_critique_mod, "_stream_judge", _impl)
    return state


def _judge_kwargs(**overrides):
    base = dict(
        client=_FakeOpenAI(), judge_model="m",
        system_prompt="s", user_message="u",
        issued_ids={"a"}, expected_chunks=1, label="t",
        judge_attempts=3,
        judge_timeout_seconds=10, judge_per_chunk_timeout_seconds=5,
        judge_max_tokens=100,
    )
    base.update(overrides)
    return base


def test_run_one_judgement_succeeds_first_try(monkeypatch):
    state = _patch_stream(monkeypatch, [_GOOD_JSON])
    jo = _run_one_judgement(**_judge_kwargs())
    assert jo.parse_ok is True
    assert jo.retried is False
    assert jo.error is None
    assert state["calls"] == 1


def test_run_one_judgement_retries_then_succeeds(monkeypatch):
    state = _patch_stream(monkeypatch, ["not json", "still bad", _GOOD_JSON])
    jo = _run_one_judgement(**_judge_kwargs())
    assert jo.parse_ok is True
    assert jo.retried is True
    assert state["calls"] == 3


def test_run_one_judgement_exhausts_attempts(monkeypatch):
    state = _patch_stream(monkeypatch, ["bad"] * 5)
    jo = _run_one_judgement(**_judge_kwargs())
    assert jo.parse_ok is False
    assert jo.retried is True
    assert jo.error and "json_decode" in jo.error
    assert state["calls"] == 3, "must stop after attempts budget"


def test_run_one_judgement_transport_error_retried(monkeypatch):
    class _ReadError(Exception):
        pass

    state = _patch_stream(monkeypatch, [_ReadError("reset"), _ReadError("reset"), _GOOD_JSON])
    jo = _run_one_judgement(**_judge_kwargs())
    assert jo.parse_ok is True
    assert state["calls"] == 3


def test_run_one_judgement_stream_abort_retried(monkeypatch):
    state = _patch_stream(monkeypatch, [StreamAbort("loop"), _GOOD_JSON])
    jo = _run_one_judgement(**_judge_kwargs())
    assert jo.parse_ok is True
    assert state["calls"] == 2


def test_run_one_judgement_case_deadline_already_past(monkeypatch):
    state = _patch_stream(monkeypatch, [_GOOD_JSON])
    past = time.monotonic() - 1.0
    jo = _run_one_judgement(**_judge_kwargs(case_deadline=past))
    assert jo.parse_ok is False
    assert jo.error == "case_wallclock_exceeded"
    assert state["calls"] == 0, "must not call _stream_judge after deadline"


# ── runner ───────────────────────────────────────────────────────────────────


def _run(test_fn):
    name = test_fn.__name__
    try:
        test_fn()
    except AssertionError as exc:
        print(f"FAIL  {name}: {exc}")
        return False
    except Exception as exc:
        print(f"ERROR {name}: {exc.__class__.__name__}: {exc}")
        return False
    print(f"ok    {name}")
    return True


if __name__ == "__main__":
    # Tests that need pytest's monkeypatch fixture are skipped here.
    # Run the full suite via:  .venv/bin/pytest tests/test_judge_streaming.py
    tests = [
        test_loop_detection_positive_repeats,
        test_loop_detection_short_input_safe,
        test_loop_detection_varied_content,
        test_loop_detection_whitespace_tail_skipped,
        test_drain_normal_completion,
        test_drain_aborts_on_thinking_loop,
        test_drain_aborts_on_per_chunk_inactivity,
        test_drain_aborts_on_total_wallclock,
        test_drain_propagates_reader_exception,
        test_line_loop_two_line_block_repeat,
        test_line_loop_single_line_repeat_with_min_block_2,
        test_line_loop_varied_content_no_flag,
        test_line_loop_blank_lines_ignored,
        test_drain_aborts_on_line_loop,
        test_strip_wrappers_think_and_fence,
        test_strip_wrappers_open_think_only,
        test_validate_parsed_happy_path,
        test_validate_parsed_unknown_id,
    ]
    results = [_run(t) for t in tests]
    failed = sum(1 for r in results if not r)
    total = len(results)
    if failed:
        print(f"\n{failed}/{total} failed")
        sys.exit(1)
    print(f"\n{total}/{total} passed")
