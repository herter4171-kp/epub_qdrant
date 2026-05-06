"""LLM judge critique of a retrieval set."""

import json
import logging
import os
import queue
import re
import sys
import threading
import time

from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from .schemas import Critique, CritiqueChunk, JudgeOutput, RetrievalSet

logger = logging.getLogger(__name__)

_ANSI_DIM_GREY = "\033[90m\033[2m"
_ANSI_RESET = "\033[0m"
_ANSI_CLEAR_LINE = "\r\033[2K"
_LIVE_PREVIEW_CHARS = 60
_LIVE_REFRESH_HZ = 5.0

# Wall-clock cap for one full judge response.
_JUDGE_TIMEOUT_SECONDS = 180.0
# Cap on inactivity between SSE chunks. Catches proxy-hold scenarios where
# the TCP socket is healthy (keepalives) but no tokens arrive.
_PER_CHUNK_TIMEOUT_SECONDS = 30.0
# Hard cap on tokens the judge may emit. Stops runaway thinking server-side.
_JUDGE_MAX_TOKENS = 2048
# Total attempts per judgement. One flat retry loop catches stream-aborts,
# transport errors, and parse failures uniformly.
_JUDGE_ATTEMPTS = 3
# Hard ceiling per (prompt, sparse_k) case across all judges + retries.
_CASE_TIMEOUT_SECONDS = 600.0

# Char-level loop detection: tail substring repeats N+ times in lookback window.
_LOOP_TAIL_CHARS = 200
_LOOP_LOOKBACK_CHARS = 1200
_LOOP_REPEAT_THRESHOLD = 4
# Line-level loop detection: last K lines == K lines before that, K >= 2.
_LOOP_LINE_TAIL = 20
_LOOP_LINE_MIN_BLOCK = 2

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_STREAM_SENTINEL = object()


class StreamAbort(Exception):
    """Raised when the judge stream is killed for timeout/inactivity/loop."""


def _is_thinking_loop(
    content: str,
    *,
    tail_chars: int = _LOOP_TAIL_CHARS,
    lookback_chars: int = _LOOP_LOOKBACK_CHARS,
    repeat_threshold: int = _LOOP_REPEAT_THRESHOLD,
) -> bool:
    """True when the tail substring repeats threshold+ times in the lookback window.

    Char-level pattern: catches single-line or sub-line repeats verbatim.
    Cheap O(N) substring count over a bounded window so it's safe to call
    on every delta.
    """
    if len(content) < lookback_chars:
        return False
    needle = content[-tail_chars:]
    if not needle.strip():
        return False
    window = content[-lookback_chars:]
    return window.count(needle) >= repeat_threshold


def _is_line_loop(
    content: str,
    *,
    n_lines: int = _LOOP_LINE_TAIL,
    min_block: int = _LOOP_LINE_MIN_BLOCK,
) -> bool:
    """True when the last K-line block equals the K lines before it, K >= min_block.

    Catches multi-line block repeats that the char-level heuristic misses
    when newline noise (whitespace, indentation drift) breaks substring
    matching. Considers only non-empty lines so a flurry of blank lines
    doesn't mask the repeat.
    """
    raw_lines = content.splitlines()
    lines = [ln for ln in raw_lines[-n_lines:] if ln.strip()]
    n = len(lines)
    if n < min_block * 2:
        return False
    for k in range(min_block, n // 2 + 1):
        if lines[-k:] == lines[-2 * k:-k]:
            return True
    return False


def _drain_stream(
    stream: Iterable,
    label: str,
    use_tty: bool,
    *,
    total_timeout: float = _JUDGE_TIMEOUT_SECONDS,
    per_chunk_timeout: float = _PER_CHUNK_TIMEOUT_SECONDS,
) -> str:
    """Pull deltas off ``stream`` via a daemon reader thread.

    Aborts (StreamAbort) when any of:
      - total wall-clock exceeds ``total_timeout``
      - no chunk arrives for ``per_chunk_timeout``
      - tail of accumulated content repeats (thinking loop)

    Daemon thread isolates network reads so socket-level blocks never
    starve the main loop's timing checks. On abort, attempts ``stream.close()``
    so the upstream model stops generating; the daemon thread exits when
    its iteration breaks (or the process ends).
    """
    q: "queue.Queue" = queue.Queue()

    def reader():
        try:
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content or ""
                except (AttributeError, IndexError):
                    delta = ""
                if delta:
                    q.put(delta)
        except Exception as exc:  # bubble to main thread
            q.put(("__error__", exc))
        finally:
            q.put(_STREAM_SENTINEL)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    content = ""
    last_render = 0.0
    refresh_period = 1.0 / _LIVE_REFRESH_HZ
    deadline = time.monotonic() + total_timeout
    abort_reason: Optional[str] = None

    try:
        while True:
            now = time.monotonic()
            if now > deadline:
                abort_reason = f"total wall-clock {total_timeout:.0f}s exceeded"
                break
            remaining = deadline - now
            wait = min(per_chunk_timeout, remaining)
            try:
                item = q.get(timeout=wait)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    abort_reason = f"total wall-clock {total_timeout:.0f}s exceeded"
                else:
                    abort_reason = (
                        f"no chunk arrived for {per_chunk_timeout:.0f}s — stream stuck"
                    )
                break
            if item is _STREAM_SENTINEL:
                break
            if isinstance(item, tuple) and item and item[0] == "__error__":
                raise item[1]
            delta = item if isinstance(item, str) else ""
            if not delta:
                continue
            content += delta
            if _is_thinking_loop(content):
                abort_reason = "thinking-loop detected — char tail repeats"
                break
            if "\n" in delta and _is_line_loop(content):
                abort_reason = "thinking-loop detected — line block repeats"
                break
            if not use_tty:
                continue
            now2 = time.monotonic()
            if now2 - last_render < refresh_period:
                continue
            last_render = now2
            tail = content.replace("\n", " ").replace("\r", " ")[-_LIVE_PREVIEW_CHARS:]
            sys.stderr.write(
                f"{_ANSI_CLEAR_LINE}{_ANSI_DIM_GREY}{label} [{len(content):>5}c]: "
                f"{tail}{_ANSI_RESET}"
            )
            sys.stderr.flush()
    finally:
        if use_tty:
            sys.stderr.write(_ANSI_CLEAR_LINE)
            sys.stderr.flush()

    if abort_reason is not None:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        raise StreamAbort(abort_reason)

    return content


def _stream_judge(
    client: OpenAI,
    model: str,
    messages: list,
    label: str,
    *,
    total_timeout: float = _JUDGE_TIMEOUT_SECONDS,
    per_chunk_timeout: float = _PER_CHUNK_TIMEOUT_SECONDS,
    max_tokens: int = _JUDGE_MAX_TOKENS,
) -> str:
    """One streamed judge call. Returns accumulated content.

    Raises ``StreamAbort`` on timeout / inactivity / loop-detection, or
    transport exceptions (httpx ReadError, ConnectError, etc.) on
    network failure. Caller owns retry policy.

    Sampling parameters (temperature, top_p, etc.) are intentionally not
    sent — the inference server's CLI configuration is the source of truth.
    """
    use_tty = sys.stderr.isatty() and os.environ.get("EVAL_NO_STREAM") != "1"
    timed_client = client.with_options(timeout=total_timeout)

    try:
        stream = timed_client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
        )
    except TypeError:
        # SDK or backend without stream/max_tokens support — fall back blocking.
        resp = timed_client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"enable_thinking": False},
        )
        return resp.choices[0].message.content or ""

    return _drain_stream(
        stream,
        label=label,
        use_tty=use_tty,
        total_timeout=total_timeout,
        per_chunk_timeout=per_chunk_timeout,
    )


def _strip_wrappers(content: str) -> str:
    """Strip qwen <think>...</think> blocks and ```json ... ``` fences."""
    content = content.strip()
    content = _THINK_RE.sub("", content)
    if "</think>" in content and "<think>" not in content:
        content = _OPEN_THINK_RE.sub("", content, count=1)
    content = content.strip()
    if content.startswith("```"):
        first_nl = content.find("\n")
        if first_nl != -1:
            content = content[first_nl + 1:]
        content = content.rstrip()
        if content.endswith("```"):
            content = content[:-3].rstrip()
    return content.strip()


def _validate_parsed(parsed, issued_ids: set, expected_chunks: int):
    """Return (parse_ok, error_str_or_None). Walks the schema once."""
    if not isinstance(parsed, dict) or "chunks" not in parsed:
        return False, "missing 'chunks' key in parsed JSON"
    actual_len = len(parsed["chunks"])
    if actual_len != expected_chunks:
        return False, (
            f"chunks array length mismatch: expected {expected_chunks}, got {actual_len}"
        )
    seen_ids = set()
    for entry in parsed["chunks"]:
        eid = entry.get("id")
        if not isinstance(eid, str):
            return False, f"id is not str in entry: {entry}"
        if eid not in issued_ids:
            return False, f"unknown id {eid!r} in entry: {entry}"
        if eid in seen_ids:
            return False, f"duplicate id {eid!r} in entry: {entry}"
        seen_ids.add(eid)
        rel = entry.get("relevance")
        if not isinstance(rel, int) or rel < 1 or rel > 10:
            return False, f"relevance not int 1..10 in entry: {entry}"
    if "reply" not in parsed or not isinstance(parsed.get("reply"), str):
        return False, "missing or invalid 'reply' field"
    sat = parsed.get("satisfaction")
    if not isinstance(sat, int) or sat < 1 or sat > 10:
        return False, f"satisfaction not int 1..10: {sat}"
    return True, None


def _run_one_judgement(
    *,
    client: OpenAI,
    judge_model: str,
    system_prompt: str,
    user_message: str,
    issued_ids: set,
    expected_chunks: int,
    label: str,
    judge_attempts: int,
    judge_timeout_seconds: float,
    judge_per_chunk_timeout_seconds: float,
    judge_max_tokens: int,
    case_deadline: Optional[float] = None,
) -> JudgeOutput:
    """Drive ONE judgement with a flat retry loop.

    Up to ``judge_attempts`` total attempts. Each attempt runs ``_stream_judge``
    once and validates the parsed JSON. Any failure (StreamAbort, transport
    exception, JSON parse error, schema violation) consumes one attempt and
    is retried until the budget is exhausted. Returns a single JudgeOutput
    reflecting the final attempt's state.

    If ``case_deadline`` (monotonic) is provided and reached mid-attempt,
    abort early and return a wallclock-exceeded JudgeOutput.
    """
    last_content = ""
    last_error: Optional[str] = None
    last_parsed = None

    for attempt in range(1, judge_attempts + 1):
        if case_deadline is not None and time.monotonic() >= case_deadline:
            return JudgeOutput(
                raw=last_content,
                parsed=None,
                parse_ok=False,
                retried=attempt > 1,
                error="case_wallclock_exceeded",
            )
        attempt_label = label if attempt == 1 else f"{label} attempt {attempt}/{judge_attempts}"
        try:
            content = _stream_judge(
                client=client,
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                label=attempt_label,
                total_timeout=judge_timeout_seconds,
                per_chunk_timeout=judge_per_chunk_timeout_seconds,
                max_tokens=judge_max_tokens,
            )
        except StreamAbort as exc:
            last_error = f"stream_abort: {exc}"
            last_content = ""
            continue
        except Exception as exc:  # transport (ReadError, ConnectError, etc.)
            last_error = f"{type(exc).__name__}: {exc}"
            last_content = ""
            continue

        content = _strip_wrappers(content)
        last_content = content
        if not content:
            last_error = "empty_response"
            continue

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            last_error = f"json_decode: {exc}"
            last_parsed = None
            continue

        last_parsed = parsed
        ok, err = _validate_parsed(parsed, issued_ids, expected_chunks)
        if ok:
            return JudgeOutput(
                raw=content,
                parsed=parsed,
                parse_ok=True,
                retried=attempt > 1,
                error=None,
            )
        last_error = err

    return JudgeOutput(
        raw=last_content,
        parsed=last_parsed,
        parse_ok=False,
        retried=judge_attempts > 1,
        error=last_error,
    )


def critique(
    retrieval_set: RetrievalSet,
    system_prompt: str,
    judge_base_url: str,
    judge_model: str,
    judge_api_key: str = "",
    judge_timeout_seconds: float = _JUDGE_TIMEOUT_SECONDS,
    judge_per_chunk_timeout_seconds: float = _PER_CHUNK_TIMEOUT_SECONDS,
    judge_max_tokens: int = _JUDGE_MAX_TOKENS,
    judge_attempts: int = _JUDGE_ATTEMPTS,
    judges_per_case: int = 1,
    case_timeout_seconds: float = _CASE_TIMEOUT_SECONDS,
) -> Critique:
    """Run LLM judge on retrieval_set N times. Returns one Critique with
    ``judge_outputs`` holding all N samples. Used to characterize judge noise
    at fixed temperature.

    A single flat retry budget (``judge_attempts``) covers every failure mode:
    timeout, loop, transport reset, JSON decode, schema mismatch.

    A case-level wall-clock cap (``case_timeout_seconds``) backstops the
    whole call. If exceeded, remaining judge slots are filled with
    ``error='case_wallclock_exceeded'`` JudgeOutputs.
    """
    chunk_lines = []
    for chunk in retrieval_set.merged:
        type_code = "D" if chunk.source == "dense" else ("S" if chunk.source == "sparse" else "?")
        header = f"id: {chunk.docket_id}\ntitle: {chunk.title}\ntype: {type_code}"
        chunk_lines.append(f"{header}\ncontent:\n{chunk.text}")
    user_message = (
        f"QUERY: {retrieval_set.prompt_text}\n\n"
        "RETRIEVED CHUNKS:\n"
        + "\n\n---\n\n".join(chunk_lines)
    )

    issued_ids = {mc.docket_id for mc in retrieval_set.merged}
    expected_chunks = len(retrieval_set.merged)
    client = OpenAI(base_url=judge_base_url, api_key=judge_api_key or "not-set")

    base_label = f"judge p{retrieval_set.prompt_index} sk={retrieval_set.sparse_k}"
    case_deadline = time.monotonic() + case_timeout_seconds

    judge_outputs: List[JudgeOutput] = []
    for j in range(judges_per_case):
        label = base_label if judges_per_case == 1 else f"{base_label} j={j+1}/{judges_per_case}"
        if time.monotonic() >= case_deadline:
            judge_outputs.append(JudgeOutput(
                raw="", parsed=None, parse_ok=False, retried=False,
                error="case_wallclock_exceeded",
            ))
            continue
        jo = _run_one_judgement(
            client=client,
            judge_model=judge_model,
            system_prompt=system_prompt,
            user_message=user_message,
            issued_ids=issued_ids,
            expected_chunks=expected_chunks,
            label=label,
            judge_attempts=judge_attempts,
            judge_timeout_seconds=judge_timeout_seconds,
            judge_per_chunk_timeout_seconds=judge_per_chunk_timeout_seconds,
            judge_max_tokens=judge_max_tokens,
            case_deadline=case_deadline,
        )
        judge_outputs.append(jo)

    critique_chunks = [
        CritiqueChunk(
            rank=mc.rank,
            id=mc.id,
            source=mc.source,
            token_count=mc.token_count,
            text=mc.text,
            title=mc.title,
            docket_id=mc.docket_id,
        )
        for mc in retrieval_set.merged
    ]

    return Critique(
        schema_version=1,
        prompt_index=retrieval_set.prompt_index,
        prompt_text=retrieval_set.prompt_text,
        system_prompt_text="",  # filled in by persist.py from config
        topk=retrieval_set.topk,
        sparse_k=retrieval_set.sparse_k,
        dense_k=retrieval_set.dense_k,
        sparse_fraction=retrieval_set.sparse_fraction,
        dense_collection=retrieval_set.dense_collection,
        sparse_collection=retrieval_set.sparse_collection,
        dense_vector_name=retrieval_set.dense_vector_name,
        embed_model_endpoint="",  # filled in by persist.py from config
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        timestamp_utc=retrieval_set.timestamp_utc,
        chunks=critique_chunks,
        judge_outputs=judge_outputs,
    )