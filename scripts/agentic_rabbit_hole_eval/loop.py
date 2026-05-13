"""Agentic message loop — model drives its own retrieval via search_corpus tool."""

import json
import logging
import os
import queue
import re
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional, Set

from openai import OpenAI

from .schemas import CaseResult
from .search import format_chunks_for_model

logger = logging.getLogger(__name__)

# ── ANSI ─────────────────────────────────────────────────────────────────────

_ANSI_RESET      = "\033[0m"
_ANSI_MAGENTA    = "\033[35m"
_ANSI_DIM_GREY   = "\033[90m\033[2m"
_ANSI_DIM_YELLOW = "\033[33m\033[2m"
_ANSI_CLEAR_LINE = "\r\033[2K"
_LIVE_PREVIEW_CHARS = 60
_LIVE_REFRESH_HZ    = 5.0

_THINK_RE      = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)

_STREAM_SENTINEL = object()
_MAX_TOKENS = 8192


# ── Tool definition ───────────────────────────────────────────────────────────

SEARCH_CORPUS_TOOL = {
    "type": "function",
    "function": {
        "name": "search_corpus",
        "description": (
            "Search the paper corpus for chunks relevant to a query. "
            "Returns raw retrieved chunks with text and metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query string."}
            },
            "required": ["query"],
        },
    },
}


# ── Streaming ─────────────────────────────────────────────────────────────────

def _strip_wrappers(content: str) -> str:
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


def _drain_stream(
    stream,
    label: str,
    use_tty: bool,
    *,
    total_timeout: float = 180.0,
    per_chunk_timeout: float = 30.0,
):
    """Drain a streaming chat completion.

    Returns (content, tool_calls) where tool_calls is a list of accumulated
    tool-call dicts (empty if the model answered with text).

    Shows reasoning_content in dim yellow on stderr and content in dim grey.
    """
    q: "queue.Queue" = queue.Queue()

    def reader():
        try:
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta
                    content_delta = delta.content or ""
                    think_delta   = getattr(delta, "reasoning_content", None) or ""
                    tc_list       = delta.tool_calls or []
                except (AttributeError, IndexError):
                    content_delta = think_delta = ""
                    tc_list = []
                if content_delta:
                    q.put(("content", content_delta))
                if think_delta:
                    q.put(("think", think_delta))
                for tc_d in tc_list:
                    q.put(("tool_call_delta", tc_d))
        except Exception as exc:
            q.put(("__error__", exc))
        finally:
            q.put(_STREAM_SENTINEL)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    content    = ""
    thinking   = ""
    tc_acc     = {}   # index -> {id, type, function: {name, arguments}}
    last_render = 0.0
    refresh_period = 1.0 / _LIVE_REFRESH_HZ
    deadline   = time.monotonic() + total_timeout
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
                    abort_reason = f"no chunk for {per_chunk_timeout:.0f}s — stream stuck"
                break
            if item is _STREAM_SENTINEL:
                break
            if isinstance(item, tuple) and item[0] == "__error__":
                raise item[1]

            kind, data = item

            if kind == "think":
                thinking += data
                if use_tty and any(c in data for c in ".?!"):
                    tail = (thinking.splitlines()[-1] if thinking.splitlines() else data).strip()
                    tail = tail[-_LIVE_PREVIEW_CHARS:]
                    sys.stderr.write(
                        f"{_ANSI_CLEAR_LINE}{_ANSI_DIM_YELLOW}{label} "
                        f"[think {len(thinking):>5}c]: {tail}{_ANSI_RESET}"
                    )
                    sys.stderr.flush()

            elif kind == "content":
                content += data
                if use_tty:
                    now2 = time.monotonic()
                    if now2 - last_render >= refresh_period:
                        last_render = now2
                        tail = content.replace("\n", " ")[-_LIVE_PREVIEW_CHARS:]
                        sys.stderr.write(
                            f"{_ANSI_CLEAR_LINE}{_ANSI_DIM_GREY}{label} "
                            f"[{len(content):>5}c]: {tail}{_ANSI_RESET}"
                        )
                        sys.stderr.flush()

            elif kind == "tool_call_delta":
                tc_d = data
                idx  = tc_d.index
                if idx not in tc_acc:
                    tc_acc[idx] = {"id": "", "type": "function",
                                   "function": {"name": "", "arguments": ""}}
                if tc_d.id:
                    tc_acc[idx]["id"] = tc_d.id
                if tc_d.function:
                    if tc_d.function.name:
                        tc_acc[idx]["function"]["name"] += tc_d.function.name
                    if tc_d.function.arguments:
                        tc_acc[idx]["function"]["arguments"] += tc_d.function.arguments

    finally:
        if use_tty:
            sys.stderr.write(_ANSI_CLEAR_LINE)
            sys.stderr.flush()

    if abort_reason:
        logger.warning("Stream aborted: %s", abort_reason)
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    tool_calls = [tc_acc[i] for i in sorted(tc_acc)] if tc_acc else []
    return content, tool_calls


def _call_streaming(
    client: OpenAI,
    kwargs: dict,
    label: str,
    timeout_seconds: float,
    per_chunk_timeout: float = 30.0,
):
    """Make a streaming chat completion call.

    Returns (content, tool_calls). Falls back to non-streaming on TypeError.
    """
    use_tty = sys.stderr.isatty() and os.environ.get("EVAL_NO_STREAM") != "1"
    timed_client = client.with_options(timeout=timeout_seconds)

    try:
        stream = timed_client.chat.completions.create(**kwargs, stream=True)
        return _drain_stream(stream, label, use_tty,
                             total_timeout=timeout_seconds,
                             per_chunk_timeout=per_chunk_timeout)
    except TypeError:
        # Backend doesn't support stream — fall back.
        response = timed_client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        content    = msg.content or ""
        tool_calls = [json.loads(tc.model_dump_json()) for tc in (msg.tool_calls or [])]
        return content, tool_calls
    except Exception as exc:
        logger.warning("API call failed (%s): %s", label, exc)
        return "", []


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_case(
    seed_prompt: str,
    *,
    prompt_index: int = 0,
    tag: str = "",
    client: OpenAI,
    model: str,
    system_prompt: str,
    max_query_depth: int,
    execute_search_fn: Callable[[str, Set, Set], List[dict]],
    temperature: float = 0.1,
    timeout_seconds: float = 180.0,
    case_deadline: float = 0.0,
) -> CaseResult:
    """Seed prompt → tool loop → final answer.

    Each LLM call is streamed so thinking tokens appear on stderr.
    Tool queries are printed in magenta to stdout.
    """
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": seed_prompt},
    ]
    turns: List[dict] = []
    tool_calls_made = 0
    tag_label = f"[{tag}] " if tag else ""
    base_label = f"{tag_label}p{prompt_index + 1:03d}"
    seen_dense_ids: Set = set()
    seen_sparse_ids: Set = set()

    while True:
        offer_tool = tool_calls_made < max_query_depth

        if case_deadline > 0 and time.monotonic() >= case_deadline:
            logger.warning("%s case deadline exceeded", base_label)
            turns.append({"type": "final_answer", "reply": ""})
            break

        kwargs: dict = {
            "model":       model,
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  _MAX_TOKENS,
        }
        if offer_tool:
            kwargs["tools"]       = [SEARCH_CORPUS_TOOL]
            kwargs["tool_choice"] = "auto"

        call_n  = tool_calls_made + 1
        label   = f"{base_label} call={call_n}"
        content, tool_calls = _call_streaming(client, kwargs, label, timeout_seconds)

        if offer_tool and tool_calls:
            # Model called the tool.
            tc   = tool_calls[0]
            args = json.loads(tc["function"]["arguments"])
            query = args.get("query", "")
            tool_calls_made += 1

            # Magenta query line to stdout.
            print(f"\033[35m[tool {tool_calls_made}/{max_query_depth}] query: {query}\033[0m")

            chunks      = execute_search_fn(query, seen_dense_ids, seen_sparse_ids)
            result_text = format_chunks_for_model(chunks)

            # Update seen-ID sets for next call.
            for c in chunks:
                if c.get("source") == "dense":
                    seen_dense_ids.add(c["id"])
                else:
                    seen_sparse_ids.add(c["id"])

            messages.append({
                "role":       "assistant",
                "content":    None,
                "tool_calls": [tc],
            })
            messages.append({
                "role":        "tool",
                "tool_call_id": tc["id"],
                "content":     result_text,
            })

            turns.append({
                "type":   "tool_call",
                "depth":  tool_calls_made,
                "query":  query,
                "chunks": chunks,
            })

        else:
            # No tool called or not offered — final answer.
            reply = _strip_wrappers(content)
            turns.append({"type": "final_answer", "reply": reply})
            break

        if case_deadline > 0 and time.monotonic() >= case_deadline:
            turns.append({"type": "final_answer", "reply": ""})
            break

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    completed = bool(turns and turns[-1].get("type") == "final_answer"
                     and turns[-1].get("reply", ""))
    return CaseResult(
        turns=turns,
        tool_calls_made=tool_calls_made,
        completed=completed,
        timestamp_utc=ts,
    )
