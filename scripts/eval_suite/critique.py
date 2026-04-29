"""LLM judge critique of a retrieval set."""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .schemas import Critique, CritiqueChunk, JudgeOutput, RetrievalSet

logger = logging.getLogger(__name__)


def critique(
    retrieval_set: RetrievalSet,
    system_prompt: str,
    judge_base_url: str,
    judge_model: str,
    judge_api_key: str = "",
    max_retries: int = 1,  # one retry on empty only
) -> Critique:
    """Run LLM judge on retrieval_set. Returns Critique with judge_output."""

    # Build user message: each chunk carries an opaque hash id, publication title,
    # type code (D/S), and text. No rank, no qdrant id, no score.
    chunk_lines = []
    for chunk in retrieval_set.merged:
        type_code = "D" if chunk.source == "dense" else ("S" if chunk.source == "sparse" else "?")
        header = f"id: {chunk.judge_id}\ntitle: {chunk.title}\ntype: {type_code}"
        chunk_lines.append(f"{header}\ncontent:\n{chunk.text}")
    user_message = (
        f"QUERY: {retrieval_set.prompt_text}\n\n"
        "RETRIEVED CHUNKS:\n"
        + "\n\n---\n\n".join(chunk_lines)
    )

    client = OpenAI(base_url=judge_base_url, api_key=judge_api_key or "not-set")

    content = ""
    retry_count = 0
    empty_after_retry = False
    parse_failure = False

    while retry_count <= max_retries:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content or ""
        content = content.strip()

        # Strip code fences if present
        if content and content.startswith("```"):
            if "\n```" in content:
                fence_marker = content.find("\n```")
                content = content[:fence_marker].strip()
        content = content.strip()

        if content:
            # Try parse — if it fails, retry once more
            try:
                json.loads(content)
                break  # valid JSON, done
            except json.JSONDecodeError:
                if retry_count >= max_retries:
                    parse_failure = True
                    break  # out of retries, will handle below
                retry_count += 1
                continue  # retry with parse failure
        else:
            retry_count += 1

    if not content:
        empty_after_retry = True
    elif parse_failure:
        empty_after_retry = False  # content exists but failed parse

    # Final strip (safety net for non-fence empty content)
    if content:
        content = content.strip()

    # Try to parse JSON (already stripped of fences above)
    parsed = None
    parse_ok = False
    error = None
    retried = (retry_count > 0)

    if not empty_after_retry and content:
        try:
            parsed = json.loads(content)
            # Validate: must have 'chunks' array matching merged length
            if not isinstance(parsed, dict) or "chunks" not in parsed:
                parse_ok = False
                error = "missing 'chunks' key in parsed JSON"
            else:
                expected_len = len(retrieval_set.merged)
                actual_len = len(parsed["chunks"])
                if actual_len != expected_len:
                    parse_ok = False
                    error = (f"chunks array length mismatch: expected {expected_len}, "
                             f"got {actual_len}")
                else:
                    # Validate chunks: each must have a string id matching one of the
                    # issued judge_id hashes, and integer relevance 1..10.
                    issued_ids = {mc.judge_id for mc in retrieval_set.merged}
                    seen_ids = set()
                    valid = True
                    for entry in parsed["chunks"]:
                        eid = entry.get("id")
                        if not isinstance(eid, str):
                            valid = False
                            error = f"id is not str in entry: {entry}"
                            break
                        if eid not in issued_ids:
                            valid = False
                            error = f"unknown id {eid!r} in entry: {entry}"
                            break
                        if eid in seen_ids:
                            valid = False
                            error = f"duplicate id {eid!r} in entry: {entry}"
                            break
                        seen_ids.add(eid)
                        rel = entry.get("relevance")
                        if not isinstance(rel, int) or rel < 1 or rel > 10:
                            valid = False
                            error = f"relevance not int 1..10 in entry: {entry}"
                            break
                    if valid:
                        # Validate top-level fields
                        if "reply" not in parsed or not isinstance(parsed.get("reply"), str):
                            valid = False
                            error = "missing or invalid 'reply' field"
                        sat = parsed.get("satisfaction")
                        if not isinstance(sat, int) or sat < 1 or sat > 10:
                            valid = False
                            error = f"satisfaction not int 1..10: {sat}"
                    parse_ok = valid
                    if not valid:
                        parsed = None
        except json.JSONDecodeError as e:
            parse_ok = False
            error = str(e)

    # Build judge_output
    judge_output = JudgeOutput(
        raw=content,
        parsed=parsed if parse_ok else (parsed if not empty_after_retry else None),
        parse_ok=parse_ok and not empty_after_retry,
        retried=retried,
        error=None if (parse_ok and not empty_after_retry) else
        ("empty_response_after_retry" if empty_after_retry else
         "parse_failure_on_retry" if parse_failure else error),
    )

    # Build CritiqueChunk list from merged
    critique_chunks = [
        CritiqueChunk(
            rank=mc.rank,
            id=mc.id,
            source=mc.source,
            token_count=mc.token_count,
            text=mc.text,
            title=mc.title,
            judge_id=mc.judge_id,
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
        collection=retrieval_set.collection,
        embed_model_endpoint="",  # filled in by persist.py from config
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        timestamp_utc=retrieval_set.timestamp_utc,
        chunks=critique_chunks,
        judge_output=judge_output,
    )