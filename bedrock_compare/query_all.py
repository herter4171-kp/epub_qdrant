#!/usr/bin/env python3
"""Query all 50 prompts against three retrieval sources and save results.

Sources:
  - agent-lookup on `papers` collection (dense, k=topk)
  - agent-lookup on `papers-semantic` collection (hybrid dense+sparse, k=topk*2)
  - papers-bedrock KB `RQMBIXUSXH` (k=topk)

Usage:
    python bedrock_compare/query_all.py --topk 10
    python bedrock_compare/query_all.py --topk 10 --prompt-id 5
    python bedrock_compare/query_all.py --topk 10 --dry-run
"""

import argparse
import asyncio
import httpx
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from bedrock_client import BedrockKBClient

PROMPTS_PATH = Path(__file__).parent / "prompts.json"
OUTPUT_DIR = Path(__file__).parent / "query_results"
AGENT_LOOKUP_URL = "http://localhost:8090/mcp"
BEDROCK_KB_ID = "RQMBIXUSXH"
BEDROCK_REGION = "us-gov-west-1"


# ── Agent-lookup MCP client (HTTP streamable) ──────────────────────────────

class AgentLookupClient:
    """JSON-RPC 2.0 over HTTP POST to agent-lookup MCP endpoint."""

    def __init__(self, url: str = AGENT_LOOKUP_URL):
        self.url = url
        self._initialized = False
        self._request_id = 0

    async def start(self):
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                    },
                },
            )
            resp.raise_for_status()
            self._request_id += 1
            await client.post(
                self.url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": "notifications/initialized",
                    "params": {},
                },
            )
            self._request_id += 1
        self._initialized = True

    async def query(self, collection: str, query: str, top_k: int) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                self.url,
                json={
                    "jsonrpc": "2.0",
                    "id": self._request_id,
                    "method": "tools/call",
                    "params": {
                        "name": "query",
                        "arguments": {
                            "query": query,
                            "mode": "search",
                            "top_k": top_k,
                            "collection": collection,
                        },
                    },
                },
            )
            resp.raise_for_status()
            body = resp.json()
            self._request_id += 1
            if "result" in body:
                result = body["result"]
                if isinstance(result, dict):
                    return result
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get("content", {})
                    if isinstance(content, dict):
                        return content
                    if isinstance(content, str):
                        try:
                            return json.loads(content)
                        except json.JSONDecodeError:
                            return {"raw": content}
            return body

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self.url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 0,
                        "method": "initialize",
                        "params": {"protocolVersion": "2024-11-05", "capabilities": {}},
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def stop(self):
        """No-op: HTTP client doesn't need explicit cleanup."""
        pass


# ── Prompt processing ──────────────────────────────────────────────────────

def load_prompts() -> dict:
    with open(PROMPTS_PATH, "r") as f:
        return json.load(f)


def format_prompt_result(
    prompt_id: int,
    category: str,
    proficiency: int,
    prompt_text: str,
    topk: int,
    sources: dict,
) -> dict:
    return {
        "id": prompt_id,
        "category": category,
        "proficiency": proficiency,
        "prompt": prompt_text,
        "topk": topk,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
    }


def output_filename(prompt_id: int, category: str, proficiency: int, topk: int) -> str:
    safe_cat = category.replace(" ", "_").replace("/", "_")
    return f"{prompt_id}_{safe_cat}_{proficiency}_{topk}.json"


# ── Response normalization ─────────────────────────────────────────────────

def normalize_bedrock_results(
    bedrock_results: list, topk: int
) -> dict:
    """Normalize papers-bedrock KB results to match agent-lookup content format.

    Transforms the flat bedrock response list:
        [{"content": {"text": "...", "type": "TEXT"}, "location": {...}, "score": N}, ...]
    into the same wrapped format used by agent-lookup:
        {"content": [{"type": "TEXT", "text": "..."}, ...], "total_results": N, ...}

    This allows downstream analysis to treat all three sources uniformly.
    """
    if not bedrock_results:
        return {"content": [], "total_results": 0, "source": "bedrock"}

    content_items = []
    for hit in bedrock_results[:topk]:
        content_obj = hit.get("content", {})
        text = content_obj.get("text", "") if isinstance(content_obj, dict) else str(content_obj)
        item_type = content_obj.get("type", "TEXT") if isinstance(content_obj, dict) else "TEXT"
        content_items.append({"type": item_type, "text": text})

    return {
        "content": content_items,
        "total_results": len(bedrock_results),
        "source": "bedrock",
    }


# ── Main ───────────────────────────────────────────────────────────────────

async def run_query(
    prompt_text: str,
    topk: int,
    agent_client: AgentLookupClient,
    bedrock_client: BedrockKBClient,
) -> dict:
    """Run all three sources for a single prompt."""
    sources = {}

    # 1. agent-lookup on papers (dense, k=topk)
    start = time.time()
    papers_result = await agent_client.query(
        collection="papers",
        query=prompt_text,
        top_k=topk,
    )
    sources["papers"] = {**papers_result, "lookup_time": time.time() - start}

    # 2. agent-lookup on papers-semantic (hybrid, k=topk*2)
    start = time.time()
    papers_semantic_result = await agent_client.query(
        collection="papers-semantic",
        query=prompt_text,
        top_k=topk * 2,
    )
    sources["papers_semantic"] = {**papers_semantic_result, "lookup_time": time.time() - start}

    # 3. papers-bedrock (k=topk) — runs synchronously in background thread
    start = time.time()
    bedrock_raw = await asyncio.to_thread(
        bedrock_client.query, prompt_text, topk
    )
    sources["bedrock"] = {**normalize_bedrock_results(bedrock_raw, topk), "lookup_time": time.time() - start}

    return sources


async def main():
    parser = argparse.ArgumentParser(
        description="Query all prompts against three retrieval sources."
    )
    parser.add_argument(
        "--topk",
        type=int,
        required=True,
        help="Number of results per source. Must be even.",
    )
    parser.add_argument(
        "--prompt-id",
        type=int,
        default=None,
        help="Run only a single prompt by ID (debugging).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate prompts.json and test connections, then exit.",
    )
    args = parser.parse_args()

    if args.topk % 2 != 0:
        print(f"Error: --topk must be even, got {args.topk}", file=sys.stderr)
        sys.exit(1)

    # Load prompts
    prompts_data = load_prompts()
    prompts = prompts_data["prompts"]

    if args.prompt_id is not None:
        prompts = [p for p in prompts if p["id"] == args.prompt_id]
        if not prompts:
            print(f"Error: prompt id {args.prompt_id} not found", file=sys.stderr)
            sys.exit(1)

    print(f"Loaded {len(prompts)} prompts. topk={args.topk}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Test connections
        print("\nTesting agent-lookup connection...")
        al = AgentLookupClient()
        ok = await al.health_check()
        print(f"  agent-lookup: {'OK' if ok else 'FAILED'}")
        if not ok:
            print("  Please ensure agent-lookup MCP server is running on localhost:8090",
                  file=sys.stderr)
            sys.exit(1)

        print("\nTesting bedrock connection...")
        try:
            bc = BedrockKBClient(
                kb_id=BEDROCK_KB_ID,
                region=BEDROCK_REGION,
            )
            health = bc.query("test query", number_of_results=1)
            print(f"  papers-bedrock: OK ({len(health)} results)")
        except Exception as e:
            print(f"  papers-bedrock: FAILED - {e}", file=sys.stderr)
            sys.exit(1)

        print("\nDry run passed. All connections OK.")
        return

    # Full run
    al = AgentLookupClient()
    await al.start()

    # Create bedrock client (reused across all queries)
    bc = BedrockKBClient(
        kb_id=BEDROCK_KB_ID,
        region=BEDROCK_REGION,
    )

    total = len(prompts)
    success = 0
    failures = 0

    for i, prompt in enumerate(prompts, 1):
        pid = prompt["id"]
        category = prompt["category"]
        proficiency = prompt["proficiency"]
        text = prompt["prompt"]

        fname = output_filename(pid, category, proficiency, args.topk)
        fpath = OUTPUT_DIR / fname

        if fpath.exists():
            print(f"[{i}/{total}] {pid} {category} p{proficiency}: SKIP (exists)")
            success += 1
            continue

        print(f"[{i}/{total}] {pid} {category} p{proficiency}: querying...", end=" ", flush=True)

        try:
            start = time.time()
            sources = await run_query(text, args.topk, al, bc)
            elapsed = time.time() - start

            result = format_prompt_result(pid, category, proficiency, text, args.topk, sources)
            with open(fpath, "w") as f:
                json.dump(result, f, indent=2, default=str)

            # Count results per source
            papers_count = len(sources.get("papers", {}).get("groups", []))
            semantic_count = len(sources.get("papers_semantic", {}).get("groups", []))
            bedrock_count = len(sources.get("bedrock", {}).get("content", []))

            print(f"OK ({elapsed:.1f}s) papers={papers_count} semantic={semantic_count} bedrock={bedrock_count}")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            # Write error file
            error_result = {
                "id": pid,
                "category": category,
                "proficiency": proficiency,
                "prompt": text,
                "topk": args.topk,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sources": {},
            }
            with open(fpath, "w") as f:
                json.dump(error_result, f, indent=2, default=str)
            failures += 1

    await al.stop()

    print(f"\nDone. {success} succeeded, {failures} failed out of {total}.")
    print(f"Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())