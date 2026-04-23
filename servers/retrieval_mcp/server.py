"""FastAPI-based MCP server with Streamable HTTP transport for EPUB knowledge base retrieval.

Supports multi-collection search across all configured Qdrant collections.
"""

import asyncio
import json
import logging
import sys
import pickle
import multiprocessing
from typing import Any, Dict, List, Optional

# Global reference to the retriever and llm client for multiprocessing
_retriever_for_pickle: Any = None
_llm_client_for_pickle: Any = None

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from src.storage import Storage
from mcp_server.config import settings
from mcp_server.retriever import Retriever
from mcp_server.llm_client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("epub-retrieval-mcp")

# ─── Globals ─────────────────────────────────────────────────────────
_storage: Optional[Storage] = None
_retriever: Optional[Retriever] = None
_llm_client: Optional[LLMClient] = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


def get_retriever(collection: Optional[str] = None) -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(collection=collection)
    return _retriever


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


# ─── Tool Definitions (MCP JSON Schema) ──────────────────────────────

TOOLS = [
    {
        "name": "query",
        "description": (
            "Query the EPUB knowledge base. Use mode='search' to retrieve raw "
            "chunk results grouped by section or book with similarity scores. "
            "Use mode='answer' to generate an LLM answer grounded in retrieved "
            "evidence. Supports single-collection or cross-collection search. "
            "Use 'collections' parameter to search multiple collections at once, "
            "or 'collection' to target a specific one."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or question.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["search", "answer"],
                    "description": "Query mode: 'search' returns raw chunk results. 'answer' generates an LLM answer from retrieved evidence.",
                    "default": "search",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results per collection (default 15).",
                    "default": 15,
                },
                "group_by": {
                    "type": "string",
                    "enum": ["section", "book"],
                    "description": "How to group results (default: section).",
                    "default": "section",
                },
                "collection": {
                    "type": "string",
                    "description": "Target a specific collection. If omitted, searches the default collection.",
                },
                "collections": {
                    "type": "string",
                    "description": "Comma-separated list of collections to search together. Overrides 'collection'.",
                },
                "filter_by": {
                    "type": "string",
                    "description": "JSON object of metadata filters, e.g. '{\"doc_type\": \"paper\"}'.",
                },
                "sparse_weight": {
                    "type": "number",
                    "description": "Multiplier for sparse vector in hybrid RRF fusion (default 0.25, set 0 for dense-only).",
                    "default": 0.25,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_context",
        "description": (
            "Get surrounding chunks around a specific section or topic within a source file. "
            "Tries exact section_title match first, falls back to semantic intra-file search. "
            "Accepts a natural language 'query' to anchor the lookup when section_title is unreliable. "
            "Useful for reading the full context of a chapter or topic."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source_file": {
                    "type": "string",
                    "description": "Filename (EPUB or PDF).",
                },
                "section_title": {
                    "type": "string",
                    "description": "Chapter/section title for exact-match lookup (tried first, falls back to semantic search).",
                },
                "query": {
                    "type": "string",
                    "description": "Optional natural language query to anchor the section lookup instead of section_title. Either section_title or query should be provided.",
                },
                "radius": {
                    "type": "integer",
                    "description": "Surrounding chunks per side (default 2).",
                    "default": 2,
                },
                "collection": {
                    "type": "string",
                    "description": "Target a specific collection.",
                },
            },
            "required": ["source_file"],
        },
    },
    {
        "name": "list_collections",
        "description": "List all available Qdrant collections with metadata (point count, vector config).",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ─── Tool Handlers ───────────────────────────────────────────────────

def _parse_filter(filter_str: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse a JSON filter string into a dict."""
    if not filter_str:
        return None
    try:
        return json.loads(filter_str)
    except json.JSONDecodeError:
        return None


def _run_search(query: str, top_k: int, group_by: str,
                collection: Optional[str], collections_str: Optional[str],
                filter_by: Optional[Dict[str, str]], retriever: Retriever,
                sparse_weight: float = 0.25):
    """Run the shared search logic and return an EvidenceBundle."""
    if collections_str:
        col_list = [c.strip() for c in collections_str.split(",") if c.strip()]
        return retriever.search_collections(
            query=query, top_k=top_k, group_by=group_by,
            collections=col_list, filter_by=filter_by, sparse_weight=sparse_weight,
        )
    elif collection:
        return retriever.search(
            query=query, top_k=top_k, group_by=group_by,
            collection=collection, filter_by=filter_by,
        )
    else:
        if settings.has_collections and len(settings.collections) > 1:
            return retriever.search_collections(
                query=query, top_k=top_k, group_by=group_by,
                filter_by=filter_by, sparse_weight=sparse_weight,
            )
        else:
            return retriever.search(
                query=query, top_k=top_k, group_by=group_by,
                filter_by=filter_by,
            )


def _build_groups_output(bundle) -> list:
    """Build groups output dict from EvidenceBundle."""
    groups_output = []
    for g in bundle.groups:
        groups_output.append({
            "group_key": g.group_key,
            "group_label": g.group_label,
            "title": g.title,
            "source_file": g.source_file,
            "best_score": round(g.best_score, 4),
            "avg_score": round(g.avg_score, 4),
            "chunk_count": len(g.results),
            "chunks": [
                {
                    "score": round(r.score, 4),
                    "text": r.text[:500],
                    "doc_type": r.doc_type,
                    "title": r.title or r.book_title or r.section_title,
                    "section": r.section or r.section_title,
                    "source_file": r.source_file,
                    "chunk_index": r.chunk_index,
                    "token_count": r.token_count,
                    "arxiv_id": r.arxiv_id,
                    "category": r.category,
                    "book_title": r.book_title,
                    "section_title": r.section_title,
                    "authors": r.authors if hasattr(r, "authors") else None,
                    "year": r.year if hasattr(r, "year") else None,
                }
                for r in g.results
            ],
        })
    return groups_output


def _build_sources_output(bundle) -> list:
    """Build sources output dict from EvidenceBundle."""
    sources_output = []
    for src in bundle.sources:
        sources_output.append({
            "id": src.id,
            "authors": src.authors,
            "title": src.title,
            "year": src.year,
            "arxiv_id": src.arxiv_id,
            "source_file": src.source_file,
            "formatted": src.format(),
        })
    return sources_output


def _handle_query(args: dict) -> dict:
    """Handle the query tool call (merged search + answer)."""
    query = args.get("query", "")
    mode = args.get("mode", "search")
    top_k = args.get("top_k", 20)
    group_by = args.get("group_by", "section")
    collection = args.get("collection")
    collections_str = args.get("collections")
    filter_by = _parse_filter(args.get("filter_by"))
    sparse_weight = args.get("sparse_weight", 0.25)

    retriever = get_retriever()
    bundle = _run_search(query, top_k, group_by, collection, collections_str, filter_by, retriever, sparse_weight)

    if mode == "answer":
        return _handle_query_answer(query, bundle)
    else:
        return _handle_query_search(bundle, query)


def _handle_query_search(bundle, query: str) -> dict:
    """Handle query with mode='search' — return raw chunk results."""
    return {
        "query": query,
        "mode": "search",
        "total_chunks": bundle.total_chunks,
        "groups": _build_groups_output(bundle),
        "prompt_context": bundle.prompt_context,
        "collections_queried": bundle.collections_queried,
        "sources": _build_sources_output(bundle),
    }


async def _handle_query_answer(query: str, bundle) -> dict:
    """Handle query with mode='answer' — return LLM-generated answer."""
    if not bundle.groups:
        return {
            "query": query,
            "mode": "answer",
            "answer": "No relevant documents found in the knowledge base.",
            "total_chunks": 0,
        }

    llm = get_llm_client()
    answer = await llm.answer(query=query, context=bundle.prompt_context)

    return {
        "query": query,
        "mode": "answer",
        "answer": answer,
        "total_chunks": bundle.total_chunks,
        "groups": [
            {
                "group_label": g.group_label,
                "title": g.title,
                "chunk_count": len(g.results),
            }
            for g in bundle.groups[:5]
        ],
        "sources": _build_sources_output(bundle),
    }


def _handle_get_context(args: dict) -> dict:
    """Handle the get_context tool call with semantic fallback anchoring."""
    source_file = args.get("source_file", "")
    section_title = args.get("section_title")
    query = args.get("query")
    radius = args.get("radius", 2)
    collection = args.get("collection")

    retriever = get_retriever()
    bundle = retriever.get_context(
        source_file=source_file,
        section_title=section_title,
        query=query,
        radius=radius,
        collection=collection,
    )

    groups_output = []
    for g in bundle.groups:
        groups_output.append({
            "group_label": g.group_label,
            "title": g.title,
            "chunks": [
                {
                    "score": round(r.score, 4),
                    "text": r.text,
                    "chunk_index": r.chunk_index,
                }
                for r in g.results
            ],
        })

    return {
        "query": bundle.query,
        "total_chunks": bundle.total_chunks,
        "groups": groups_output,
    }


def _handle_list_collections(_args: dict) -> dict:
    """Handle the list_collections tool call."""
    storage = get_storage()
    return {"collections": storage.list_collections_info()}


HANDLERS = {
    "query": _handle_query,
    "get_context": _handle_get_context,
    "list_collections": _handle_list_collections,
}


# ─── FastAPI App ─────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title="EPUB Knowledge Base Retrieval MCP",
        description=(
            "MCP server for EPUB knowledge base retrieval. "
            "Exposes query (search + answer), get_context (semantic fallback), "
            "and list_collections tools over Streamable HTTP transport. "
            f"Configured collections: {', '.join(settings.collections) or '(none)'}. "
            f"Default collection: {settings.DEFAULT_COLLECTION or '(none)'}.",
        ),
        version="0.3.0",
    )

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "collections": settings.collections,
            "default_collection": settings.DEFAULT_COLLECTION,
            "version": "0.3.0",
        }

    @app.get("/mcp/info")
    async def mcp_info() -> dict:
        return {
            "protocol": "StreamableHTTP",
            "spec_version": "2025-03-26",
            "collections": settings.collections,
            "default_collection": settings.DEFAULT_COLLECTION,
            "tools": [t["name"] for t in TOOLS],
        }

    @app.get("/collections")
    async def list_collections_endpoint() -> JSONResponse:
        """REST endpoint for listing available collections with stats."""
        storage = get_storage()
        return JSONResponse(content={
            "collections": storage.list_collections_info(),
        })

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> JSONResponse:
        """Main MCP endpoint. Accepts JSON-RPC 2.0 messages."""
        body = await request.json()

        # ── JSON-RPC 2.0 format ──────────────────────────────────
        if "jsonrpc" in body:
            return _handle_jsonrpc(body)

        # ── Legacy/compat format ─────────────────────────────────
        if "method" in body:
            method = body["method"]
            args = body.get("arguments", body.get("params", {}))

            if method not in HANDLERS:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"Unknown method: {method}", "status": "error"},
                )

            try:
                handler = HANDLERS[method]
                result = _run_sync_handler(handler, args if isinstance(args, dict) else {})
                return JSONResponse(content={"status": "ok", "result": result})
            except Exception as e:
                logger.error(f"Method '{method}' failed: {e}", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "status": "error"},
                )

        return JSONResponse(
            status_code=400,
            content={"error": "Invalid message format. Expected JSON-RPC 2.0 or legacy format."},
        )

    return app


def _run_async_handler(worker_args: dict) -> dict:
    """Run the async answer handler in a fresh event loop.

    This function runs in a separate process via ProcessPoolExecutor,
    so it can safely create its own event loop without conflicts.
    """
    import asyncio

    global _storage, _retriever, _llm_client

    handler_name = worker_args["handler"]
    args = worker_args["args"]

    # Initialize globals fresh in this process
    _storage = Storage()
    _retriever = Retriever()
    _llm_client = LLMClient()

    if handler_name == "query":

        query = args.get("query", "")
        mode = args.get("mode", "search")
        top_k = args.get("top_k", 20)
        group_by = args.get("group_by", "section")
        collection = args.get("collection")
        collections_str = args.get("collections")
        filter_by = _parse_filter(args.get("filter_by"))
        sparse_weight = args.get("sparse_weight", 0.25)

        retriever = get_retriever()
        bundle = _run_search(query, top_k, group_by, collection, collections_str, filter_by, retriever, sparse_weight)

        if mode == "answer":
            return asyncio.run(_handle_query_answer(query, bundle))
        else:
            return _handle_query_search(bundle, query)

    elif handler_name == "get_context":

        source_file = args.get("source_file", "")
        section_title = args.get("section_title")
        q = args.get("query")
        radius = args.get("radius", 2)
        collection = args.get("collection")

        retriever = get_retriever()
        bundle = retriever.get_context(
            source_file=source_file, section_title=section_title,
            query=q, radius=radius, collection=collection,
        )

        groups_output = []
        for g in bundle.groups:
            groups_output.append({
                "group_label": g.group_label,
                "title": g.title,
                "chunks": [
                    {
                        "score": round(r.score, 4),
                        "text": r.text,
                        "chunk_index": r.chunk_index,
                    }
                    for r in g.results
                ],
            })

        return {
            "query": bundle.query,
            "total_chunks": bundle.total_chunks,
            "groups": groups_output,
        }

    elif handler_name == "list_collections":
        storage = Storage()
        return {"collections": storage.list_collections_info()}

    return {"error": "Unknown handler"}


def _run_sync_handler(handler, args: dict):
    """Run a handler. For async handlers, run in a separate process.

    When already inside a running event loop (e.g. called from another MCP server),
    we cannot use run_until_complete. Instead we spawn a fresh process via
    ProcessPoolExecutor that creates its own event loop.
    """
    result = handler(args)
    if not asyncio.iscoroutine(result):
        return result

    # We're in an async context — use ProcessPoolExecutor to get a fresh event loop.
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run_async_handler, {"handler": "query", "args": args})
        return future.result()


def _handle_jsonrpc(body: dict) -> JSONResponse:
    """Handle a JSON-RPC 2.0 MCP message."""
    method = body.get("method")
    params = body.get("params", {})
    msg_id = body.get("id")

    def _error(code: int, message: str) -> JSONResponse:
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": code, "message": message},
            }
        )

    if method == "tools/call":
        tool_name = params.get("name", "")
        args = params.get("arguments", {})

        if tool_name not in HANDLERS:
            return _error(-32601, f"Tool not found: {tool_name}")

        try:
            result = _run_sync_handler(HANDLERS[tool_name], args)
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(result, default=str)}
                        ]
                    },
                }
            )
        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {e}", exc_info=True)
            return _error(-32603, str(e))

    elif method == "tools/list":
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS},
            }
        )

    elif method == "initialize":
        return JSONResponse(
            content={
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "epub-retrieval-mcp",
                        "version": "0.3.0",
                    },
                },
            }
        )

    return _error(-32601, f"Method not found: {method}")


# ─── Entry Point ─────────────────────────────────────────────────────

def main():
    """Start the MCP server as a long-running HTTP service."""
    logger.info("Starting EPUB Retrieval MCP Server...")
    logger.info("  Collections:     %s", ", ".join(settings.collections) or "(none)")
    logger.info("  Default:         %s", settings.DEFAULT_COLLECTION)
    logger.info("  Qdrant URL:      %s", settings.QDRANT_URL)
    logger.info("  Ollama URL:      %s", settings.OLLAMA_URL)
    logger.info("  LiteLLM URL:     %s", settings.LITELLM_API_URL)
    logger.info("  MCP Port:        %d", settings.MCP_PORT)

    if not settings.has_collections and not settings.QDRANT_COLLECTION:
        logger.error(
            "No collections configured. Set QDRANT_COLLECTIONS (comma-separated) "
            "or QDRANT_COLLECTION and try again."
        )
        sys.exit(1)

    if not settings.LITELLM_API_KEY:
        logger.warning(
            "LITELLM_API_KEY not set. The 'query' tool with mode='answer' will fail without it."
        )

    app = create_app()

    try:
        import uvicorn
        uvicorn.run(
            app,
            host=settings.MCP_HOST,
            port=settings.MCP_PORT,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Server shut down.")


if __name__ == "__main__":
    main()