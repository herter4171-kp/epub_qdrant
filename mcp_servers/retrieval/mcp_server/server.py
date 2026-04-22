"""FastAPI-based MCP server with Streamable HTTP transport for EPUB knowledge base retrieval.

Supports multi-collection search across all configured Qdrant collections.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

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
        "name": "search",
        "description": (
            "Search the EPUB knowledge base. Retrieves semantically relevant "
            "chunks grouped by section or book with similarity scores. "
            "Use 'collections' parameter to search multiple collections at once, "
            "or 'collection' to target a specific one."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results per collection (default 20).",
                    "default": 20,
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
            },
            "required": ["query"],
        },
    },
    {
        "name": "answer",
        "description": (
            "Answer a question using the knowledge base. Retrieves relevant "
            "chunks, assembles evidence, and generates an LLM answer. "
            "Supports single-collection or cross-collection search."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to answer.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results per collection (default 20).",
                    "default": 20,
                },
                "group_by": {
                    "type": "string",
                    "enum": ["section", "book"],
                    "description": "How to group results (default: section).",
                    "default": "section",
                },
                "collection": {
                    "type": "string",
                    "description": "Target a specific collection.",
                },
                "collections": {
                    "type": "string",
                    "description": "Comma-separated list of collections to search together.",
                },
                "filter_by": {
                    "type": "string",
                    "description": "JSON object of metadata filters, e.g. '{\"doc_type\": \"epub\"}'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_context",
        "description": (
            "Get surrounding chunks around a specific section. Useful for "
            "reading the full context of a known chapter or section."
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
                    "description": "Chapter/section title.",
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
            "required": ["source_file", "section_title"],
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


def _handle_search(args: dict) -> dict:
    """Handle the search tool call."""
    query = args.get("query", "")
    top_k = args.get("top_k", 20)
    group_by = args.get("group_by", "section")
    collection = args.get("collection")
    collections_str = args.get("collections")
    filter_by = _parse_filter(args.get("filter_by"))

    retriever = get_retriever()

    # Determine search mode
    if collections_str:
        # Cross-collection search
        col_list = [c.strip() for c in collections_str.split(",") if c.strip()]
        bundle = retriever.search_collections(
            query=query, top_k=top_k, group_by=group_by,
            collections=col_list, filter_by=filter_by,
        )
    elif collection:
        # Single collection
        bundle = retriever.search(
            query=query, top_k=top_k, group_by=group_by,
            collection=collection, filter_by=filter_by,
        )
    else:
        # Default: use configured collections or single
        if settings.has_collections and len(settings.collections) > 1:
            bundle = retriever.search_collections(
                query=query, top_k=top_k, group_by=group_by,
                filter_by=filter_by,
            )
        else:
            bundle = retriever.search(
                query=query, top_k=top_k, group_by=group_by,
                filter_by=filter_by,
            )

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

    return {
        "query": bundle.query,
        "total_chunks": bundle.total_chunks,
        "groups": groups_output,
        "prompt_context": bundle.prompt_context,
        "collections_queried": bundle.collections_queried,
    }


async def _handle_answer(args: dict) -> dict:
    """Handle the answer tool call."""
    query = args.get("query", "")
    top_k = args.get("top_k", 20)
    group_by = args.get("group_by", "section")
    collection = args.get("collection")
    collections_str = args.get("collections")
    filter_by = _parse_filter(args.get("filter_by"))

    retriever = get_retriever()

    # Same collection resolution logic as search
    if collections_str:
        col_list = [c.strip() for c in collections_str.split(",") if c.strip()]
        bundle = retriever.search_collections(
            query=query, top_k=top_k, group_by=group_by,
            collections=col_list, filter_by=filter_by,
        )
    elif collection:
        bundle = retriever.search(
            query=query, top_k=top_k, group_by=group_by,
            collection=collection, filter_by=filter_by,
        )
    else:
        if settings.has_collections and len(settings.collections) > 1:
            bundle = retriever.search_collections(
                query=query, top_k=top_k, group_by=group_by,
                filter_by=filter_by,
            )
        else:
            bundle = retriever.search(
                query=query, top_k=top_k, group_by=group_by,
                filter_by=filter_by,
            )

    if not bundle.groups:
        return {
            "query": query,
            "answer": "No relevant documents found in the knowledge base.",
            "total_chunks": 0,
        }

    llm = get_llm_client()
    answer = await llm.answer(query=query, context=bundle.prompt_context)

    return {
        "query": query,
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
    }


def _handle_get_context(args: dict) -> dict:
    """Handle the get_context tool call."""
    source_file = args.get("source_file", "")
    section_title = args.get("section_title", "")
    radius = args.get("radius", 2)
    collection = args.get("collection")

    retriever = get_retriever()
    bundle = retriever.get_context(
        source_file=source_file,
        section_title=section_title,
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
    "search": _handle_search,
    "answer": _handle_answer,
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
            "Exposes search, answer, get_context, and list_collections tools "
            "over Streamable HTTP transport. "
            f"Configured collections: {', '.join(settings.collections) or '(none)'}. "
            f"Default collection: {settings.DEFAULT_COLLECTION or '(none)'}.",
        ),
        version="0.2.0",
    )

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "collections": settings.collections,
            "default_collection": settings.DEFAULT_COLLECTION,
            "version": "0.2.0",
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


def _run_sync_handler(handler, args: dict):
    """Run a handler, awaiting it if it's a coroutine."""
    result = handler(args)
    if asyncio.iscoroutine(result):
        return asyncio.get_event_loop().run_until_complete(result)
    return result


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
                        "version": "0.2.0",
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
            "LITELLM_API_KEY not set. The 'answer' tool will fail without it."
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