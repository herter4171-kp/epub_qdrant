# Project Framing: EPUB Knowledge Base with Qdrant and Ollama

## Overview

This project builds a local knowledge base from a library of EPUB books.

The system will extract text from EPUB files, split that text into structured chunks, generate embeddings using a local Ollama embedding model, and store the resulting vectors and payload metadata in Qdrant.

The goal is not merely to index books for nearest-neighbor search. The goal is to make the collection behave like a knowledge base: when asked a question, the system should retrieve relevant chunk-level evidence, aggregate those results back up to a meaningful document structure such as a chapter or book, and provide coherent supporting context to an LLM.

## Problem Statement

A simple vector store of disconnected chunks is not sufficient for this use case.

The desired behavior is closer to this:

- a question is asked in natural language,
- the system retrieves a generous top-k set of semantically relevant chunks,
- those hits are grouped and ranked by their originating document structure,
- nearby context is added where appropriate,
- the resulting evidence bundle is passed to the LLM,
- the LLM answers as though it is consulting a knowledge base rather than a pile of fragments.

This means the project is fundamentally a retrieval-and-aggregation system, not just an ingestion pipeline.

## Core Goal

Create a local document-aware retrieval system for EPUB books that supports knowledge-base-style question answering.

## Desired Outcome

At the end of this project, it should be possible to:

1. Ingest a folder of EPUB files into a Qdrant-backed semantic index.
2. Preserve enough metadata to trace each chunk back to its book and chapter.
3. Ask natural-language questions about the library.
4. Retrieve top-k chunk matches.
5. Aggregate those matches by chapter or book.
6. Assemble coherent evidence for an LLM.
7. Return an answer grounded in retrieved passages.
8. Resolve any retrieved result back to the original EPUB file on disk.

## Non-Goals

This project is not intended to:

- depend on OpenWebUI for ingestion or retrieval,
- treat Qdrant as the primary storage location for books,
- replace the original EPUB files,
- solve every document format in the first version,
- perform OCR on image-heavy or scanned books,
- build a polished multi-user production application in its first phase.

## Project Framing

This project has three distinct layers:

### 1. Source Library
The original EPUB files remain the source of truth and stay on disk.

### 2. Semantic Retrieval Index
Qdrant stores chunk vectors and payload metadata for semantic retrieval. All EPUBs live in a **single shared collection**.

### 3. Knowledge Base Retrieval Layer
A local retrieval layer sits in front of Qdrant and turns chunk-level hits into grouped, coherent evidence suitable for LLM use.

This separation matters. Qdrant is not the bookshelf. It is the retrieval engine.

## Implementation

The retrieval layer is implemented as a standalone MCP server at `mcp_servers/retrieval/`:

- `mcp_server/server.py` — MCP server backbone (Streamable HTTP on configurable port)
- `mcp_server/retriever.py` — Retrieval logic: search Qdrant → expand context → group by chapter/book → assemble evidence bundle
- `mcp_server/llm_client.py` — LiteLLM streaming chat completion for answer generation
- `mcp_server/config.py` — MCP-specific settings

The retriever uses the shared `src.storage` and `src.embedder` modules for Qdrant access and embedding generation.

## High-Level Architecture

```text
EPUB files on disk
   ↓
EPUB parsing and text extraction
   ↓
Section-aware chunking
   ↓
Ollama embeddings
   ↓
Qdrant chunk index (shared single collection)
   ↓
Retrieval MCP server (mcp_servers/retrieval/)
   ↓
Top-k retrieval + context expansion
   ↓
Aggregation by chapter/book
   ↓
Context assembly (EvidenceBundle)
   ↓
LLM answer grounded in evidence
```

## Retrieval Flow

1. **Query** — natural-language question arrives via MCP tool call
2. **Embed** — query text converted to vector via Ollama
3. **Search** — Qdrant returns top-k chunk matches by cosine similarity
4. **Expand** — surrounding chunks from same chapters added for context (`_expand_with_context`)
5. **Group** — results grouped by chapter or book (`_group_results`)
6. **Assemble** — evidence bundle formatted as prompt context (`_build_prompt_context`)
7. **Answer** — LiteLLM streams final answer grounded in retrieved evidence