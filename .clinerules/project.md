# Project Framing: EPUB-to-Qdrant Ingestion Pipeline

## Overview

This project builds a standalone ingestion pipeline that reads a library of EPUB files, extracts their text content, generates embeddings using a local Ollama embedding model, and stores the resulting vectors and metadata in Qdrant for semantic search and retrieval.

The system is intentionally decoupled from OpenWebUI and other chat-facing tools. Its purpose is not to provide a user interface, but to create a clean, repeatable, inspectable document indexing workflow that can later be consumed by whatever application or agent stack is appropriate.

## Why This Project Exists

The immediate need is to put a collection of EPUB books into Qdrant without depending on community-supported integration layers. The preferred architecture is direct and boring in the best sense:

- EPUB files are treated as source documents.
- A standalone script or small service handles ingestion.
- Ollama is used only for embedding generation.
- Qdrant is used only for vector storage and retrieval.

This keeps the indexing pipeline understandable, testable, and under local control.

## Core Goal

Create a reliable ingestion tool that can take a directory of EPUB files and populate a Qdrant collection with well-structured vectorized chunks, preserving enough metadata to support useful retrieval later.

## Desired Outcome

At the end of this project, it should be possible to:

1. Point the tool at a folder of EPUB files.
2. Extract readable text from each book.
3. Split that text into meaningful chunks.
4. Generate embeddings through a local Ollama model.
5. Upsert those chunks into Qdrant with stable IDs and useful metadata.
6. Run semantic queries against the resulting collection and retrieve relevant passages.

## Non-Goals

This project is not intended to:

- depend on OpenWebUI for ingestion,
- build a full end-user search interface,
- summarize or rewrite books during ingestion,
- perform OCR on image-based books,
- solve every document format at once.

The initial target is EPUB only, with a clean enough design that other document types could be added later.

## Design Principles

### 1. Standalone over integrated
The ingestion pipeline should stand on its own. It should not require a chat UI, plugin ecosystem, or community adapter in order to function.

### 2. Preserve structure where possible
Books have chapters, sections, and headings. The pipeline should preserve these boundaries when practical rather than flattening everything into anonymous text.

### 3. Metadata matters
Each stored chunk should carry enough metadata to identify where it came from and make filtering practical.

### 4. Re-runnable without chaos
Re-ingesting the same library should not create uncontrolled duplication. Point IDs should be deterministic.

### 5. Test behavior along the way
Extraction, chunking, embedding, and storage should each be validated with small functional tests rather than treated as a black box.

## High-Level Architecture

```text
EPUB files
   ↓
EPUB parser / XHTML extraction
   ↓
Text cleaning and section handling
   ↓
Chunking with overlap
   ↓
Ollama embedding requests
   ↓
Qdrant upsert with payload metadata
   ↓
Semantic retrieval
