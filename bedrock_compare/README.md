# Bedrock Compare

Compare retrieval results across three sources:

1. **agent-lookup `papers`** — Qdrant dense semantic search (768-d Ollama embeddings)
2. **agent-lookup `papers-semantic`** — Qdrant hybrid dense + sparse search (MiniCOIL)
3. **papers-bedrock KB `RQMBIXUSXH`** — Amazon Bedrock Knowledge Base retrieval

## Quick Start

```bash
# Run all 50 prompts across topK values 2, 4, 8, 16, 32
bash 0_retrieve_using_prompts.sh

# Run a single prompt (debugging)
python3 query_all.py --topk 10 --prompt-id 1

# Dry run (validate connections)
python3 query_all.py --topk 10 --dry-run
```

## Setup

- `agent-lookup` MCP server must be running on `localhost:8090`
- `papers-bedrock` MCP server via `uvx awslabs.bedrock-kb-retrieval-mcp-server@latest`
- AWS credentials configured for `us-gov-west-1` region

## Collections

### agent-lookup (Qdrant)

| Collection | Points | Vector Size | Notes |
|---|---|---|---|
| `papers` | 90,256 | 768 | Dense semantic (Ollama) |
| `papers-semantic` | 117,750 | 768 | Hybrid dense + sparse (MiniCOIL) |

### papers-bedrock (Amazon Bedrock)

| Item | Value |
|---|---|
| Knowledge Base ID | `RQMBIXUSXH` |
| Data Source ID | `QNK2BHS61Y` |
| Region | `us-gov-west-1` |
| Inclusion Tag | `knowledge-base-quick-start-PAPERS` |
| Reranking | Disabled |

## Output Format

Results are saved as JSON in `query_results/`:

```
query_results/{prompt_id}_{category}_{proficiency}_{topk}.json
```

Each file contains results from all three sources:

```json
{
  "id": 1,
  "category": "spatial_orientation",
  "proficiency": 1,
  "prompt": "...",
  "topk": 2,
  "timestamp": "...",
  "sources": {
    "papers": { ... },           // agent-lookup papers collection
    "papers_semantic": { ... },  // agent-lookup papers-semantic (hybrid)
    "bedrock": [ ... ]           // papers-bedrock KB results
  }
}
```

## Prompt Categories

| Category | Prompts | Description |
|---|---|---|
| `spatial_orientation` | 1–10 | Chunk-index scrolling and contiguous context retrieval |
| `structural_aggregation` | 11–20 | Heading hierarchy reconstruction and section-level filtering |
| `lexical_vs_semantic` | 21–30 | Dense vs sparse vector signal differentiation |
| `corpus_lineage` | 31–40 | Cross-document metadata traversal and timeline building |
| `analytical_deep_dives` | 41–50 | Metadata auditing, token density, and context window management |

## Source Response Formats

| Source | Shape | Notes |
|---|---|---|
| `papers` / `papers_semantic` | `{"content": [{"type": "text", "text": "{JSON string}"}]}` | Groups of chunks with scores |
| `bedrock` | `[{"content": {"text": "..."}, "location": {...}, "score": N}]` | Flat list with S3 location metadata |