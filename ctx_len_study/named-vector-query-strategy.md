# Named Vector Query Strategy

## Background

Collections store **named vectors**: `dense` (semantic embeddings from embeddinggemma-300m) and `sparse` (SAE-SPLADE from gemma-scope-2-270m-pt, JumpReLU, 65k latent concepts). The plain `papers` collection stores a single unnamed vector.

## Collections

| Collection | Vectors | Size | Points | Sparse Model |
|---|---|---|---|---|
| `papers` | unnamed dense | 768-d Cosine | ~90K | N/A |
| `papers-2048ctx-SAE` | dense (768) + sparse (IDF) | 768-d + variable | ~78K | SAE-SPLADE JumpReLU |

`papers-2048ctx-SAE` uses SAE-SPLADE: sparse keyword embeddings from a Sparse Autoencoder applied to the resid_post layer 12 activations of gemma-scope-2-270m-pt. Active dimensions correspond to learned semantic concepts, not surface tokens.

## Query Approach

### Unnamed vector collections (e.g., `papers`)

Single query, no `using=` parameter. Set `limit` to desired result count.

```python
points = client.query_points(
    collection_name="papers",
    query=query_vec,
    limit=topk,
    query_filter=filter,
)
```

### Named vector collections (e.g., `papers-2048ctx-SAE`)

Query **both named vectors with the same filter**, each with `limit=topk//2`, then concatenate:

- Dense query: `using="dense"`, `limit=topk//2`, filter → results
- Sparse query: `using="sparse"`, `limit=topk//2`, filter → results
- Concatenate dense first, sparse second → `topk` results total

No deduplication needed — results come from different vector spaces.

### Config

Hardcoded in `generate_study.py`:

```python
QDRANT_COLLECTIONS = ["papers", "papers-2048ctx-SAE"]
NAMED_VECTOR_COLLECTIONS = {"papers-2048ctx-SAE"}
```

All collections in `QDRANT_COLLECTIONS` are queried per prompt. Collections in `NAMED_VECTOR_COLLECTIONS` use hybrid dense+sparse. Others use single dense.

Bedrock KB (RQMBIXUSXH) is always queried as a fourth contender.

## CLI

```bash
python3 generate_study.py [--limit N] [--topk M]
```

- `--limit`: Number of papers to process (sorted, first N)
- `--topk`: Results per Qdrant query per collection (must be divisible by 4)

## Server Testing

The only source of truth for a server being operational is testing it yourself. See `server-testing.md`.