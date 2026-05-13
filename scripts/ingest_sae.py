"""Bulk ingest SAE sparse vectors into Qdrant.

Reads pre-computed SPLADE activations from sae_data/activations.npy,
runs them through the SAE encoder, and upserts into a new collection.

Run:
    python3 scripts/ingest_sae.py
    python3 scripts/ingest_sae.py --batch-size 2048 --limit 1000  # smoke test
"""

import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
)
from safetensors import safe_open

# Ensure the project root is on sys.path so `src.config` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT   = "/tank/sae-splade/sae_data_good_2x/901120/sae_weights.safetensors"
DEC_NORMS    = Path("sae_data/dec_norms.pt")
ACTIVATIONS  = Path("sae_data/activations.npy")
CHUNK_IDS    = Path("sae_data/chunk_ids.npy")
COLLECTION   = "sae-sparse"
SAE_K        = 165
PROTECTED    = frozenset({"books", "books-named", "papers", "papers-named"})


def load_sae_encoder():
    """Load W_enc, b_enc, b_dec, dec_norms onto GPU in fp16."""
    logger.info("Loading SAE encoder weights...")
    with safe_open(CHECKPOINT, framework="pt") as f:
        W_enc = f.get_tensor("W_enc").to(torch.float16).cuda()
        b_enc = f.get_tensor("b_enc").to(torch.float16).cuda()
        b_dec = f.get_tensor("b_dec").to(torch.float16).cuda()
    dec_norms = torch.load(DEC_NORMS, weights_only=True).to(torch.float16).cuda()
    logger.info("SAE encoder ready. VRAM: %.2f GB", torch.cuda.memory_allocated() / 1e9)
    return W_enc, b_enc, b_dec, dec_norms


@torch.no_grad()
def encode_batch(activations_np, W_enc, b_enc, b_dec, dec_norms):
    """
    activations_np: numpy array [batch, 30522] float32
    returns: list of {"indices": [...], "values": [...]}
    """
    x = torch.from_numpy(activations_np.copy()).to(torch.float16).cuda()
    x_centered  = x - b_dec
    hidden_pre  = x_centered @ W_enc + b_enc
    hidden_pre  = hidden_pre * dec_norms
    top_values, top_indices = hidden_pre.topk(SAE_K, dim=-1)
    values = top_values.relu().to(torch.float32).cpu()
    indices = top_indices.cpu()

    results = []
    for idx, val in zip(indices, values):
        results.append({
            "indices": idx.tolist(),
            "values":  val.tolist(),
        })
    return results


def ensure_collection(client, name):
    if name in PROTECTED:
        raise ValueError(f"Refusing to touch protected collection '{name}'")
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        logger.info("Collection '%s' already exists, will append/overwrite.", name)
        return
    client.create_collection(
        collection_name=name,
        vectors_config={},
        sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)},
    )
    logger.info("Created collection '%s'.", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N rows (for smoke testing)")
    args = parser.parse_args()

    # Load the activations as a memory-mapped array — won't pull 22 GB into RAM
    acts   = np.load(ACTIVATIONS, mmap_mode="r")   # (189216, 30522) float32
    ids    = np.load(CHUNK_IDS)                     # (189216,) int64
    total  = len(ids) if args.limit is None else min(args.limit, len(ids))
    logger.info("Total rows to process: %d", total)

    client = QdrantClient(url=settings.QDRANT_URL)
    ensure_collection(client, args.collection)

    W_enc, b_enc, b_dec, dec_norms = load_sae_encoder()

    written = 0
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        batch_acts = acts[start:end]                # numpy slice from mmap
        batch_ids  = ids[start:end]

        sparse_vecs = encode_batch(batch_acts, W_enc, b_enc, b_dec, dec_norms)

        points = []
        for point_id, vec in zip(batch_ids, sparse_vecs):
            pid = int(point_id)
            points.append(PointStruct(
                id=pid,
                vector={"sparse": SparseVector(
                    indices=vec["indices"],
                    values=vec["values"],
                )},
                payload={"dense_chunk_ids": [pid]},
            ))

        client.upsert(collection_name=args.collection, points=points)
        written += len(points)
        logger.info("Upserted rows %d–%d (%d total written)", start, end - 1, written)

    logger.info("Done. %d points in '%s'.", written, args.collection)


if __name__ == "__main__":
    main()