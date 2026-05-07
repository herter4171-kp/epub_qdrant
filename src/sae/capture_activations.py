"""Full activation capture pipeline for Phase 1.

This module orchestrates the end-to-end process of:
1. Scrolling the sparse collection
2. Extracting SPLADE activations
3. Saving to memmap format
4. Computing corpus statistics
"""

import logging
import os
from pathlib import Path
from typing import Tuple

import numpy as np

from src.sae.extract_payload import scroll_collection
from src.sae.splade_extractor import SpladeExtractor

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.getenv("SAE_OUTPUT_DIR", "./sae_data")


def run(collection_name: str, output_dir: str = OUTPUT_DIR) -> Tuple[np.ndarray, np.ndarray]:  # noqa: E501
    """Run the full activation capture pipeline.
    
    Args:
        collection_name: Name of the Qdrant collection to scroll.
        output_dir: Directory to save output files.
        
    Returns:
        Tuple of (activations_memmap, chunk_ids_array)
        
    Output files:
        - chunk_ids.npy: Array of Qdrant point IDs in scroll order
        - activations.npy: (N, 30522) float32 memmap of SPLADE activations
        - corpus_mean.npy: (30522,) float32 mean vector for SAE pre-bias
    """
    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Output directory: %s", output_dir)
    
    # 1. Scroll collection
    logger.info("Scrolling collection %s...", collection_name)
    point_id_text_pairs = scroll_collection(collection_name)
    
    if not point_id_text_pairs:
        raise ValueError(f"No points found in collection '{collection_name}'")
    
    logger.info("Scrolled %d points", len(point_id_text_pairs))
    
    # Extract point IDs in scroll order
    chunk_ids = np.array([p[0] for p in point_id_text_pairs], dtype=np.int64)
    chunk_ids_path = os.path.join(output_dir, "chunk_ids.npy")
    np.save(chunk_ids_path, chunk_ids)
    logger.info("Saved chunk_ids.npy (%d IDs)", len(chunk_ids))
    
    # Extract texts in same order
    texts = [p[1] for p in point_id_text_pairs]
    
    # 2. SPLADE extraction
    logger.info("Extracting SPLADE activations...")
    import torch
    extractor = SpladeExtractor(batch_size=128)
    activations = extractor.extract_all(texts)
    
    # Verify shape
    assert activations.shape[1] == 30522, \
        f"Expected 30522 dims, got {activations.shape[1]}"
    logger.info("Activation shape: %s", activations.shape)
    
    # 3. Save as memmap
    memmap_path = os.path.join(output_dir, "activations.npy")
    activations_np = activations.numpy()
    activations_memmap = np.memmap(
        memmap_path,
        dtype='float32',
        mode='w+',
        shape=activations_np.shape,
    )
    activations_memmap[:] = activations_np
    activations_memmap.flush()
    del activations_memmap  # Flush to disk
    logger.info("Saved activations.npy (%.2f MB)", 
                os.path.getsize(memmap_path) / 1e6)
    
    # 4. Compute corpus mean
    logger.info("Computing corpus mean...")
    corpus_mean = np.array(activations.mean(dim=0), dtype=np.float32)
    mean_path = os.path.join(output_dir, "corpus_mean.npy")
    np.save(mean_path, corpus_mean)
    
    # Verify mean vector
    logger.info("Corpus mean stats: min=%.4f max=%.4f mean=%.4f",
                corpus_mean.min(), corpus_mean.max(), corpus_mean.mean())
    
    # 5. Log NNZ distribution
    logger.info("Computing nonzero counts...")
    nnz_counts = (activations != 0).sum(dim=1).numpy()  # noqa: E501
    logger.info("NNZ: mean=%.2f median=%.2f p10=%.2f p90=%.2f min=%d max=%d",
                nnz_counts.mean(), np.median(nnz_counts),
                np.percentile(nnz_counts, 10), np.percentile(nnz_counts, 90),
                nnz_counts.min(), nnz_counts.max())
    
    return activations_np, chunk_ids


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Capture SPLADE activations from Qdrant collection."
    )
    parser.add_argument(
        "--collection",
        default="sparse-only-256len",
        help="Qdrant collection to scroll (default: sparse-only-256len)"
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    run(args.collection, args.output_dir)


if __name__ == "__main__":
    main()