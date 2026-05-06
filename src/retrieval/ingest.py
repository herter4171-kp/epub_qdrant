"""Ingestion orchestrator. Runs dense, sparse, or both."""

import argparse
import logging
from pathlib import Path
from typing import List

from src.ingestion.loader import DocumentLoader
from src.ingestion.loader import DocumentChunk
from src.retrieval.collection_config import CollectionConfig
from src.retrieval.retrieval_config import RetrievalConfig
from src.retrieval import ingest_dense, ingest_sparse


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ingestion orchestrator for hybrid retrieval.")
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse", "both"],
        required=True,
        help="Ingestion mode: dense, sparse, or both.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing documents to ingest.",
    )
    parser.add_argument(
        "--dense-collection",
        required=True,
        help="Name of the dense collection.",
    )
    parser.add_argument(
        "--sparse-collection",
        default=None,
        help="Name of the sparse collection.",
    )
    # In a real implementation, these would be loaded from settings/config
    parser.add_argument("--dense-model", default="embeddinggemma:300m")
    parser.add_argument("--sparse-model", default="naver/splade-v3")
    parser.add_argument("--dense-chunk-size", type=int, default=2048)
    parser.add_argument("--sparse-chunk-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks to process.")

    args = parser.parse_args()

    # 1. Setup Configs
    dense_config = CollectionConfig(
        name=args.dense_collection,
        chunk_size=args.dense_chunk_size,
        embedding_model=args.dense_model,
        vector_type="dense",
        min_chunk_tokens=50,
    )

    if args.mode in ("sparse", "both") and not args.sparse_collection:
        parser.error("--sparse-collection is required for sparse and both modes")

    sparse_config = None
    if args.sparse_collection:
        sparse_config = CollectionConfig(
            name=args.sparse_collection,
            chunk_size=args.sparse_chunk_size,
            embedding_model=args.sparse_model,
            vector_type="sparse",
            min_chunk_tokens=50,
        )

    # 2. Load Documents
    input_path = Path(args.input_dir)
    if not input_path.is_dir():
        logger.error("Input directory does not exist: %s", args.input_dir)
        return

    all_chunks: List[DocumentChunk] = []
    for file_path in input_path.rglob("*"):
        if file_path.is_file():
            try:
                loader = DocumentLoader.for_path(file_path)
                chunks = loader.load(file_path)
                all_chunks.extend(chunks)
                logger.info("Loaded %d chunks from %s", len(chunks), file_path.name)
            except ValueError as e:
                # Ignore files that don't have a recognized loader (e.g. non-doc files)
                continue
            except Exception as e:
                logger.error("Failed to load %s: %s", file_path.name, e)

    if not all_chunks:
        logger.warning("No chunks found in %s. Nothing to ingest.", args.input_dir)
        return

    # 3. Execute Ingestion
    if args.limit:
        all_chunks = all_chunks[:args.limit]
        logger.info("Limiting ingestion to first %d chunks.", args.limit)

    if args.mode == "dense":
        ingest_dense.run(dense_config, all_chunks)

    elif args.mode == "sparse":
        if not sparse_config:
            raise ValueError("Sparse collection name must be provided for sparse mode.")
        ingest_sparse.run(sparse_config, args.dense_collection, all_chunks, limit=args.limit)

    elif args.mode == "both":
        if not sparse_config:
            raise ValueError("Sparse collection name must be provided for both mode.")
        ingest_dense.run(dense_config, all_chunks, limit=args.limit)
        ingest_sparse.run(sparse_config, args.dense_collection, all_chunks, limit=args.limit)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
