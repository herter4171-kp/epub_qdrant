"""Top-level retrieval configuration pairing dense and sparse collections."""

from dataclasses import dataclass

from src.retrieval.collection_config import CollectionConfig


@dataclass
class RetrievalConfig:
    """Pairs a dense and a sparse collection for hybrid retrieval.

    Neither collection has any structural dependency on the other.
    They share source documents but differ in chunk granularity,
    embedding model, and index type.

    The dense collection name and the sparse collection name must both
    be specified explicitly at ingestion and query time. There is no
    inferred pairing. This allows an existing dense collection to be
    paired with a newly built sparse collection without rebuilding dense.

    Attributes:
        dense: Configuration for the dense embedding collection.
            Stores full MinerU sections as retrieval units.
        sparse: Configuration for the SPLADE sparse collection.
            Stores 256-token chunks that point to dense chunk IDs.
    """

    dense: CollectionConfig
    sparse: CollectionConfig
