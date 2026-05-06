from dataclasses import dataclass
from typing import Literal


@dataclass
class CollectionConfig:
    """Parameters describing one retrieval collection.

    Attributes:
        name: Qdrant collection name. Always specified explicitly.
            There is no default. Different ingestion runs may use
            different names to preserve prior indexes during rebuilds.
        chunk_size: Maximum real tokens per chunk, measured by the project
            tokenizer (load_tokenizer()). For dense this is the section
            size ceiling. For sparse this is the SPLADE document window.
        embedding_model: Model identifier passed to the embedding client.
        vector_type: Whether this collection holds dense float vectors or
            sparse SPLADE vectors.
        min_chunk_tokens: Chunks below this threshold are merged into
            adjacent chunks by the runt merger. Applies to both types.
    """

    name: str
    chunk_size: int
    embedding_model: str
    vector_type: Literal["dense", "sparse"]
    min_chunk_tokens: int = 50
