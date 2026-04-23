"""Dense and sparse embedder wrappers for the unified embedding server."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

DENSE_MODEL = "/tank/huggingface/embeddinggemma-300m"
SPARSE_MODEL = "Qdrant/minicoil-v1"


class DenseEmbedder:
    """Wraps Snowflake Arctic Embed M v2.0 via sentence-transformers on GPU."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading dense model: %s", DENSE_MODEL)
        self.model = SentenceTransformer(DENSE_MODEL, device="cuda")
        logger.info("Dense model loaded.")

    def encode(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Encode texts into 768-d dense vectors."""
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()


class SparseEmbedder:
    """Wraps MiniCOIL via fastembed-gpu for sparse embeddings."""

    def __init__(self):
        from fastembed import SparseTextEmbedding

        logger.info("Loading sparse model: %s", SPARSE_MODEL)
        self.model = SparseTextEmbedding(
            model_name=SPARSE_MODEL,
            providers=["CUDAExecutionProvider"],
        )
        logger.info("Sparse model loaded.")

    def encode(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Encode texts into sparse vectors with indices and values."""
        if is_query:
            results = list(self.model.query_embed(texts))
        else:
            results = list(self.model.passage_embed(texts))
        return [
            {"indices": r.indices.tolist(), "values": r.values.tolist()}
            for r in results
        ]
