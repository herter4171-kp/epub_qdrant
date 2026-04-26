"""Configuration loaded from environment variables."""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env file if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class Settings:
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://192.168.68.75:6333")

    # Comma-separated list of collection names (e.g. "epub_kb,papers")
    QDRANT_COLLECTIONS: str = os.getenv("QDRANT_COLLECTIONS", "")

    # Legacy single-collection settings (backwards compat)
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "books")
    QDRANT_PAPERS_COLLECTION: str = os.getenv("QDRANT_PAPERS_COLLECTION", "papers")

    # Default collection: first from comma-separated list, then legacy fallback
    @property
    def DEFAULT_COLLECTION(self) -> str:
        collections = self.collections
        if collections:
            return collections[0]
        # Fall back to QDRANT_COLLECTION
        return self.QDRANT_COLLECTION or ""

    @property
    def collections(self) -> List[str]:
        """Parse the comma-separated QDRANT_COLLECTIONS into a list."""
        if not self.QDRANT_COLLECTIONS:
            return []
        return [c.strip() for c in self.QDRANT_COLLECTIONS.split(",") if c.strip()]

    @property
    def has_collections(self) -> bool:
        return len(self.collections) > 0

    # Unified embedding server
    EMBEDDING_SERVER_URL: str = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8100")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "2048"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Semantic chunking
    TOKENIZER_JSON: str = os.getenv("TOKENIZER_JSON", "")
    CHUNK_OVERLAP_RATIO: float = float(os.getenv("CHUNK_OVERLAP_RATIO", "0.0"))
    SIMILARITY_PERCENTILE: float = float(os.getenv("SIMILARITY_PERCENTILE", "95.0"))
    MIN_DISTANCE_FLOOR: float = float(os.getenv("MIN_DISTANCE_FLOOR", "0.1"))
    MIN_SENTENCES_FOR_SEMANTIC: int = int(os.getenv("MIN_SENTENCES_FOR_SEMANTIC", "10"))
    MIN_CHUNK_TOKENS: int = int(os.getenv("MIN_CHUNK_TOKENS", "50"))
    SEMANTIC_CHUNKING_ENABLED: bool = os.getenv("SEMANTIC_CHUNKING_ENABLED", "true").lower() in ("true", "1", "yes")

    # Qdrant
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "768"))
    DISTANCE: str = os.getenv("DISTANCE", "Cosine")


settings = Settings()
