"""Configuration for the retrieval MCP server."""

import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

# Load parent project .env if available
_parent_env = Path(__file__).parent.parent.parent.parent / ".env"
if _parent_env.exists():
    load_dotenv(_parent_env)


class Settings:
    # ── Qdrant ──────────────────────────────────────────────────────
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    # Comma-separated list of collection names (e.g. "epub_kb,papers")
    QDRANT_COLLECTIONS: str = os.getenv("QDRANT_COLLECTIONS", "")

    # ── Embedding ───────────────────────────────────────────────────
    EMBEDDING_SERVER_URL: str = os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8100")

    # ── LiteLLM ─────────────────────────────────────────────────────
    LITELLM_API_URL: str = os.getenv("LITELLM_API_URL", "https://litellm.twr.church/v1")
    LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "")
    LITELLM_MODEL: str = os.getenv("LITELLM_MODEL", "qwen36")

    # ── MCP server ──────────────────────────────────────────────────
    MCP_PORT: int = int(os.getenv("MCP_PORT", "8090"))
    MCP_HOST: str = os.getenv("MCP_HOST", "0.0.0.0")

    # ── Retrieval defaults ──────────────────────────────────────────
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "15"))
    RETRIEVAL_CONTEXT_RADIUS: int = int(os.getenv("RETRIEVAL_CONTEXT_RADIUS", "2"))
    RETRIEVAL_GROUP_BY: str = os.getenv("RETRIEVAL_GROUP_BY", "section")

    # ── Qdrant ──────────────────────────────────────────────────────
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "768"))
    DISTANCE: str = os.getenv("DISTANCE", "Cosine")

    # ── Multi-collection ────────────────────────────────────────────
    # Default collection when none specified (first in the list).
    @property
    def DEFAULT_COLLECTION(self) -> str:
        collections = self.collections
        return collections[0] if collections else ""

    # ── Metadata filter defaults ────────────────────────────────────
    # Default doc_type filter when searching a single collection
    DEFAULT_DOC_TYPE: str = os.getenv("DEFAULT_DOC_TYPE", "")

    # ── Helpers ─────────────────────────────────────────────────────
    @property
    def collections(self) -> List[str]:
        """Parse the comma-separated QDRANT_COLLECTIONS into a list, stripping blanks."""
        if not self.QDRANT_COLLECTIONS:
            return []
        return [c.strip() for c in self.QDRANT_COLLECTIONS.split(",") if c.strip()]

    @property
    def has_collections(self) -> bool:
        return len(self.collections) > 0


settings = Settings()
