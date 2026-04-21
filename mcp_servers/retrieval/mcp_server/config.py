"""Configuration for the retrieval MCP server."""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load parent project .env if available
_parent_env = Path(__file__).parent.parent.parent.parent / ".env"
if _parent_env.exists():
    load_dotenv(_parent_env)


class Settings:
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")

    # LiteLLM
    LITELLM_API_URL: str = os.getenv("LITELLM_API_URL", "https://litellm.twr.church/v1")
    LITELLM_API_KEY: str = os.getenv("LITELLM_API_KEY", "")
    LITELLM_MODEL: str = os.getenv("LITELLM_MODEL", "meta-llama/llama-3.1-70b-instruct")

    # MCP server
    MCP_PORT: int = int(os.getenv("MCP_PORT", "8090"))
    MCP_HOST: str = os.getenv("MCP_HOST", "0.0.0.0")

    # Retrieval defaults
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    RETRIEVAL_CONTEXT_RADIUS: int = int(os.getenv("RETRIEVAL_CONTEXT_RADIUS", "2"))
    RETRIEVAL_GROUP_BY: str = os.getenv("RETRIEVAL_GROUP_BY", "chapter")

    # Qdrant
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "768"))
    DISTANCE: str = os.getenv("DISTANCE", "Cosine")


settings = Settings()