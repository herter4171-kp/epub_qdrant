"""Configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class Settings:
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://192.168.68.75:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "books")
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://192.168.68.75:11434")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m")

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Qdrant
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", "768"))
    DISTANCE: str = os.getenv("DISTANCE", "Cosine")


settings = Settings()