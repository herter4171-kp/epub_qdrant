"""Generate embeddings via Ollama API."""

import logging
from typing import List

import requests

from src.chunker import Chunk

logger = logging.getLogger(__name__)


class Embedder:
    """Calls Ollama to generate embedding vectors."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def embed_single(self, text: str) -> List[float]:
        """Get embedding for a single text string.

        Args:
            text: Input text to embed.

        Returns:
            List of floats (the embedding vector).

        Raises:
            requests.exceptions.RequestException: If the Ollama request fails.
        """
        url = f"{self.base_url}/api/embed"
        payload = {
            "model": self.model,
            "input": text,
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()

        # Ollama /api/embed returns {"embeddings": [[...]]}
        embeddings = data.get("embeddings", [])
        if not embeddings:
            raise RuntimeError(f"No embedding returned for model {self.model}")
        return embeddings[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts.

        Sends each text individually to Ollama (some models only support single input).
        Logs failures but continues processing.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        results: List[List[float]] = []
        for i, text in enumerate(texts):
            try:
                vec = self.embed_single(text)
                results.append(vec)
            except Exception as e:
                logger.error(f"Embedding failed for text {i+1}/{len(texts)}: {e}")
                # Skip this text - caller should handle missing vectors
                results.append([])

        return results

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Embed all chunks and attach vectors.

        Processes chunks in batches. Returns chunks with vector field set.

        Args:
            chunks: List of Chunk objects (without vectors).

        Returns:
            Same chunks but with vector field populated.
        """
        texts = [c.text for c in chunks]
        vectors = self.embed_batch(texts)

        for chunk, vec in zip(chunks, vectors):
            if vec:
                chunk.vector = vec
            else:
                logger.warning(f"Skipping chunk {chunk.id} - no vector generated")

        return chunks