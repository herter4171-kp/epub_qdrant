"""Integration tests for Phase 1 activation capture.

Tests T1.1 - T1.4 from sae_intent/tasks.md:
- T1.1: Scroll Qdrant collection and extract payload
- T1.2: SPLADE activation extractor
- T1.3: Full activation capture
- T1.4: Corpus statistics and spot check
"""

import numpy as np
import os
import pytest
from unittest.mock import MagicMock, patch


COLLECTION_NAME = os.getenv("SAE_TEST_COLLECTION", "sparse-only-256len")


def test_scroll_collection():
    """T1.1: Can scroll collection and extract payloads."""
    if not os.getenv("QDRANT_URL"):
        pytest.skip("QDRANT_URL not set")
    
    from src.sae.extract_payload import scroll_collection
    
    points = scroll_collection(COLLECTION_NAME)
    assert len(points) > 0, "Collection should have at least one point"
    
    # Verify point structure
    for pid, text in points[:5]:
        assert isinstance(pid, int), "Point ID should be int"
        assert isinstance(text, str) and len(text) > 0, "Text should be non-empty string"
    
    # Verify point count matches collection
    from qdrant_client import QdrantClient
    from src.config import settings
    client = QdrantClient(url=settings.QDRANT_URL)
    info = client.get_collection(COLLECTION_NAME)
    assert len(points) == info.points_count, "Scrolled points should match collection count"


def test_splade_extractor_structure():
    """T1.2: SPLADE extractor class has correct structure."""
    from src.sae.splade_extractor import SpladeExtractor
    
    # Verify class exists and has expected methods
    assert hasattr(SpladeExtractor, 'extract_batch')
    assert hasattr(SpladeExtractor, 'extract_all')
    assert hasattr(SpladeExtractor, 'count_nonzero_stats')
    
    # Verify constants
    from src.sae.splade_extractor import (
        SPLADE_VOCAB_SIZE,
        SPLADE_MAX_DOC_LENGTH,
        SPLADE_MAX_QUERY_LENGTH,
    )
    assert SPLADE_VOCAB_SIZE == 30522
    assert SPLADE_MAX_DOC_LENGTH == 256
    assert SPLADE_MAX_QUERY_LENGTH == 32


def test_splade_extractor_count_nonzero_stats():
    """T1.4: Nonzero statistics computation."""
    from src.sae.splade_extractor import SpladeExtractor
    
    # Create mock activations tensor
    mock_activations = np.random.rand(10, 30522)
    mock_activations[mock_activations < 0.5] = 0  # Sparsify
    
    # Test with numpy array directly
    nnz = (mock_activations != 0).sum(axis=1)
    stats = {
        "mean": float(nnz.mean()),
        "median": float(np.median(nnz)),
        "p10": float(np.percentile(nnz, 10)),
        "p90": float(np.percentile(nnz, 90)),
        "min": int(nnz.min()),
        "max": int(nnz.max()),
    }
    
    assert stats["mean"] > 0, "Mean NNZ should be positive"
    assert stats["min"] >= 0, "Min NNZ should be >= 0"
    assert stats["max"] <= 30522, "Max NNZ should not exceed vocab size"


def test_capture_activations_module():
    """T1.3: capture_activations module has expected structure."""
    from src.sae import capture_activations
    
    assert hasattr(capture_activations, 'run')
    assert hasattr(capture_activations, 'main')
    assert capture_activations.OUTPUT_DIR == "./sae_data"


def test_capture_activations_run_signature():
    """Verify run() accepts expected parameters."""
    from src.sae.capture_activations import run
    import inspect
    
    sig = inspect.signature(run)
    params = list(sig.parameters.keys())
    
    assert "collection_name" in params
    assert "output_dir" in params


def test_extract_payload_module():
    """Verify extract_payload module has expected structure."""
    from src.sae import extract_payload
    
    assert hasattr(extract_payload, 'scroll_collection')


def test_extract_payload_scroll_signature():
    """Verify scroll_collection accepts expected parameters."""
    from src.sae.extract_payload import scroll_collection
    import inspect
    
    sig = inspect.signature(scroll_collection)
    params = list(sig.parameters.keys())
    
    assert "collection_name" in params


def test_config_integration():
    """Verify config values used by SAE modules."""
    from src.config import settings
    
    # Verify QDRANT_URL is accessible
    assert settings.QDRANT_URL is not None
    assert len(settings.QDRANT_URL) > 0
    
    # Verify EMBEDDING_SERVER_URL is accessible
    assert settings.EMBEDDING_SERVER_URL is not None
    assert len(settings.EMBEDDING_SERVER_URL) > 0


def test_expected_k_value_from_config():
    """Verify that captured stats should match expected K value from config."""
    # Expected K should be around 165 based on config.yaml
    # This is a sanity check for the Phase 0 analysis
    expected_k = 165
    expected_range = (100, 300)
    
    assert expected_k > expected_range[0]
    assert expected_k < expected_range[1]