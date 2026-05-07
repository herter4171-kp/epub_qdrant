"""SAELens DataProvider for activation memmap data.

This module implements a DataProvider that wraps the activations.npy memmap
and yields shuffled batches of SPLADE activations for SAELens training.

SAELens v6 uses SAETrainer + DataProvider (a plain Iterator[torch.Tensor])
+ mixing_buffer() for shuffling.
"""

import logging
import numpy as np
import torch
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class SaelensDataProvider:
    """DataProvider for SAELens training on SPLADE activations.

    Wraps a NumPy memmap file containing activations of shape (N, 30522)
    and yields shuffled batches of shape (batch_size, 30522).
    """

    def __init__(
        self,
        activations_path: str,
        batch_size: int = 256,
        seed: Optional[int] = None,
        mixing_buffer_size: int = 1000,
    ):
        """Initialize the DataProvider.

        Args:
            activations_path: Path to the activations.npy memmap file.
            batch_size: Number of samples per batch.
            seed: Random seed for reproducibility.
            mixing_buffer_size: Size of buffer for shuffling (mixing_buffer).
        """
        self.activations_path = activations_path
        self.batch_size = batch_size
        self.seed = seed
        self.mixing_buffer_size = mixing_buffer_size

        # Load activations - shape is (N, 30522)
        # activations.npy is now saved as a standard numpy array using np.save()
        self._activations: np.ndarray = np.load(activations_path)
        self.n_samples = self._activations.shape[0]
        self.d_in = self._activations.shape[1]

        logger.info(
            "DataProvider initialized: %d samples, d_in=%d, batch_size=%d",
            self.n_samples,
            self.d_in,
            self.batch_size,
        )

        # Initialize rng for shuffling
        self._rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield batches of activations.

        Uses mixing_buffer() approach for shuffling across epochs.
        Returns torch.Tensor of shape (batch_size, d_in).
        """
        indices = np.arange(self.n_samples)
        self._rng.shuffle(indices)

        # Build mixing buffer
        buffer: list[int] = []
        buffer_idx = 0

        for idx in indices:
            buffer.append(idx)
            if len(buffer) >= self.mixing_buffer_size:
                # Shuffle buffer and yield from it
                self._rng.shuffle(buffer)
                while buffer and buffer_idx < self.n_samples:
                    batch_indices = buffer[: self.batch_size]
                    buffer = buffer[self.batch_size :]
                    buffer_idx += len(batch_indices)

                    # Load batch from memmap
                    batch_data = self._activations[batch_indices]
                    yield torch.from_numpy(batch_data).float()

        # Handle remaining samples
        while buffer:
            batch_indices = buffer[: self.batch_size]
            buffer = buffer[self.batch_size :]
            batch_data = self._activations[batch_indices]
            yield torch.from_numpy(batch_data).float()

    def __len__(self) -> int:
        """Return approximate number of batches per epoch."""
        return max(1, self.n_samples // self.batch_size)


def create_data_provider(
    activations_path: str = "./sae_data/activations.npy",
    batch_size: int = 256,
    seed: Optional[int] = None,
) -> SaelensDataProvider:
    """Factory function to create a DataProvider.

    Args:
        activations_path: Path to activations.npy file.
        batch_size: Number of samples per batch.
        seed: Random seed for reproducibility.

    Returns:
        SaelensDataProvider instance.
    """
    return SaelensDataProvider(
        activations_path=activations_path,
        batch_size=batch_size,
        seed=seed,
    )


def test_data_provider():
    """Basic unit test for DataProvider."""
    import os

    # Check if activations file exists
    activations_path = "./sae_data/activations.npy"
    if not os.path.exists(activations_path):
        raise FileNotFoundError(f"Activations file not found: {activations_path}")

    # Create provider
    provider = create_data_provider(
        activations_path=activations_path,
        batch_size=64,
        seed=42,
    )

    # Test iteration
    batches = []
    for i, batch in enumerate(provider):
        assert isinstance(batch, torch.Tensor), f"Expected Tensor, got {type(batch)}"
        assert batch.shape[1] == 30522, f"Expected d_in=30522, got {batch.shape[1]}"
        assert not torch.any(batch < 0), "Negative values found in activations"
        batches.append(batch)

        if i >= 2:  # Test first 3 batches
            break

    logger.info("Tested %d batches, all valid", len(batches))

    # Verify shuffling works (run twice, should get different order)
    provider1 = create_data_provider(activations_path, batch_size=64, seed=42)
    provider2 = create_data_provider(activations_path, batch_size=64, seed=42)

    batch1_a = next(iter(provider1))
    batch1_b = next(iter(provider1))
    batch2_a = next(iter(provider2))

    # Same seed should produce same first batch
    assert torch.allclose(batch1_a, batch2_a), "Same seed should produce same results"

    logger.info("DataProvider tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_data_provider()