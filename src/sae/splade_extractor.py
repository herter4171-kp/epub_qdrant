"""SPLADE activation extractor for Phase 1 capture."""

import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

SPLADE_LOCAL_PATH = "/tank/huggingface/splade-cocondenser-ensembledistil"
SPLADE_VOCAB_SIZE = 30522
SPLADE_MAX_DOC_LENGTH = 256
SPLADE_MAX_QUERY_LENGTH = 32


class SpladeExtractor:
    """SPLADE activation extractor for Phase 1.
    
    Extracts pre-threshold activations from SPLADE model output.
    The pooling operation is: log1p(relu(logits)).max(dim=sequence)
    
    Returns float32 tensors of shape (batch_size, 30522).
    """

    def __init__(self, batch_size: int = 128):
        self.batch_size = batch_size
        self._load_model()

    def _load_model(self):
        """Load SPLADE model and tokenizer from local path."""
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        logger.info("Loading SPLADE from %s", SPLADE_LOCAL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(
            SPLADE_LOCAL_PATH, local_files_only=True,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            SPLADE_LOCAL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cuda",
        ).eval()
        logger.info("SPLADE loaded on GPU")

    def extract_batch(self, texts: List[str]) -> "torch.Tensor":
        """Extract pre-threshold SPLADE activations from a batch.
        
        Args:
            texts: List of document strings to embed.
            
        Returns:
            Tensor of shape (batch_size, 30522) with float32 dtype.
            All values are >= 0 (ReLU applied).
        """
        import torch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SPLADE_MAX_DOC_LENGTH,
        ).to("cuda")

        with torch.no_grad():
            output = self.model(**inputs)
            # output.logits: [batch, seq_len, vocab_size]

            # SPLADE pooling: log1p(relu(logits)).max(dim=seq_len)
            vecs = torch.log1p(torch.relu(output.logits))
            mask = inputs["attention_mask"].unsqueeze(-1).to(vecs.dtype)
            sparse_vecs = (vecs * mask).max(dim=1).values
            # sparse_vecs: [batch, vocab_size]

        # Free GPU memory
        del inputs, output, vecs, mask, sparse_vecs
        torch.cuda.empty_cache()

        return sparse_vecs.cpu().float()

    def extract_all(self, texts: List[str]) -> "torch.Tensor":
        """Process all texts in batches, return concatenated results.
        
        Args:
            texts: List of document strings to embed.
            
        Returns:
            Tensor of shape (N, 30522) with float32 dtype.
        """
        import torch
        all_batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_vecs = self.extract_batch(batch)
            all_batches.append(batch_vecs)

        return torch.cat(all_batches, dim=0)

    def count_nonzero_stats(self, activations) -> dict:
        """Compute per-document nonzero count statistics.
        
        Args:
            activations: Tensor or array of shape (N, 30522).
            
        Returns:
            Dict with 'mean', 'median', 'p10', 'p90', 'min', 'max'.
        """
        nnz = (activations != 0).sum(dim=1).numpy()
        return {
            "mean": float(nnz.mean()),
            "median": float(np.median(nnz)),
            "p10": float(np.percentile(nnz, 10)),
            "p90": float(np.percentile(nnz, 90)),
            "min": int(nnz.min()),
            "max": int(nnz.max()),
        }