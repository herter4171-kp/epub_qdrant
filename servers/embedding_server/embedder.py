"""Dense and sparse embedder wrappers for the unified embedding server."""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ── VRAM cap ─────────────────────────────────────────────────────────────────
_vram_fraction = float(os.getenv("EMBEDDING_VRAM_FRACTION", "0.90"))
torch.cuda.set_per_process_memory_fraction(_vram_fraction, device=0)
logger.info("VRAM cap set to %.0f%% of device 0", _vram_fraction * 100)

os.environ.setdefault("SBERT_DISABLE_ONNX", "1")

_executor = ThreadPoolExecutor(max_workers=1)

DENSE_MODEL = "/tank/huggingface/embeddinggemma-300m"

SPLADE_LOCAL_PATH = "/tank/huggingface/splade-cocondenser-ensembledistil"
SPLADE_VOCAB_SIZE = 30522
SPLADE_MAX_DOC_LENGTH = 256
SPLADE_MAX_QUERY_LENGTH = 32
SPLADE_THRESHOLD = 0.0

# Micro-batch size for GPU inference. Keep small — dense + SPLADE + IT all
# share the same VRAM budget. 2 is safest; raise to 4 only if you have headroom.
INTERNAL_BATCH_SIZE = int(os.getenv("SPLADE_INTERNAL_BATCH", "4"))

IT_MODEL_LOCAL_PATH = "/tank/huggingface/gemma-3-270m-it"
IT_MAX_NEW_TOKENS = 512
IT_TEMPERATURE = float(os.getenv("IT_TEMPERATURE", "0.05"))

# SAE checkpoint
SAE_CHECKPOINT = "/tank/sae-splade/sae_data_good_2x/901120/sae_weights.safetensors"
SAE_DEC_NORMS = "sae_data/dec_norms.pt"
SAE_D_SAE = 61044
SAE_K = 165

_PROMPT_FILE = Path(__file__).parent / "rewrite_prompt.txt"


# ── DenseEmbedder ─────────────────────────────────────────────────────────────

class DenseEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        logger.info("Loading dense model: %s", DENSE_MODEL)
        self.model = SentenceTransformer(DENSE_MODEL, device="cuda")
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("Dense model loaded. VRAM allocated: %.2f GB", allocated)

    def encode(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True,
        )
        return embeddings.tolist()


# ── SparseEmbedder (SPLADE) ───────────────────────────────────────────────────

class SparseEmbedder:
    """SPLADE sparse embedder — naver/splade-cocondenser-ensemble-distil.

    INTERNAL_BATCH_SIZE controls how many texts go through the GPU at once.
    The server endpoint already receives sliced batches from the client, so
    this is a second level of slicing purely for VRAM safety.
    """

    def __init__(self, internal_batch_size: int = INTERNAL_BATCH_SIZE):
        self.internal_batch_size = internal_batch_size
        self._load_model()
        self.sae = SAEEncoder()

    def _load_model(self):
        from transformers import AutoModelForMaskedLM
        logger.info("Loading SPLADE model from %s", SPLADE_LOCAL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(
            SPLADE_LOCAL_PATH, local_files_only=True,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            SPLADE_LOCAL_PATH,
            local_files_only=True,
            torch_dtype=torch.float32,
            device_map="cuda",
        ).eval()

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            "SPLADE loaded | vocab=%d | VRAM alloc=%.2fGB res=%.2fGB",
            SPLADE_VOCAB_SIZE, allocated, reserved,
        )
        assert next(self.model.parameters()).device.type == "cuda", \
            "FATAL: SPLADE model is on CPU."

    def _encode_batch(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Run one micro-batch synchronously on GPU."""
        max_length = SPLADE_MAX_QUERY_LENGTH if is_query else SPLADE_MAX_DOC_LENGTH

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to("cuda")

        with torch.no_grad():
            output = self.model(**inputs)
            vecs = torch.log1p(torch.relu(output.logits))
            mask = inputs["attention_mask"].unsqueeze(-1).to(vecs.dtype)
            sparse_vecs = (vecs * mask).max(dim=1).values
            # sparse_vecs: [batch, vocab_size]

        results = []
        for vec in sparse_vecs:
            nonzero_mask = vec > SPLADE_THRESHOLD
            # flatten() ensures 1-D even if vec is somehow scalar-degenerate
            indices = nonzero_mask.nonzero(as_tuple=False).flatten()
            results.append({
                "indices": indices.cpu().tolist(),
                "values": vec[indices].cpu().float().tolist(),
            })

        # Free intermediate tensors immediately — three models share VRAM
        del inputs, output, vecs, mask, sparse_vecs
        torch.cuda.empty_cache()

        return results

    def encode(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Encode texts into SPLADE sparse vectors.

        is_query=True  → 32-token truncation  (rewritten queries)
        is_query=False → 256-token truncation (documents)
        """
        all_results = []
        for i in range(0, len(texts), self.internal_batch_size):
            batch = texts[i : i + self.internal_batch_size]
            all_results.extend(self._encode_batch(batch, is_query=is_query))

        nnz_counts = [len(r["indices"]) for r in all_results]
        if nnz_counts:
            logger.info(
                "SPLADE encode: %d texts | NNZ min=%d max=%d mean=%.1f | is_query=%s",
                len(texts), min(nnz_counts), max(nnz_counts),
                sum(nnz_counts) / len(nnz_counts), is_query,
            )
        return all_results

    async def encode_async(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.encode(texts, is_query),
        )

    def encode_sae(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Encode texts through SPLADE → SAE. Returns 61044-dim sparse vectors."""
        all_results = []
        for i in range(0, len(texts), self.internal_batch_size):
            batch = texts[i : i + self.internal_batch_size]
            all_results.extend(self._encode_sae_batch(batch, is_query))
        return all_results

    def _encode_sae_batch(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """One micro-batch: SPLADE pool → SAE encoder → topk sparse."""
        max_length = SPLADE_MAX_QUERY_LENGTH if is_query else SPLADE_MAX_DOC_LENGTH

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to("cuda")

        with torch.no_grad():
            output = self.model(**inputs)
            vecs = torch.log1p(torch.relu(output.logits))
            mask = inputs["attention_mask"].unsqueeze(-1).to(vecs.dtype)
            sparse_vecs = (vecs * mask).max(dim=1).values   # [batch, 30522]

        top_indices, values = self.sae.encode(sparse_vecs)  # [batch, 165] each

        results = []
        for idx, val in zip(top_indices, values):
            results.append({
                "indices": idx.cpu().tolist(),
                "values":  val.cpu().tolist(),
            })

        del inputs, output, vecs, mask, sparse_vecs, top_indices, values
        torch.cuda.empty_cache()

        return results

    async def encode_sae_async(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.encode_sae(texts, is_query),
        )


# ── SAEEncoder ────────────────────────────────────────────────────────────────

class SAEEncoder:
    """Encoder-only wrapper for the trained TopK SAE.

    Loads W_enc, b_enc, b_dec, and dec_norms (W_dec row norms).
    W_dec itself is NOT loaded — saves ~7.4 GB VRAM.

    Input:  SPLADE activations tensor, shape [batch, 30522], on CUDA, float32
    Output: (indices, values) each shape [batch, SAE_K], on CUDA
    """

    def __init__(self):
        from safetensors import safe_open
        logger.info("Loading SAE encoder from %s", SAE_CHECKPOINT)
        with safe_open(SAE_CHECKPOINT, framework="pt") as f:
            # Cast to fp16 to cut VRAM from 7.4 GB → 3.7 GB
            self.W_enc = f.get_tensor("W_enc").to(torch.float16).cuda()  # (30522, 61044)
            self.b_enc = f.get_tensor("b_enc").to(torch.float16).cuda()  # (61044,)
            self.b_dec = f.get_tensor("b_dec").to(torch.float16).cuda()  # (30522,)

        self.dec_norms = torch.load(SAE_DEC_NORMS, weights_only=True) \
                              .to(torch.float16).cuda()                  # (61044,)

        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("SAE encoder loaded. VRAM allocated: %.2f GB", allocated)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> tuple:
        """
        x: SPLADE activations [batch, 30522] float32 on CUDA
        returns: (top_indices [batch, 165], values [batch, 165]) — both on CUDA
        """
        x16 = x.to(torch.float16)
        x_centered  = x16 - self.b_dec                          # [batch, 30522]
        hidden_pre  = x_centered @ self.W_enc + self.b_enc      # [batch, 61044]
        hidden_pre  = hidden_pre * self.dec_norms                # [batch, 61044]
        top_values, top_indices = hidden_pre.topk(SAE_K, dim=-1) # [batch, 165] each
        values = top_values.relu().to(torch.float32)
        return top_indices, values


# ── QueryRewriter ─────────────────────────────────────────────────────────────

class QueryRewriter:
    """Rewrites user prompts using gemma-3-270m-it for better retrieval."""

    def __init__(self):
        self._load_model()
        self._load_prompt()

    def _load_model(self):
        logger.info("Loading IT model from %s", IT_MODEL_LOCAL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(IT_MODEL_LOCAL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            IT_MODEL_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("IT model loaded. VRAM allocated: %.2f GB", allocated)
        assert next(self.model.parameters()).device.type == "cuda", \
            "FATAL: IT model is on CPU."

    def _load_prompt(self):
        if _PROMPT_FILE.exists():
            self.system_prompt = _PROMPT_FILE.read_text(encoding="utf-8").strip()
        else:
            self.system_prompt = (
                "You are a technical query reformulation engine. Transform the user's "
                "natural-language input into a precise, effective search query optimized "
                "for retrieving AI/ML research papers. Return ONLY the reformulated query "
                "text. No explanation, no preamble, no quotes."
            )
        logger.info("System prompt loaded (%d characters)", len(self.system_prompt))

    def rewrite(self, query: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to("cuda")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=IT_MAX_NEW_TOKENS,
                do_sample=IT_TEMPERATURE > 0.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if IT_TEMPERATURE > 0.1:
                gen_kwargs["temperature"] = IT_TEMPERATURE
            outputs = self.model.generate(**gen_kwargs)

        generated = outputs[0][input_ids.shape[1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        if "\n\n" in result:
            result = result.split("\n\n")[0].strip()
        if "\n" in result:
            result = result.split("\n")[0].strip()

        logger.info("Rewrote query (%d → %d chars)", len(query), len(result))
        return result

    async def rewrite_async(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.rewrite(query),
        )