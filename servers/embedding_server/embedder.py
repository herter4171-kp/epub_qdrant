"""Dense and sparse embedder wrappers for the unified embedding server."""

import asyncio
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from safetensors.torch import safe_open
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ── VRAM cap ─────────────────────────────────────────────────────────────────
# Set EMBEDDING_VRAM_FRACTION env var to override (0.0–1.0).
# Default 0.90 leaves headroom for ONNX/CUDA kernels on top of model weights.
# TODO: Revamp when we go multi-gpu (device=0 hardcoded).
_vram_fraction = float(os.getenv("EMBEDDING_VRAM_FRACTION", "0.90"))
torch.cuda.set_per_process_memory_fraction(_vram_fraction, device=0)
logger.info("VRAM cap set to %.0f%% of device 0", _vram_fraction * 100)

# Force sentence-transformers off the ONNX path — ONNX manages its own BFC
# arena separately from PyTorch and fights for the same VRAM budget.
os.environ.setdefault("SBERT_DISABLE_ONNX", "1")

# Thread pool for non-blocking GPU inference (one worker — GPU is single-threaded)
_executor = ThreadPoolExecutor(max_workers=1)

DENSE_MODEL = "/tank/huggingface/embeddinggemma-300m"

# SAE-SPLADE paths
SAE_LOCAL_PATH = "/tank/huggingface/gemma-scope-2-270m-pt"
BACKBONE_LOCAL_PATH = "/tank/huggingface/gemma-3-270m"
SAE_ID = "layer_12_width_65k_l0_medium"
SAE_HOOK_LAYER = 12          # resid_post after layer 12
SPLADE_THRESHOLD = 0.01      # prune near-zero activations before returning
INTERNAL_BATCH_SIZE = 32     # micro-batch size for GPU inference

# Query Rewriter (IT model) paths
IT_MODEL_LOCAL_PATH = "/tank/huggingface/gemma-3-270m-it"
IT_MAX_NEW_TOKENS = 512
IT_TEMPERATURE = float(os.getenv("IT_TEMPERATURE", "0.05"))

# Path to the system prompt file (same directory as this module)
_PROMPT_FILE = Path(__file__).parent / "rewrite_prompt.txt"


# ── Minimal JumpReLU SAE ──────────────────────────────────────────────────────
# Gemma-Scope-2 checkpoints use lowercase key naming (w_enc, w_dec) which
# sae-lens's JumpReLUSAE does not expect.  We implement a tiny loader here
# that handles the weight keys directly, avoiding the entire cfg ceremony.

class JumpReLUSAE(nn.Module):
    """Minimal JumpReLU sparse autoencoder — no sae-lens dependency needed."""

    def __init__(self, d_in: int, d_sae: int, threshold: float = 1.0):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        # threshold is a scalar in standard JumpReLU, but Gemma-Scope 2 checkpoints
        # store a per-feature threshold of shape [d_sae]. We init as scalar here and
        # let the loader override it from the checkpoint.
        self.threshold = torch.nn.Parameter(
            torch.tensor(threshold, dtype=torch.float32),
            requires_grad=False,
        )
        self.w_enc = nn.Parameter(torch.empty(d_sae, d_in, dtype=torch.float32))
        self.b_enc = nn.Parameter(torch.empty(d_sae, dtype=torch.float32))
        self.w_dec = nn.Parameter(torch.empty(d_sae, d_in, dtype=torch.float32))
        # No decoder bias for JumpReLU (w_dec is not bias-shifted)

        # Normalise decoder columns to unit length (standard JumpReLU convention)
        self._norm_dec()

    def _norm_dec(self):
        norms = self.w_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.w_dec.data = self.w_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] → activations: [..., d_sae]"""
        # F.linear(x, weight) computes x @ weight.T + bias
        # w_enc is [d_sae, d_in], so weight.T is [d_in, d_sae]
        # x @ weight.T = [..., d_in] @ [..., d_in, d_sae] = [..., d_sae] ✓
        acts = torch.nn.functional.linear(x, self.w_enc, self.b_enc)
        # Jump ReLU: clamp at threshold, then ReLU
        activated = torch.clamp(acts - self.threshold, min=0.0)
        return activated

    def decode(self, activated: torch.Tensor) -> torch.Tensor:
        """activated: [..., d_sae] → reconstruction: [..., d_in]"""
        # w_dec is [d_sae, d_in], so weight.T is [d_in, d_sae]
        # activated @ weight.T = [..., d_sae] @ [..., d_sae, d_in] = [..., d_in] ✓
        return torch.nn.functional.linear(activated, self.w_dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = self.encode(x)
        return self.decode(activated)


def _load_jumprelu_sae(path: str, d_in: int, device: str = "cuda") -> JumpReLUSAE:
    """Load a Gemma-Scope-2 JumpReLU SAE from a local checkpoint directory.

    Handles the lowercase key naming (w_enc/w_dec/b_enc) used by the
    gemma-scope-2 HuggingFace repo.

    Args:
        path: Checkpoint directory containing config.json + weight file.
        d_in: Encoder input dimension (backbone hidden size, e.g. 1152).
        device: Device to load SAE onto.
    """
    weight_path = os.path.join(path, "sae_weights.safetensors")
    if not os.path.exists(weight_path):
        weight_path = os.path.join(path, "params.safetensors")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"No weight file found in {path}")

    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)

    d_sae = cfg["width"]

    sae = JumpReLUSAE(d_in, d_sae, threshold=1.0).to(device)

    with safe_open(weight_path, framework="pt", device="cpu") as f:
        state = {}
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key == "w_enc":
                # Checkpoint stores [d_in, d_sae], transpose to [d_sae, d_in]
                state["w_enc"] = tensor.t().contiguous()
            elif key == "w_dec":
                # Already [d_sae, d_in] — no transpose needed
                state["w_dec"] = tensor.contiguous()
            elif key == "b_enc":
                state["b_enc"] = tensor
            elif key == "b_dec":
                # b_dec in Gemma-Scope 2 checkpoints is the decoder bias-shift;
                # JumpReLU decoders are bias-free.  Absorb it into b_enc.
                if "b_enc" in state:
                    state["b_enc"] = state["b_enc"] + tensor
            elif key == "threshold":
                # Checkpoint may store per-feature threshold [d_sae] or scalar []
                if tensor.dim() == 0:
                    state["threshold"] = tensor
                else:
                    # Replace scalar threshold with per-feature threshold
                    sae.threshold = torch.nn.Parameter(
                        tensor.clone(), requires_grad=False
                    )

    sae.load_state_dict(state, strict=False)
    sae._norm_dec()
    return sae.to(device)


# ── DenseEmbedder ─────────────────────────────────────────────────────────────

class DenseEmbedder:
    """Wraps the dense embedding model via sentence-transformers on GPU.

    Loaded first so it claims VRAM before the larger sparse backbone.
    ONNX is disabled via SBERT_DISABLE_ONNX to avoid BFC arena conflicts.
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        logger.info("Loading dense model: %s", DENSE_MODEL)
        self.model = SentenceTransformer(DENSE_MODEL, device="cuda")
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("Dense model loaded. VRAM allocated: %.2f GB", allocated)

    def encode(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        """Encode texts into 768-d dense vectors."""
        embeddings = self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True
        )
        return embeddings.tolist()


# ── SparseEmbedder ────────────────────────────────────────────────────────────

class SparseEmbedder:
    """SAE-SPLADE sparse embedder using gemma-scope-2-270m-pt.

    Loaded after DenseEmbedder so dense claims VRAM first.
    Drop-in: same encode(texts, is_query) -> List[Dict] interface.
    """

    def __init__(self, internal_batch_size: int = INTERNAL_BATCH_SIZE):
        self.internal_batch_size = internal_batch_size
        self._load_models()
        self._validate_dimensions()

    def _load_models(self):
        logger.info("Loading Gemma 3 270M PT backbone from %s", BACKBONE_LOCAL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(BACKBONE_LOCAL_PATH)
        self.backbone = AutoModel.from_pretrained(
            BACKBONE_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            output_hidden_states=True,
        ).eval()

        sae_checkpoint_path = os.path.join(SAE_LOCAL_PATH, "resid_post", SAE_ID)
        logger.info("Loading Gemma Scope 2 SAE from local path: %s", sae_checkpoint_path)

        backbone_hidden = self.backbone.config.hidden_size
        self.sae = _load_jumprelu_sae(sae_checkpoint_path, d_in=backbone_hidden, device="cuda")

        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            "Sparse models loaded: backbone=%s, SAE d_in=%d d_sae=%d | "
            "VRAM allocated: %.2f GB, reserved: %.2f GB",
            BACKBONE_LOCAL_PATH,
            self.sae.d_in,
            self.sae.d_sae,
            allocated,
            reserved,
        )

        assert next(self.backbone.parameters()).device.type == "cuda", \
            "FATAL: Backbone is on CPU. Check device_map."
        assert next(self.sae.parameters()).device.type == "cuda", \
            "FATAL: SAE is on CPU."

    def _validate_dimensions(self):
        """Confirm backbone hidden dim matches SAE d_in."""
        backbone_hidden = self.backbone.config.hidden_size
        assert backbone_hidden == self.sae.d_in, (
            f"Dimension mismatch: backbone hidden_size={backbone_hidden} "
            f"but SAE d_in={self.sae.d_in}."
        )
        logger.info(
            "Dimension check passed: backbone hidden_size=%d, SAE d_in=%d, SAE width=%d",
            backbone_hidden,
            self.sae.d_in,
            self.sae.d_sae,
        )

    def _encode_batch(self, texts: List[str]) -> List[Dict]:
        """Run one micro-batch synchronously on GPU. Called from thread pool."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to("cuda")

        with torch.no_grad():
            outputs = self.backbone(**inputs)
            # hidden_states is a tuple of (n_layers + 1) tensors, each [batch, seq, hidden]
            # Index SAE_HOOK_LAYER + 1 because index 0 is the embedding layer
            hidden = outputs.hidden_states[SAE_HOOK_LAYER + 1]

            # SAE encode: hidden [batch, seq, hidden] -> activations [batch, seq, sae_width]
            batch_size, seq_len, hidden_dim = hidden.shape
            flat_hidden = hidden.reshape(-1, hidden_dim).float()  # cast to float32 for SAE
            flat_acts = self.sae.encode(flat_hidden)
            acts = flat_acts.reshape(batch_size, seq_len, -1)

            # SPLADE max-pooling with log saturation, masked to real tokens
            logged = torch.log1p(acts)
            mask = inputs["attention_mask"].unsqueeze(-1).to(logged.dtype)
            sparse_vecs = (logged * mask).max(dim=1).values

        results = []
        for vec in sparse_vecs:
            nonzero_mask = vec > SPLADE_THRESHOLD
            indices = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
            results.append({
                "indices": indices.cpu().tolist(),
                "values": vec[indices].cpu().float().tolist(),
            })
        return results

    def encode(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Encode texts into SAE-SPLADE sparse vectors.

        `is_query` is accepted for API compatibility but has no effect here —
        the same SAE is used for both documents and queries, consistent with
        how SPLADE symmetric models work.

        Args:
            texts: List of strings to embed.
            is_query: Ignored (kept for interface compatibility).

        Returns:
            List of dicts with 'indices' and 'values' keys.
        """
        all_results = []
        for i in range(0, len(texts), self.internal_batch_size):
            batch = texts[i : i + self.internal_batch_size]
            all_results.extend(self._encode_batch(batch))
        return all_results

    async def encode_async(self, texts: List[str], is_query: bool = False) -> List[Dict]:
        """Async wrapper for use in FastAPI async routes.

        Pushes GPU work to the thread pool so the event loop is not blocked.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.encode(texts, is_query),
        )


# ── QueryRewriter ─────────────────────────────────────────────────────────────

class QueryRewriter:
    """Rewrites user prompts using gemma-scope-2-270m-it for better retrieval.

    Loads the Instruct-Tuned Gemma Scope 2 model from local disk and uses it
    to reformulate natural-language queries into precise technical search queries
    optimized for retrieving AI/ML research papers.
    """

    def __init__(self):
        self._load_model()
        self._load_prompt()

    def _load_model(self):
        """Load the IT model tokenizer and causal LM from local disk."""
        logger.info("Loading Gemma Scope 2 IT model from %s", IT_MODEL_LOCAL_PATH)
        self.tokenizer = AutoTokenizer.from_pretrained(IT_MODEL_LOCAL_PATH)
        self.model = AutoModelForCausalLM.from_pretrained(
            IT_MODEL_LOCAL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        allocated = torch.cuda.memory_allocated() / 1e9
        logger.info("IT model loaded. VRAM allocated: %.2f GB", allocated)

        assert next(self.model.parameters()).device.type == "cuda", \
            "FATAL: IT model is on CPU. Check device_map."

    def _load_prompt(self):
        """Load the system prompt from the accompanying text file."""
        if _PROMPT_FILE.exists():
            self.system_prompt = _PROMPT_FILE.read_text(encoding="utf-8").strip()
        else:
            # Fallback if the file is missing (shouldn't happen, but be safe)
            self.system_prompt = (
                "You are a technical query reformulation engine. Transform the user's "
                "natural-language input into a precise, effective search query optimized "
                "for retrieving AI/ML research papers. Return ONLY the reformulated query "
                "text. No explanation, no preamble, no quotes."
            )
        logger.info("System prompt loaded (%d characters)", len(self.system_prompt))

    def rewrite(self, query: str) -> str:
        """Reformulate a user query into a precise technical search query.

        Args:
            query: The raw user input (natural language, potentially imprecise).

        Returns:
            Reformulated query string optimized for retrieval.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
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

        # Decode only the generated tokens (skip the prompt)
        generated = outputs[0][input_ids.shape[1]:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Clean up: take only the first meaningful line if the model adds extras
        if "\n\n" in result:
            result = result.split("\n\n")[0].strip()
        if "\n" in result:
            result = result.split("\n")[0].strip()

        logger.info("Rewrote query (%d → %d chars)", len(query), len(result))
        return result

    async def rewrite_async(self, query: str) -> str:
        """Async wrapper for use in FastAPI async routes.

        Pushes GPU inference to the thread pool so the event loop is not blocked.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: self.rewrite(query),
        )