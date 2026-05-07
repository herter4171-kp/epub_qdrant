"""
Train a Top-K Sparse Autoencoder on pre-computed SPLADE activations
using SAELens 6.x.x (tested on 6.43.0).

No LLM, no ActivationsStore, no model_name required.
DataProvider in v6 is just Iterator[torch.Tensor].

Hardware targets:
  - RTX 5090 (single-GPU, primary path)
  - DGX Spark (multi-GPU, set NUM_GPUS > 1 and enable DDP flag)

Install:
    pip install sae-lens==6.43.0
"""

import logging
import os
from collections.abc import Iterator

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACTIVATIONS_PATH = os.getenv("SAE_ACTIVATIONS_PATH", "./sae_data/activations.npy")
CORPUS_MEAN_PATH = os.getenv("SAE_CORPUS_MEAN_PATH", "./sae_data/corpus_mean.npy")
OUTPUT_DIR = os.getenv("SAE_OUTPUT_DIR", "./sae_data/sae_v6_output")

# SAE architecture
D_IN = 30522            # BERT / SPLADE vocabulary size
EXPANSION_FACTOR = 2    # 4x → 122,088 latent dims  (image recommends 32k–64k min)
D_SAE = D_IN * EXPANSION_FACTOR
K = 165                 # top-k active features per vector

# Training
BATCH_SIZE = 256        # 256–512 is comfortable on an RTX 5090 at float32
LEARNING_RATE = 2e-4
EPOCHS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SAELens 6.x trainer knobs
LR_WARM_UP_STEPS = 500
LR_DECAY_STEPS = 0
N_CHECKPOINTS = 3       # intermediate checkpoints
LOG_TO_WANDB = False    # flip to True if you want W&B logging


# ---------------------------------------------------------------------------
# DataProvider: Iterator[torch.Tensor]  ← the entire v6 contract
# ---------------------------------------------------------------------------

class SpladeActivationProvider:
    """
    Wraps a pre-computed (N, D_IN) float32 numpy array and yields
    batches of shape (BATCH_SIZE, D_IN) on the target device.

    SAELens v6 DataProvider contract: Iterator[torch.Tensor].
    No position dimension, no context window — exactly what we have.
    """

    def __init__(
        self,
        activations: np.ndarray,
        batch_size: int,
        epochs: int,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.activations = activations          # (N, D_IN)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.n_samples = len(activations)
        self.total_samples = epochs * self.n_samples

    def __iter__(self) -> Iterator[torch.Tensor]:
        samples_yielded = 0
        for _ in range(self.epochs):
            idx = self.rng.permutation(self.n_samples)
            shuffled = self.activations[idx]
            for start in range(0, self.n_samples - self.batch_size + 1, self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                yield torch.from_numpy(batch).to(self.device, non_blocking=True)
                samples_yielded += self.batch_size
                if samples_yielded >= self.total_samples:
                    return


# ---------------------------------------------------------------------------
# Build SAE
# ---------------------------------------------------------------------------

def build_sae(corpus_mean: np.ndarray | None = None):
    """
    Construct a TopKTrainingSAE.

    corpus_mean is used to initialise the pre-encoder bias (b_dec),
    which dramatically speeds up convergence on SPLADE's sparse non-negative
    distribution.
    """
    from sae_lens.saes.topk_sae import TopKTrainingSAE, TopKTrainingSAEConfig

    cfg = TopKTrainingSAEConfig(
        d_in=D_IN,
        d_sae=D_SAE,
        k=K,
        dtype="float32",
        device=DEVICE,
        apply_b_dec_to_input=True,
        normalize_activations="none",   # SPLADE is already sparse; no extra norm
        # aux_loss_coefficient encourages dead-neuron recovery
        aux_loss_coefficient=1.0 / 32,
        rescale_acts_by_decoder_norm=True,
        decoder_init_norm=0.1,          # Anthropic heuristic init
    )

    sae = TopKTrainingSAE(cfg)

    # Ensure model is on the target device (SAELens may init on CPU)
    sae = sae.to(DEVICE)

    # Initialise b_dec with negative corpus mean → encoder sees zero-mean input
    if corpus_mean is not None:
        with torch.no_grad():
            sae.b_dec.data = torch.from_numpy(-corpus_mean).to(
                dtype=torch.float32, device=DEVICE
            )
        logger.info("Initialized b_dec from corpus mean.")

    return sae


# ---------------------------------------------------------------------------
# Build trainer config
# ---------------------------------------------------------------------------

def build_trainer_config(total_samples: int):
    from sae_lens.config import LoggingConfig, SAETrainerConfig

    return SAETrainerConfig(
        total_training_samples=total_samples,
        train_batch_size_samples=BATCH_SIZE,
        lr=LEARNING_RATE,
        lr_end=LEARNING_RATE,           # constant schedule (set decay steps for cosine)
        lr_scheduler_name="constant",
        lr_warm_up_steps=LR_WARM_UP_STEPS,
        lr_decay_steps=LR_DECAY_STEPS,
        adam_beta1=0.9,
        adam_beta2=0.999,
        device=DEVICE,
        autocast=False,                 # keep float32; SPLADE vecs are sparse floats
        n_checkpoints=N_CHECKPOINTS,
        checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoints"),
        save_final_checkpoint=True,
        dead_feature_window=1000,
        feature_sampling_window=2000,
        n_batches_for_norm_estimate=500,
        logger=LoggingConfig(
            log_to_wandb=LOG_TO_WANDB,
            wandb_project="splade-sae",
            eval_every_n_wandb_logs=200,
        ),
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from sae_lens.training.sae_trainer import SAETrainer

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 1. Load data
    logger.info("Loading activations from %s", ACTIVATIONS_PATH)
    activations = np.load(ACTIVATIONS_PATH).astype(np.float32)  # (N, 30522)
    assert activations.ndim == 2 and activations.shape[1] == D_IN, (
        f"Expected (N, {D_IN}), got {activations.shape}"
    )
    n_samples = len(activations)
    total_samples = EPOCHS * n_samples
    logger.info("  %d samples × %d epochs = %d total training samples",
                n_samples, EPOCHS, total_samples)

    # 2. Corpus mean for bias init
    corpus_mean = None
    if os.path.exists(CORPUS_MEAN_PATH):
        corpus_mean = np.load(CORPUS_MEAN_PATH).astype(np.float32)
        logger.info("Corpus mean loaded: shape=%s", corpus_mean.shape)

    # 3. Data iterator  — v6 contract: Iterator[Tensor]
    data_provider = SpladeActivationProvider(
        activations=activations,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=DEVICE,
        seed=SEED,
    )

    # 4. Build SAE
    sae = build_sae(corpus_mean=corpus_mean)
    logger.info("SAE: d_in=%d  d_sae=%d  k=%d  device=%s", D_IN, D_SAE, K, DEVICE)

    # 5. Build trainer config
    trainer_cfg = build_trainer_config(total_samples=total_samples)

    # 6. Train
    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae,
        data_provider=iter(data_provider),
    )

    logger.info("Starting training (%d steps)…", trainer_cfg.total_training_steps)
    trainer.fit()

    # 7. Save
    final_path = os.path.join(OUTPUT_DIR, "sae_final")
    os.makedirs(final_path, exist_ok=True)
    sae.save_model(final_path)
    logger.info("Saved SAE to %s", final_path)

    # 8. Export TorchScript encoder for inference (no sae-lens dep at serve time)
    example = torch.randn(1, D_IN, device=DEVICE)
    with torch.no_grad():
        ts = torch.jit.trace(sae.encode, example)
    ts_path = os.path.join(OUTPUT_DIR, "sae_encoder.pt")
    ts.save(ts_path)
    logger.info("TorchScript encoder saved to %s", ts_path)

    return sae


# ---------------------------------------------------------------------------
# Optional: multi-GPU wrapper for DGX Spark
# ---------------------------------------------------------------------------
# For DGX Spark, launch with:
#   torchrun --nproc_per_node=<NUM_GPUS> splade_sae_train_v6.py --ddp
#
# SAETrainer itself is single-process; the recommended approach is data
# parallelism at the DataProvider level: each rank reads a non-overlapping
# shard of the activation matrix and the gradients are all-reduced.
#
# Minimal DDP wrapper:

def run_ddp_training():
    """Entry point for torchrun-based multi-GPU training."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    activations = np.load(ACTIVATIONS_PATH).astype(np.float32)
    # Shard: rank r owns rows [r::world_size]
    shard = activations[rank::world_size]

    corpus_mean = None
    if os.path.exists(CORPUS_MEAN_PATH):
        corpus_mean = np.load(CORPUS_MEAN_PATH).astype(np.float32)

    data_provider = SpladeActivationProvider(
        activations=shard,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=f"cuda:{local_rank}",
        seed=SEED + rank,
    )

    sae = build_sae(corpus_mean=corpus_mean).to(f"cuda:{local_rank}")
    sae_ddp = DDP(sae, device_ids=[local_rank])

    # SAETrainer operates on the underlying module; gradients are synced by DDP
    from sae_lens.training.sae_trainer import SAETrainer
    from sae_lens.config import SAETrainerConfig, LoggingConfig

    total_samples = EPOCHS * len(shard)
    trainer_cfg = build_trainer_config(total_samples=total_samples)
    # Only rank 0 logs / saves
    trainer_cfg.logger.log_to_wandb = LOG_TO_WANDB and rank == 0
    trainer_cfg.save_final_checkpoint = rank == 0

    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae_ddp.module,     # pass the unwrapped module; DDP handles grad sync
        data_provider=iter(data_provider),
    )
    trainer.fit()

    if rank == 0:
        final_path = os.path.join(OUTPUT_DIR, "sae_final")
        os.makedirs(final_path, exist_ok=True)
        sae.save_model(final_path)
        logger.info("Rank 0: saved SAE to %s", final_path)

    dist.destroy_process_group()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if "--ddp" in sys.argv:
        run_ddp_training()
    else:
        run_training()
