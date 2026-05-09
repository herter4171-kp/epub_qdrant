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
import sys
from collections.abc import Iterator

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ACTIVATIONS_PATH = os.getenv("SAE_ACTIVATIONS_PATH", "./sae_data/activations.npy")
CORPUS_MEAN_PATH = os.getenv("SAE_CORPUS_MEAN_PATH", "./sae_data/corpus_mean.npy")
OUTPUT_DIR = os.getenv("SAE_OUTPUT_DIR", "./sae_data/sae_v6_output")

D_IN = 30522
EXPANSION_FACTOR = 2
D_SAE = D_IN * EXPANSION_FACTOR
K = 165

BATCH_SIZE = 8192
LEARNING_RATE = 2e-4
LR_END = LEARNING_RATE * 0.1   # cosine decays to here
EPOCHS = 20
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CHECKPOINTS = EPOCHS
LOG_TO_WANDB = False

# LR schedule fractions — tuned at the batch-size/epoch level, not raw step counts
WARMUP_FRAC  = 0.05   # 5% of steps warming up
DECAY_FRAC   = 0.40   # cosine decay covers final 40% of steps

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
        epoch_file: str = "",
    ):
        self.activations = activations          # (N, D_IN)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.n_samples = len(activations)
        self.total_samples = epochs * self.n_samples
        self.epoch_file = epoch_file

    def __iter__(self) -> Iterator[torch.Tensor]:
        samples_yielded = 0
        samples_per_epoch = (self.n_samples // self.batch_size) * self.batch_size
        total_batches = samples_per_epoch // self.batch_size
        for epoch in range(self.epochs):
            logger.info(
                "Epoch %d/%d: permuting %d sample indices (activations shape=%s, dtype=%s)...",
                epoch + 1, self.epochs, self.n_samples,
                self.activations.shape, self.activations.dtype,
            )
            idx = self.rng.permutation(self.n_samples)
            logger.info(
                "Epoch %d: ready, yielding batches of %d (per-batch fancy indexing)...",
                epoch + 1, self.batch_size,
            )
            pbar = tqdm(
                total=total_batches,
                desc=f"Epoch {epoch + 1}",
                unit="batch",
                leave=False,
                bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
            for start in range(0, samples_per_epoch, self.batch_size):
                batch_idx = idx[start : start + self.batch_size]
                batch = np.ascontiguousarray(self.activations[batch_idx])
                yield torch.from_numpy(batch).to(self.device, non_blocking=True)
                del batch
                samples_yielded += self.batch_size
                pbar.update(1)
                # Log progress every 10 batches
                if (samples_yielded // self.batch_size) % 10 == 0:
                    logger.info(
                        "Epoch %d: %d/%d batches done (%.1f%%)",
                        epoch + 1,
                        samples_yielded // self.batch_size,
                        total_batches,
                        100.0 * samples_yielded / self.total_samples,
                    )
            # Save epoch progress for resume (once per epoch)
            if self.epoch_file:
                with open(self.epoch_file, "w") as f:
                    f.write(str(epoch + 1))
                logger.info("Saved epoch %d progress to %s", epoch + 1, self.epoch_file)
            pbar.close()
            # Yield a sentinel StopIteration by exhausting the generator naturally
            # rather than returning early, so SAELens's next() call works correctly
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
        # aux_loss_coefficient encourages dead-neuron recovery
        aux_loss_coefficient=1.0 / 8,
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

    total_steps = total_samples // BATCH_SIZE
    warmup_steps = max(1, int(total_steps * WARMUP_FRAC))
    decay_steps  = int(total_steps * DECAY_FRAC)

    logger.info(
        "LR schedule: total_steps=%d  warmup=%d  cosine_decay=%d  lr %.2e → %.2e",
        total_steps, warmup_steps, decay_steps, LEARNING_RATE, LR_END,
    )

    return SAETrainerConfig(
        total_training_samples=total_samples,
        train_batch_size_samples=BATCH_SIZE,
        lr=LEARNING_RATE,
        lr_end=LR_END,
        lr_scheduler_name="cosineannealing",
        lr_warm_up_steps=warmup_steps,
        lr_decay_steps=0, # decay_steps in this library means a linear decay to zero tacked on after the main scheduler finishes — it's not part of the cosine
        adam_beta1=0.9,
        adam_beta2=0.999,
        device=DEVICE,
        autocast=False,
        n_checkpoints=N_CHECKPOINTS,
        checkpoint_path=os.path.join(OUTPUT_DIR, "checkpoints"),
        save_final_checkpoint=True,
        dead_feature_window=500,
        feature_sampling_window=2000,
        n_batches_for_norm_estimate=100,   # was 500; meaningless at 461 total steps
        logger=LoggingConfig(
            log_to_wandb=LOG_TO_WANDB,
            wandb_project="splade-sae",
            eval_every_n_wandb_logs=200,
        ),
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(resume: bool = False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from sae_lens.training.sae_trainer import SAETrainer

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Resume: load epoch progress from file
    epoch_file = os.path.join(OUTPUT_DIR, "epoch.txt")
    if resume and os.path.exists(epoch_file):
        with open(epoch_file) as f:
            completed_epochs = int(f.read().strip())
        logger.info("Resuming from epoch %d (completed epochs: %d)",
                    completed_epochs + 1, completed_epochs)
        SEED_OFFSET = completed_epochs
    else:
        SEED_OFFSET = 0

    def save_epoch(epoch_num):
        with open(epoch_file, "w") as f:
            f.write(str(epoch_num))
        logger.info("Saved epoch %d progress to %s", epoch_num, epoch_file)

    # 1. Load data
    logger.info("Loading activations from %s", ACTIVATIONS_PATH)
    activations = np.load(ACTIVATIONS_PATH, mmap_mode='r')  # (N, 30522)
    #activations = activations[:2*BATCH_SIZE] # TODO: REMOVE!
    if activations.dtype != np.float32:
        raise TypeError(
            f"Expected float32 activations on disk, got {activations.dtype}. "
            "Convert the .npy once offline rather than casting here (an in-memory "
            "astype would materialize the entire ~22 GB array)."
        )
    logger.info("  activations shape=%s, dtype=%s, nbytes=%.1f GB, mmap_mode=%s",
                activations.shape, activations.dtype, activations.nbytes / 1e9, "r")
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
    epoch_file = os.path.join(OUTPUT_DIR, "epoch.txt")
    data_provider = SpladeActivationProvider(
        activations=activations,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        device=DEVICE,
        seed=SEED + SEED_OFFSET,
        epoch_file=epoch_file,
    )

    # 4. Build SAE
    sae = build_sae(corpus_mean=corpus_mean)
    logger.info("SAE: d_in=%d  d_sae=%d  k=%d  device=%s", D_IN, D_SAE, K, DEVICE)

    # 5. Build trainer config
    trainer_cfg = build_trainer_config(total_samples=total_samples)

    # 6. Build trainer and hook in the convergence tracker.
    #
    # Note on metric capture: SAELens v6 only computes/exports per-step metrics
    # via wandb (gated by cfg.logger.log_to_wandb). With wandb off we'd see
    # nothing. To keep wandb optional, we wrap _train_step — the real per-step
    # method — and read the TrainStepOutput it returns.
    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae,
        data_provider=iter(data_provider),
    )

    samples_per_epoch = (n_samples // BATCH_SIZE) * BATCH_SIZE
    tracker = _ConvergenceTracker(
        output_dir=OUTPUT_DIR,
        samples_per_epoch=samples_per_epoch,
        total_epochs=EPOCHS,
    )

    original_train_step = trainer._train_step
    def _train_step_with_tracking(sae, sae_in):
        out = original_train_step(sae=sae, sae_in=sae_in)
        tracker.record(out, trainer)
        tracker.flush_epoch_if_complete(trainer)
        return out
    trainer._train_step = _train_step_with_tracking

    original_save_checkpoint = trainer.save_checkpoint
    def _save_checkpoint_with_prune(*args, **kwargs):
        original_save_checkpoint(*args, **kwargs)
        prune_checkpoints(trainer_cfg.checkpoint_path, keep_last=2)
    trainer.save_checkpoint = _save_checkpoint_with_prune

    logger.info("Starting training (%d steps)…", trainer_cfg.total_training_steps)
    # Route logging through tqdm.write while bars are alive — otherwise log
    # lines land mid-bar because logging doesn't \r before writing.
    from tqdm.contrib.logging import logging_redirect_tqdm
    with logging_redirect_tqdm():
        trainer.fit()
        tracker.finalize(trainer)

    # 7. Save final epoch progress
    save_epoch(EPOCHS)

    # 8. Save
    final_path = os.path.join(OUTPUT_DIR, "sae_final")
    os.makedirs(final_path, exist_ok=True)
    sae.save_model(final_path)
    logger.info("Saved SAE to %s", final_path)

    # 9. Export TorchScript encoder for inference (no sae-lens dep at serve time)
    example = torch.randn(1, D_IN, device=DEVICE)
    with torch.no_grad():
        ts = torch.jit.trace(sae.encode, example)
    ts_path = os.path.join(OUTPUT_DIR, "sae_encoder.pt")
    ts.save(ts_path)
    logger.info("TorchScript encoder saved to %s", ts_path)

    return sae


# ---------------------------------------------------------------------------
# Per-epoch convergence tracker
# ---------------------------------------------------------------------------

class _ConvergenceTracker:
    """Aggregates per-step training metrics into one row per epoch.

    Hook this in by wrapping SAETrainer._train_step: call record() with the
    returned TrainStepOutput, then flush_epoch_if_complete() to detect epoch
    boundaries from trainer.n_training_samples. At end of fit(), call finalize()
    to flush any trailing partial epoch and print the summary.

    Per-epoch row is the *mean* of per-step values for losses/EV; for
    dead_features and lr we keep the last step's value (running state, not
    per-batch).

    Rows are appended to <output_dir>/convergence.tsv. The header is written
    only when the file does not yet exist, so resumed runs append cleanly.
    """

    COLUMNS = (
        "epoch", "step", "samples",
        "loss", "mse", "aux", "explained_var",
        "dead_features", "lr",
    )

    def __init__(self, output_dir: str, samples_per_epoch: int, total_epochs: int):
        self.samples_per_epoch = samples_per_epoch
        self.total_epochs = total_epochs
        self.tsv_path = os.path.join(output_dir, "convergence.tsv")
        self.last_flushed_epoch = -1
        self.rows: list[dict] = []
        self._reset_buffer()
        if not os.path.exists(self.tsv_path):
            with open(self.tsv_path, "w") as f:
                f.write("\t".join(self.COLUMNS) + "\n")

    def _reset_buffer(self) -> None:
        self.buf_loss: list[float] = []
        self.buf_mse: list[float] = []
        self.buf_aux: list[float] = []
        self.buf_ev: list[float] = []
        self.buf_dead = 0
        self.buf_lr = 0.0
        self.buf_step = 0

    @torch.no_grad()
    def record(self, step_output, trainer) -> None:
        self.buf_loss.append(float(step_output.loss.item()))

        losses = step_output.losses
        if "mse_loss" in losses:
            self.buf_mse.append(float(losses["mse_loss"].item()))
        aux_total = 0.0
        for k, v in losses.items():
            if k == "mse_loss":
                continue
            aux_total += float(v.item()) if hasattr(v, "item") else float(v)
        if len(losses) > 1 or "mse_loss" not in losses:
            self.buf_aux.append(aux_total)

        # Explained variance — same formulation as SAELens' wandb log dict.
        sae_in = step_output.sae_in
        sae_out = step_output.sae_out
        per_token_l2 = (sae_out - sae_in).pow(2).sum(dim=-1)
        total_var = (sae_in - sae_in.mean(0)).pow(2).sum(-1)
        ev = (1.0 - per_token_l2.mean() / total_var.mean()).item()
        self.buf_ev.append(float(ev))

        self.buf_dead = int(trainer.dead_neurons.sum().item())
        self.buf_lr = float(trainer.optimizer.param_groups[0]["lr"])
        # n_training_steps is incremented *after* _train_step in the fit loop,
        # so add 1 to reflect the step that just completed.
        self.buf_step = int(trainer.n_training_steps + 1)

    def flush_epoch_if_complete(self, trainer) -> None:
        completed = trainer.n_training_samples // self.samples_per_epoch
        while self.last_flushed_epoch + 1 < completed:
            target = self.last_flushed_epoch + 1
            self._write_row(
                epoch_index=target,
                samples_at_end=(target + 1) * self.samples_per_epoch,
            )
            self._reset_buffer()
            self.last_flushed_epoch = target

    def finalize(self, trainer) -> None:
        if self.buf_loss:
            target = self.last_flushed_epoch + 1
            self._write_row(epoch_index=target, samples_at_end=trainer.n_training_samples)
            self.last_flushed_epoch = target
        self._print_summary()

    def _write_row(self, epoch_index: int, samples_at_end: int) -> None:
        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        row = {
            "epoch": epoch_index + 1,
            "step": self.buf_step,
            "samples": samples_at_end,
            "loss": _mean(self.buf_loss),
            "mse": _mean(self.buf_mse),
            "aux": _mean(self.buf_aux),
            "explained_var": _mean(self.buf_ev),
            "dead_features": self.buf_dead,
            "lr": self.buf_lr,
        }
        self.rows.append(row)

        logger.info(
            "[epoch %d/%d] step=%d  loss=%.5f  mse=%.5f  aux=%.5f  EV=%.4f  dead=%d  lr=%.2e",
            row["epoch"], self.total_epochs, row["step"],
            row["loss"], row["mse"], row["aux"], row["explained_var"],
            row["dead_features"], row["lr"],
        )

        with open(self.tsv_path, "a") as f:
            f.write(
                f"{row['epoch']}\t{row['step']}\t{row['samples']}\t"
                f"{row['loss']:.6f}\t{row['mse']:.6f}\t{row['aux']:.6f}\t"
                f"{row['explained_var']:.6f}\t{row['dead_features']}\t{row['lr']:.6e}\n"
            )

    def _print_summary(self) -> None:
        if not self.rows:
            logger.info("No epoch rows recorded; skipping summary.")
            return
        bar = "=" * 78
        logger.info("")
        logger.info(bar)
        logger.info("FINAL CONVERGENCE SUMMARY  (%d rows -> %s)",
                    len(self.rows), self.tsv_path)
        logger.info(bar)
        logger.info(f"{'Metric':<16s} {'Start':>14s} {'End':>14s} {'Best':>14s} {'BestEp':>8s}")
        logger.info("-" * 78)
        # explained_var is higher-is-better; the rest are lower-is-better.
        higher_is_better = {"explained_var"}
        for m in ("loss", "mse", "aux", "explained_var", "dead_features"):
            vals = [r[m] for r in self.rows]
            start, end = vals[0], vals[-1]
            if m in higher_is_better:
                best = max(vals)
            else:
                best = min(vals)
            best_ep = self.rows[vals.index(best)]["epoch"]
            logger.info(
                f"{m:<16s} {start:>14.5f} {end:>14.5f} {best:>14.5f} {best_ep:>8d}"
            )
        logger.info(bar)


# ---------------------------------------------------------------------------
# Checkpoint pruning
# ---------------------------------------------------------------------------

def prune_checkpoints(checkpoint_dir: str, keep_last: int = 2):
    """Keep only the last `keep_last` checkpoint directories in checkpoint_dir."""
    import glob
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))
    if len(checkpoints) <= keep_last:
        return
    to_delete = checkpoints[:-keep_last]
    for cp in to_delete:
        import shutil
        shutil.rmtree(cp, ignore_errors=True)
        logger.info("Pruned checkpoint: %s", cp)


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

    activations = np.load(ACTIVATIONS_PATH, mmap_mode='r')
    if activations.dtype != np.float32:
        raise TypeError(
            f"Expected float32 activations on disk, got {activations.dtype}."
        )
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

    if "--resume" in sys.argv:
        run_training(resume=True)
    elif "--ddp" in sys.argv:
        run_ddp_training()
    else:
        run_training()
