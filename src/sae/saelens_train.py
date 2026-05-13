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

# Cap Inductor compile parallelism BEFORE importing torch — default is
# min(cpu_count, 32), which fans out N worker processes each holding a copy
# of the FX graph. On a unified-memory box that easily wedges the kernel.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS",
                      os.getenv("SAE_INDUCTOR_THREADS", "1"))

# Reduce PyTorch caching-allocator fragmentation. Without this, freed
# blocks can leave multi-GB holes that the next allocation can't reuse —
# fatal at the unified-memory ceiling. Must be set before CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
EXPANSION_FACTOR = 3
D_SAE = D_IN * EXPANSION_FACTOR
K = 165

BATCH_SIZE = 8192
LEARNING_RATE = 2e-4
LR_END = LEARNING_RATE * 0.1   # cosine decays to here
EPOCHS = 30
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_CHECKPOINTS = EPOCHS
LOG_TO_WANDB = False
USE_ADAM8BIT = os.getenv("SAE_USE_ADAM8BIT", "0").lower() in ("1", "true", "yes")
# Adafactor is the default at EF=3 — fp32 Adam state is ~45 GB, 8-bit Adam
# is fragile on Blackwell, Adafactor's factored 2nd moment + no 1st moment
# is ~1 MB. Set SAE_USE_ADAFACTOR=0 to fall back to fp32 Adam.
USE_ADAFACTOR = os.getenv("SAE_USE_ADAFACTOR", "1").lower() in ("1", "true", "yes")

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
                # Sort within batch — turns 8192 random page reads into
                # disk-monotonic reads. SGD doesn't care about order within
                # a batch; gradients are summed regardless.
                batch_idx = np.sort(idx[start : start + self.batch_size])
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
# OS hints
# ---------------------------------------------------------------------------

def _log_mem(stage: str) -> None:
    """Snapshot process VM and system MemAvailable. Cheap; safe to call often."""
    vmsize = vmrss = vmdata = "?"
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmSize:"):
                    vmsize = line.split()[1]
                elif line.startswith("VmRSS:"):
                    vmrss = line.split()[1]
                elif line.startswith("VmData:"):
                    vmdata = line.split()[1]
    except OSError:
        pass
    avail = "?"
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = line.split()[1]
                    break
    except OSError:
        pass
    logger.info(
        "[mem:%s] VmSize=%s kB, VmRSS=%s kB, VmData=%s kB, MemAvailable=%s kB",
        stage, vmsize, vmrss, vmdata, avail,
    )


def _preinit_cuda() -> None:
    """Force CUDA context init now, before mmap'ing the 23 GB activations file.

    On Grace Blackwell unified memory, the CUDA driver reserves a sizable
    chunk of virtual address space (and physical RAM) when its context is
    first created. If we do that *after* mmap'ing 23 GB and constructing
    multi-billion-param tensors, the driver can fail with 'out of memory'
    even on a 122 KB allocation. Pre-init solves it: the driver claims its
    slot when the process is at its smallest.
    """
    if not torch.cuda.is_available():
        return
    _ = torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    logger.info(
        "CUDA pre-initialized. Device memory: %.1f GB free / %.1f GB total.",
        free / 1e9, total / 1e9,
    )


def _apply_ram_safeguards() -> None:
    """Two safeguards, neither incompatible with CUDA's expandable_segments.

    1) oom_score_adj=+1000 (configurable via SAE_OOM_SCORE_ADJ): tells the
       kernel to OOM-kill *us* before anything else. sshd, dbus, etc. retain
       their normal scores and stay up, so a memory blowup in training never
       costs you your shell.

    2) Optional startup gate via SAE_MIN_FREE_RAM_MB: if MemAvailable is
       below the threshold at the moment we start, we refuse to launch.
       This catches "you forgot to drop_caches" / "another job is still
       running" before we burn 30 minutes initializing.

    Why not RLIMIT_AS: with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    the CUDA driver pre-reserves 100+ GB of virtual address space on init,
    so any AS cap below that blocks legitimate mmap (e.g. the 23 GB
    activations file). RLIMIT_AS is the wrong tool for unified memory.
    """
    # Startup gate (optional)
    threshold_mb = os.getenv("SAE_MIN_FREE_RAM_MB")
    if threshold_mb:
        avail_kb = 0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                        break
        except OSError:
            pass
        threshold_kb = int(float(threshold_mb)) * 1024
        if avail_kb and avail_kb < threshold_kb:
            raise RuntimeError(
                f"Refusing to start: MemAvailable={avail_kb // 1024} MB "
                f"< SAE_MIN_FREE_RAM_MB={threshold_mb} MB. "
                "Drop caches or stop other jobs first."
            )

    # OOM priority
    score = os.getenv("SAE_OOM_SCORE_ADJ", "1000")
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(score + "\n")
        logger.info(
            "oom_score_adj=%s — this process is the OOM killer's preferred "
            "target; sshd will survive memory blowups here.",
            score,
        )
    except OSError as e:
        logger.warning("Could not set oom_score_adj: %s", e)


def _madvise_random(arr: np.ndarray) -> None:
    """Issue MADV_RANDOM on a mmap'd numpy array to suppress kernel readahead.

    Default Linux readahead is ~128 KiB per fault, which is wasted I/O for
    random row sampling — the speculatively-loaded pages get evicted before
    we ever touch them, and they squeeze hot pages out of the cache.

    np.load(..., mmap_mode='r') puts a 128-byte .npy header at the start of
    the mapping, so arr.ctypes.data is mid-page. madvise requires a
    page-aligned address, so we round down — the bytes before our slice are
    still part of the same mapping (the header), so it's safe to advise them.
    """
    import ctypes
    import mmap as _mmap

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    MADV_RANDOM = 1
    page = _mmap.PAGESIZE
    addr = arr.ctypes.data
    length = arr.nbytes
    aligned_addr = addr - (addr % page)
    aligned_length = length + (addr - aligned_addr)
    rc = libc.madvise(
        ctypes.c_void_p(aligned_addr),
        ctypes.c_size_t(aligned_length),
        ctypes.c_int(MADV_RANDOM),
    )
    if rc != 0:
        logger.warning("madvise(MADV_RANDOM) failed: errno=%d", ctypes.get_errno())
    else:
        logger.info("MADV_RANDOM applied to activations mmap (%.1f GB).",
                    arr.nbytes / 1e9)


def _swap_to_adam8bit(trainer, trainer_cfg, sae) -> None:
    """Replace SAETrainer's fp32 Adam with torchao Adam8bit, eager.

    SAELens hard-codes torch.optim.Adam in SAETrainer.__init__; we override
    after construction. The grad scaler reads trainer.optimizer lazily each
    step, so swapping the attribute is sufficient. The LR scheduler binds
    to the optimizer via param_groups, so we rebuild it.

    Why torchao instead of bitsandbytes: bnb's optimizer8bit_blockwise kernel
    declares the element-count parameter as int32, which overflows for
    weight matrices >2.1B elements (i.e. EXPANSION_FACTOR ≳ 2.3 at D_IN=30522).
    torchao's 8-bit Adam is pure-PyTorch, no int32 cap.

    Why dynamo.disable on .step: torchao's default path wraps the per-param
    update in torch.compile, which spawns Inductor + Triton subprocess
    workers at first call. On a unified-memory box this can wedge the kernel
    before training even begins. Eager step is slower per call but the
    optimizer is a small fraction of step time vs. the 30522×91566 matmuls.
    """
    import torch._dynamo
    from torchao.optim import Adam8bit
    from sae_lens.training.optim import get_lr_scheduler

    optimizer = Adam8bit(
        sae.parameters(),
        lr=trainer_cfg.lr,
        betas=(trainer_cfg.adam_beta1, trainer_cfg.adam_beta2),
    )
    trainer.optimizer = optimizer
    # Build the LR scheduler first — it patches optimizer.step to a bound
    # method that tracks step calls. If we wrap with dynamo.disable before
    # this, the scheduler's patch fails (expects a bound method to unwrap
    # via .__func__).
    trainer.lr_scheduler = get_lr_scheduler(
        scheduler_name=trainer_cfg.lr_scheduler_name,
        optimizer=trainer.optimizer,
        training_steps=trainer_cfg.total_training_steps,
        lr=trainer_cfg.lr,
        warm_up_steps=trainer_cfg.lr_warm_up_steps,
        decay_steps=trainer_cfg.lr_decay_steps,
        lr_end=trainer_cfg.lr_end,
        num_cycles=trainer_cfg.n_restart_cycles,
    )
    # Now wrap the scheduler-patched step with dynamo.disable; tracking
    # still happens inside the disabled region.
    optimizer.step = torch._dynamo.disable(optimizer.step)
    logger.info("Optimizer: torchao Adam8bit (8-bit moments, eager — torch.compile disabled).")


def _swap_to_adafactor(trainer, trainer_cfg, sae) -> None:
    """Replace SAETrainer's fp32 Adam with HuggingFace Adafactor.

    Adafactor uses a factored second moment (O(d_in + d_out) per tensor)
    and no first moment by default. State at EF=3 totals ~1 MB across
    the SAE, vs 30-45 GB for fp32 Adam.

    scale_parameter=False, relative_step=False, warmup_init=False — so
    Adafactor honors our explicit lr and the cosine schedule rather than
    its built-in relative-step heuristic.
    """
    from transformers.optimization import Adafactor
    from sae_lens.training.optim import get_lr_scheduler

    trainer.optimizer = Adafactor(
        sae.parameters(),
        lr=trainer_cfg.lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    trainer.lr_scheduler = get_lr_scheduler(
        scheduler_name=trainer_cfg.lr_scheduler_name,
        optimizer=trainer.optimizer,
        training_steps=trainer_cfg.total_training_steps,
        lr=trainer_cfg.lr,
        warm_up_steps=trainer_cfg.lr_warm_up_steps,
        decay_steps=trainer_cfg.lr_decay_steps,
        lr_end=trainer_cfg.lr_end,
        num_cycles=trainer_cfg.n_restart_cycles,
    )
    logger.info("Optimizer: HuggingFace Adafactor (factored 2nd moment, no 1st moment).")


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
    _log_mem("startup")
    _apply_ram_safeguards()
    _preinit_cuda()
    _log_mem("after_cuda_init")

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
    activations = activations = np.load(ACTIVATIONS_PATH) #np.load(ACTIVATIONS_PATH, mmap_mode='r')  # (N, 30522)
    #activations = activations[:2*BATCH_SIZE] # TODO: REMOVE!
    _madvise_random(activations)
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
    _log_mem("after_mmap")

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
    _log_mem("after_sae_build")

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

    if USE_ADAM8BIT and USE_ADAFACTOR:
        raise ValueError("Pick one: --adam8bit OR --adafactor (or unset SAE_USE_ADAFACTOR).")
    if USE_ADAM8BIT:
        _swap_to_adam8bit(trainer, trainer_cfg, sae)
    elif USE_ADAFACTOR:
        _swap_to_adafactor(trainer, trainer_cfg, sae)
    _log_mem("after_optim_setup")


    samples_per_epoch = (n_samples // BATCH_SIZE) * BATCH_SIZE
    tracker = _ConvergenceTracker(
        output_dir=OUTPUT_DIR,
        samples_per_epoch=samples_per_epoch,
        total_epochs=EPOCHS,
    )

    original_train_step = trainer._train_step
    step_state = {"first_step_done": False}
    def _train_step_with_tracking(sae, sae_in):
        import time as _time
        is_first = not step_state["first_step_done"]
        if is_first:
            logger.info("Step 1 starting…")
            t0 = _time.perf_counter()
        out = original_train_step(sae=sae, sae_in=sae_in)
        if is_first:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = _time.perf_counter() - t0
            logger.info("Step 1 done in %.1fs.", elapsed)
            step_state["first_step_done"] = True
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

    _apply_ram_safeguards()
    _preinit_cuda()
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
    if USE_ADAM8BIT and USE_ADAFACTOR:
        raise ValueError("Pick one: --adam8bit OR --adafactor (or unset SAE_USE_ADAFACTOR).")
    if USE_ADAM8BIT:
        _swap_to_adam8bit(trainer, trainer_cfg, sae_ddp.module)
    elif USE_ADAFACTOR:
        _swap_to_adafactor(trainer, trainer_cfg, sae_ddp.module)
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

    if "--adam8bit" in sys.argv:
        USE_ADAM8BIT = True
        USE_ADAFACTOR = False
    if "--adafactor" in sys.argv:
        USE_ADAFACTOR = True
        USE_ADAM8BIT = False
    if "--no-adafactor" in sys.argv or "--fp32-adam" in sys.argv:
        USE_ADAFACTOR = False
        USE_ADAM8BIT = False

    if "--resume" in sys.argv:
        run_training(resume=True)
    elif "--ddp" in sys.argv:
        run_ddp_training()
    else:
        run_training()
