"""
Print per-epoch convergence tables from an existing SAE training results directory.

Usage:
    python -m src.sae.sae_convergence <results_dir>
    python -m src.sae.sae_convergence ./sae_data_63_MSE/sae_v6_output
    python -m src.sae.sae_convergence ./sae_data/sae_v6_output --latest
"""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_checkpoint_cfg(checkpoint_dir: str) -> dict:
    """Load cfg.json from a checkpoint directory."""
    cfg_path = os.path.join(checkpoint_dir, "cfg.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def load_sparsity(checkpoint_dir: str) -> dict:
    """Load sparsity.safetensors from a checkpoint directory."""
    import torch
    sparsity_path = os.path.join(checkpoint_dir, "sparsity.safetensors")
    if os.path.exists(sparsity_path):
        return torch.load(sparsity_path, weights_only=True)
    return {}


def load_activation_scaler(checkpoint_dir: str) -> dict:
    """Load activation_scaler.json from a checkpoint directory."""
    scaler_path = os.path.join(checkpoint_dir, "activation_scaler.json")
    if os.path.exists(scaler_path):
        with open(scaler_path) as f:
            return json.load(f)
    return {}


def load_trainer_state(checkpoint_dir: str) -> dict:
    """Load trainer_state.pt from a checkpoint directory."""
    import torch
    state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if os.path.exists(state_path):
        return torch.load(state_path, weights_only=False)
    return {}


def extract_metrics_from_trainer_state(trainer_state: dict) -> dict:
    """Extract per-epoch metrics from trainer_state.pt."""
    metrics = {}

    # Common keys in SAELens trainer_state
    for key in ["mse_loss", "l1_loss", "aux_loss", "sparsity",
                 "dead_feature_rate", "liveness", "decoder_norm", "encoder_norm"]:
        if key in trainer_state:
            val = trainer_state[key]
            if isinstance(val, (int, float)):
                metrics[key] = float(val)
            elif isinstance(val, (list, tuple)):
                metrics[key] = float(val[-1])  # latest value
            elif hasattr(val, 'item'):
                metrics[key] = float(val.item())

    return metrics


def extract_metrics_from_sparsity(sparsity_data: dict) -> dict:
    """Extract sparsity-related metrics from sparsity.safetensors."""
    metrics = {}
    for key in ["dead_feature_rate", "liveness", "sparsity"]:
        if key in sparsity_data:
            val = sparsity_data[key]
            if isinstance(val, (int, float)):
                metrics[key] = float(val)
            elif hasattr(val, 'item'):
                metrics[key] = float(val.item())
    return metrics


def get_checkpoint_dirs(results_dir: str) -> list:
    """Get sorted list of checkpoint directories."""
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        return []

    checkpoint_dirs = []
    for entry in os.listdir(checkpoints_dir):
        full_path = os.path.join(checkpoints_dir, entry)
        if os.path.isdir(full_path) and entry.startswith("checkpoint-") or entry.isdigit():
            checkpoint_dirs.append(full_path)

    return sorted(checkpoint_dirs)


def print_convergence_table(results_dir: str, latest: bool = False):
    """Print a convergence table from an existing results directory."""
    checkpoint_dirs = get_checkpoint_dirs(results_dir)
    if not checkpoint_dirs:
        print(f"No checkpoints found in {results_dir}")
        return

    if latest:
        checkpoint_dirs = [checkpoint_dirs[-1]]

    print(f"\n{'=' * 80}")
    print(f"CONVERGENCE TABLE — {results_dir}")
    print(f"{'=' * 80}")
    print(f"Checkpoints: {len(checkpoint_dirs)}")
    print()

    # Collect metrics from each checkpoint
    all_metrics = []
    for cp_dir in checkpoint_dirs:
        trainer_state = load_trainer_state(cp_dir)
        sparsity_data = load_sparsity(cp_dir)
        cfg = load_checkpoint_cfg(cp_dir)

        metrics = extract_metrics_from_trainer_state(trainer_state)
        metrics.update(extract_metrics_from_sparsity(sparsity_data))
        metrics['checkpoint'] = os.path.basename(cp_dir)
        metrics['cfg'] = cfg

        all_metrics.append(metrics)

    # Define canonical metric columns
    metric_columns = [
        ("mse_loss", "MSE"),
        ("l1_loss", "L1"),
        ("aux_loss", "Aux"),
        ("sparsity", "Sparsity"),
        ("dead_feature_rate", "Dead%"),
        ("liveness", "Liveness"),
        ("decoder_norm", "DecNorm"),
        ("encoder_norm", "EncNorm"),
    ]

    # Print header
    header = f"{'#':>4s}  {'Checkpoint':>16s}"
    for _, label in metric_columns:
        header += f"  {label:>10s}"
    print(header)
    print("-" * len(header))

    # Print rows
    for i, metrics in enumerate(all_metrics):
        row = f"{i + 1:>4d}  {metrics['checkpoint']:>16s}"
        for metric_name, _ in metric_columns:
            val = metrics.get(metric_name)
            if val is not None:
                row += f"  {val:>10.4f}"
            else:
                row += f"  {'—':>10s}"
        print(row)

    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for metric_name, label in metric_columns:
        values = [m.get(metric_name) for m in all_metrics if m.get(metric_name) is not None]
        if values:
            start = values[0]
            end = values[-1]
            best = min(values)
            best_idx = values.index(best) + 1
            print(f"  {label:<15s}  start={start:>10.4f}  end={end:>10.4f}  "
                  f"best={best:>10.4f}  (epoch {best_idx})")

    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Print SAE convergence table from results dir")
    parser.add_argument("results_dir", help="Path to SAE training results directory")
    parser.add_argument("--latest", action="store_true", help="Show only the latest checkpoint")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: {args.results_dir} is not a directory")
        sys.exit(1)

    print_convergence_table(args.results_dir, latest=args.latest)


if __name__ == "__main__":
    main()