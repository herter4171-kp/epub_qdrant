"""Extract W_dec row norms from the SAE checkpoint. Run once."""

import torch
from safetensors import safe_open
from pathlib import Path

CHECKPOINT = "/tank/sae-splade/sae_data_good_2x/901120/sae_weights.safetensors"
OUT = Path("sae_data/dec_norms.pt")

with safe_open(CHECKPOINT, framework="pt") as f:
    W_dec = f.get_tensor("W_dec")  # shape (61044, 30522), float32

dec_norms = W_dec.norm(dim=-1)  # shape (61044,), float32
torch.save(dec_norms, OUT)
print(f"Saved dec_norms to {OUT}  shape={dec_norms.shape}  "
      f"min={dec_norms.min():.4f}  max={dec_norms.max():.4f}")