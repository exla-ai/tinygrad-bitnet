"""
Quick sanity‑check script for the BitNet tinygrad port.

It does the absolute minimum:
  • Loads the Hugging‑Face (PyTorch) and tinygrad models.
  • Compares a single row of the embedding matrix (BOS token).
  • Runs a 4‑token prompt through both models and compares the final‑step logits.

Prints max‑error and L2‑error so you can immediately see if / where the weights diverge.

Adjust `HF_PATH` below if your local snapshot lives somewhere else.
"""

import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tinygrad import Tensor, dtypes
from extra.models.bitnet import BitNetConfig, build_transformer

#####################################################################
# Paths — edit these two lines if your snapshot sits somewhere else  #
#####################################################################
HF_PATH = (
    "/Users/viraat/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/"
    "snapshots/5494d2858154ceb2b3854430366bf6d43d6ba5b5"
)
SAFETENSOR = os.path.join(HF_PATH, "model.safetensors")


def to_numpy(x):
    """Return **float32** NumPy array from torch / tinygrad / NumPy input."""
    # tinygrad Tensor → NumPy
    if isinstance(x, Tensor):
        if x.dtype == dtypes.bfloat16:
            x = x.cast(dtypes.float32)
        return x.numpy()

    # torch Tensor → NumPy
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()

    # already NumPy
    return np.asarray(x, dtype=np.float32)


def diff(a, b):
    """max‑error & L2 between two tensors (after ensuring same dtype)"""
    a, b = to_numpy(a), to_numpy(b)
    max_err = np.abs(a - b).max()
    l2_err = np.linalg.norm(a - b)
    return max_err, l2_err


def main():
    ########################################
    # 1. Load models                       #
    ########################################
    hf_model = AutoModelForCausalLM.from_pretrained(HF_PATH, torch_dtype=torch.float32)

    config = BitNetConfig()
    tg_model, raw_weights = build_transformer(Path(SAFETENSOR))

    ########################################
    # 2. Compare a single weight row       #
    ########################################
    tid = config.bos_token_id  # BOS = 128_000 for this checkpoint

    row_hf = hf_model.base_model.embed_tokens.weight[tid]
    row_raw = raw_weights["model.embed_tokens.weight"][tid]
    row_tg = tg_model.model.embed_tokens.weight[tid]

    print("\nEmbedding row check (token id %d):" % tid)
    print("  HF  vs raw  : max %.6e | L2 %.6e" % diff(row_hf, row_raw))
    print("  raw vs tiny : max %.6e | L2 %.6e" % diff(row_raw, row_tg))
    print("  HF  vs tiny : max %.6e | L2 %.6e" % diff(row_hf, row_tg))

    ########################################
    # 3. Quick forward‑pass sanity check   #
    ########################################
    prompt = [tid, 42, 123, 77]

    with torch.no_grad():
        logits_hf = hf_model(torch.tensor([prompt])).logits[0, -1]

    # temperature=0 → return (token, logprob, logits)
    _, _, tg_logits = tg_model(Tensor([prompt], dtype=dtypes.int64), None, 0.0)

    logits_max_err, _ = diff(logits_hf, tg_logits)
    print("\nLogits delta after prompt : %.6e" % logits_max_err)


if __name__ == "__main__":
    main()
