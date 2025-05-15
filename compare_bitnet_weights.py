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
import sys
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tinygrad import Tensor, dtypes, Device
from extra.models.bitnet import BitNetConfig, build_transformer

# Force CPU for all operations to ensure consistent comparisons
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["METAL_DEVICE_WRAPPER_TYPE"] = "0"
os.environ["DISABLE_METAL"] = "1"
os.environ["DISABLE_CUDA"] = "1"
os.environ["DISABLE_OPENCL"] = "1"
os.environ["DISABLE_CLSPV"] = "1"

# Set tinygrad to use CPU explicitly before any imports
Device.DEFAULT = "CPU"

# Force CPU device for all tensors
torch.set_default_device("cpu")

# Reduce debug verbosity
os.environ["DEBUG_PRINT"] = "0"

# Suppress all debug logging
logging.basicConfig(level=logging.ERROR)
# Suppress torch warnings
torch.set_warn_always(False)
# Also redirect stderr temporarily to suppress debug prints
class StderrSuppressor:
    def __enter__(self):
        self.stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self.stderr


#####################################################################
# Paths — edit these two lines if your snapshot sits somewhere else  #
#####################################################################
HF_PATH = (
    "/Users/viraat/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/"
    "snapshots/5494d2858154ceb2b3854430366bf6d43d6ba5b5"
)
SAFETENSORS_PATH = os.path.join(HF_PATH, "model.safetensors")


def to_numpy(x):
    """Return **float32** NumPy array from torch / tinygrad / NumPy input."""
    # tinygrad Tensor → NumPy
    if isinstance(x, Tensor):
        if x.dtype == dtypes.bfloat16:
            x = x.cast(dtypes.float32)
        # Ensure CPU device before numpy conversion
        if x.device != "CPU":
            x = x.to("CPU")
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
    print("\nLoading models on CPU device...")
    # Suppress debug output during model loading
    with StderrSuppressor():
        # Explicitly load PyTorch model on CPU
        hf_model = AutoModelForCausalLM.from_pretrained(HF_PATH, torch_dtype=torch.float32, device_map="cpu")
        print(f"PyTorch model device: {next(hf_model.parameters()).device}")
        
        # Set tinygrad device to CPU before model creation
        print(f"Setting tinygrad default device to: {Device.DEFAULT}")
        config = BitNetConfig()
        tg_model, raw_weights = build_transformer(Path(SAFETENSORS_PATH))
    
    print("\n--- Models loaded successfully ---")

    ########################################
    # 2. Compare a single weight row       #
    ########################################
    tid = config.bos_token_id  # BOS = 128_000 for this checkpoint

    row_hf = hf_model.base_model.embed_tokens.weight[tid]
    row_raw = raw_weights["model.embed_tokens.weight"][tid]
    row_tg = tg_model.model.embed_tokens.weight[tid]
    
    # Print device information for debugging
    print(f"\nWeight device check:")
    print(f"  PyTorch embedding device: {row_hf.device}")
    print(f"  TinyGrad embedding device: {row_tg.device}")

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
    # Ensure the input tensor is on CPU
    input_tensor = Tensor([prompt], dtype=dtypes.int64, device="CPU")
    print(f"TinyGrad input tensor device: {input_tensor.device}")
    
    # Run inference and verify the device of outputs
    _, _, tg_logits = tg_model(input_tensor, None, 0.0)
    print(f"TinyGrad output logits device: {tg_logits.device}")

    # Ensure the logits are compared on the same device
    print(f"\nLogits device check:")
    print(f"  PyTorch logits device: {logits_hf.device if torch.is_tensor(logits_hf) else 'numpy'}")
    print(f"  TinyGrad logits device: {tg_logits.device}")
    
    # Compare logits and print error metrics
    logits_max_err, logits_l2_err = diff(logits_hf, tg_logits)
    print("\nLogits comparison:")
    print(f"  Max error: {logits_max_err:.6e}")
    print(f"  L2 error : {logits_l2_err:.6e}")
    print(f"  PyTorch shape: {logits_hf.shape if hasattr(logits_hf, 'shape') else 'N/A'}")
    print(f"  TinyGrad shape: {tg_logits.shape if hasattr(tg_logits, 'shape') else 'N/A'}")


if __name__ == "__main__":
    print("\n==== BitNet Weight Comparison (CPU Device) =====")
    # Report device configurations
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"TinyGrad default device: {Device.DEFAULT}")
    
    # Execute main function with debug output suppressed
    try:
        main()
        print("\n==== Comparison Complete =====")
    except Exception as e:
        print(f"\n==== Error during comparison: {e} =====")
