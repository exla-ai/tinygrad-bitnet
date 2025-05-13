import torch, transformers
from tinygrad import Tensor, dtypes
from extra.models.bitnet import BitNetForCausalLM, BitNetConfig, convert_from_huggingface, build_transformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from tinygrad.nn.state import safe_load, load_state_dict
import os
import numpy as np

# 1. load HF reference
PATH = "/Users/viraat/.cache/huggingface/hub/models--microsoft--bitnet-b1.58-2B-4T/snapshots/5494d2858154ceb2b3854430366bf6d43d6ba5b5"

SAFETENSOR = os.path.join(PATH, "model.safetensors")

# Path to local weights (needed for tinygrad loading

# Load tokenizer directly from the HF repo
tokenizer = AutoTokenizer.from_pretrained(PATH)
hf_model = AutoModelForCausalLM.from_pretrained(
    PATH,
    torch_dtype=torch.bfloat16
)

# 2. port weights into tinygrad
config = BitNetConfig()

tg_model, raw_weights = build_transformer(Path(SAFETENSOR))


tid = config.bos_token_id           # 128_000 for BitNet-B1.58-2B-4T

# 1) Hugging-Face reference
row_hf = (hf_model.base_model            # <-- only ONE ".model" level
                    .embed_tokens.weight[tid]
                    .detach().cpu())

# 2) raw tensor straight from *.safetensors*
# Convert BFloat16 tensor to float32 for NumPy compatibility
raw_tensor = raw_weights["model.embed_tokens.weight"]

if hasattr(raw_tensor, 'float'):
    raw_tensor = raw_tensor.float()
row_raw = raw_tensor.numpy()[tid]

# 3) tensor that lives inside tinygrad after load_state_dict
# Let's add a print here to see the type before .numpy() call
print(f"[DEBUG] tg_model.model.embed_tokens.weight dtype: {tg_model.model.embed_tokens.weight.dtype}")
row_tg = tg_model.model.embed_tokens.weight[tid]
if row_tg.dtype == dtypes.bfloat16:
    print("[DEBUG] Casting row_tg from bfloat16 to float32 before numpy conversion")
    row_tg = row_tg.cast(dtypes.float32)
row_tg = row_tg.numpy()

def diff(a, b):
    print(f"[DEBUG] diff input a type: {type(a)}, dtype: {getattr(a, 'dtype', 'N/A')}")
    print(f"[DEBUG] diff input b type: {type(b)}, dtype: {getattr(b, 'dtype', 'N/A')}")
    # Convert torch tensors to numpy if needed
    if hasattr(a, 'numpy'):
        print(f"[DEBUG] Converting a to numpy. Original dtype: {a.dtype}")
        if isinstance(a, Tensor) and a.dtype == dtypes.bfloat16:
             print("[DEBUG] Found tinygrad bfloat16 in a, casting to float32")
             a = a.cast(dtypes.float32).numpy()
        elif hasattr(a, 'detach') and hasattr(a, 'dtype') and a.dtype == torch.bfloat16:
             print("[DEBUG] Found torch bfloat16 in a, casting to float32")
             a = a.detach().cpu().float().numpy()
        elif hasattr(a, 'detach'):
             a = a.detach().cpu().numpy()
        else:
             a = a.numpy()
        print(f"[DEBUG] Converted a to numpy array, dtype: {a.dtype}")
    if hasattr(b, 'numpy'):
        print(f"[DEBUG] Converting b to numpy. Original dtype: {b.dtype}")
        if isinstance(b, Tensor) and b.dtype == dtypes.bfloat16:
             print("[DEBUG] Found tinygrad bfloat16 in b, casting to float32")
             b = b.cast(dtypes.float32).numpy()
        elif hasattr(b, 'detach') and hasattr(b, 'dtype') and b.dtype == torch.bfloat16:
             print("[DEBUG] Found torch bfloat16 in b, casting to float32")
             b = b.detach().cpu().float().numpy()
        elif hasattr(b, 'detach'):
             b = b.detach().cpu().numpy()
        else:
             b = b.numpy()
        print(f"[DEBUG] Converted b to numpy array, dtype: {b.dtype}")

    # Ensure both are numpy arrays with same dtype
    print("[DEBUG] Ensuring both arrays are float32 for comparison")
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    maxerr = np.max(np.abs(a - b))
    l2err  = np.sqrt(((a - b) ** 2).sum())
    print(f"[DEBUG] Calculated diff: maxerr={maxerr}, l2err={l2err}")
    return maxerr, l2err


w = raw_weights["model.embed_tokens.weight"]
print(w.shape, w.dtype)          # should say (128256, 2560)  bfloat16
print(w[0, :10])                 # eyeball a handful of numbers


print("HF  vs raw  : max = %.6f   L2 = %.6f" % diff(row_hf, row_raw))
print("raw vs tiny : max = %.6f   L2 = %.6f" % diff(row_raw, row_tg))
print("HF  vs tiny : max = %.6f   L2 = %.6f" % diff(row_hf, row_tg))


with torch.no_grad():
    # Make sure prompt_ids is defined before use
    prompt_ids = [config.bos_token_id, 42, 123, 77] # Example prompt_ids
    logits_hf = hf_model(torch.tensor([prompt_ids])).logits[0, -1]

# Pass temperature parameter to ensure we get 3 return values
print(f"[DEBUG] tg_model input dtype: {dtypes.int64}") # Debug: Check input type
tg_token, _, tg_logits = tg_model(Tensor([prompt_ids], dtype=dtypes.int64), None, 0.0)

print(f"[DEBUG] tg_logits dtype before numpy(): {tg_logits.dtype}")
if tg_logits.dtype == dtypes.bfloat16:
    print("[DEBUG] Casting tg_logits from bfloat16 to float32")
    tg_logits_np = tg_logits.cast(dtypes.float32).numpy()
else:
    tg_logits_np = tg_logits.numpy()

print(f"[DEBUG] logits_hf dtype before numpy(): {logits_hf.dtype}")
if logits_hf.dtype == torch.bfloat16:
    print("[DEBUG] Casting logits_hf from bfloat16 to float32")
    logits_hf_np = logits_hf.float().numpy()
else:
    logits_hf_np = logits_hf.numpy()

diff_val = abs(tg_logits_np - logits_hf_np).max()
print("max |Δ| =", diff_val)

def hf_hidden_states(ids):
    out = hf_model(torch.tensor([ids]), output_hidden_states=True)
    # out.hidden_states is a tuple: (embed, layer0, layer1, … final_norm)
    # Need to cast to float32 if they are bfloat16
    if out.hidden_states is None:
        print("[ERROR] hf_model did not return hidden_states. Ensure output_hidden_states=True was passed.")
        return []
    return [(h[0].detach().cpu().float().numpy() if h.dtype == torch.bfloat16 else h[0].detach().cpu().numpy()) for h in out.hidden_states]

def forward_debug(self, input_ids):
    states = []
    x = self.embed_tokens(input_ids)
    print(f"[DEBUG forward_debug] embed output dtype: {x.dtype}")
    states.append(x.cast(dtypes.float32).numpy() if x.dtype == dtypes.bfloat16 else x.numpy())

    batch, seq = input_ids.shape
    pos = Tensor.arange(seq, device=x.device)[None, :]
    cos, sin = self.rotary(x, pos)

    for i, layer in enumerate(self.layers):
        x = layer(x, cos, sin)
        print(f"[DEBUG forward_debug] layer {i} output dtype: {x.dtype}")
        states.append(x.cast(dtypes.float32).numpy() if x.dtype == dtypes.bfloat16 else x.numpy())

    x = self.norm(x)
    print(f"[DEBUG forward_debug] final_norm output dtype: {x.dtype}")
    states.append(x.cast(dtypes.float32).numpy() if x.dtype == dtypes.bfloat16 else x.numpy())
    return states


prompt = [config.bos_token_id, 42, 123, 77]

hf_states = hf_hidden_states(prompt)
tg_states = forward_debug(tg_model.model, Tensor([prompt], dtype=dtypes.int64))


for i, (h, t) in enumerate(zip(hf_states, tg_states)):
    delta = abs(h - t).max()
    print(f"layer {i:02d}   max|Δ| = {delta:.6f}")