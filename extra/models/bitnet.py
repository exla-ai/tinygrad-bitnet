from typing import Union, Optional, Any
import collections
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.helpers import getenv, DEBUG
import sys
import math
import struct
import numpy as np

# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return Tensor.stack(freqs.cos(), freqs.sin(), dim=-1).reshape(1, end, 1, dim//2, 2)

# matches meta, non hugging face weights
# (a+i*b) * (c+i*d) = (ac-bd) + i*(ad+bc)
def complex_mult(A, c, d):
  a,b = A[..., 0:1], A[..., 1:2]
  ro = a*c - b*d
  co = a*d + b*c
  return ro.cat(co, dim=-1)

def apply_rotary_emb(xq:Tensor, xk:Tensor, freqs_cis:Tensor) -> tuple[Tensor, Tensor]:
  assert freqs_cis.shape[1] == xq.shape[1] == xk.shape[1], f"freqs_cis shape mismatch {freqs_cis.shape} xq:{xq.shape} xk:{xk.shape}"
  xq = xq.reshape(*xq.shape[0:-1], -1, 2)
  xk = xk.reshape(*xk.shape[0:-1], -1, 2)
  assert len(xq.shape) == len(xk.shape) == len(freqs_cis.shape) == 5
  c, d = freqs_cis[..., 0:1], freqs_cis[..., 1:2]
  xq_out = complex_mult(xq, c, d)
  xk_out = complex_mult(xk, c, d)
  return xq_out.flatten(3), xk_out.flatten(3)

def repeat_kv(x:Tensor, n_rep:int) -> Tensor:
  bs, seqlen, n_kv_heads, head_dim = x.shape
  if n_rep == 1: return x
  # NOTE: this is different from x.repeat((1, 1, n_rep, 1))
  return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim)

class Attention:
  def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_context: int, linear: type = nn.BitLinear):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
    self.head_dim = dim // n_heads
    self.n_rep = self.n_heads // self.n_kv_heads
    self.max_context = max_context

    self.wq = linear(dim, self.n_heads * self.head_dim, bias=False)
    self.wk = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wv = linear(dim, self.n_kv_heads * self.head_dim, bias=False)
    self.wo = linear(self.n_heads * self.head_dim, dim, bias=False)

    self.cache_k = Tensor.zeros(1, self.max_context, self.n_kv_heads, self.head_dim)
    self.cache_v = Tensor.zeros(1, self.max_context, self.n_kv_heads, self.head_dim)

  def __call__(self, x: Tensor, start_pos: Union[Variable, int], freqs_cis: Tensor, mask: Optional[Tensor]) -> Tensor:
    if getenv("WQKV"):
      if not hasattr(self, 'wqkv'): self.wqkv = Tensor.cat(self.wq.weight, self.wk.weight, self.wv.weight)
      xqkv = x @ self.wqkv.T
      xq, xk, xv = xqkv.split([self.wq.weight.shape[0], self.wk.weight.shape[0], self.wv.weight.shape[0]], dim=2)
    else:
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    xq = xq.reshape(xq.shape[0], xq.shape[1], self.n_heads, self.head_dim)
    xk = xk.reshape(xk.shape[0], xk.shape[1], self.n_kv_heads, self.head_dim)
    xv = xv.reshape(xv.shape[0], xv.shape[1], self.n_kv_heads, self.head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
    bsz, seqlen, _, _ = xq.shape

    # create kv cache
    if not hasattr(self, "cache_kv"):
      self.cache_kv = Tensor.zeros(2, bsz, self.max_context, self.n_kv_heads, self.head_dim, dtype=x.dtype).contiguous().realize()
      if isinstance(x.device, tuple):
        # TODO: instead of specifying how to shard, it can follow how xk and xv are being sharded
        self.cache_kv.shard_((x.device), axis=3 if getenv("SHARD_KVCACHE") else None).realize()

    # update the cache
    assert xk.dtype == xv.dtype == self.cache_kv.dtype, f"{xk.dtype=}, {xv.dtype=}, {self.cache_kv.dtype=}"
    self.cache_kv.shrink((None, None, (start_pos, start_pos+seqlen), None, None)).assign(Tensor.stack(xk, xv)).realize()

    keys = self.cache_kv[0].shrink((None, (0, start_pos+seqlen), None, None))
    values = self.cache_kv[1].shrink((None, (0, start_pos+seqlen), None, None))

    keys, values = repeat_kv(keys, self.n_rep), repeat_kv(values, self.n_rep)
    xq, keys, values = xq.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2)
    attn = xq.scaled_dot_product_attention(keys, values, mask).transpose(1, 2)
    attn = attn.reshape(bsz, seqlen, -1)

    # Apply final projection layer wo using attention output
    return self.wo(attn)

class FeedForward:
  def __init__(self, dim: int, hidden_dim: int, linear: type = nn.BitLinear):
    self.w1 = linear(dim, hidden_dim, bias=False) # gate_proj
    self.w2 = linear(hidden_dim, dim, bias=False) # down_proj
    self.w3 = linear(dim, hidden_dim, bias=False) # up_proj

  def __call__(self, x: Tensor) -> Tensor:
    # Original BitNet 1.58: SwiGLU -> Linear
    # Apply SwiGLU activation using normalized input x
    swiglu_out = self.w1(x).silu() * self.w3(x)
    # Apply the final linear layer w2 using the intermediate result
    return self.w2(swiglu_out)

class TransformerBlock:
  def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_kv_heads: int, norm_eps: float, max_context: int, linear: type = nn.Linear,
               ffn_dim_multiplier=None):
    # Use LayerNorm, pass eps. Tinygrad LayerNorm doesn't have elementwise_affine, assumes True.
    self.attention_norm = nn.LayerNorm(dim, eps=norm_eps)
    self.ffn_norm = nn.LayerNorm(dim, eps=norm_eps)
    self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear=linear)
    self.feed_forward = FeedForward(dim, hidden_dim, linear=linear)

  def __call__(self, x: Tensor, start_pos: int, freqs_cis: Tensor, mask: Tensor) -> Tensor:
    # Apply attention_norm *before* attention
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    # Apply ffn_norm *before* feed_forward
    out = h + self.feed_forward(self.ffn_norm(h))
    return out

# standard openai sampling
def initialize_iq2s_grid():
    """
    Initialize the grid lookup table for I2_S quantization.
    This populates BitLinear.iq2s_grid_packed with values for dequantization.
    
    Based on the C++ implementation from llama.cpp/ggml-bitnet.cpp:
    - In C++, this is stored in a grid of values
    - For the BitNet I2_S format, we use direct mapping rather than grid lookups
    """
    from tinygrad.nn import BitLinear
    import numpy as np
    
    if BitLinear.iq2s_grid_packed is not None:
        return  # Already initialized
    
    # For I2_S (2-bit quantization), we're directly mapping:
    # 00 -> -1.0, 01 -> 0.0, 10 -> 1.0, 11 -> 0.0 (unused)
    # This is enough for our dequantization to work correctly
    BitLinear.iq2s_grid_packed = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    
    print(f"Initialized I2_S value mapping table: 00 -> -1.0, 01 -> 0.0, 10 -> 1.0, 11 -> 0.0 (unused)")
    print(f"Using simplified dequantization without grid lookups based on C++ implementation")


def sample(logits: Tensor, temp: float, k: int, p: float, af: float, ap: float):
  assert logits.ndim == 1, "only works on 1d tensors"
  assert 0 <= p <= 1, "p must be between 0 and 1"
  assert 0 <= k <= logits.numel(), "k must be between 0 and numel"

  # if temperature is very low just use argmax
  if temp < 1e-6: return logits.argmax()

  logits = logits.to(Device.DEFAULT)

  # alpha sampling
  if af or ap:
    if not hasattr(sample, "alpha_counter"):
      setattr(sample, "alpha_counter", Tensor.zeros_like(logits, dtype=dtypes.int32).contiguous())
    logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

  # replace NaNs with -inf
  logits = (logits != logits).where(-float("inf"), logits)

  # softmax
  t = (logits / temp).softmax()

  counter, counter2 = Tensor.arange(t.numel(), device=logits.device).contiguous(), Tensor.arange(t.numel() - 1, -1, -1, device=logits.device).contiguous()
  # top k
  if k:
    output, output_indices = Tensor.zeros(k, device=logits.device).contiguous(), Tensor.zeros(k, device=logits.device, dtype=dtypes.int32).contiguous()
    for i in range(k):
      t_argmax = (t.numel() - ((t == (t_max := t.max())) * counter2).max() - 1).cast(dtypes.default_int)
      output = output + t_max.unsqueeze(0).pad(((i, k - i - 1),))
      output_indices = output_indices + t_argmax.unsqueeze(0).pad(((i, k - i - 1),))
      t = (counter == t_argmax).where(0, t)

    # approximate top p
    # because we are already limited to top k elements we can do top p "without sorting"
    output_cumsum = output[::-1].cumsum()[::-1] + t.sum()
    output = (output_cumsum >= (1 - p)) * output
    output_indices = (output_cumsum >= (1 - p)) * output_indices

    # sample
    output_idx = output.multinomial()
    output_token = output_indices[output_idx]

    if DEBUG >= 2:
      print(f"  Sample[topk]: output_indices[:10] = {output_indices.numpy()[:10]}", file=sys.stderr)
      print(f"  Sample[topk]: output_idx (index into topk) = {output_idx.item()}", file=sys.stderr)
  else:
    output_token_raw = t.multinomial()

    if DEBUG >= 2:
      print(f"  Sample[no topk]: multinomial result = {output_token_raw.item()}", file=sys.stderr)

    # Clamp the result just in case
    output_token = output_token_raw.clip(0, t.shape[-1]-1)

    if DEBUG >= 2 and output_token.item() != output_token_raw.item():
      print(f"  Sample[no topk]: CLAMPED multinomial result to {output_token.item()}", file=sys.stderr)

  # increase alpha counter
  if af or ap:
    sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

  if DEBUG >= 1:
    print(f"  Sample function selected token ID: {output_token.item()}", file=sys.stderr)

  return output_token

class Transformer:
  def __init__(self, dim: int, hidden_dim: int, n_heads: int, n_layers: int, norm_eps: float, vocab_size: int, n_kv_heads=None,
               max_context=1024, jit=False, ffn_dim_multiplier=None, linear: type = nn.Linear, rope_theta: float = 10000.0,
               embedding: type = nn.Embedding):
    self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context, linear=linear,
                                    ffn_dim_multiplier=ffn_dim_multiplier) for _ in range(n_layers)] 
    self.norm = nn.LayerNorm(dim, eps=norm_eps)
    self.tok_embeddings = embedding(vocab_size, dim) 
    self.output = linear(dim, vocab_size, bias=False) 
    self.max_context = max_context
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, self.max_context * 2, rope_theta).contiguous().to(Device.DEFAULT)
    self.forward_jit = TinyJit(self.forward) if jit else None
    self.dim = dim
    self.n_kv_heads = n_kv_heads
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    
  
    

  def forward(self, tokens: Tensor, start_pos: Union[Variable, int], temperature: float, top_k: int, top_p: float, alpha_f: float, alpha_p: float):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)

    self.freqs_cis = self.freqs_cis.cast(h.dtype).realize()
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos+seqlen),None,None,None))

    mask = Tensor.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), dtype=h.dtype, device=h.device).triu(start_pos+1).realize() if seqlen > 1 else None
    for layer in self.layers: h = layer(h, start_pos, freqs_cis, mask)
    logits = self.output(self.norm(h)).float()[:, -1, :]
    if DEBUG >= 1:
      print(f"  Logits shape: {logits.shape}", file=sys.stderr)
      print(f"  Logits stats: min={logits.min().item():.2f}, max={logits.max().item():.2f}, mean={logits.mean().item():.2f}", file=sys.stderr)
      top_logits, top_indices = logits.flatten().topk(5)
      print(f"  Top 5 logits: {top_logits.numpy()}", file=sys.stderr)
      print(f"  Top 5 indices: {top_indices.numpy()}", file=sys.stderr)

    return sample(logits.flatten(), temperature, top_k, top_p, alpha_f, alpha_p).realize()

  def __call__(self, tokens: Tensor, start_pos: int, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.8, alpha_f: float = 0.0, alpha_p: float = 0.0):
    # TODO: better way to handle the first call v.s. the rest?
    if tokens.shape[0:2] == (1,1) and self.forward_jit is not None and start_pos != 0:
      return self.forward_jit(tokens, Variable("start_pos", 1, self.max_context).bind(start_pos), temperature, top_k, top_p, alpha_f, alpha_p)
    return self.forward(tokens, start_pos, temperature, top_k, top_p, alpha_f, alpha_p)

# *** helpers ***

import collections
from tinygrad import dtypes, Device, Tensor
from tinygrad.helpers import DEBUG

def convert_from_gguf(weights: dict[str, Tensor], model: Transformer) -> dict[str, Tensor]:
  """
  Convert GGUF tensor data to Tinygrad Transformer state dict.
    
  Args:
      weights: source weight dict from GGUF.
      model: instance of Tinygrad Transformer.
      
  Returns:
      sd: target state dict mapping Tinygrad param names to Tensors.
  """
  from tinygrad.nn import BitLinear
  import numpy as np
  from tinygrad.helpers import prod
    
  # Initialize the I2S grid lookup table
  if BitLinear.iq2s_grid_packed is None:
    print("Initializing IQ2_S grid lookup table...")
    initialize_iq2s_grid()
    
  # Expected dimensions directly from GGUF metadata, not calculated dimensions
  # The GGUF tensors are stored in [output_dim, input_dim] layout
  # Need to exactly match the dimensions in the GGUF file, then transpose if needed
  expected_shapes = {
    # Layer weights - matching actual GGUF dimensions
    "attention.wq.weight": (model.dim, model.dim),              # GGUF shows as [2560, 2560]
    "attention.wk.weight": (model.dim, 512),                   # GGUF shows as [2560, 512] 
    "attention.wv.weight": (model.dim, 512),                   # GGUF shows as [2560, 512]
    "attention.wo.weight": (model.dim, model.dim),              # GGUF shows as [2560, 2560]
    "feed_forward.w1.weight": (model.dim, model.hidden_dim),    # GGUF shows as [2560, 6912]
    "feed_forward.w2.weight": (model.hidden_dim, model.dim),    # GGUF shows as [6912, 2560]
    "feed_forward.w3.weight": (model.dim, model.hidden_dim),    # GGUF shows as [2560, 6912]
    # Embeddings
    "tok_embeddings.weight": (model.dim, model.vocab_size)      # GGUF shows as [2560, 128256]
  }
  
  # Which tensors need transposing after loading
  needs_transpose = {
    "attention.wk.weight": True,
    "attention.wv.weight": True,
    "feed_forward.w1.weight": True, 
    "feed_forward.w3.weight": True,
    "tok_embeddings.weight": True
  }
  
  # Map GGUF keys to Tinygrad keys
  keymap = {
    "token_embd.weight": "tok_embeddings.weight",
    **{f"blk.{l}.attn_norm.weight": f"layers.{l}.attention_norm.weight" for l in range(len(model.layers))},
    **{f"blk.{l}.attn_q.weight": f"layers.{l}.attention.wq.weight" for l in range(len(model.layers))},
    **{f"blk.{l}.attn_k.weight": f"layers.{l}.attention.wk.weight" for l in range(len(model.layers))},
    **{f"blk.{l}.attn_v.weight": f"layers.{l}.attention.wv.weight" for l in range(len(model.layers))},
    **{f"blk.{l}.attn_output.weight": f"layers.{l}.attention.wo.weight" for l in range(len(model.layers))},
    **{f"blk.{l}.ffn_norm.weight": f"layers.{l}.ffn_norm.weight" for l in range(len(model.layers))},
    **{f"blk.{l}.ffn_gate.weight": f"layers.{l}.feed_forward.w1.weight" for l in range(len(model.layers))},
    **{f"blk.{l}.ffn_down.weight": f"layers.{l}.feed_forward.w2.weight" for l in range(len(model.layers))},
    **{f"blk.{l}.ffn_up.weight": f"layers.{l}.feed_forward.w3.weight" for l in range(len(model.layers))},
    "output_norm.weight": "norm.weight",
  }
  
  sd = {}
  for k, v in weights.items():
    if k in keymap:
      target_key = keymap[k]
            
      # Handle I2_S quantized weights (returned as tuple from ggml_data_to_tensor)
      if isinstance(v, tuple) and len(v) == 2:
        # This is a raw bytes and shape tuple from I2_S format
        raw_bytes, shape = v
                
        # Find the target shape for this tensor
        expected_shape = None
        name_parts = target_key.split('.')
        if len(name_parts) >= 3:
          base_name = '.'.join(name_parts[2:]) if name_parts[0] == 'layers' else '.'.join(name_parts)
          for shape_name, shape_val in expected_shapes.items():
            if shape_name in base_name:
              expected_shape = shape_val
              break
                
        if expected_shape is None:
          raise ValueError(f"Unknown shape for {target_key}")
                
        print(f"Unpacking I2_S weights for {target_key} with shape {expected_shape}")
                
        # Use the unpack_i2_weights method to properly unpack the binary weights
        weight_tensor, scale_tensor = BitLinear.unpack_i2_weights(raw_bytes, expected_shape)
        
        # Transpose weight tensor if needed for correct matrix multiplication  
        base_name = None
        for shape_name in expected_shapes.keys():
          if shape_name in target_key:
            base_name = shape_name
            break
            
        # Apply transpose if this tensor type needs it
        if base_name and base_name in needs_transpose:
          print(f"Transposing {target_key} from shape {weight_tensor.shape}")
          weight_tensor = weight_tensor.transpose()
          print(f"  to shape {weight_tensor.shape}")
          
        # Store the weight tensor
        sd[target_key] = weight_tensor

        # If scale_tensor is a scalar but we need per-output-feature scales
        if scale_tensor.shape == (1,) and target_key.endswith('.weight'):
          # After potential transpose, the output features are the first dimension
          if base_name and base_name in needs_transpose:
            out_features = weight_tensor.shape[0]  # After transpose
          else:
            out_features = expected_shape[0]  # Original shape
          
          expanded_scale = scale_tensor.expand(out_features)
          sd[target_key.replace('.weight', '.scale')] = expanded_scale
        else:
          sd[target_key.replace('.weight', '.scale')] = scale_tensor
      else:
        # Regular tensor, just store it
        sd[target_key] = v
    
  # Handle output layer (use token embeddings for weight tying)
  if "tok_embeddings.weight" in sd:
    sd["output.weight"] = sd["tok_embeddings.weight"]
    if "tok_embeddings.scale" in sd:
      sd["output.scale"] = sd["tok_embeddings.scale"]
    
  return sd

def fix_bf16(weights:dict[Any, Tensor]):
    if getenv("SUPPORT_BF16", 1):
        # TODO: without casting to float16, 70B llama OOM on tinybox.
        return {k:v.cast(dtypes.float32).cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k,v in weights.items()}
    # TODO: check if device supports bf16
    return {k:v.llvm_bf16_cast(dtypes.half).to(v.device) if v.dtype == dtypes.bfloat16 else v for k,v in weights.items()}