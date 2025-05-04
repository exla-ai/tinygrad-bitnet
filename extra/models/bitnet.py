from typing import Union, Optional, Any
import collections
from tinygrad import Tensor, Variable, TinyJit, dtypes, nn, Device
from tinygrad.nn import BitLinear
from tinygrad.helpers import getenv, DEBUG
from tinygrad import dtypes # Ensure dtypes is imported if used
import sys
import numpy as np

# https://github.com/facebookresearch/llama/blob/1076b9c51c77ad06e9d7ba8a4c6df775741732bd/llama/model.py#L47
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, dtype=dtypes.bfloat16, device=None) -> Tensor:
  # Create initial tensors with the specified device
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2, dtype=dtypes.float32, device=device)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end, device=device).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  # All intermediate operations will preserve the device
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
  # Preserve the device of the input tensor
  device = x.device
  # NOTE: this is different from x.repeat((1, 1, n_rep, 1))
  # Ensure the result is on the same device as the input
  return x.repeat((1, 1, 1, n_rep)).reshape(bs, seqlen, n_kv_heads * n_rep, head_dim).to(device).realize()

class Attention:
  def __init__(self, dim, n_heads, n_kv_heads, max_context, linear=BitLinear, qk_norm:float|None=None):
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads # n_kv_heads != n_heads implies MQA [arxiv/2307.09288, A.2.1]
    self.head_dim = dim // n_heads # 2560 / 20 = 128
    self.n_rep = self.n_heads // self.n_kv_heads
    self.max_context = max_context

    self.wq = linear(dim, self.n_heads * self.head_dim) # 2560 -> 2560
    self.wk = linear(dim, self.n_kv_heads * self.head_dim) # 2560 -> 640
    self.wv = linear(dim, self.n_kv_heads * self.head_dim) # 2560 -> 640
    self.wo = linear(self.n_heads * self.head_dim, dim) 

    print("WQ:", self.wq.weight.shape)
    print("WK:", self.wk.weight.shape)
    print("WV:", self.wv.weight.shape)
    print("WO:", self.wo.weight.shape)

    self.q_norm = nn.RMSNorm(dim, qk_norm) if qk_norm is not None else None
    self.k_norm = nn.RMSNorm(dim, qk_norm) if qk_norm is not None else None

  def __call__(self, x:Tensor, start_pos:Union[Variable,int], freqs_cis:Tensor, mask:Optional[Tensor]) -> Tensor:
    if getenv("WQKV"):
      if not hasattr(self, 'wqkv'): self.wqkv = Tensor.cat(self.wq.weight, self.wk.weight, self.wv.weight)
      xqkv = x @ self.wqkv.T
      xq, xk, xv = xqkv.split([self.wq.weight.shape[0], self.wk.weight.shape[0], self.wv.weight.shape[0]], dim=2)
    else:
      xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

    if self.q_norm is not None and self.k_norm is not None:
      xq = self.q_norm(xq)
      xk = self.k_norm(xk)

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
    return self.wo(attn)

class FeedForward:
  def __init__(self, dim:int, hidden_dim:int, linear=BitLinear):
    self.w1 = linear(dim, hidden_dim)
    self.w2 = linear(hidden_dim, dim)
    self.w3 = linear(dim, hidden_dim) # the gate in Gated Linear Unit

  def __call__(self, x:Tensor) -> Tensor:
    gate_activated = self.w1(x).relu().square()
    gated_result = gate_activated * self.w3(x)
    return self.w2(gated_result)


class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, max_context:int, linear=BitLinear,
               feed_forward=FeedForward, qk_norm=None):
    self.attention = Attention(dim, n_heads, n_kv_heads, max_context, linear, qk_norm)
    self.feed_forward = feed_forward(dim, hidden_dim, linear)
    self.attention_norm = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm = nn.RMSNorm(dim, norm_eps)

  def __call__(self, x:Tensor, start_pos:Union[Variable,int], freqs_cis:Tensor, mask:Optional[Tensor]):
    h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
    return (h + self.feed_forward(self.ffn_norm(h))).contiguous()

# standard openai sampling
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
  else:
    output_token = t.multinomial()

  # increase alpha counter
  if af or ap:
    sample.alpha_counter = (counter == output_token).where(sample.alpha_counter + 1, sample.alpha_counter)

  return output_token


class Transformer:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_layers:int, norm_eps:float, vocab_size, linear=BitLinear, embedding=nn.Embedding,
               n_kv_heads=None, rope_theta=10000, max_context=1024, jit=True, feed_forward=FeedForward, qk_norm=None):
    self.layers = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, max_context, linear, feed_forward=feed_forward, qk_norm=qk_norm) for _ in range(n_layers)]
    self.norm = nn.RMSNorm(dim, norm_eps)
    self.tok_embeddings = embedding(vocab_size, dim)
    self.output = linear(dim, vocab_size)
    self.max_context = max_context
    self.freqs_cis = precompute_freqs_cis(dim // n_heads, self.max_context * 2, rope_theta).contiguous()
    self.forward_jit = TinyJit(self.forward) if jit else None

  def forward(self, tokens:Tensor, start_pos:Union[Variable,int], temperature:float, top_k:int, top_p:float, alpha_f:float, alpha_p:float):
    _bsz, seqlen = tokens.shape
    h = self.tok_embeddings(tokens)

    self.freqs_cis = self.freqs_cis.cast(h.dtype).realize()
    freqs_cis = self.freqs_cis.shrink((None, (start_pos, start_pos+seqlen),None,None,None))

    mask = Tensor.full((1, 1, seqlen, start_pos+seqlen), float("-inf"), dtype=h.dtype, device=h.device).triu(start_pos+1).realize() if seqlen > 1 else None
    for layer in self.layers: h = layer(h, start_pos, freqs_cis, mask)
    logits = self.output(self.norm(h)).float()[:, -1, :]

    return sample(logits.flatten(), temperature, top_k, top_p, alpha_f, alpha_p).realize()

  def __call__(self, tokens:Tensor, start_pos:int, temperature:float=0.0, top_k:int=0, top_p:float=0.8, alpha_f:float=0.0, alpha_p:float=0.0):
    # TODO: better way to handle the first call v.s. the rest?
    if tokens.shape[0:2] == (1,1) and self.forward_jit is not None and start_pos != 0:
      return self.forward_jit(tokens, Variable("start_pos", 1, self.max_context).bind(start_pos), temperature, top_k, top_p, alpha_f, alpha_p)
    return self.forward(tokens, start_pos, temperature, top_k, top_p, alpha_f, alpha_p)

def convert_from_huggingface(weights:dict[str, Tensor], model: Transformer, n_heads: int, n_kv_heads: int, permute_layers: bool = True):
  # huggingface stores Q and K permuted! it is mostly correct without this, but without it makes RoPE different, so it will diverge after 10+ toks.
  def permute(v: Tensor, n_heads: int):
    return v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, v.shape[1] if len(v.shape) > 1 else 1).transpose(1, 2).reshape(*v.shape[:2])

  num_layers = len(model.layers) # Keep this to define the keymap range

  keymap = {
      # embeddings
      "model.embed_tokens.weight": "tok_embeddings.weight",
      # --- per-layer attention sub-module ---
       **{
        f"model.layers.{l}.input_layernorm.weight":
        f"layers.{l}.input_layernorm.weight"
        for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.q_proj.weight":
          f"layers.{l}.self_attn.q_proj.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.q_proj.weight_scale":
          f"layers.{l}.self_attn.q_proj.weight_scale"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.k_proj.weight":
          f"layers.{l}.self_attn.k_proj.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.k_proj.weight_scale":
          f"layers.{l}.self_attn.k_proj.weight_scale"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.v_proj.weight":
          f"layers.{l}.self_attn.v_proj.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.v_proj.weight_scale":
          f"layers.{l}.self_attn.v_proj.weight_scale"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.o_proj.weight":
          f"layers.{l}.self_attn.o_proj.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.o_proj.weight_scale":
          f"layers.{l}.self_attn.o_proj.weight_scale"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.mlp.gate_proj.weight":
          f"layers.{l}.mlp.gate_proj.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.mlp.gate_proj.weight_scale":
          f"layers.{l}.mlp.gate_proj.weight_scale"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.mlp.down_proj.weight":
          f"layers.{l}.mlp.down_proj.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.mlp.down_proj.weight_scale":
          f"layers.{l}.mlp.down_proj.weight_scale"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.mlp.up_proj.weight":
          f"layers.{l}.mlp.up_proj.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.mlp.up_proj.weight_scale":
          f"layers.{l}.mlp.up_proj.weight_scale"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.post_attention_layernorm.weight":
          f"layers.{l}.post_attention_layernorm.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.mlp.ffn_sub_norm.weight":
          f"layers.{l}.mlp.ffn_sub_norm.weight"
          for l in range(num_layers)
      },
      **{
          f"model.layers.{l}.self_attn.attn_sub_norm.weight":
          f"layers.{l}.self_attn.attn_sub_norm.weight"
          for l in range(num_layers)
      },
      "model.norm.weight": "norm.weight",
  }

  sd = {}

  for k, v in weights.items():
    if ".rotary_emb." in k: continue
    v = v.to(Device.DEFAULT)
    if "model.layers" in k:
      if "q_proj" in k and permute_layers and v.shape[0] % (n_heads * 2) == 0:
        v = permute(v, n_heads)
      elif "k_proj" in k and permute_layers and v.shape[0] % (n_kv_heads * 2) == 0:
        v = permute(v, n_kv_heads)

    sd[keymap[k]] = v
  return sd

  

def fix_bf16(weights:dict[Any, Tensor]):
  if getenv("SUPPORT_BF16", 1):
    # TODO: without casting to float16, 70B llama OOM on tinybox.
    return {k:v.cast(dtypes.float32).cast(dtypes.float16) if v.dtype == dtypes.bfloat16 else v for k,v in weights.items()}
  # TODO: check if device supports bf16
  return {k:v.llvm_bf16_cast(dtypes.half).to(v.device) if v.dtype == dtypes.bfloat16 else v for k,v in weights.items()}
