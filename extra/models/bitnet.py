import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tinygrad import Tensor, Device, dtypes
from tinygrad.nn import Embedding, Linear  # Linear only for lm_head
from tinygrad.nn.state import load_state_dict, get_state_dict
from tinygrad.helpers import getenv, DEBUG

# Global debug flag to control detailed output
DEBUG_PRINT = getenv("DEBUG_PRINT", 0) == 1

# ────────────────────────────────────────────────────────────
# Debug utilities
# ────────────────────────────────────────────────────────────
DEBUG_PRINT = True

def debug(msg: str) -> None:
    if DEBUG_PRINT:
        print(f"[DEBUG] {msg}")

# ────────────────────────────────────────────────────────────
# Quantisation helpers (4‑bit packed)
# ────────────────────────────────────────────────────────────
VALUES_PER_ITEM = 4  # 2‑bit per value → 4 vals in a uint8


def rotate_half(t: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input."""
    # Match HF implementation exactly
    x1 = t[..., : t.shape[-1] // 2]
    x2 = t[..., t.shape[-1] // 2 :]
    return (-x2).cat(x1, dim=-1)


class WeightQuant:
    """Grok‑simple symmetric 1‑bit quant (for experimentation)."""

    @staticmethod
    def forward(weight: Tensor) -> Tensor:
        debug(f"WeightQuant.forward: weight.shape={weight.shape}")
        w = weight.float()
        scale = 1.0 / w.abs().mean().clamp(min=1e-5)
        q = (w * scale).round().clamp(-1, 1) / scale
        return q.cast(weight.dtype)


class ActQuant:
    """Per‑token dynamic int8 quantisation."""

    @staticmethod
    def forward(x: Tensor) -> Tuple[Tensor, Tensor]:
        debug(f"ActQuant: x.shape={x.shape}, x.dtype={x.dtype}")
        Qn, Qp = -(2 ** 7), 2 ** 7 - 1
        debug(f"ActQuant: Quantization range: [{Qn}, {Qp}]")
        
        # Calculate max for scaling
        max_abs = x.abs().max(axis=-1, keepdim=True).clamp(min_=1e-5)
        debug(f"ActQuant: max_abs.shape={max_abs.shape}, min={max_abs.min().item() if max_abs.numel() > 0 else 'N/A'}, max={max_abs.max().item() if max_abs.numel() > 0 else 'N/A'}")
        
        # Scale calculation
        scale = Qp / max_abs
        debug(f"ActQuant: scale.shape={scale.shape}, min={scale.min().item() if scale.numel() > 0 else 'N/A'}, max={scale.max().item() if scale.numel() > 0 else 'N/A'}")
        
        # Quantization
        q = (x * scale).round().clamp(Qn, Qp)
        debug(f"ActQuant: q.shape={q.shape} (before cast), min={q.min().item() if q.numel() > 0 else 'N/A'}, max={q.max().item() if q.numel() > 0 else 'N/A'}")
        
        # Cast to int8 and return
        q_int8 = q.cast(dtypes.int8)
        debug(f"ActQuant: returning q_int8.dtype={q_int8.dtype}, scale.dtype={scale.dtype}")
        
        return q_int8, scale


# ────────────────────────────────────────────────────────────
# 2‑bit weight pack / unpack helpers
# ────────────────────────────────────────────────────────────

def unpack_weights(packed: Tensor, dtype) -> Tensor:
    """Expand 2‑bit packed weights to full uint8 then cast."""
    debug(f"unpack_weights: packed.shape={packed.shape}, packed.dtype={packed.dtype}, target_dtype={dtype}")
    p0, *prest = packed.shape
    out_shape = (p0 * VALUES_PER_ITEM, *prest) if prest else (p0 * VALUES_PER_ITEM,)
    debug(f"unpack_weights: out_shape={out_shape}, VALUES_PER_ITEM={VALUES_PER_ITEM}")

    unpacked = Tensor.zeros(out_shape, dtype=dtypes.uint8, device=packed.device).contiguous().realize()
    for i in range(VALUES_PER_ITEM):
        chunk = ((packed & (3 << (2 * i))) >> (2 * i)).cast(dtypes.uint8)
        debug(f"unpack_weights: chunk[{i}] shape={chunk.shape}, min={chunk.min().item() if chunk.numel() > 0 else 'N/A'}, max={chunk.max().item() if chunk.numel() > 0 else 'N/A'}")
        unpacked[i * p0 : (i + 1) * p0] = chunk
    result = unpacked.cast(dtype) - 1  # map {0,1,2,3} → {‑1,0,1,2}
    debug(f"unpack_weights: unpacked result shape={result.shape}, dtype={result.dtype}, min={result.min().item() if result.numel() > 0 else 'N/A'}, max={result.max().item() if result.numel() > 0 else 'N/A'}")
    if result.numel() > 0 and DEBUG_PRINT:
        # Sample some values for inspection
        sample_idx = min(100, result.numel()-1)
        debug(f"unpack_weights: sample values[0:3]={result.flatten()[:3].numpy().tolist() if result.numel() >= 3 else 'too small'}, value[{sample_idx}]={result.flatten()[sample_idx].item()}")
    return result


# ────────────────────────────────────────────────────────────
# Token sampling
# ────────────────────────────────────────────────────────────

def sample(
    logits: Tensor,
    temp: float = 0.0,
    k: int = 0,
    p: float = 0.0,
    af: float = 0.0,
    ap: float = 0.0,
):
    """Return **int** token id chosen from `logits`."""

    debug(
        f"sample: logits.shape={logits.shape}, temp={temp}, k={k}, p={p}, af={af}, ap={ap}"
    )
    assert logits.ndim == 1, "logits must be 1‑D (vocab)"

    # Greedy / argmax path
    if temp < 1e-6:
        return int(logits.argmax().item())

    if af or ap:
        if not hasattr(sample, "alpha_counter"):
            sample.alpha_counter = Tensor.zeros_like(logits, dtype=dtypes.int32)
        logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

    # mask NaNs
    logits = (logits != logits).where(-float("inf"), logits)
    probs = (logits / temp).softmax()

    if k:
        values, indices = probs.topk(k)
        cum = values[::-1].cumsum()[::-1]
        mask = cum >= (1 - p)
        values = values * mask
        indices = indices * mask
        choice = int(values.multinomial().item())
        token = int(indices[choice].item())
    else:
        token = int(probs.multinomial().item())

    if af or ap:
        counter = Tensor.arange(probs.numel(), device=logits.device)
        sample.alpha_counter = (counter == token).where(
            sample.alpha_counter + 1, sample.alpha_counter
        )

    debug(f"sample: token={token}")
    return token


# ────────────────────────────────────────────────────────────
# Config & Tiny‑grad model definition
# ────────────────────────────────────────────────────────────
class BitNetConfig:
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_attention_heads: int = 20
    num_key_value_heads: int = 5
    num_hidden_layers: int = 30
    rms_norm_eps: float = 1e-05  # Exact match with HF config
    vocab_size: int = 128_256
    max_position_embeddings: int = 4096
    hidden_act: str = "relu2"
    initializer_range: float = 0.02
    rope_theta: float = 500_000.0
    bos_token_id: int = 128_000
    eos_token_id: int = 128_001
    pad_token_id: int = 0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"  # Added from HF config
    use_cache: bool = True  # Added from HF config
    # Quantization config details
    quant_method: str = "bitnet"
    linear_class: str = "autobitlinear"
    quantization_mode: str = "offline"

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads


# ────────────────────────────────────────────────────────────
# Low‑level building blocks
# ────────────────────────────────────────────────────────────
class BitLinear:
    """Linear layer with 2‑bit packed weights & int8 activations."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        device=None,
        dtype=dtypes.float32,
    ):
        debug(f"BitLinear.__init__: in={in_features}, out={out_features}, bias={bias}")
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device or Device.DEFAULT

        self.weight = Tensor.zeros(
            (out_features // VALUES_PER_ITEM, in_features), dtype=dtypes.uint8, device=self.device
        )
        
        self.weight_scale = Tensor.ones((1,), dtype=self.dtype, device=self.device)

        self.bias = (
            Tensor.zeros((out_features,), dtype=self.dtype, device=self.device) if bias else None
        )

    def __call__(self, x: Tensor) -> Tensor:
        # If weights are packed (uint8), follow ternary quantised path,
        # otherwise fall back to a standard float matmul for HF fp weights.
        debug(f"BitLinear.__call__: input x.shape={x.shape}, x.dtype={x.dtype}, weight.shape={self.weight.shape}, weight.dtype={self.weight.dtype}")
        debug(f"BitLinear.__call__: weight_scale={self.weight_scale.item() if self.weight_scale.numel()==1 else 'multiple'}")
        
        # Always quantize activations for BitLinear, matching HF's approach
        x_q, x_scale = ActQuant.forward(x)  # int8, scale shape [B,1,1]
        debug(f"BitLinear.__call__: quantized x_q.shape={x_q.shape}, x_q.dtype={x_q.dtype}, x_scale.shape={x_scale.shape}")
        
        if self.weight.dtype == dtypes.uint8:
            debug(f"BitLinear.__call__: USING QUANTIZED PATH for {self.out_features}x{self.in_features} layer")
            # Unpack the weights from uint8 to full tensors with -1, 0, 1 values
            w_full = unpack_weights(self.weight, dtype=self.dtype)
            debug(f"BitLinear.__call__: unpacked w_full.shape={w_full.shape}, w_full.dtype={w_full.dtype}")
            w = w_full * self.weight_scale
        else:
            # If weights are not in uint8 format, treat them as regular weights with scaling
            # In BitNet, we should ideally quantize these to match HF's implementation
            w = self.weight * self.weight_scale
            debug(f"BitLinear.__call__: using standard weight path, w.shape={w.shape}, w.min={w.min().item():.6f}, w.max={w.max().item():.6f}")
        
        # Perform matrix multiplication using quantized activations
        out = x_q.dot(w.T)
        debug(f"BitLinear.__call__: dot product shape={out.shape}")
        
        # Rescale outputs using the activation scale
        out = out * x_scale
        debug(f"BitLinear.__call__: after x_scale, out.shape={out.shape}, out.min={out.min().item():.6f}, out.max={out.max().item():.6f}")
        
        if self.bias is not None:
            out = out + self.bias
            
        return out


class BitNetRMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = Tensor.ones((dim,), device=Device.DEFAULT)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        # Match HF's BitNetRMSNorm.forward exactly
        input_dtype = x.dtype
        # Always compute in float32 for numerical stability
        hidden_states = x.cast(dtypes.float32)
        # Use power instead of ** operator, matching HF's pow(2)
        variance = hidden_states.pow(2).mean(axis=-1, keepdim=True)
        hidden_states = hidden_states * Tensor.rsqrt(variance + self.eps)
        # Apply weight and cast back to input dtype
        return (self.weight * hidden_states).cast(input_dtype)


class BitNetMLP:
    def __init__(self, config: BitNetConfig):
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, bias=False)
        # Add ffn_sub_norm like in HF implementation
        self.ffn_sub_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)
        # Match HF's activation function property - we're using relu^2 which is ACT2FN['relu2'] in HF
        # This makes it explicit and aligns with HF naming convention
        self.act_fn = lambda x: x.relu().square()

    def __call__(self, x: Tensor) -> Tensor:
        # Add detailed debug logging to track numerical precision
        gate_out = self.gate_proj(x)
        activated_gate = self.act_fn(gate_out)
        up_out = self.up_proj(x)
        gate_up_product = activated_gate * up_out
        normed_product = self.ffn_sub_norm(gate_up_product)
        # Debug stats for critical tensors

        debug(f"BitNetMLP forward: shapes: gate={gate_out.shape} up={up_out.shape} product={gate_up_product.shape}")
        debug(f"BitNetMLP forward: gate stats: min={gate_out.min().item():.6f}, max={gate_out.max().item():.6f}")
        debug(f"BitNetMLP forward: activated stats: min={activated_gate.min().item():.6f}, max={activated_gate.max().item():.6f}")
        debug(f"BitNetMLP forward: up stats: min={up_out.min().item():.6f}, max={up_out.max().item():.6f}")
        debug(f"BitNetMLP forward: product stats: min={gate_up_product.min().item():.6f}, max={gate_up_product.max().item():.6f}")
        debug(f"BitNetMLP forward: normed stats: min={normed_product.min().item():.6f}, max={normed_product.max().item():.6f}")
        
        # Matching HF's exact forward: down_proj(ffn_sub_norm(act_fn(gate_proj) * up_proj))
        return self.down_proj(normed_product)


class BitNetRotaryEmbedding:
    def __init__(self, config: BitNetConfig):
        inv = Tensor.arange(0, config.head_dim, 2, dtype=dtypes.float32)
        self.inv_freq = 1.0 / (config.rope_theta ** (inv / config.head_dim))

    def __call__(self, x: Tensor, pos_ids: Tensor):
        freqs = pos_ids.unsqueeze(-1).float() * self.inv_freq
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return cos.cast(x.dtype), sin.cast(x.dtype)


class BitNetAttention:
    def __init__(self, config: BitNetConfig):
        self.nh = config.num_attention_heads
        self.nkv = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5

        self.q_proj = BitLinear(config.hidden_size, self.nh * self.head_dim, bias=False)
        self.k_proj = BitLinear(config.hidden_size, self.nkv * self.head_dim, bias=False)
        self.v_proj = BitLinear(config.hidden_size, self.nkv * self.head_dim, bias=False)
        self.o_proj = BitLinear(self.nh * self.head_dim, config.hidden_size, bias=False)
        # Add attention sub-norm just like HF implementation
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # Direct translation of HF's apply_rotary_pos_emb with unsqueeze_dim=1
        cos = cos.unsqueeze(1) # Matches unsqueeze_dim=1 in HF implementation
        sin = sin.unsqueeze(1)
        # Exactly matches HF implementation: (q * cos) + (rotate_half(q) * sin)
        return (x * cos) + (rotate_half(x) * sin)

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor):
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.nh, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, s, self.nkv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, s, self.nkv, self.head_dim).transpose(1, 2)

        repeat = self.nh // self.nkv
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-1, -2)) * self.scale
        mask = Tensor.tril(Tensor.ones(attn.shape[-2:], device=attn.device))
        attn = attn.where(mask, Tensor.full_like(attn, -1e9))
        probs = attn.softmax()

        out = (probs @ v).transpose(1, 2).reshape(b, s, -1)
        # Apply attention sub-norm before the output projection, matching HF implementation
        out = self.attn_sub_norm(out)
        return self.o_proj(out)


class BitNetDecoderLayer:
    def __init__(self, config: BitNetConfig):
        self.input_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = BitNetAttention(config)
        self.post_attention_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BitNetMLP(config)

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor):
        # Explicit residual connections matching HF implementation
        residual = x
        hidden_states = self.input_layernorm(x)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states
        
        # Second residual block for MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class BitNetModel:
    def __init__(self, config: BitNetConfig):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BitNetDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary = BitNetRotaryEmbedding(config)

    def __call__(self, input_ids: Tensor, past=None):
        x = self.embed_tokens(input_ids)
        batch, seq = input_ids.shape
        pos = Tensor.arange(seq, device=x.device)[None, :].expand(batch, -1)
        cos, sin = self.rotary(x, pos)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class BitNetForCausalLM:
    def __init__(self, config: BitNetConfig):
        self.config = config
        self.model = BitNetModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids: Tensor, past=None, *sample_args):
        hidden_states = self.model(input_ids, past)
        logits = self.lm_head(hidden_states[:, -1, :].cast(dtypes.float32))[0]
        if sample_args:
            token = sample(logits, *sample_args)
            return token, past, logits 
        else:
            return logits, past


# ────────────────────────────────────────────────────────────
# HF → Tiny‑grad weight converter
# ────────────────────────────────────────────────────────────

def _permute_qkv(v: Tensor, n_heads: int):
    # (F, I) where F = out_features
    return (
        v.reshape(n_heads, 2, v.shape[0] // n_heads // 2, -1)
        .transpose(1, 2)
        .reshape(*v.shape[:2])
    )

def convert_from_huggingface(
    raw: Dict[str, Tensor],
    config: BitNetConfig,
) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    # No keymap needed anymore if model names align with HF names after prefix stripping
    debug(f"convert_from_huggingface: Processing {len(raw)} keys")

    for hf_key, hf_tensor in raw.items():
        debug(f"Processing HF key for direct loading: {hf_key}, shape={hf_tensor.shape}, dtype={hf_tensor.dtype}")
        v = hf_tensor.to(Device.DEFAULT)
        
        if v.dtype == dtypes.bfloat16:
            # Cast to float32 for tinygrad operations if bf16 is not fully supported on backend
            debug(f"  Converting {hf_key} from bfloat16 to float32")
            v = Tensor(v.cast(dtypes.float32).numpy(), dtype=dtypes.float32)
        
        debug(f"  After device transfer: {v.shape}, {v.dtype}")

        # Check if weight needs to be quantized or already is
        if 'weight' in hf_key and v.dtype in [dtypes.float32, dtypes.float16, dtypes.bfloat16] and 'lm_head' not in hf_key:
            debug(f"  Weight tensor found: {hf_key} - checking if it should be packed/quantized")
            if v.dtype != dtypes.uint8:
                debug(f"  Weight is in floating point format ({v.dtype}), not currently quantized")

        # Permute Q/K weights based on original HF key name
        if hf_key.endswith("self_attn.q_proj.weight"):
            debug(f"  Permuting Q weights for {hf_key}, before shape={v.shape}")
            v = _permute_qkv(v, config.num_attention_heads)
            debug(f"  After permutation: shape={v.shape}")
        elif hf_key.endswith("self_attn.k_proj.weight"):
            debug(f"  Permuting K weights for {hf_key}, before shape={v.shape}")
            v = _permute_qkv(v, config.num_key_value_heads)
            debug(f"  After permutation: shape={v.shape}")
        
        out[hf_key] = v # Use original HF key

    debug(f"convert_from_huggingface: Finished processing, returning {len(out)} keys")
    return out



# ────────────────────────────────────────────────────────────
# Convenience loader
# ────────────────────────────────────────────────────────────

def build_transformer(model_path: Path, load_weights: bool = True):
    debug(f"build_transformer: Creating model with path {model_path}, load_weights={load_weights}")
    config = BitNetConfig()
    net = BitNetForCausalLM(config)
    debug(f"build_transformer: Created BitNetForCausalLM instance")

    if not load_weights:
        debug(f"build_transformer: Skipping weight loading as requested")
        return net, None

    sf_path = (
        model_path
        if model_path.is_file()
        else model_path / "model.safetensors"
    )
    debug(f"build_transformer: SafeTensors path: {sf_path}")
    assert sf_path.exists(), f"weights not found at {sf_path}"

    from tinygrad.nn.state import safe_load
    debug(f"build_transformer: Loading weights from {sf_path}")

    raw = safe_load(str(sf_path))
    debug(f"build_transformer: Loaded {len(raw)} raw tensors from safetensors file")
    
    # Print key shapes/dtypes for inspection
    if DEBUG_PRINT:
        for k, v in list(raw.items())[:5]:  # Show first 5 for brevity
            debug(f"  Raw tensor: {k}, shape={v.shape}, dtype={v.dtype}")
    
    debug(f"build_transformer: Converting weights from huggingface format")
    weights = convert_from_huggingface(raw, config)
    debug(f"build_transformer: Converted {len(weights)} tensors")

    # Use consume_prefix to strip "model." and load into net.model
    # strict=False will report missing/unexpected keys via consume_prefix's behavior
    debug(f"build_transformer: Loading state dict into model")
    # Check for weight format/dtype mismatches
    if DEBUG_PRINT:
        # Sample a MLP gate projection weight to check if format matches
        if "model.layers.0.mlp.gate_proj.weight" in weights:
            debug(f"Gate projection weight dtype: {weights['model.layers.0.mlp.gate_proj.weight'].dtype}")
            debug(f"Gate projection weight shape: {weights['model.layers.0.mlp.gate_proj.weight'].shape}")
            gate_weight = weights['model.layers.0.mlp.gate_proj.weight']
            debug(f"Weight stats: min={gate_weight.min().item():.6f}, max={gate_weight.max().item():.6f}, mean={gate_weight.mean().item():.6f}")
    
    load_state_dict(net, weights, strict=False) 
    debug(f"build_transformer: State dict loaded")

    # Check if any layers have uint8 weights (should be quantized)
    if DEBUG_PRINT:
        has_uint8 = False
        linear_count = 0
        float_count = 0
        # Examine BitLinear weights specifically
        for name, module in net.__dict__.items():
            if hasattr(module, 'weight') and hasattr(module.weight, 'dtype'):
                dtype_str = str(module.weight.dtype)
                debug(f"  Module weight dtype check: {name}.weight.dtype = {dtype_str}")
                if module.weight.dtype == dtypes.uint8:
                    has_uint8 = True
                    linear_count += 1
                elif module.weight.dtype in [dtypes.float32, dtypes.float16, dtypes.bfloat16]:
                    float_count += 1
        
        # Also check layers inside model
        if hasattr(net, 'model') and hasattr(net.model, 'layers'):
            for i, layer in enumerate(net.model.layers[:2]):  # Just check first two layers
                debug(f"Layer {i} self_attn.q_proj weight dtype: {layer.self_attn.q_proj.weight.dtype}")
                debug(f"Layer {i} mlp.gate_proj weight dtype: {layer.mlp.gate_proj.weight.dtype}")
        
        debug(f"build_transformer weight stats: uint8={linear_count}, float={float_count}, has_uint8={has_uint8}")

    return net, raw
