import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tinygrad import Tensor, Device, dtypes
from tinygrad.nn import Embedding, Linear  # Linear only for lm_head
from tinygrad.nn.state import load_state_dict
from tinygrad.helpers import DEBUG

# ────────────────────────────────────────────────────────────
# Debug utilities
# ────────────────────────────────────────────────────────────
DEBUG_PRINT = bool(DEBUG)

def debug(msg: str) -> None:
    if DEBUG_PRINT:
        print(f"[DEBUG] {msg}")

# ────────────────────────────────────────────────────────────
# Quantisation helpers (4‑bit packed)
# ────────────────────────────────────────────────────────────
VALUES_PER_ITEM = 4  # 2‑bit per value → 4 vals in a uint8


def rotate_half(t: Tensor) -> Tensor:
    d = t.shape[-1] // 2
    return (-t[..., d:]).cat(t[..., :d], dim=-1)


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
        debug(f"ActQuant: x.shape={x.shape}")
        Qn, Qp = -(2 ** 7), 2 ** 7 - 1
        max_abs = x.abs().max(axis=-1, keepdim=True).clamp(min_=1e-5)
        scale = Qp / max_abs
        q = (x * scale).round().clamp(Qn, Qp)
        return q.cast(dtypes.int8), scale


# ────────────────────────────────────────────────────────────
# 2‑bit weight pack / unpack helpers
# ────────────────────────────────────────────────────────────

def unpack_weights(packed: Tensor, dtype) -> Tensor:
    """Expand 2‑bit packed weights to full uint8 then cast."""
    debug(f"unpack_weights: packed.shape={packed.shape}")
    p0, *prest = packed.shape
    out_shape = (p0 * VALUES_PER_ITEM, *prest) if prest else (p0 * VALUES_PER_ITEM,)

    unpacked = Tensor.zeros(out_shape, dtype=dtypes.uint8, device=packed.device).contiguous().realize()
    for i in range(VALUES_PER_ITEM):
        chunk = ((packed & (3 << (2 * i))) >> (2 * i)).cast(dtypes.uint8)
        unpacked[i * p0 : (i + 1) * p0] = chunk
    return unpacked.cast(dtype) - 1  # map {0,1,2,3} → {‑1,0,1,2}


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
    rms_norm_eps: float = 1e-5
    vocab_size: int = 128_256
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rope_theta: float = 500_000.0
    bos_token_id: int = 128_000
    eos_token_id: int = 128_001
    pad_token_id: int = 0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True

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
        packed = self.weight
        w = unpack_weights(packed, dtype=self.dtype)
        x_q, x_scale = ActQuant.forward(x)
        mat = x_q.cast(self.dtype) @ w.T
        y = mat * self.weight_scale / x_scale
        if self.bias is not None:
            y = y + self.bias
        return y


class BitNetRMSNorm:
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        self.weight = Tensor.ones((hidden_size,), device=Device.DEFAULT)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        var = (x.cast(dtypes.float32) ** 2).mean(axis=-1, keepdim=True)
        x_hat = x * Tensor.rsqrt(var + self.eps)
        return (self.weight * x_hat).cast(x.dtype)


class BitNetMLP:
    def __init__(self, config: BitNetConfig):
        self.gate_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = BitLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = BitLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.ffn_norm = BitNetRMSNorm(config.intermediate_size, eps=config.rms_norm_eps)

    def __call__(self, x: Tensor) -> Tensor:
        g = self.gate_proj(x).relu().square()
        u = self.up_proj(x)
        inter = self.ffn_norm(g * u)
        return self.down_proj(inter)


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
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return x * cos + rotate_half(x) * sin

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
        return self.o_proj(self.attn_sub_norm(out))


class BitNetDecoderLayer:
    def __init__(self, config: BitNetConfig):
        self.input_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = BitNetAttention(config)
        self.post_attn_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = BitNetMLP(config)

    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor):
        x = x + self.self_attn(self.input_norm(x), cos, sin)
        x = x + self.mlp(self.post_attn_norm(x))
        return x


class BitNetModel:
    def __init__(self, config: BitNetConfig):
        self.embed = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [BitNetDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.final_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary = BitNetRotaryEmbedding(config)

    def __call__(self, input_ids: Tensor, past=None):
        x = self.embed(input_ids)
        batch, seq = input_ids.shape
        pos = Tensor.arange(seq, device=x.device)[None, :].expand(batch, -1)
        cos, sin = self.rotary(x, pos)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.final_norm(x)


class BitNetForCausalLM:
    def __init__(self, config: BitNetConfig):
        self.config = config
        self.model = BitNetModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.model.embed.weight

    def __call__(self, input_ids: Tensor, past=None, start_pos: int = 0, *sample_args):
        hidden = self.model(input_ids, past)
        logits = self.lm_head(hidden[:, -1, :].cast(dtypes.float32))[0]
        if sample_args:
            token = sample(logits, *sample_args)
            return token, None, logits  # K/V cache not implemented yet
        return logits  # Return just logits if no sampling args provided


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
    weights: Dict[str, Tensor], n_heads: int, n_kv_heads: int, model: Any
) -> Dict[str, Tensor]:
    debug("convert_from_huggingface")
    sd: Dict[str, Tensor] = {}
    for k, v in weights.items():
        if k == "model.embed_tokens.weight":  # HuggingFace naming
            k = "model.embed.weight"
        v = v.to(Device.DEFAULT)
        if v.dtype == dtypes.bfloat16:
            v = Tensor(v.cast(dtypes.float32).numpy(), dtype=dtypes.float32)
        if k.endswith("q_proj.weight"):
            v = _permute_qkv(v, n_heads)
        elif k.endswith("k_proj.weight"):
            v = _permute_qkv(v, n_kv_heads)
        sd[k] = v
    return sd


# ────────────────────────────────────────────────────────────
# Convenience loader
# ────────────────────────────────────────────────────────────

def build_transformer(model_path: Path, load_weights: bool = True):
    config = BitNetConfig()
    net = BitNetForCausalLM(config)

    if not load_weights:
        return net

    sf_path = (
        model_path
        if model_path.is_file()
        else model_path / "model.safetensors"
    )
    assert sf_path.exists(), f"weights not found at {sf_path}"

    from tinygrad.nn.state import safe_load

    raw = safe_load(str(sf_path))
    weights = convert_from_huggingface(
        raw, config.num_attention_heads, config.num_key_value_heads, net
    )
    load_state_dict(net, weights, strict=False, consume=True)
    return net
