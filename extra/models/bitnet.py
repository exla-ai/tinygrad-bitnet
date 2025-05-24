import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import sys
import re

from tinygrad import Tensor, Device, dtypes
from tinygrad.nn import Embedding, Linear  # Linear only for lm_head
from tinygrad.nn.state import load_state_dict, get_state_dict
from tinygrad.helpers import getenv, DEBUG



# ────────────────────────────────────────────────────────────
# Debug utilitie s
# ────────────────────────────────────────────────────────────
DEBUG_PRINT = True

def debug(msg: str) -> None:
    if DEBUG_PRINT:
        print(f"[DEBUG] {msg}")
        sys.stdout.flush()

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


# ────────────────────────────────────────────────────────────
# 2‑bit weight pack / unpack helpers
# ────────────────────────────────────────────────────────────

def unpack_ternary_weights(arr: np.ndarray, target_dtype=dtypes.float32) -> Tensor:
    """
    Unpacks 2-bit weights (packed into uint8) into ternary {-1, 0, 1} tensor following the transformers implementation.
    
    This function matches the transformers unpack_weights function exactly to ensure compatibility.
    
    Args:
        arr: uint8 NumPy array with shape [out_features_packed, in_features]
        target_dtype: The desired tinygrad dtype for the output tensor
        
    Returns:
        Tensor with shape [out_features_unpacked, in_features] and dtype=target_dtype, values are {-1, 0, 1}.
    """
    debug(f"unpack_ternary_weights: packed_np.shape={arr.shape}, packed_np.dtype={arr.dtype}, target_dtype={target_dtype}")
    
    packed_shape = arr.shape
    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * VALUES_PER_ITEM
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    # Initialize output array
    unpacked = np.zeros(unpacked_shape, dtype=np.uint8)

    # Unpack using the same bit manipulation as transformers
    for i in range(VALUES_PER_ITEM):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = 3 << (2 * i)  # Create mask: 3 = 0b11
        unpacked[start:end] = (arr & mask) >> (2 * i)

    # Convert to target dtype and subtract 1 to get {-1, 0, 1} range
    # This matches the transformers implementation: unpacked.to(dtype) - 1
    result_np = unpacked.astype(np.float32) - 1.0
    
    # Ensure values are exactly {-1, 0, 1}
    result_np = np.clip(result_np, -1.0, 1.0)
    
    # Create tensor and cast to target dtype
    result = Tensor(result_np, dtype=dtypes.float32, requires_grad=False, device=Device.DEFAULT)
    if target_dtype != dtypes.float32:
        result = result.cast(target_dtype)
    
    result = result.realize()
    
    debug(f"unpack_ternary_weights: final result shape={result.shape}, dtype={result.dtype}")
    assert result.shape == unpacked_shape, f"Shape mismatch: expected {unpacked_shape}, got {result.shape}"
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
    print(f"[SAMPLE] Input logits shape: {logits.shape}, temp={temp}, top_k={k}, top_p={p}")
    """Return **int** token id chosen from `logits`."""

    debug(
        f"sample: logits.shape={logits.shape}, temp={temp}, k={k}, p={p}, af={af}, ap={ap}"
    )
    assert logits.ndim == 1, "logits must be 1‑D (vocab)"

    # Greedy / argmax path
    if temp < 1e-6:
        token = int(logits.argmax().realize().to("CPU").numpy())
        print(f"[SAMPLE] Greedy sampling result: token={token}")
        return token

    if af or ap:
        if not hasattr(sample, "alpha_counter"):
            sample.alpha_counter = Tensor.zeros_like(logits, dtype=dtypes.int32)
        logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

    # mask NaNs
    logits = (logits != logits).where(-float("inf"), logits)
    probs = (logits / temp).softmax()
    print(f"[SAMPLE] After softmax: shape={probs.shape}")
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    print(f"[SAMPLE] Top 5 probs: shape={probs_sort[:5].shape}")
    print(f"[SAMPLE] Top 5 indices: shape={probs_idx[:5].shape}")

    if k:
        values, indices = probs.topk(k)
        cum = values[::-1].cumsum()[::-1]
        mask = cum >= (1 - p)
        values = values * mask
        indices = indices * mask
        # Use CPU-based sampling instead of multinomial
        values_np = values.realize().to("CPU").numpy()
        indices_np = indices.realize().to("CPU").numpy()
        # Normalize probabilities
        if values_np.sum() > 0:
            values_np = values_np / values_np.sum()
            # Sample using numpy
            choice = np.random.choice(len(values_np), p=values_np)
        else:
            choice = 0
        choice = min(choice, k-1)  # Ensure choice is within bounds
        token = int(indices_np[choice])
    else:
        # Use CPU-based sampling instead of multinomial
        probs_np = probs.realize().to("CPU").numpy()
        # Sample using numpy
        token = int(np.random.choice(len(probs_np), p=probs_np))

    if af or ap:
        counter = Tensor.arange(probs.numel(), device=logits.device)
        sample.alpha_counter = (counter == token).where(
            sample.alpha_counter + 1, sample.alpha_counter
        )

    debug(f"sample: token={token}")

    print(f"[SAMPLE] Final sampled token: {token}")
    return token


# ────────────────────────────────────────────────────────────
# Config & Tiny‑grad model definition
# ────────────────────────────────────────────────────────────

class BitNetConfig:
    # From HF JSON config
    hidden_size: int = 2560
    intermediate_size: int = 6912
    num_attention_heads: int = 20
    num_key_value_heads: int = 5 # Explicitly from JSON
    num_hidden_layers: int = 30
    rms_norm_eps: float = 1e-05
    vocab_size: int = 128256
    max_position_embeddings: int = 4096
    hidden_act: str = "relu2" # As per JSON
    initializer_range: float = 0.02
    rope_theta: float = 500000.0 # As per JSON
    bos_token_id: int = 128000
    eos_token_id: int = 128001

    
    use_cache: bool = True # As per JSON
    
    # Quantization specific from JSON
    quant_method: str = "bitnet" 
    linear_class: str = "autobitlinear" # As per JSON: quantization_config.linear_class
    quantization_mode: str = "offline" # As per JSON: quantization_config.quantization_mode

    # Retained from previous tinygrad config structure, not in this specific HF JSON
    pad_token_id: Optional[int] = None 
    attention_dropout: float = 0.0 # Common, though BitNet might not use it heavily
    use_bias: bool = False # Consistent with BitNet papers typically omitting biasesrty
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
        device=None,
        dtype=dtypes.float32,
        transposed=False,
    ):
        debug(f"BitLinear.__init__: in={in_features}, out={out_features}")
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device or Device.DEFAULT
        self.transposed = transposed

        # self.weight is initialized on self.device (e.g., CUDA) and will be int8.
        # It will be filled by load_state_dict with data that originates from CPU uchar,
        # then transferred to self.device and cast to int8 (lazily if realize=False in load_state_dict).
        if transposed:
            self.weight = Tensor.empty(
                (in_features, out_features), dtype=dtypes.int8, requires_grad=False, device=self.device
            )
        else:
            self.weight = Tensor.empty(
                (self.out_features // VALUES_PER_ITEM, self.in_features), dtype=dtypes.int8, requires_grad=False, device=self.device
            )
        
        # self.weight_scale is always a CPU float32 tensor.
        # It's loaded from a CPU float32 tensor from the weights dict.
        self.weight_scale = Tensor([1.0], dtype=dtypes.float32, requires_grad=False, device="CPU")

    def activation_quant(self, x: Tensor, num_bits: int = 8) -> Tuple[Tensor, Tensor]:
        """
        Activation quantization: Performs symmetric, per-token quantization on the input activations.
        Maps activations to int8 range [-128, 127] with per-token scaling.
        
        Args:
            x: Input activations to be quantized
            num_bits: Number of bits to use for quantization (default: 8)
            
        Returns:
            Tuple of (quantized_activations, scale_factors)
        """
        Qn = -(2 ** (num_bits - 1))  # -128 for 8-bit
        Qp = 2 ** (num_bits - 1) - 1  # 127 for 8-bit
        
        # Compute per-token scale factors (max absolute value along last dimension)
        abs_max = x.abs().max(axis=-1, keepdim=True).clamp(min=1e-5)
        scale = Qp / abs_max
        
        # Quantize and clamp to valid range
        quantized = (x * scale).round().clamp(Qn, Qp)
        
        # Convert to int8 (represented as float in tinygrad)
        quantized_int8 = quantized.cast(dtypes.float32)  # Tinygrad doesn't have native int8, use float32
        
        return quantized_int8, scale

    def post_quant_process(self, x: Tensor, input_scale: Tensor, weight_scale: Tensor) -> Tensor:
        """
        Post-quantization processing: Applies proper scaling to dequantize the output.
        
        Args:
            x: Output from quantized linear operation
            input_scale: Scale factors from activation quantization
            weight_scale: Scale factors from weight quantization
            
        Returns:
            Properly scaled output tensor
        """
        # Combine input and weight scales
        combined_scale = input_scale * weight_scale.to(input_scale.device)
        
        # Apply inverse scaling to dequantize
        out = x / combined_scale
        
        return out

    def __call__(self, x: Tensor) -> Tensor:
        # Step 1: Unpack ternary weights from packed format
        packed_weights_np = self.weight.to('CPU').realize().numpy()
        dequantized_ternary_weights = unpack_ternary_weights(packed_weights_np, self.dtype)
        
        # Step 2: Quantize input activations to int8 (crucial step that was missing!)
        input_quant, input_scale = self.activation_quant(x)
        
        # Step 3: Perform linear operation with quantized inputs and ternary weights
        if self.transposed:
            # For transposed weights, x @ W
            y = input_quant @ dequantized_ternary_weights
        else:
            # For non-transposed weights, x @ W.T
            y = input_quant @ dequantized_ternary_weights.T
        
        # Step 4: Apply proper post-quantization scaling (this was incorrect before!)
        weight_scale = self.weight_scale.to(y.device)
        y = self.post_quant_process(y, input_scale, weight_scale)
        
        return y

class BitNetRMSNorm:
    """
    HF-equivalent rms-norm.

    - works in float32 for the math,  
    - keeps a single learned weight vector (loaded from the checkpoint).
    """
    def __init__(self, dim: int, eps: float = 1e-6, device=None):
        # will be overwritten by load_state_dict, so no grad is fine
        self.weight = Tensor.ones((dim,), dtype=dtypes.float32, requires_grad=False, device=device)
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        in_dtype  = x.dtype
        h         = x.cast(dtypes.float32)                       # 1. promote
        var       = h.pow(2).mean(axis=-1, keepdim=True)         # 2. variance
        h         = h * Tensor.rsqrt(var + self.eps)             # 3. normalize
        return (self.weight * h).cast(in_dtype)                  # 4. scale & cast back




class BitNetMLP:
    def __init__(self, config: BitNetConfig, device=None):
        """
        Initialize BitNetMLP following HuggingFace's implementation pattern.
        
        This MLP structure consists of three parts:
        1. up_proj - projects hidden states to intermediate size
        2. gate_proj - also projects to intermediate size, then gets activated
        3. down_proj - projects back from intermediate to hidden size
        
        The activation pattern is SwiGLU-like: (gate_proj * relu^2(up_proj))
        """
        hidden_size = config.hidden_size         # 2560
        intermediate_size = config.intermediate_size # 6912
        
        self.gate_proj = BitLinear(hidden_size, intermediate_size, device=device)
        self.up_proj = BitLinear(hidden_size, intermediate_size, device=device)
        
        # Layer normalization is applied to the intermediate_size features
        self.ffn_ln = BitNetRMSNorm(intermediate_size, eps=config.rms_norm_eps, device=device)
        
        # down_proj: intermediate_size (6912) -> hidden_size (2560)
        # Weight shape from file is (640, 6912) uchar.
        # If not transposed, BitLinear(6912, 2560) expects weight (2560//4, 6912) = (640, 6912). This matches.
        self.down_proj = BitLinear(intermediate_size, hidden_size, transposed=False, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        # x shape: (batch, seq_len, hidden_size)
        
        gate = self.gate_proj(x)          # Output: (batch, seq_len, intermediate_size)
        gate = gate.relu() ** 2
        
        up = self.up_proj(x)             # Output: (batch, seq_len, intermediate_size)
        
        h = gate * up                    # Shape: (batch, seq_len, intermediate_size)
        
        h = self.ffn_ln(h)               # Shape: (batch, seq_len, intermediate_size)
        
        # down_proj expects input of shape (..., intermediate_size)
        # and outputs (..., hidden_size)
        return self.down_proj(h)


class BitNetRotaryEmbedding:
    """
    Pre-computes inverse frequencies once, serves cos/sin on demand.
    """

    def __init__(self, config: BitNetConfig, device=None):
        hd = config.head_dim()               # scalar value
        inv = Tensor.arange(0, hd, 2, dtype=dtypes.float32, device=device)
        self.inv_freq = 1.0 / (config.rope_theta ** (inv / hd))

    def __call__(self, x: Tensor, pos_ids: Tensor):
        # x: [*, seq, dim] – only dtype/device matter
        freqs = pos_ids.unsqueeze(-1).float() * self.inv_freq     # [B,S,hd/2]
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return cos.cast(x.dtype), sin.cast(x.dtype)



class BitNetAttention:
    """Multi-headed attention from 'Attention Is All You Need' paper, aligned with HuggingFace implementation."""
    
    def __init__(self, config: BitNetConfig, layer_idx: int, device=None):
        super().__init__()
        self._config = config
        self.layer_idx = layer_idx
        
        # Exactly following HuggingFace's implementation
        # Make sure we get the head_dim as a value, not a method
        if hasattr(config, "head_dim") and callable(getattr(config, "head_dim")):
            self.head_dim = config.head_dim()
        else:
            self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5  # Scale factor for division before softmax
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        
        debug(f"BitNetAttention.__init__: layer_idx={layer_idx}, head_dim={self.head_dim}, num_key_value_groups={self.num_key_value_groups}")
        
        # Initialize projections with exact dimensions matching the HuggingFace weights
        # From the logs, we can see the actual dimensions of the weights
        # q_proj.weight: (640, 2560)
        # k_proj.weight/v_proj.weight: (160, 2560)
        # o_proj.weight: (640, 2560) - Unpacked from original (640, 2560)
        self.q_proj = BitLinear(config.hidden_size, config.num_attention_heads * self.head_dim, device=device)
        self.k_proj = BitLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, device=device)
        self.v_proj = BitLinear(config.hidden_size, config.num_key_value_heads * self.head_dim, device=device)
        
        # o_proj needs to match the EXACT shape of the weights in the state dict (640, 2560)
        # From the error logs, we see the state dict expects (640, 2560) for o_proj.weight
        self.o_proj = BitLinear(config.num_attention_heads * self.head_dim, config.hidden_size, device=device)
        
        # Add attention sub-normalization for stabilizing the input to the output projection
        # This matches the HF implementation exactly
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotary position embeddings to a single tensor (query or key).
        Direct translation of HF's apply_rotary_pos_emb with unsqueeze_dim=1.
        
        Args:
            x: Input tensor to apply rotary embeddings to (shape [batch, heads, seq_len, head_dim])
            cos: Cosine part of rotary embeddings (shape [batch, seq_len, rope_dim])
            sin: Sine part of rotary embeddings (shape [batch, seq_len, rope_dim])
            
        Returns:
            Tensor with rotary embeddings applied.
        """
        # Get dimensions for proper rotary embedding application
        head_dim = x.shape[-1]
        rope_dim = min(cos.shape[-1], head_dim)
        
        # If the rotary embedding dimension is larger than needed, slice it
        if cos.shape[-1] > rope_dim:
            cos = cos[..., :rope_dim]
            sin = sin[..., :rope_dim]
        
        # Unsqueeze dim=1 to add a head dimension for broadcasting
        cos = cos.unsqueeze(1)  # Shape becomes [batch, 1, seq_len, rope_dim]
        sin = sin.unsqueeze(1)  # Shape becomes [batch, 1, seq_len, rope_dim]
        
        # Debug dimensions after processing
        debug(f"_apply_rope - x: {x.shape}, cos: {cos.shape}, sin: {sin.shape}, rope_dim: {rope_dim}")
        
        # Exactly matches HF implementation: (x * cos) + (rotate_half(x) * sin)
        # But only apply to the first rope_dim dimensions if head_dim is larger
        if rope_dim < head_dim:
            # Split tensor into parts that need rotation and parts that remain unchanged
            x_roped = (x[..., :rope_dim] * cos) + (rotate_half(x[..., :rope_dim]) * sin)
            # Concatenate with unmodified part
            result = x_roped.cat(x[..., rope_dim:], dim=-1)
        else:
            # Standard case - apply to all dimensions
            result = (x * cos) + (rotate_half(x) * sin)
        
        return result
        
    def _apply_rope_to_qk(self, query_states: Tensor, key_states: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to both query and key states.
        Handles cases where the rotary embedding dimension doesn't match head dimensions.
        
        Args:
            query_states: Query tensor (shape [batch, heads, seq_len, head_dim])
            key_states: Key tensor (shape [batch, heads, seq_len, head_dim]) 
            cos: Cosine part of rotary embeddings (shape [batch, seq_len, rope_dim])
            sin: Sine part of rotary embeddings (shape [batch, seq_len, rope_dim])
            
        Returns:
            Tuple of (rotated_query_states, rotated_key_states)
        """
        # Debug the shapes before applying rotary embeddings
        debug(f"_apply_rope_to_qk - query: {query_states.shape}, key: {key_states.shape}, cos: {cos.shape}, sin: {sin.shape}")
        
        # Get dimensions for proper rotary embedding application
        rope_dim = min(cos.shape[-1], query_states.shape[-1])
        head_dim = query_states.shape[-1]
        debug(f"_apply_rope_to_qk - rope_dim: {rope_dim}, head_dim: {head_dim}")
        
        # Apply rotary embeddings using separate calls for query and key states
        if rope_dim == head_dim:
            # Simple case - dimensions match exactly
            query_states_rotary = self._apply_rope(query_states, cos, sin)
            key_states_rotary = self._apply_rope(key_states, cos, sin)
        else:
            # Complex case - handle partial rotation when dimensions don't match
            # Only rotate the first rope_dim dimensions, leave the rest unchanged
            query_partial_rotated = self._apply_rope(
                query_states[..., :rope_dim], 
                cos[..., :rope_dim], 
                sin[..., :rope_dim]
            )
            key_partial_rotated = self._apply_rope(
                key_states[..., :rope_dim], 
                cos[..., :rope_dim], 
                sin[..., :rope_dim]
            )
            
            # Recombine the rotated and non-rotated parts
            if rope_dim < head_dim:
                # Use instance method cat() rather than static method
                query_states_rotary = query_partial_rotated.cat(
                    query_states[..., rope_dim:], dim=-1
                )
                key_states_rotary = key_partial_rotated.cat(
                    key_states[..., rope_dim:], dim=-1
                )
            else:
                query_states_rotary = query_partial_rotated
                key_states_rotary = key_partial_rotated
        
        debug(f"_apply_rope_to_qk - output query: {query_states_rotary.shape}, key: {key_states_rotary.shape}")
        return query_states_rotary, key_states_rotary

    def forward(self, 
              hidden_states: Tensor,
              position_embeddings: Tuple[Tensor, Tensor],
              attention_mask: Optional[Tensor] = None,
              past_key_value = None,
              cache_position: Optional[Tensor] = None,
              output_attentions: bool = False):
        """Forward pass for BitNetAttention, following HuggingFace implementation.
        
        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            attention_mask: Optional attention mask
            past_key_value: Optional past key-value state for incremental decoding
            cache_position: Optional tensor indicating position in the cache
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (attn_output, attn_weights)
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # Project inputs to query, key, and value
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Project inputs to query, key, and value with dynamic dimension handling
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Debug the actual output shapes
        debug(f"BitNetAttention shapes - query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}")
        
        # Calculate head dimensions dynamically based on actual output shapes
        # For queries, we have 16 attention heads
        q_head_dim = query_states.shape[-1] // 16
        # For keys and values, we have 4 key/value heads
        k_head_dim = key_states.shape[-1] // 4
        v_head_dim = value_states.shape[-1] // 4
        
        debug(f"BitNetAttention head dimensions - q_head_dim: {q_head_dim}, k_head_dim: {k_head_dim}, v_head_dim: {v_head_dim}")
        
        # Reshape with dynamically calculated dimensions
        query_states = query_states.reshape(batch_size, seq_len, 16, q_head_dim).transpose(1, 2)
        key_states = key_states.reshape(batch_size, seq_len, 4, k_head_dim).transpose(1, 2)
        value_states = value_states.reshape(batch_size, seq_len, 4, v_head_dim).transpose(1, 2)
        
        # Extract cos and sin from position embeddings
        cos, sin = position_embeddings
        
        # Apply rotary embeddings using the new method that properly handles both query and key states
        query_states, key_states = self._apply_rope_to_qk(query_states, key_states, cos, sin)
        
        # Update cache if provided
        if past_key_value is not None:
            # Extract past keys and values
            past_key, past_value = past_key_value
            # Concatenate with current keys and values
            if past_key is not None:
                key_states = past_key.cat(key_states, dim=2)  # Using instance method cat
            if past_value is not None:
                value_states = past_value.cat(value_states, dim=2)  # Using instance method cat
        
        # Handle grouped query attention (GQA) for our specific case
        # We have 16 query heads and 4 key/value heads, so repeat ratio is 4
        # Always repeat for this model since heads are hardcoded
        key_states = self._repeat_kv(key_states, 4)  # 16/4 = 4 repeat factor
        value_states = self._repeat_kv(value_states, 4)
            
        # Compute scaled dot-product attention
        attn_weights = query_states @ key_states.transpose(2, 3) * self.scaling
        
        # Apply attention mask if provided
        if attention_mask is not None and not isinstance(attention_mask, tuple):
            # Process attention mask to get correct shape
            debug(f"attention_mask shape: {attention_mask.shape}, key_states shape: {key_states.shape}")
            try:
                if attention_mask.shape[-1] != key_states.shape[-2]:
                    attention_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + attention_mask
            except Exception as e:
                debug(f"Error processing attention mask: {e}")
                # Use default causal mask by not applying the attention_mask
        
        # Apply softmax and dropout
        attn_weights = attn_weights.softmax()
        if self.attention_dropout > 0 and self.training:
            # Simple dropout implementation (can be improved later)
            dropout_mask = Tensor.rand(*attn_weights.shape) > self.attention_dropout
            attn_weights = attn_weights * dropout_mask * (1.0 / (1.0 - self.attention_dropout))
        
        # Apply attention to values
        attn_output = attn_weights @ value_states
        
        # Reshape back to original dimensions
        # Transpose from [batch, heads, seq_len, head_dim] to [batch, seq_len, heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        
        # Debug shape before reshape
        debug(f"attn_output before reshape: {attn_output.shape}")
        
        # Flatten the heads dimension
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Debug shape after reshape
        debug(f"attn_output after reshape: {attn_output.shape}")
        
        # Force reshape to hidden_size (2560) if dimensions don't match
        if attn_output.shape[-1] != self._config.hidden_size:
            debug(f"Padding attn_output from {attn_output.shape[-1]} to {self._config.hidden_size}")
            
            # Create a full-sized tensor filled with zeros
            hidden_size = self._config.hidden_size
            attn_dim = attn_output.shape[-1]
            
            # Use concatenation instead of slice assignment
            # Create a zero tensor for the padding portion
            padding = Tensor.zeros(batch_size, seq_len, hidden_size - attn_dim, dtype=attn_output.dtype, device=self.device)
            
            # Concatenate the attention output with the padding along the last dimension
            attn_output = attn_output.cat(padding, dim=-1)
            
            debug(f"Padded attn_output shape: {attn_output.shape}")
            
            # Ensure we have the exact shape required
            assert attn_output.shape[-1] == hidden_size, f"Expected shape {hidden_size} but got {attn_output.shape[-1]}"
        
        # Apply attention sub-norm (specific to BitNet implementation)
        attn_output = self.attn_sub_norm(attn_output)
        
        # Apply output projection
        attn_output = self.o_proj(attn_output)
        
        # Return appropriate outputs
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs
    
    # For backward compatibility with existing code
    def __call__(self, x, cos, sin, past_key=None, past_value=None, attention_mask=None):
        """Legacy interface for backward compatibility"""
        debug(f"BitNetAttention.__call__: Using legacy interface, redirecting to forward()")
        position_embeddings = (cos, sin)
        past_key_value = (past_key, past_value) if past_key is not None else None
        
        outputs = self.forward(
            hidden_states=x, 
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        
        attn_output = outputs[0]
        present_key = key_states if 'key_states' in locals() else None
        present_value = value_states if 'value_states' in locals() else None
        
        return attn_output, present_key, present_value
    
    # This space previously contained a duplicate _apply_rope implementation
    # We've standardized on a single implementation with the signature (self, x, cos, sin)
    
    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat key and value states for grouped query attention."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        
        if n_rep == 1:
            return hidden_states
            
        # Expand and reshape to repeat the key/value heads
        expanded = hidden_states.reshape(batch, num_key_value_heads, 1, slen, head_dim)
        expanded = expanded.repeat(1, 1, n_rep, 1, 1)
        return expanded.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class BitNetDecoderLayer:
    def __init__(self, config: BitNetConfig, layer_idx: int, device=None):
        self.hidden_size = config.hidden_size
        self.gradient_checkpointing = False  # For compatibility with HF's implementation
        
        # Initialize layers as in HuggingFace implementation
        self.self_attn = BitNetAttention(config=config, layer_idx=layer_idx, device=device)
        self.mlp = BitNetMLP(config, device=device)
        self.input_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.post_attention_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
    
    def forward(self, 
              hidden_states: Tensor,
              position_embeddings: Tuple[Tensor, Tensor],
              attention_mask: Optional[Tensor] = None,
              past_key_value = None,
              cache_position: Optional[Tensor] = None,
              output_attentions: bool = False,
              use_cache: bool = False):
        """Forward pass for BitNetDecoderLayer, following HuggingFace implementation.
        
        Args:
            hidden_states: Input tensor
            position_embeddings: Tuple of (cos, sin) for rotary embeddings
            attention_mask: Optional attention mask
            past_key_value: Optional past key-value state for incremental decoding
            cache_position: Optional tensor indicating position in the cache
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cache for incremental decoding
            
        Returns:
            Tuple of (hidden_states, present_key_value, attentions)
        """
        # Residual connection pattern follows standard Transformer architecture
        # Layer norm -> self-attention -> add residual -> layer norm -> MLP -> add residual
        
        # Apply first layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Apply self-attention
        # The self_attn.forward method handles the KV cache internally
        attn_outputs = self.self_attn.forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            output_attentions=output_attentions
        )

        # Get the attention output and possibly the attention weights
        attn_output = attn_outputs[0]
        
        # Update KV cache if needed
        if use_cache:
            present_key_value = attn_outputs[1] if len(attn_outputs) > 1 else None
        else:
            present_key_value = None
            
        # Add residual connection
        hidden_states = residual + attn_output
        
        # Apply second layer norm
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Apply MLP
        hidden_states = self.mlp(hidden_states)
        
        # Add second residual connection
        hidden_states = residual + hidden_states
        
        # Prepare output tuple
        outputs = (hidden_states,)
        if use_cache:
            outputs += (present_key_value,)
        if output_attentions and len(attn_outputs) > 1:
            attention = attn_outputs[1]
            outputs += (attention,)
            
        return outputs
    
    # For backward compatibility with existing code
    def __call__(self, x: Tensor, cos: Tensor, sin: Tensor, attention_mask: Optional[Tensor] = None, past=None):
        """Legacy interface for backward compatibility"""
        debug(f"BitNetDecoderLayer.__call__: Using legacy interface, redirecting to forward()")
        position_embeddings = (cos, sin)
        use_cache = past is not None
        
        outputs = self.forward(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past,
            use_cache=use_cache
        )
        
        # Always return two values (hidden_states and present_key_value)
        # to maintain consistent return signature
        if isinstance(outputs, tuple) and len(outputs) > 1:
            return outputs[0], outputs[1]  # hidden_states, present_key_value
        else:
            # If no cache is used, return the hidden states and None as present_key_value
            debug(f"BitNetDecoderLayer.__call__: No cache used, returning hidden_states and None")
            return outputs if not isinstance(outputs, tuple) else outputs[0], None

class BitNetModel:
    def __init__(self, config: BitNetConfig, device=None):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Initialize token embeddings
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        if device is not None:
            self.embed_tokens.weight = self.embed_tokens.weight.to(device)
        
        # Create decoder layers with proper layer_idx for each
        self.layers = [
            BitNetDecoderLayer(config, layer_idx, device=device) 
            for layer_idx in range(config.num_hidden_layers)
        ]
        
        # Final normalization and rotary embeddings
        self.norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps, device=device)
        self.rotary = BitNetRotaryEmbedding(config, device=device)
        
        # For compatibility with HF's implementation
        self.gradient_checkpointing = False

    def __call__(self, input_ids: Tensor, past=None):
        debug(f"BitNetModel.__call__: input_ids.shape={input_ids.shape}, past is {'present' if past is not None else 'None'}")
        print(f"[MODEL] BitNetModel call with input shape {input_ids.shape}, past is {'present' if past is not None else 'None'}")
        
        x = self.embed_tokens(input_ids)
        batch, seq = input_ids.shape
        pos = Tensor.arange(seq, device=x.device)[None, :].expand(batch, -1)
        cos, sin = self.rotary(x, pos)
        
        # Initialize or retrieve key-value cache
        if past is None:
            past_length = 0
            # Initialize KV cache for all layers
            past = [(None, None) for _ in range(len(self.layers))]
        else:
            # More robust handling of past to avoid NoneType errors
            past_length = 0
            try:
                if isinstance(past, tuple) or isinstance(past, list):
                    if len(past) > 0 and past[0] is not None:
                        if isinstance(past[0], tuple) and len(past[0]) > 0 and past[0][0] is not None:
                            past_length = past[0][0].shape[2]
                        # If the structure is different, we'll just use 0 as a safe default
            except (IndexError, AttributeError, TypeError) as e:
                debug(f"Error getting past_length in BitNetModel: {e}, falling back to 0")
                past_length = 0
        
        print(f"[MODEL] Processing with past_length={past_length}, input_seq_length={seq}")
        
        # Track the updated KV cache
        present = []
        
        # Process through all layers
        for i, layer in enumerate(self.layers):
            layer_past = past[i] if past is not None else None
            x, layer_present = layer(x, cos, sin, layer_past)
            present.append(layer_present)
        
        # Normalize the final hidden states
        normalized_x = self.norm(x)
        
        debug(f"BitNetModel.__call__: completed with hidden_states.shape={normalized_x.shape}, cache size={len(present)}")
        print(f"[MODEL] Returning hidden_states shape={normalized_x.shape}, cache size={len(present)}")
        
        return normalized_x, present


class BitNetForCausalLM:
    def __init__(self, config: BitNetConfig, device=None):
        print(f"[MODEL] Initializing BitNetForCausalLM with config: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
        self.model = BitNetModel(config, device=device)
        # Create Linear layer without device parameter
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
        # Move the weight to the specified device if needed
        if device is not None:
            self.lm_head.weight = self.lm_head.weight.to(device)
        print(f"[MODEL] BitNetForCausalLM initialized")
        self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(self, input_ids: Tensor, past=None, *sample_args):
        debug(f"BitNetForCausalLM.__call__: input_ids.shape={input_ids.shape}")
        print(f"[MODEL-CALL] Input shape: {input_ids.shape}, past is {'present' if past is not None else 'None'}, args count: {len(sample_args)}")
        
        # More robust checking of past to avoid NoneType errors
        past_len = 0
        if past is not None:
            try:
                if isinstance(past, tuple) and len(past) > 0 and past[0] is not None:
                    if isinstance(past[0], tuple) and len(past[0]) > 0 and past[0][0] is not None:
                        past_len = past[0][0].shape[2]
            except (IndexError, AttributeError) as e:
                debug(f"Error getting past_len: {e}, falling back to 0")
        
        print(f"[MODEL-CALL] Past length: {past_len}")
        outputs = self.model(input_ids, past)
        hidden_states, present = outputs
        debug(f"BitNetForCausalLM.__call__: hidden_states.shape={hidden_states.shape}")
        print(f"[MODEL-CALL] Hidden states shape: {hidden_states.shape}")
        logits = self.lm_head(hidden_states)
        debug(f"BitNetForCausalLM.__call__: logits.shape={logits.shape}")
        print(f"[MODEL-CALL] Logits shape: {logits.shape}")
        if sample_args:
            # Just return logits if no sample_args
            token = sample(logits[0, -1, :], *sample_args) # Pass 1D logits (vocab_size,) for the current token
            return token, present, logits # Return updated KV cache 
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

def dequantize_weight(packed_weight: Tensor, weight_scale: Tensor, config: BitNetConfig) -> Tensor:
    """Properly dequantize BitNet ternary weights using weight_scale.
    This handles the BitNet b1.58 format (ternary weights with scale).
    
    Args:
        packed_weight: Packed uint8 weight tensor with shape [reduced_dim, input_dim].
        weight_scale: Scale factor for the weight, typically a scalar tensor.
        config: BitNet configuration with hidden_size and other params.
        
    Returns:
        Dequantized float32 tensor with properly scaled values.
    """
    # Expand dimensions to match the way the model was trained
    # operate on host to avoid CUDA bit-ops
    arr = packed.to("CPU").numpy()          # uint8 ndarray
    b0 =  (arr      & 3)
    b1 = ((arr>>2)  & 3)
    b2 = ((arr>>4)  & 3)
    b3 = ((arr>>6)  & 3)
    signs = np.stack([b0,b1,b2,b3]).reshape(-1, arr.shape[1]).astype("f4") - 1
    return Tensor(signs, device=Device.DEFAULT) * weight_scale


def convert_from_huggingface(raw: Dict[str, Tensor], config) -> Dict[str, Tensor]:
    """
    Converts weights from HuggingFace format to what the BitNet model expects.
    - BitLinear weights are stored as packed uchar.
    - BitLinear scales are stored as float32 tensors.
    - Other weights (embeddings, norms) are processed as needed.
    All tensors in the output dict will be on CPU.
    """
    target_device = "CPU" # Prepare all weights on CPU initially
    out: Dict[str, Tensor] = {}
    processed_keys = set()

    debug(f"[CONVERT] Starting weight conversion. Target device for 'weights' dict: {target_device}")

    # Pass 1: Process and store scale tensors directly into 'out'.
    # These are named like '...mlp.down_proj.weight_scale'.
    for k, v_cpu in raw.items():
        if k.endswith(".weight_scale"):
            if v_cpu.dtype == dtypes.float32:
                scale_tensor = v_cpu.to(target_device)
            else:
                # Ensure scale is a float32 tensor on the target device
                scale_tensor = v_cpu.cast(dtypes.float32).to(target_device)
            
            out[k] = scale_tensor
            processed_keys.add(k)
            debug(f"[CONVERT] Processed scale tensor {k}. Shape: {out[k].shape}, Dtype: {out[k].dtype}, Device: {out[k].device}")

    # Pass 2: Process main weight tensors.
    for k, v_cpu in raw.items():
        if k in processed_keys:  # Skip if already processed (e.g., it was a scale)
            continue

        # Determine if this is a BitLinear main weight by checking if its corresponding scale key exists.
        # Example: for "model.layers.0.mlp.down_proj.weight", scale_key is "model.layers.0.mlp.down_proj.weight_scale"
        potential_scale_key = k.replace(".weight", ".weight_scale")
        is_bitlinear_main_weight = potential_scale_key in out # 'out' now contains all scale tensors

        if is_bitlinear_main_weight:
            # This is a main weight for a BitLinear layer (e.g., model.layers.0.mlp.down_proj.weight).
            # It's expected to be packed uchar data from the raw HuggingFace model.
            if v_cpu.dtype == dtypes.uchar:
                out[k] = v_cpu.to(target_device)
            else:
                # This is a warning/fallback. Ideally, HF BitLinear weights are uchar.
                print(f"[WARN-CONVERT] Expected uchar for BitLinear weight {k}, but got {v_cpu.dtype}. Casting to uchar.")
                try:
                    out[k] = v_cpu.cast(dtypes.uchar).to(target_device)
                except Exception as e_cast:
                    print(f"[ERROR-CONVERT] Failed to cast {k} to uchar: {e_cast}. Storing as is (dtype: {v_cpu.dtype}).")
                    out[k] = v_cpu.to(target_device) # Store as-is if cast fails, hoping load_state_dict handles it or errors informatively
            
            debug(f"[CONVERT] Processed BitLinear main weight {k} (packed). Shape: {out[k].shape}, Dtype: {out[k].dtype}")

        elif "lm_head.weight" == k:
            # lm_head is a standard Linear layer, often tied to embeddings or float32.
            # Ensure it's float32 as per typical lm_head requirements.
            if v_cpu.dtype == dtypes.float32:
                 out[k] = v_cpu.to(target_device)
            else:
                 out[k] = v_cpu.cast(dtypes.float32).to(target_device)
            debug(f"[CONVERT] Processed lm_head.weight {k}. Shape: {out[k].shape}, Dtype: {out[k].dtype}")

        else:
            # This covers other tensors like embeddings, layernorm weights, biases (if any).
            # These are typically bfloat16 or float32 in the raw model.
            # We move them to the target device, preserving their original dtype from raw load unless specific handling is needed.
            # The model's layers (Embedding, BitNetRMSNorm) will handle these dtypes.
            out[k] = v_cpu.to(target_device)
            debug(f"[CONVERT] Processed regular tensor {k}. Shape: {out[k].shape}, Dtype: {out[k].dtype}")
        
        processed_keys.add(k)

    # Ensure all raw keys were processed
    if len(processed_keys) != len(raw):
        unprocessed_raw_keys = set(raw.keys()) - processed_keys
        print(f"[WARN-CONVERT] Some raw keys were not processed: {unprocessed_raw_keys}")

    debug(f"[CONVERT] Weight conversion finished. Total keys processed: {len(out)}")
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

    # Load weights to CPU first as Tensors
    raw = safe_load(str(sf_path))
    debug(f"build_transformer: Loaded {len(raw)} raw tensors from safetensors file")
    
    if DEBUG_PRINT:
        for k_debug, v_debug in list(raw.items())[:5]:
            debug(f"  Raw tensor sample: {k_debug}, shape={v_debug.shape}, dtype={v_debug.dtype}, device={v_debug.device}")
    
    debug(f"build_transformer: Converting/preparing weights for the model structure.")
    # convert_from_huggingface now prepares the 'weights' dictionary in the exact format
    # that load_state_dict expects for the BitNetForCausalLM model.
    # - BitLinear '.weight' will be uchar (packed).
    # - BitLinear '.weight_scale' will be float32.
    # - Other weights will have dtypes as per raw load or specific casting (e.g., lm_head to float32).
    weights = convert_from_huggingface(raw, config)
    debug(f"build_transformer: Weight preparation complete. Number of tensors in 'weights' dict: {len(weights)}")

    # The 'weights' dictionary is now ready. The subsequent data fill/binarization loop is removed
    # as 'convert_from_huggingface' handles the necessary transformations.
    # load_state_dict will now map these correctly prepared tensors to the model's state_dict.

    debug(f"build_transformer: Loading state dict into model using strict=False")
    
    # Debug: Check a sample BitLinear weight and its scale from the `weights` dict
    if DEBUG_PRINT and "model.layers.0.mlp.down_proj.weight" in weights:
        debug(f"  Sample BitLinear weight PRE-LOAD: model.layers.0.mlp.down_proj.weight, shape={weights['model.layers.0.mlp.down_proj.weight'].shape}, dtype={weights['model.layers.0.mlp.down_proj.weight'].dtype}")
        if "model.layers.0.mlp.down_proj.weight_scale" in weights:
            debug(f"  Sample BitLinear scale PRE-LOAD: model.layers.0.mlp.down_proj.weight_scale, shape={weights['model.layers.0.mlp.down_proj.weight_scale'].shape}, dtype={weights['model.layers.0.mlp.down_proj.weight_scale'].dtype}")
    
    # Get the model's state_dict structure for comparison if needed for debugging
    model_state_dict_shapes = {k: v.shape for k, v in get_state_dict(net).items()}
    if DEBUG_PRINT and "model.layers.0.mlp.down_proj.weight" in model_state_dict_shapes:
        debug(f"  Model expected shape for model.layers.0.mlp.down_proj.weight: {model_state_dict_shapes['model.layers.0.mlp.down_proj.weight']}")


    try:
        # Pass realize=False to defer realization until later on the proper device,
        # preventing CPU-device tensors from requiring a renderer at load time.
        load_state_dict(net, weights, strict=False, realize=False)  # strict=False is important
        debug("[DEBUG-LOAD] State dictionary loaded successfully into model.")
    except Exception as e:
        print(f"[ERROR-LOAD] ERROR during load_state_dict: {e}")
        if "Shape mismatch" in str(e) or "dtype mismatch" in str(e):
            try:
                layer_name_match = re.search(r"layer `([^`]+)`", str(e))
                if layer_name_match:
                    layer_name = layer_name_match.group(1)
                    print(f"  Mismatch details for layer: {layer_name}")
                    if layer_name in weights:
                        print(f"    State dict tensor: shape={weights[layer_name].shape}, dtype={weights[layer_name].dtype}")
                    model_sd = get_state_dict(net)
                    if layer_name in model_sd:
                        print(f"    Model expected tensor: shape={model_sd[layer_name].shape}, dtype={model_sd[layer_name].dtype}")
                    else:
                        print(f"    Layer {layer_name} not found directly in model's get_state_dict(). Check nested modules.")
                else: # Fallback parsing if the above regex fails
                    parts = str(e).split("`")
                    if len(parts) > 1 : layer_name = parts[1]
                    else: layer_name = "Unknown Layer"
                    print(f"  Mismatch in layer (fallback parsing): {layer_name}")

            except Exception as e_detail:
                print(f"    Error getting detailed mismatch info: {e_detail}")
        import traceback
        traceback.print_exc()
        raise e
        
    debug(f"build_transformer: State dict loaded. Model is ready.")

    return net, raw
