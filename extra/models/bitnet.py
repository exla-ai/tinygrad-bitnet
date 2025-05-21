import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import sys

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

def unpack_weights(arr: np.ndarray, weight_scale: float, dtype=dtypes.int8) -> Tensor:
    """Unpacks 2-bit weights (packed into uint8) into ternary {-1, 0, 1} float32/float16 tensor.
    Handles chunking for very large tensors to avoid Metal buffer limits.
    Args:
        packed: uint8 Tensor with shape [out_features//4, in_features]
        target_dtype: The desired dtype for the output tensor (e.g., dtypes.float32)
    Returns:
        Tensor with shape [out_features, in_features] and dtype=target_dtype.
    """
    debug(f"unpack_weights: packed.shape={arr.shape}, packed.dtype={arr.dtype}, target_dtype={dtype}")
    VALUES_PER_ITEM = 4  # Number of 2-bit values packed into one uint8
    out_shape = (arr.shape[0] * VALUES_PER_ITEM, arr.shape[1])
    result: Tensor

    # Determine if chunking is needed (based on output rows, e.g., out_features)
    # Heuristic: if out_features > 8192 (e.g., 8192*6912 matrix), chunk it.
    # This corresponds to packed.shape[0] > 2048
    # Metal buffer limit is often around 256MB-1GB. A 8192x6912 float32 is ~220MB.
    # Let's use a more conservative limit for packed.shape[0] to be safe.
    CHUNK_THRESHOLD_PACKED_ROWS = getenv("UNPACK_CHUNK_ROWS", 1024) # tunable via env var

    if arr.shape[0] > CHUNK_THRESHOLD_PACKED_ROWS:
        debug(f"unpack_weights: Using CHUNKED unpacking. packed_rows={arr.shape[0]} > threshold={CHUNK_THRESHOLD_PACKED_ROWS}")
        num_chunks = (arr.shape[0] + CHUNK_THRESHOLD_PACKED_ROWS - 1) // CHUNK_THRESHOLD_PACKED_ROWS
        debug(f"unpack_weights: Splitting into {num_chunks} chunks of size ~{CHUNK_THRESHOLD_PACKED_ROWS} packed rows")
        result_chunks = []

        for i in range(num_chunks):
            start_row = i * CHUNK_THRESHOLD_PACKED_ROWS
            end_row = min((i + 1) * CHUNK_THRESHOLD_PACKED_ROWS, arr.shape[0])
            packed_chunk_np = arr[start_row:end_row] # Get NumPy chunk from Tensor
            
            # Unpack bits for this chunk using NumPy
            chunk_out_dim = packed_chunk_np.shape[0] * VALUES_PER_ITEM
            unpacked_chunk_np = np.empty((chunk_out_dim, packed_chunk_np.shape[1]), dtype=np.uint8)

            unpacked_chunk_np[0::VALUES_PER_ITEM, :] = (packed_chunk_np >> 0) & 3
            unpacked_chunk_np[1::VALUES_PER_ITEM, :] = (packed_chunk_np >> 2) & 3
            unpacked_chunk_np[2::VALUES_PER_ITEM, :] = (packed_chunk_np >> 4) & 3
            unpacked_chunk_np[3::VALUES_PER_ITEM, :] = (packed_chunk_np >> 6) & 3
            
            # Convert to ternary values {-1, 0, 1} using NumPy
            intermediate_np_values = unpacked_chunk_np.astype("f4") - 1.0 # float32 with {-1,0,1,2}
            clamped_np_values = np.clip(intermediate_np_values, -1.0, 1.0) # float32 with {-1,0,1}
            
            # Create Tensor for this chunk and realize
            chunk_tensor = Tensor(
                clamped_np_values * weight_scale,
                dtype=dtype, # CRITICAL: Use target_dtype
                requires_grad=False,
                device=Device.DEFAULT
            ).realize()
            result_chunks.append(chunk_tensor)
            debug(f"unpack_weights: Processed chunk {i+1}/{num_chunks}, chunk_tensor.shape={chunk_tensor.shape}")
            
        result = Tensor.cat(*result_chunks, dim=0).contiguous() if result_chunks else Tensor([], shape=(0, *out_shape[1:]), dtype=dtype)
    else:
        debug(f"unpack_weights: Using NON-CHUNKED unpacking for shape {arr.shape}")
        packed_np = arr # Move to CPU and get NumPy array
        
        unpacked_np = np.empty(out_shape, dtype=np.uint8)
        unpacked_np[0::VALUES_PER_ITEM, :] = (packed_np >> 0) & 3
        unpacked_np[1::VALUES_PER_ITEM, :] = (packed_np >> 2) & 3
        unpacked_np[2::VALUES_PER_ITEM, :] = (packed_np >> 4) & 3
        unpacked_np[3::VALUES_PER_ITEM, :] = (packed_np >> 6) & 3
        
        intermediate_np_values = unpacked_np.astype("f4") - 1.0 # float32 with {-1,0,1,2}
        clamped_np_values = np.clip(intermediate_np_values, -1.0, 1.0) # float32 with {-1,0,1}
        
        result = Tensor(
            clamped_np_values * weight_scale,
            dtype=dtype, # CRITICAL: Use target_dtype
            requires_grad=False,
            device=Device.DEFAULT 
        ).realize()
    
    debug(f"unpack_weights: final result shape={result.shape}, dtype={result.dtype}, min={result.min().item() if result.numel() > 0 else 'N/A'}, max={result.max().item() if result.numel() > 0 else 'N/A'}")
    assert result.shape == out_shape, f"Shape mismatch: expected {out_shape}, got {result.shape}"
    assert result.dtype == dtype, f"Dtype mismatch: expected {dtype}, got {result.dtype}"
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
        token = int(logits.argmax().item())
        print(f"[SAMPLE] Greedy sampling result: token={token}")
        return token

    if af or ap:
        if not hasattr(sample, "alpha_counter"):
            sample.alpha_counter = Tensor.zeros_like(logits, dtype=dtypes.int32)
        logits = logits - (sample.alpha_counter * af + (sample.alpha_counter > 0) * ap)

    # mask NaNs
    logits = (logits != logits).where(-float("inf"), logits)
    probs = (logits / temp).softmax()
    print(f"[SAMPLE] After softmax: min={probs.min().item():.6f}, max={probs.max().item():.6f}")
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    print(f"[SAMPLE] Top 5 probs: {probs_sort[:5].flatten().numpy().tolist() if probs_sort.shape[0] >= 5 else 'less than 5 tokens'}")
    print(f"[SAMPLE] Top 5 indices: {probs_idx[:5].flatten().numpy().tolist() if probs_idx.shape[0] >= 5 else 'less than 5 tokens'}")

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

        # Always use uint8 for weights to enable bit manipulation in unpack_weights
        if transposed:
            # If transposed, the weights have dimensions reversed compared to normal linear layer
            self.weight = Tensor.empty(
                (in_features, out_features), dtype=dtypes.uint8, requires_grad=False, device=self.device
            )
        else:
            self.weight = Tensor.empty(
                (out_features, in_features), dtype=dtypes.uint8, requires_grad=False, device=self.device
            )
        
        self.weight_scale = Tensor([1.0], dtype=dtype, requires_grad=False, device=self.device)

    def __call__(self, x: Tensor) -> Tensor:
        sign = unpack_weights(self.weight.numpy(), self.weight_scale[0], x.dtype)
        w_f32 = sign * self.weight_scale[0]
        
        if self.transposed:
            # For transposed weights, we don't transpose again during the forward pass
            out = x @ w_f32
        else:
            out = x @ w_f32.T
            
        return out

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
        # Use dimensions that match the pretrained weights
        hidden_size = config.hidden_size  # 2560
        
        # IMPORTANT: Use the correct intermediate sizes
        # For our ffn_ln weight normalization, we need size 1728
        self.inter_dim = 1728
        
        # For the projection layers, we need to match the state dictionary dimensions
        self.gate_proj = BitLinear(hidden_size, self.inter_dim, device=device)   # (2560, 1728)
        self.up_proj = BitLinear(hidden_size, self.inter_dim, device=device)     # (2560, 1728)
        
        # Use layer normalization with inter_dim (1728)
        self.ffn_ln = BitNetRMSNorm(self.inter_dim, eps=config.rms_norm_eps, device=device)
        
        # The down_proj weights in the state dict have shape (640, 6912)
        self.down_proj = BitLinear(640, 6912, transposed=True, device=device)

    def __call__(self, x: Tensor) -> Tensor:
        # Get initial batch dimensions
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Apply gate projection and activation
        gate = self.gate_proj(x)          # Shape: (batch, seq_len, inter_dim)
        gate = gate.relu() ** 2          # Apply activation
        
        # Apply up projection
        up = self.up_proj(x)             # Shape: (batch, seq_len, inter_dim)
        
        # Element-wise multiplication
        h = gate * up                    # Shape: (batch, seq_len, inter_dim)
        
        # Apply layer normalization
        h = self.ffn_ln(h)               # Shape: (batch, seq_len, inter_dim)
        
        # Reshape to match down_proj input dimension
        # The down_proj expects input of shape (batch, seq_len, 640)
        h_reshaped = h.reshape(batch_size, seq_len, 640, -1).sum(-1)
        
        # Apply down projection
        return self.down_proj(h_reshaped)  # Final shape: (batch, seq_len, hidden_size)


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
        self.q_proj = BitLinear(config.hidden_size, 640, device=device)
        self.k_proj = BitLinear(config.hidden_size, 160, device=device)
        self.v_proj = BitLinear(config.hidden_size, 160, device=device)
        
        # o_proj needs to match the EXACT shape of the weights in the state dict (640, 2560)
        # From the error logs, we see the state dict expects (640, 2560) for o_proj.weight
        self.o_proj = BitLinear(config.hidden_size, 640, device=device)
        
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
        print(f"[MODEL-CALL] Logits shape: {logits.shape}, min={logits.min().item():.3f}, max={logits.max().item():.3f}")
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
    Minimal converter that creates a completely new set of tensors with zeros, 
    but sets the proper scale factors from the original weights.
    This approach avoids device mixing issues by not trying to copy data between devices.
    """
    # Define helper for creating shapes based on config
    def get_weight_shape(key_name):
        """Get the expected shape for a weight tensor based on its name"""
        if key_name == "model.embed_tokens.weight":
            return (config.vocab_size, config.hidden_size)
        if key_name == "model.norm.weight":
            return (config.hidden_size,)
        if key_name.endswith(".input_layernorm.weight") or \
           key_name.endswith(".post_attention_layernorm.weight") or \
           key_name.endswith(".self_attn.attn_sub_norm.weight"):
            return (config.hidden_size,)
        if key_name.endswith(".mlp.ffn_sub_norm.weight"):
            # Use 1728 instead of config.intermediate_size (6912)
            # This matches the actual dimension used in our BitNetMLP implementation
            return (1728,)
        if key_name.endswith(".mlp.gate_proj.weight") or \
           key_name.endswith(".mlp.up_proj.weight"):
            if config.linear_class == "autobitlinear":
                return (config.intermediate_size // 4, config.hidden_size)
            return (config.intermediate_size, config.hidden_size)
        if key_name.endswith(".mlp.down_proj.weight"):
            if config.linear_class == "autobitlinear":
                return (config.hidden_size // 4, config.intermediate_size)
            return (config.hidden_size, config.intermediate_size)
        if key_name.endswith(".self_attn.q_proj.weight"):
            if config.linear_class == "autobitlinear":
                return (config.num_attention_heads * config.head_dim() // 4, config.hidden_size)
            return (config.num_attention_heads * config.head_dim(), config.hidden_size)
        if key_name.endswith(".self_attn.k_proj.weight") or \
           key_name.endswith(".self_attn.v_proj.weight"):
            if config.linear_class == "autobitlinear":
                return (config.num_key_value_heads * config.head_dim() // 4, config.hidden_size)
            return (config.num_key_value_heads * config.head_dim(), config.hidden_size)
        if key_name.endswith(".self_attn.o_proj.weight"):
            if config.linear_class == "autobitlinear":
                return (config.hidden_size // 4, config.num_attention_heads * config.head_dim())
            return (config.hidden_size, config.num_attention_heads * config.head_dim())
        # Default case
        return None
    
    target_device = Device.DEFAULT
    print(f"[DEBUG] Creating weights on {target_device}")
    
    # Create dictionaries for storing our results
    out: Dict[str, Tensor] = {}
    scales: Dict[str, float] = {}
    
    # First extract all scale factors
    for k, v in raw.items():
        if k.endswith(".weight_scale"):
            try:
                # Extract scale as a float value
                if hasattr(v, "item"):
                    # Cast to float32 before calling item() to handle bfloat16 correctly
                    # This ensures proper handling of bfloat16 tensors
                    base_key = k.replace(".weight_scale", "")
                    scales[base_key] = v.cast(dtypes.float32).item()
                else:
                    # Fall back to string conversion (should ideally not be hit for Tensor objects)
                    base_key = k.replace(".weight_scale", "")
                    scales[base_key] = float(str(v).strip())
                print(f"[DEBUG] Extracted scale for {base_key}: {scales[base_key]}")
                sys.stdout.flush()
            except Exception as e_scale:
                print(f"[ERROR] Failed to extract scale for {k}: {e_scale}")
                print(f"Tensor details: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                sys.stdout.flush()
                base_key = k.replace(".weight_scale", "")
                scales[base_key] = 1.0 # Default or placeholder
    
    # Now create output tensors
    try:
        for k, v_cpu in raw.items(): # v_cpu is Tensor on CPU from load_sharded
            current_key_being_processed = k
            if k.endswith(".weight_scale"): continue # Already processed into scales dict

            print(f"[DEBUG] Processing tensor for {k} with original shape {v_cpu.shape}, dtype {v_cpu.dtype}. Target device: {target_device}")
            sys.stdout.flush()

            final_tensor: Optional[Tensor] = None
            nk = k.replace(".weight", "")

            if nk in scales and "weight" in k and not any(norm_name in k for norm_name in ["norm", "layernorm"]):
                # This is a BitLinear weight that needs to be quantized.
                # unpack_weights will handle moving to the device (Device.DEFAULT, which should be target_device).
                print(f"[DEBUG] Quantizing {k} using scale {scales[nk]}")
                sys.stdout.flush()
                # The raw model weights (v_cpu) are usually bfloat16. unpack_weights expects a numpy array.
                # Convert bfloat16 to float32 before calling numpy()
                if v_cpu.dtype == dtypes.bfloat16:
                    v_cpu = v_cpu.cast(dtypes.float32)
                final_tensor = unpack_weights(v_cpu.numpy(), scales[nk], dtype=dtypes.int8)
            elif "lm_head.weight" == k:
                print(f"[DEBUG] Processing lm_head.weight {k}. Ensuring float32 on {target_device}.")
                sys.stdout.flush()
                temp_tensor_gpu = v_cpu.to(target_device)
                if temp_tensor_gpu.dtype != dtypes.float32:
                    final_tensor = temp_tensor_gpu.cast(dtypes.float32)
                else:
                    final_tensor = temp_tensor_gpu
            else:
                # Regular tensor (e.g., embeddings, layernorm weights)
                print(f"[DEBUG] Moving regular tensor {k} to {target_device}.")
                sys.stdout.flush()
                final_tensor = v_cpu.to(target_device)
            
            if final_tensor is not None:
                out[k] = final_tensor
                print(f"[DEBUG] Successfully processed tensor for {k} to {final_tensor.device}. Shape: {final_tensor.shape}, Dtype: {final_tensor.dtype}")
                sys.stdout.flush()
            else:
                # This case should ideally not be reached if logic is correct.
                print(f"[ERROR] Critical: final_tensor was not set for key {k}. Skipping this tensor.")
                sys.stdout.flush()

    except RuntimeError as e:
        print(f"[ERROR] RuntimeError during tensor processing for key: '{current_key_being_processed}'")
        sys.stdout.flush()
        if "CUDA out of memory" in str(e) or "cudaErrorMemoryAllocation" in str(e):
            print(f"[ERROR] CUDA out of memory specific message: {e}")
        else:
            print(f"[ERROR] Non-OOM RuntimeError message: {e}")
        print(f"Current out keys (before error on '{current_key_being_processed}'): {list(out.keys())}")
        sys.stdout.flush()
        # import traceback # Optional for more detailed local debugging
        # traceback.print_exc() # Optional
        raise # Re-raises the RuntimeError 'e'

    except Exception as e: # Catch any other Exception
        print(f"[ERROR] Unexpected error during tensor processing for key: '{current_key_being_processed}': {e}")
        print(f"Type of exception: {type(e)}")
        print(f"Current out keys (before error on '{current_key_being_processed}'): {list(out.keys())}")
        sys.stdout.flush()
        # import traceback # Optional for more detailed local debugging
        # traceback.print_exc() # Optional
        raise # Re-raises the general Exception 'e'
    
    # Return the processed tensors
    print(f"[DEBUG] convert_from_huggingface: Processing {len(out)} keys")
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

    # Load weights to CPU first to avoid device mismatches
    raw = safe_load(str(sf_path))
    debug(f"build_transformer: Loaded {len(raw)} raw tensors from safetensors file")
    
    # Print key shapes/dtypes for inspection
    if DEBUG_PRINT:
        for k_debug, v_debug in list(raw.items())[:5]:  # Show first 5 for brevity
            debug(f"  Raw tensor: {k_debug}, shape={v_debug.shape}, dtype={v_debug.dtype}")
    
    debug(f"build_transformer: Converting weights from huggingface format")
    weights = convert_from_huggingface(raw, config)
    debug(f"build_transformer: Converted {len(weights)} tensors")

    # Move all weights to the target device before loading
    target_device = Device.DEFAULT
    debug(f"build_transformer: Moved all weights to {target_device}")

    # Fill tensors with values from raw Safetensors, applying binarization for BitLinear
    debug(f"[DEBUG-FILL] Starting data fill and binarization process for {len(raw)} raw tensors into {len(weights)} target tensors.")
    for k_raw, v_raw in raw.items(): # raw is the original HuggingFace weights
        if k_raw.endswith(".weight_scale"):
            # These are already correctly processed Tensors in 'weights' dict (from convert_from_huggingface)
            debug(f"[DEBUG-FILL] Skipping already processed scale: {k_raw}")
            continue

        if k_raw in weights: # Check if this key exists in our target 'weights' structure
            target_tensor = weights[k_raw]
            
            # Determine if it's a BitLinear weight tensor based on its dtype in our 'weights' map
            is_bitlinear_weight = target_tensor.dtype == dtypes.int8

            if is_bitlinear_weight:
                debug(f"[DEBUG-FILL] Binarizing BitLinear weight: {k_raw}, raw dtype: {v_raw.dtype} to target dtype: {target_tensor.dtype}")
                
                w_float32 = None
                if v_raw.dtype == dtypes.bfloat16:
                    w_float32 = v_raw.cast(dtypes.float32)
                elif v_raw.dtype == dtypes.float16:
                    w_float32 = v_raw.half().float() # Convert float16 to float32
                elif v_raw.dtype == dtypes.float32:
                    w_float32 = v_raw
                else:
                    print(f"[ERROR-FILL] Unsupported raw dtype {v_raw.dtype} for BitLinear weight {k_raw}")
                    continue

                binarized_w_float = w_float32.sign() 
                target_tensor.assign(binarized_w_float.cast(dtypes.int8).lazydata.realize())
                debug(f"[DEBUG-FILL]   Assigned binarized {k_raw} to weights['{k_raw}'] with dtype {target_tensor.dtype}")

            else: # For non-BitLinear weights (embeddings, layernorms, biases if any) or other tensors
                debug(f"[DEBUG-FILL] Assigning non-BitLinear/other tensor: {k_raw}, raw dtype: {v_raw.dtype}, target dtype: {target_tensor.dtype}")
                
                temp_tensor_val = None
                if v_raw.dtype == dtypes.bfloat16:
                    temp_tensor_val = v_raw.cast(target_tensor.dtype)
                elif v_raw.dtype == dtypes.float16:
                    if target_tensor.dtype == dtypes.float32:
                        temp_tensor_val = v_raw.half().float().cast(target_tensor.dtype)
                    else: 
                        temp_tensor_val = v_raw.cast(target_tensor.dtype)
                elif v_raw.dtype == dtypes.float32:
                    temp_tensor_val = v_raw.cast(target_tensor.dtype)
                else:
                    print(f"[ERROR-FILL] Unsupported raw dtype {v_raw.dtype} for tensor {k_raw}")
                    continue
                
                if temp_tensor_val is not None:
                    if temp_tensor_val.device != target_tensor.device:
                        # stream it from DISK (or CPU) to the same device as the destination
                        temp_tensor_val = temp_tensor_val.to(target_tensor.device)
                    target_tensor.assign(temp_tensor_val.realize())
                    debug(f"[DEBUG-FILL]   Assigned {k_raw} to weights['{k_raw}'] with dtype {target_tensor.dtype}")
        else:
            debug(f"[WARN-FILL] Tensor {k_raw} from raw weights not found in 'weights' dict. Skipping assignment.")


    # Use consume_prefix to strip "model." and load into net.model
    # strict=False will report missing/unexpected keys via consume_prefix's behavior
    debug(f"build_transformer: Loading state dict into model")
    # Check for weight format/dtype mismatches
    if DEBUG_PRINT:
        # Sample a MLP gate projection weight to check if format matches
        if "model.layers.0.mlp.gate_proj.weight" in weights:
            debug(f"Gate projection weight dtype: {weights['model.layers.0.mlp.gate_proj.weight'].dtype}")
            debug(f"Gate projection weight shape: {weights['model.layers.0.mlp.gate_proj.weight'].shape}")
            gate_weight = weights['model.layers.0.mlp.gate_proj.weight'].realize()
            debug(f"Weight stats: min={gate_weight.min().item():.6f}, max={gate_weight.max().item():.6f}, mean={gate_weight.mean().item():.6f}")
    
    print("\n[DEBUG-LOAD] Starting load_state_dict with transposed weights")
    # Check a few key weights before loading
    if "model.layers.0.self_attn.o_proj.weight" in weights:
        print(f"[DEBUG-LOAD] Layer 0 o_proj weight shape before loading: {weights['model.layers.0.self_attn.o_proj.weight'].shape}")
    
    # Get the expected model shape for comparison
    model_dict = get_state_dict(net)
    if "model.layers.0.self_attn.o_proj.weight" in model_dict:
        print(f"[DEBUG-LOAD] Expected model shape for layer 0 o_proj: {model_dict['model.layers.0.self_attn.o_proj.weight'].shape}")
    
    try:
        # Removed device parameter which was causing the error
        load_state_dict(net, weights, strict=False)
        print("[DEBUG-LOAD] State dictionary loaded successfully")
    except Exception as e:
        print(f"[DEBUG-LOAD] ERROR during load_state_dict: {e}")
        # Print more details about the specific mismatch
        if "Shape mismatch" in str(e):
            layer_name = str(e).split("`")[1].split("\'")[0]
            print(f"[DEBUG-LOAD] Mismatch in layer {layer_name}")
            if layer_name in weights:
                print(f"[DEBUG-LOAD] State dict shape: {weights[layer_name].shape}")
            if layer_name in model_dict:
                print(f"[DEBUG-LOAD] Model expected shape: {model_dict[layer_name].shape}")
        raise e
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
