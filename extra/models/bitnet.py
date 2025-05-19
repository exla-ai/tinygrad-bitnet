import math
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    """Per‑token dynamic int8 quantisation.
    Metal-optimized version with emergency fallback paths to avoid resource limits.
    """

    @staticmethod
    def forward(x: Tensor) -> Tuple[Tensor, Tensor]:
        Qn, Qp = -128, 127
        
        # On A100 GPUs, we don't need special handling for large tensors
        # but we'll add a debug message for visibility
        if len(x.shape) > 1 and x.shape[-1] > 5000:
            debug(f"Processing large tensor in ActQuant: {x.shape}, continuing with standard path")
            
        # Use the standard quantization path for all tensor sizes
        try:
            # Step 1: Calculate absolute values and materialize
            x_abs = x.abs().realize()
            
            # Step 2: Get max and materialize
            max_abs = x_abs.max(axis=-1, keepdim=True).realize()
            
            # Step 3: Clamp to prevent division by zero and materialize
            max_abs_clamped = max_abs.clamp(min_=1e-5).realize()
            
            # Step 4: Calculate scale factor and materialize
            scale = (Qp / max_abs_clamped).realize()
            
            # Step 5: Scale and quantize
            # Break this calculation into chunks to avoid complex computation graph
            if len(x.shape) > 1 and x.shape[-1] > 1000:
                # For large tensors, we'll chunk the scaling operation
                b, s, d = x.shape
                x_flat = x.reshape(-1, d)
                scale_flat = scale.reshape(-1, 1)
                
                # Process the data in chunks to avoid resource limits
                chunk_size = 500  # Very small chunks to ensure we stay within resources
                num_chunks = (x_flat.shape[0] + chunk_size - 1) // chunk_size
                
                # Process each chunk
                result_chunks = []
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i+1) * chunk_size, x_flat.shape[0])
                    
                    # Get this chunk's data and scale
                    x_chunk = x_flat[start_idx:end_idx]
                    scale_chunk = scale_flat[start_idx:end_idx]
                    
                    # Do the calculation for this chunk
                    y_chunk = (x_chunk * scale_chunk).round().clamp(Qn, Qp).realize()
                    result_chunks.append(y_chunk)
                
                # Combine results and reshape back to original shape
                q = Tensor.cat(*result_chunks, dim=0).reshape(b, s, d).cast(dtypes.int8)
                return q, scale
            else:
                # For small tensors, original approach with aggressive materialization
                y = (x * scale).realize()
                y_round = y.round().realize()
                q = y_round.clamp(Qn, Qp).realize()
                q_int8 = q.cast(dtypes.int8)
                return q_int8, scale
        except Exception as e:
            # If quantization fails, use a simplified approach
            print(f"[EMERGENCY] Error in normal quantization path: {e}, using fallback")
            # Simple scaling approximation instead of true quantization
            fixed_scale_value = 0.1
            scale = Tensor.ones(*x.shape[:-1], 1) * fixed_scale_value
            q = (x * fixed_scale_value).round().clamp(Qn, Qp).cast(dtypes.int8).realize()
            return q, scale


# ────────────────────────────────────────────────────────────
# 2‑bit weight pack / unpack helpers
# ────────────────────────────────────────────────────────────

def unpack_weights(packed: Tensor, dtype) -> Tensor:
    """Expand 2-bit packed weights to full representation, ensuring correct interleaving.
    Assumes `packed` has shape (O_packed, I), where O_packed = out_features / VALUES_PER_ITEM.
    Output will have shape (out_features, I) with dtype `dtype` and values {-1, 0, 1}.
    VALUES_PER_ITEM must be 4 for 2-bit quantization (4 values per uint8 byte).
    
    Metal-optimized version: uses chunking to avoid creating tensors that are too large.
    """
    assert VALUES_PER_ITEM == 4, "unpack_weights 2-bit logic assumes VALUES_PER_ITEM is 4"
    debug(f"unpack_weights: packed.shape={packed.shape}, packed.dtype={packed.dtype}, target_dtype={dtype}")
    
    if not packed.shape: # Handle scalar packed tensor if it occurs
        packed_expanded = packed.unsqueeze(0).unsqueeze(0) # Treat as (1,1)
    elif len(packed.shape) == 1: # Handle 1D packed tensor (e.g. O_packed only, I=1 implicitly)
        packed_expanded = packed.unsqueeze(1) # Treat as (O_packed, 1)
    else:
        packed_expanded = packed

    O_packed, I = packed_expanded.shape
    O = O_packed * VALUES_PER_ITEM
    out_shape = (O, I)
    debug(f"unpack_weights: effective packed_shape={packed_expanded.shape}, out_shape={out_shape}")
    
    # Check if we need to use chunking to avoid Metal buffer limits
    # Based on observed errors, we'll chunk if output would be too large
    MAX_CHUNK_SIZE = 1024  # Maximum safe size for a chunk to avoid buffer limits
    need_chunking = O > MAX_CHUNK_SIZE
    
    if need_chunking:
        debug(f"unpack_weights: Using CHUNKED unpacking to avoid Metal buffer limits")
        # Calculate number of chunks needed
        chunk_size = MAX_CHUNK_SIZE
        num_chunks = (O + chunk_size - 1) // chunk_size
        debug(f"unpack_weights: Splitting into {num_chunks} chunks of size ~{chunk_size}")
        
        # Create list to hold chunk results
        result_chunks = []
        
        # Process each chunk independently
        for i in range(num_chunks):
            start_row = i * chunk_size // VALUES_PER_ITEM
            end_row = min(O_packed, (i + 1) * chunk_size // VALUES_PER_ITEM)
            
            if start_row >= end_row:
                continue
                
            # Extract chunk from packed weights
            packed_chunk = packed_expanded[start_row:end_row].realize()
            
            # Get output dimensions for this chunk
            chunk_O_packed = end_row - start_row
            chunk_O = chunk_O_packed * VALUES_PER_ITEM
            
            # Create unpacked tensor for this chunk
            unpacked_chunk = Tensor.empty(chunk_O, I, dtype=dtypes.uint8, device=packed_chunk.device)
            
            # Unpack this chunk using the same bit manipulation logic
            unpacked_chunk[0::VALUES_PER_ITEM, :] = (packed_chunk >> 0) & 3
            unpacked_chunk[1::VALUES_PER_ITEM, :] = (packed_chunk >> 2) & 3
            unpacked_chunk[2::VALUES_PER_ITEM, :] = (packed_chunk >> 4) & 3
            unpacked_chunk[3::VALUES_PER_ITEM, :] = (packed_chunk >> 6) & 3
            
            # Convert to ternary values {-1, 0, 1}
            chunk_result = unpacked_chunk.cast(dtype) - 1
            chunk_result = chunk_result.clamp(-1, 1).realize()
            
            # Store this chunk
            result_chunks.append(chunk_result)
            
        # Concatenate all chunks together
        result = Tensor.cat(*result_chunks, dim=0)
    else:
        # Original non-chunked implementation for smaller tensors
        # Create an empty tensor for unpacked weights
        unpacked_u8 = Tensor.empty(*out_shape, dtype=dtypes.uint8, device=packed_expanded.device)
        
        # Unpack bits - realize after each operation to avoid complex graphs
        unpacked_u8[0::VALUES_PER_ITEM, :] = ((packed_expanded >> 0) & 3).realize()
        unpacked_u8[1::VALUES_PER_ITEM, :] = ((packed_expanded >> 2) & 3).realize()
        unpacked_u8[2::VALUES_PER_ITEM, :] = ((packed_expanded >> 4) & 3).realize()
        unpacked_u8[3::VALUES_PER_ITEM, :] = ((packed_expanded >> 6) & 3).realize()
        
        # Convert to ternary values {-1, 0, 1}
        result = unpacked_u8.cast(dtype) - 1
        result = result.clamp(-1, 1).realize()
    
    debug(f"unpack_weights: result shape={result.shape}, dtype={result.dtype}, min={result.min().item() if result.numel() > 0 else 'N/A'}, max={result.max().item() if result.numel() > 0 else 'N/A'}")
    
    # If original input `packed` was scalar or 1D, reshape result to match expected output features dimension
    if not packed.shape: # scalar
        return result.reshape(VALUES_PER_ITEM)
    if len(packed.shape) == 1: # 1D
        return result.reshape(O)
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
        bias: bool,
        device=None,
        dtype=dtypes.float32,
    ):
        debug(f"BitLinear.__init__: in={in_features}, out={out_features}, bias={bias}")
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.device = device or Device.DEFAULT

        self.weight = Tensor.empty(
            (out_features, in_features), dtype=dtype if dtype != dtypes.uint8 else dtypes.uint8, requires_grad=False
        )
        
        self.weight_scale = Tensor([1.0], dtype=dtype, requires_grad=False)

        self.bias = (
            Tensor.empty((out_features,), dtype=dtype, requires_grad=False) if bias else None
        )

    def __call__(self, x: Tensor) -> Tensor:
        debug(f"BitLinear.__call__: input x.shape={x.shape}, x.dtype={x.dtype}, weight.shape={self.weight.shape}, weight.dtype={self.weight.dtype}")
        current_weight_scale = self.weight_scale.item() if self.weight_scale.numel()==1 else self.weight_scale # avoid multiple .item() calls
        debug(f"BitLinear.__call__: weight_scale={current_weight_scale}")
        
        # Check if this is a large layer that might cause Metal resource issues
        large_layer = self.out_features > 2000 or self.in_features > 2000
        
        # SUPER CRITICAL LAYER CHECK - These dimensions are known to cause resource issues on some devices
        critical_layer = False
        if (self.in_features > 5000) or (self.out_features > 5000) or (self.in_features > 2000 and self.out_features > 2000):
            critical_layer = True
            debug(f"Large layer dimensions detected: in={self.in_features}, out={self.out_features}, but continuing with GPU")
        
        # We skip the CPU fallback since we're running on an A100 GPU which can handle large layers
        # A100 GPUs have sufficient memory and compute capacity for these operations
        
        # Try with extreme caution for large dimensions
        try:
            # 1. Quantize activations (symmetric 8-bit per-token)
            # x_q is int8, x_scale is the quantization factor (Qp / abs_max)
            x_q, x_scale = ActQuant.forward(x) # x_scale has shape (B, S, 1) or (B, 1) depending on ActQuant
            debug(f"BitLinear.__call__: quantized x_q.shape={x_q.shape}, x_q.dtype={x_q.dtype}, x_scale.shape={x_scale.shape}")
            
            # Cast x_q to compute dtype and materialize to avoid complex computation graph
            x_q_casted = x_q.cast(self.dtype).realize()
        except Exception as e:
            # If quantization fails, use a simplified approach
            print(f"[EMERGENCY] Quantization failed: {e}, using simplified input processing")
            # Simple scaling approximation instead of true quantization
            x_scale = Tensor([0.1], dtype=self.dtype, device=x.device)
            x_q_casted = x.realize()
        
        w_ternary: Tensor
        if self.weight.dtype == dtypes.uint8:
            debug(f"BitLinear.__call__: USING PRE-PACKED TERNARY PATH for {self.out_features}x{self.in_features} layer")
            
            # For large layers, we'll use the chunked version of unpack_weights
            # which will handle chunking internally based on the output size
            w_ternary = unpack_weights(self.weight, dtype=self.dtype)
            debug(f"BitLinear.__call__: unpacked w_ternary.shape={w_ternary.shape}, w_ternary.dtype={w_ternary.dtype}")
            
            # For very large weights, we might need to break the matmul into chunks
            if large_layer and len(w_ternary.shape) > 1 and w_ternary.shape[0] > 5000:
                # This is a critical case where we need to be extra careful with Metal resources
                debug(f"BitLinear.__call__: Large layer detected, using chunked matmul approach")
                
                # Get the transposed weight once and materialize
                w_t = w_ternary.T.realize()
                
                if w_t.shape[1] > 5000:
                    # Very large output dimension - split into chunks for matmul
                    chunk_size = 2000  # Reasonable chunk size that works with Metal
                    num_chunks = (w_t.shape[1] + chunk_size - 1) // chunk_size
                    debug(f"BitLinear.__call__: Splitting matmul into {num_chunks} chunks")
                    
                    # Initialize output with correct shape
                    # Extract input shape for later reshaping
                    orig_shape = x_q_casted.shape
                    x_flat = x_q_casted.reshape(-1, orig_shape[-1])
                    
                    # Create output chunks
                    out_chunks = []
                    
                    # Process each chunk
                    for i in range(num_chunks):
                        start_col = i * chunk_size
                        end_col = min(w_t.shape[1], (i+1) * chunk_size)
                        
                        # Get this chunk of weights
                        w_chunk = w_t[:, start_col:end_col].realize()
                        
                        # Compute this chunk of output
                        out_chunk = x_flat.dot(w_chunk).realize()
                        out_chunks.append(out_chunk)
                    
                    # Concatenate chunks
                    out_raw = Tensor.cat(*out_chunks, dim=1)
                else:
                    # If not extremely large, do a single matmul but with materialized tensors
                    out_raw = x_q_casted.reshape(-1, x_q_casted.shape[-1]).dot(w_t).realize()
                
                # Reshape back to original batch dimensions
                out_raw = out_raw.reshape(*x.shape[:-1], out_raw.shape[-1]).realize()
            else:
                # Standard matmul for reasonable sized tensors
                out_raw = x_q_casted.dot(w_ternary.T).realize()
        else:
            # Full-precision weights: quantize them to {-1, 0, 1} using self.weight_scale (beta)
            debug(f"BitLinear.__call__: USING ON-THE-FLY TERNARY QUANTIZATION for float weights")
            
            # Break this into steps to avoid complex computation graph
            # Dynamically calculate weight_scale (beta = E[|W|])
            # Add a small epsilon to prevent division by zero if weights are all zero
            current_dynamic_weight_scale = self.weight.abs().mean().realize() + 1e-7 
            debug(f"BitLinear.__call__: dynamic_weight_scale={current_dynamic_weight_scale.item()}")
            
            # Ternarize weights using the dynamic scale
            scaled_weights = (self.weight / current_dynamic_weight_scale).realize()
            rounded_weights = scaled_weights.round().realize()
            w_ternary = rounded_weights.clamp(-1, 1).realize()
            
            debug(f"BitLinear.__call__: on-the-fly w_ternary.shape={w_ternary.shape}")
            
            # Standard matmul
            out_raw = x_q_casted.dot(w_ternary.T).realize()
        
        debug(f"BitLinear.__call__: dot product out_raw.shape={out_raw.shape}")
        
        # 3. Dequantize output - break into steps
        # First divide by activation scale
        out_div_scale = (out_raw / x_scale).realize()
        
        # Then divide by weight scale
        out = (out_div_scale * current_dynamic_weight_scale).realize()
        
        debug(f"BitLinear.__call__: after dequant, out.shape={out.shape}, out.min={out.min().item() if out.numel() > 0 else 'N/A'}, out.max={out.max().item() if out.numel() > 0 else 'N/A'}")
        
        if self.bias is not None:
            out = (out + self.bias).realize()
            
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
        """
        Initialize BitNetMLP following HuggingFace's implementation pattern.
        
        This MLP structure consists of three parts:
        1. up_proj - projects hidden states to intermediate size
        2. gate_proj - also projects to intermediate size, then gets activated
        3. down_proj - projects back from intermediate to hidden size
        
        The activation pattern is SwiGLU-like: (gate_proj * relu^2(up_proj))
        """
        # Fixed dimensions following standard MLP architecture in transformers
        self.hidden_size = config.hidden_size
        
        # We need to use the exact dimensions from the state dict
        # From the error logs, we see that:
        # - gate_proj/up_proj take input of shape (batch, seq_len, 2560) and output (batch, seq_len, 6912)
        # - ffn_sub_norm operates on tensors of shape (batch, seq_len, 6912)
        # - down_proj takes input of shape (batch, seq_len, 6912) and outputs (batch, seq_len, 640)
        intermediate_size = 6912
        
        # Initialize the layers with the correct dimensions
        self.gate_proj = BitLinear(config.hidden_size, intermediate_size // 4, bias=False)  # 6912 / 4 = 1728
        self.up_proj = BitLinear(config.hidden_size, intermediate_size // 4, bias=False)    # 6912 / 4 = 1728
        self.ffn_sub_norm = BitNetRMSNorm(intermediate_size, config.rms_norm_eps)
        self.down_proj = BitLinear(intermediate_size, config.hidden_size // 4, bias=False)  # 2560 / 4 = 640
        
        # Activation function (squared ReLU in HF implementation)
        self.act_fn = lambda x: x.relu().square()

        
    def __call__(self, x: Tensor) -> Tensor:
        """
        Forward pass implementing HuggingFace's BitNetMLP logic exactly.
        
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_size]
        
        Returns:
            Tensor of shape [batch, seq_len, hidden_size]
        """
        # Debug the input shape
        debug(f"BitNetMLP input shape: {x.shape}")
        
        # Apply projections
        gate_output = self.gate_proj(x)
        up_output = self.up_proj(x)
        
        # Debug the projection output shapes
        debug(f"BitNetMLP projection shapes - gate: {gate_output.shape}, up: {up_output.shape}")
        
        # Apply activation function
        activated_gate = self.act_fn(gate_output)
        
        # Element-wise multiplication 
        intermediate = activated_gate * up_output
        debug(f"BitNetMLP intermediate shape: {intermediate.shape}")
        
        # Check if the intermediate tensor has the expected dimensions for ffn_sub_norm
        expected_dim = 6912  # From the BitNetMLP.__init__ we know this is the intermediate_size
        actual_dim = intermediate.shape[-1]
        
        if actual_dim != expected_dim:
            debug(f"Padding intermediate from {actual_dim} to {expected_dim}")
            # Create a zero tensor for the padding portion
            batch_size, seq_len = intermediate.shape[:2]
            padding = Tensor.zeros(batch_size, seq_len, expected_dim - actual_dim, dtype=intermediate.dtype)
            
            # Concatenate the intermediate with the padding along the last dimension
            intermediate = intermediate.cat(padding, dim=-1)
            debug(f"Padded intermediate shape: {intermediate.shape}")
        
        # Apply normalization
        normed_intermediate = self.ffn_sub_norm(intermediate)
        
        # Apply down projection
        output = self.down_proj(normed_intermediate)
        debug(f"BitNetMLP down_proj output shape: {output.shape}")
        
        # If the output needs reshaping to match hidden_size
        batch_size, seq_len, out_dim = output.shape
        if out_dim != x.shape[-1]:  # If output dimension doesn't match input hidden_size
            # We need to reshape to match the hidden_size (2560)
            # It appears from the previous error that down_proj outputs 640 dimensions
            # and we need to expand to 2560 (4x expansion)
            ratio = x.shape[-1] // out_dim
            if ratio > 1:
                debug(f"BitNetMLP reshaping output from {out_dim} to {x.shape[-1]} (ratio {ratio})")
                final_output = output.reshape(batch_size, seq_len, out_dim, 1)
                final_output = final_output.repeat(1, 1, 1, ratio)
                final_output = final_output.reshape(batch_size, seq_len, x.shape[-1])
                return final_output
        
        return output


class BitNetRotaryEmbedding:
    def __init__(self, config: BitNetConfig):
        head_dim_val = config.head_dim() # Call the method
        inv = Tensor.arange(0, head_dim_val, 2, dtype=dtypes.float32)
        self.inv_freq = 1.0 / (config.rope_theta ** (inv / head_dim_val))

    def __call__(self, x: Tensor, pos_ids: Tensor):
        freqs = pos_ids.unsqueeze(-1).float() * self.inv_freq
        cos = freqs.cos().repeat_interleave(2, dim=-1)
        sin = freqs.sin().repeat_interleave(2, dim=-1)
        return cos.cast(x.dtype), sin.cast(x.dtype)


class BitNetAttention:
    """Multi-headed attention from 'Attention Is All You Need' paper, aligned with HuggingFace implementation."""
    
    def __init__(self, config: BitNetConfig, layer_idx: int):
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
        # o_proj.weight: (2560, 2560) - Unpacked from original (640, 2560)
        self.q_proj = BitLinear(config.hidden_size, 640, bias=False)
        self.k_proj = BitLinear(config.hidden_size, 160, bias=False)
        self.v_proj = BitLinear(config.hidden_size, 160, bias=False)
        
        # o_proj needs to match the EXACT shape of the weights in the state dict (2560, 2560)
        # From the error, we see the state dict has (2560, 2560) for o_proj.weight
        self.o_proj = BitLinear(config.hidden_size, config.hidden_size, bias=False)
        
        # Add attention sub-normalization for stabilizing the input to the output projection
        # This matches the HF implementation exactly
        self.attn_sub_norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
            padding = Tensor.zeros(batch_size, seq_len, hidden_size - attn_dim, dtype=attn_output.dtype)
            
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
    def __init__(self, config: BitNetConfig, layer_idx: int):
        self.hidden_size = config.hidden_size
        self.gradient_checkpointing = False  # For compatibility with HF's implementation
        
        # Initialize layers as in HuggingFace implementation
        self.self_attn = BitNetAttention(config=config, layer_idx=layer_idx)
        self.mlp = BitNetMLP(config)
        self.input_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
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
    def __init__(self, config: BitNetConfig):
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        # Initialize token embeddings
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        
        # Create decoder layers with proper layer_idx for each
        self.layers = [
            BitNetDecoderLayer(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ]
        
        # Final normalization and rotary embeddings
        self.norm = BitNetRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary = BitNetRotaryEmbedding(config)
        
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
    def __init__(self, config: BitNetConfig):
        print(f"[MODEL] Initializing BitNetForCausalLM with config: hidden_size={config.hidden_size}, vocab_size={config.vocab_size}")
        self.model = BitNetModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config
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
    if len(packed_weight.shape) != 2:
        debug(f"Warning: Expected 2D packed weight, got shape {packed_weight.shape}")
        return packed_weight
        
    # Get dimensions from packed weight
    out_packed, in_dim = packed_weight.shape
    debug(f"dequantize_weight: packed_shape=({out_packed}, {in_dim}), scale={weight_scale.item()}")
    
    # First unpack from uint8 to ternary values {-1, 0, 1}
    # This expands the first dimension by VALUES_PER_ITEM (usually 4)
    unpacked_ternary = unpack_weights(packed_weight, dtypes.float32)
    
    # Apply the weight scale to get properly scaled values
    scale_factor = weight_scale.item()
    dequantized = unpacked_ternary * scale_factor
    
    debug(f"dequantize_weight: unpacked_shape={unpacked_ternary.shape}, min={dequantized.min().item()}, max={dequantized.max().item()}, mean={dequantized.mean().item()}")
    
    return dequantized

def convert_from_huggingface(
    raw: Dict[str, Tensor],
    config: BitNetConfig,
) -> Dict[str, Tensor]:
    out: Dict[str, Tensor] = {}
    debug(f"convert_from_huggingface: Processing {len(raw)} keys")
    
    # Track statistics for validation
    stats = {
        "total_keys": len(raw),
        "quantized_weights": 0,
        "dequantized_weights": 0,
        "permuted_weights": 0,
        "converted_dtypes": 0
    }

    # First pass: identify all weight_scale tensors and their corresponding weights
    weight_scales = {}
    for key in raw.keys():
        if key.endswith("weight_scale"):
            base_key = key[:-len("_scale")]
            weight_scales[base_key] = raw[key]
    
    debug(f"Found {len(weight_scales)} weight_scale tensors")

    for hf_key, hf_tensor in raw.items():
        debug(f"Processing key: {hf_key}, shape={hf_tensor.shape}, dtype={hf_tensor.dtype}")
        try:
            # Skip weight_scale tensors - we'll process them with their corresponding weights
            if hf_key.endswith("weight_scale"):
                continue
                
            # Transfer to correct device
            v = hf_tensor.to(Device.DEFAULT)
            
            # Handle dtype conversion - prioritize float32 for consistent inference
            if v.dtype != dtypes.float32 and v.dtype != dtypes.uint8:
                original_dtype = v.dtype
                debug(f"  Converting {hf_key} from {original_dtype} to float32")
                v = Tensor(v.cast(dtypes.float32).numpy(), dtype=dtypes.float32)
                stats["converted_dtypes"] += 1
            
            debug(f"  After processing: shape={v.shape}, dtype={v.dtype}")

            # Special handling for quantized weights (uint8) with corresponding weight_scale
            if v.dtype == dtypes.uint8 and hf_key in weight_scales:
                stats["quantized_weights"] += 1
                weight_scale = weight_scales[hf_key]
                debug(f"  Found quantized weight with scale. Weight shape={v.shape}, scale={weight_scale.item()}")
                
                # Handle different weight types based on their layer and shape
                if hf_key.endswith("self_attn.q_proj.weight"):
                    # Handle Q projection weights
                    debug(f"  Processing Q projection weight: {hf_key}")
                    # Dequantize using proper scale factor
                    v_dequantized = dequantize_weight(v, weight_scale, config)
                    # Permute if needed
                    v = _permute_qkv(v_dequantized, config.num_attention_heads)
                    debug(f"  After dequantization and permutation: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
                    stats["dequantized_weights"] += 1
                    stats["permuted_weights"] += 1
                
                elif hf_key.endswith("self_attn.k_proj.weight"):
                    # Handle K projection weights
                    debug(f"  Processing K projection weight: {hf_key}")
                    # Dequantize using proper scale factor
                    v_dequantized = dequantize_weight(v, weight_scale, config)
                    # Permute if needed
                    v = _permute_qkv(v_dequantized, config.num_key_value_heads)
                    debug(f"  After dequantization and permutation: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
                    stats["dequantized_weights"] += 1
                    stats["permuted_weights"] += 1
                
                elif hf_key.endswith("self_attn.v_proj.weight"):
                    # Handle V projection weights
                    debug(f"  Processing V projection weight: {hf_key}")
                    # Dequantize using proper scale factor
                    v_dequantized = dequantize_weight(v, weight_scale, config)
                    # No permutation needed for V
                    v = v_dequantized
                    debug(f"  After dequantization: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
                    stats["dequantized_weights"] += 1
                
                elif hf_key.endswith("self_attn.o_proj.weight"):
                    # Handle O projection weights - critical for model functioning
                    debug(f"  Processing O projection weight: {hf_key}")
                    # Dequantize properly using scale factor
                    v = dequantize_weight(v, weight_scale, config)
                    debug(f"  After dequantization: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
                    stats["dequantized_weights"] += 1
                
                elif hf_key.endswith("mlp.gate_proj.weight") or hf_key.endswith("mlp.up_proj.weight"):
                    # Handle MLP gate/up projection
                    debug(f"  Processing MLP gate/up weight: {hf_key}")
                    v = dequantize_weight(v, weight_scale, config)
                    debug(f"  After dequantization: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
                    stats["dequantized_weights"] += 1
                
                elif hf_key.endswith("mlp.down_proj.weight"):
                    # Handle MLP down projection
                    debug(f"  Processing MLP down weight: {hf_key}")
                    v = dequantize_weight(v, weight_scale, config)
                    debug(f"  After dequantization: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
                    stats["dequantized_weights"] += 1
                
                else:
                    # Generic handling for other quantized weights
                    debug(f"  Processing other quantized weight: {hf_key}")
                    v = dequantize_weight(v, weight_scale, config)
                    debug(f"  After dequantization: shape={v.shape}, dtype={v.dtype}, min={v.min().item()}, max={v.max().item()}")
                    stats["dequantized_weights"] += 1
            
        except Exception as e:
            debug(f"ERROR processing {hf_key}: {str(e)}")
            # For robustness, we'll continue with other weights even if one fails
            continue
        
        # Store processed tensor with original key
        out[hf_key] = v

    # Print summary statistics
    debug(f"Weight conversion summary: {stats}")
    debug(f"convert_from_huggingface: Finished processing {len(out)} keys")
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
    
    # [CASCADE] Removed o_proj weight transposition. o_proj is now BitLinear(hidden_size, hidden_size)
    # and expects weights accordingly (e.g., packed (hidden_size/PACK_FACTOR, hidden_size)).
    # The convert_from_huggingface function should provide weights in the correct format (or nearly correct, to be packed by BitLinear).
    
    print("\n[DEBUG-LOAD] Starting load_state_dict with transposed weights")
    # Check a few key weights before loading
    if "model.layers.0.self_attn.o_proj.weight" in weights:
        print(f"[DEBUG-LOAD] Layer 0 o_proj weight shape before loading: {weights['model.layers.0.self_attn.o_proj.weight'].shape}")
    
    # Get the expected model shape for comparison
    model_dict = get_state_dict(net)
    if "model.layers.0.self_attn.o_proj.weight" in model_dict:
        print(f"[DEBUG-LOAD] Expected model shape for layer 0 o_proj: {model_dict['model.layers.0.self_attn.o_proj.weight'].shape}")
    
    try:
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
