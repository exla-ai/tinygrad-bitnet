from __future__ import annotations
import math
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.device import is_dtype_supported
from tinygrad.helpers import prod, make_tuple, flatten
from tinygrad.nn import optim, state, datasets  # noqa: F401

class BatchNorm:
  """
  Applies Batch Normalization over a 2D or 3D input.

  - Described: https://paperswithcode.com/method/batch-normalization
  - Paper: https://arxiv.org/abs/1502.03167v3

  See: `Tensor.batchnorm`

  ```python exec="true" session="tensor"
  from tinygrad import Tensor, dtypes, nn
  import numpy as np
  np.set_printoptions(precision=4)
  ```

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.BatchNorm(3)
  t = Tensor.rand(2, 3, 4, 4)
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

    self.weight: Tensor|None = Tensor.ones(sz) if affine else None
    self.bias: Tensor|None = Tensor.zeros(sz) if affine else None

    self.num_batches_tracked = Tensor.zeros(1, dtype='long' if is_dtype_supported(dtypes.long) else 'int', requires_grad=False)
    if track_running_stats: self.running_mean, self.running_var = Tensor.zeros(sz, requires_grad=False), Tensor.ones(sz, requires_grad=False)

  def calc_stats(self, x:Tensor) -> tuple[Tensor, Tensor]:
    shape_mask: list[int] = [1, -1, *([1]*(x.ndim-2))]
    if self.track_running_stats and not Tensor.training: return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)
    # This requires two full memory accesses to x
    # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
    # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
    batch_mean = x.mean(axis=(reduce_axes:=tuple(x for x in range(x.ndim) if x != 1)))
    y = (x - batch_mean.detach().reshape(shape=shape_mask))  # d(var)/d(mean) = 0
    batch_var = (y*y).mean(axis=reduce_axes)
    return batch_mean, batch_var

  def __call__(self, x:Tensor) -> Tensor:
    batch_mean, batch_var = self.calc_stats(x)
    # NOTE: wow, this is done all throughout training in most PyTorch models
    if self.track_running_stats and Tensor.training:
      self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
      self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * x.numel()/(x.numel()-x.shape[1]) * batch_var.detach())
      self.num_batches_tracked += 1
    return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt())
BatchNorm2d = BatchNorm3d = BatchNorm

def Conv1d(in_channels:int, out_channels:int, kernel_size:int, stride=1, padding:int|str=0, dilation=1, groups=1, bias=True) -> Conv2d:
  """
  Applies a 1D convolution over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.Conv1d(1, 1, 3)
  t = Tensor.rand(1, 1, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  return Conv2d(in_channels, out_channels, (kernel_size,), stride, padding, dilation, groups, bias)

class Conv2d:
  """
  Applies a 2D convolution over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.Conv2d(1, 1, 3)
  t = Tensor.rand(1, 1, 4, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride=1, padding:int|tuple[int, ...]|str=0,
               dilation=1, groups=1, bias=True):
    self.kernel_size = make_tuple(kernel_size, 2)
    if isinstance(padding, str):
      if padding.lower() != 'same': raise ValueError(f"Invalid padding string {padding!r}, only 'same' is supported")
      if stride != 1: raise ValueError("padding='same' is not supported for strided convolutions")
      pad = [(d*(k-1)//2, d*(k-1) - d*(k-1)//2) for d,k in zip(make_tuple(dilation, len(self.kernel_size)), self.kernel_size[::-1])]
      padding = tuple(flatten(pad))
    self.stride, self.dilation, self.groups, self.padding = stride, dilation, groups, padding
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale)
    self.bias: Tensor|None = Tensor.uniform(out_channels, low=-scale, high=scale) if bias else None

  def __call__(self, x:Tensor) -> Tensor: return x.conv2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding)

def ConvTranspose1d(in_channels:int, out_channels:int, kernel_size:int, stride=1, padding=0, output_padding=0, dilation=1,
                      groups=1, bias=True) -> ConvTranspose2d:
  """
  Applies a 1D transposed convolution operator over an input signal composed of several input planes.

  See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.ConvTranspose1d(1, 1, 3)
  t = Tensor.rand(1, 1, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  return ConvTranspose2d(in_channels, out_channels, (kernel_size,), stride, padding, output_padding, dilation, groups, bias)

class ConvTranspose2d(Conv2d):
  """
  Applies a 2D transposed convolution operator over an input image.

  See: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d

  ```python exec="true" source="above" session="tensor" result="python"
  conv = nn.ConvTranspose2d(1, 1, 3)
  t = Tensor.rand(1, 1, 4, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = conv(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int|tuple[int, ...], stride=1, padding=0, output_padding=0,
                dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
    self.weight = Tensor.uniform(in_channels, out_channels//groups, *self.kernel_size, low=-scale, high=scale)
    self.output_padding = output_padding

  def __call__(self, x:Tensor) -> Tensor:
    return x.conv_transpose2d(self.weight, self.bias, self.groups, self.stride, self.dilation, self.padding, self.output_padding)

class Linear:
  """
  Applies a linear transformation to the incoming data.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Linear

  ```python exec="true" source="above" session="tensor" result="python"
  lin = nn.Linear(3, 4)
  t = Tensor.rand(2, 3)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = lin(t)
  print(t.numpy())
  ```
  """
  def __init__(self, in_features:int, out_features:int, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

  def __call__(self, x:Tensor) -> Tensor: return x.linear(self.weight.transpose(), self.bias)

class BitLinear:
  """
  Applies a bit-quantized linear transformation to the incoming data.
  
  BitLinear uses binary weights (-1, 1) for efficient computation with minimal memory footprint.
  This is an implementation of the BitNet approach where weights are quantized to 1-bit.
  
  Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
  """
  # Constants for I2_S quantization from the C++ code
  QK_K = 256  # Global block size constant
  QK_I2_S = 128  # Block size (K) for quantization (according to C++ code)
  I2S_BLOCK_SIZE_BYTES = 32  # Size of one block in bytes (confirmed in C++ code)
  
  # Masks for I2S quantization
  kmask_iq2xs = [0xc0, 0x30, 0x0c, 0x03]  # Bit masks for 2-bit groups in a byte
  kshift_iq2xs = [6, 4, 2, 0]  # Bit shifts for each group
  
  # Grid lookup table for I2_S quantization (populated by initialize_iq2s_grid)
  iq2s_grid_packed = None
  
  def __init__(self, in_features: int, out_features:int, bias=True):
    bound = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
    self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

    # Scale factor - will be loaded from quantized weights
    # Use consistent naming - let's stick with "scale"
    self.scale = Tensor.ones(out_features, dtype=dtypes.float32)  # One scale per output feature
    
    # For I2_S quantized weights storage in raw block format
    self.raw_blocks = None  # Will be populated when loading weights
    
    self.in_features = in_features
    self.out_features = out_features

  def _quantize_row_q8_1_tensor(self, x: Tensor) -> tuple[Tensor, Tensor]:
    """
    Quantize a row to 8-bit integers and return scales.
    Args:
        x: Input tensor of shape [batch_size, in_features]
    Returns:
        Tuple of (quantized_tensor, scales)
    """
    print(f"[BitLinear._quantize_row_q8_1_tensor] Input shape={x.shape}, dtype={x.dtype}, stats: min={x.min().item():.6f}, max={x.max().item():.6f}, mean={x.mean().item():.6f}")
    
    # Get max absolute value for scaling
    x_abs = x.abs()
    print(f"[BitLinear._quantize_row_q8_1_tensor] x_abs shape={x_abs.shape}, stats: min={x_abs.min().item():.6f}, max={x_abs.max().item():.6f}, mean={x_abs.mean().item():.6f}")
    
    x_max = x_abs.max(axis=1, keepdim=True).clip(min=1e-5)  # Ensure non-zero scale
    print(f"[BitLinear._quantize_row_q8_1_tensor] x_max shape={x_max.shape}, stats: min={x_max.min().item():.6f}, max={x_max.max().item():.6f}, mean={x_max.mean().item():.6f}")
    
    # Compute scale (1/max) and quantize
    scale = 127.0 / x_max
    print(f"[BitLinear._quantize_row_q8_1_tensor] scale shape={scale.shape}, stats: min={scale.min().item():.6f}, max={scale.max().item():.6f}, mean={scale.mean().item():.6f}")
    
    # Apply scaling and clipping
    scaled_x = (x * scale)
    print(f"[BitLinear._quantize_row_q8_1_tensor] scaled_x shape={scaled_x.shape}, stats: min={scaled_x.min().item():.6f}, max={scaled_x.max().item():.6f}, mean={scaled_x.mean().item():.6f}")
    
    # Round to nearest integer and clip to int8 range
    rounded_x = scaled_x.round()
    print(f"[BitLinear._quantize_row_q8_1_tensor] rounded_x shape={rounded_x.shape}, stats: min={rounded_x.min().item():.6f}, max={rounded_x.max().item():.6f}, mean={rounded_x.mean().item():.6f}")
    
    clipped_x = rounded_x.clip(-127, 127)
    print(f"[BitLinear._quantize_row_q8_1_tensor] clipped_x shape={clipped_x.shape}, stats: min={clipped_x.min().item():.6f}, max={clipped_x.max().item():.6f}, mean={clipped_x.mean().item():.6f}")
    
    x_quant = clipped_x.cast(dtypes.int8)
    print(f"[BitLinear._quantize_row_q8_1_tensor] x_quant shape={x_quant.shape}, dtype={x_quant.dtype}, stats: min={x_quant.min().item()}, max={x_quant.max().item()}, mean={x_quant.mean().item():.3f}")
    
    # Return inverse scale for dequantization later
    inv_scale = 1.0 / scale
    print(f"[BitLinear._quantize_row_q8_1_tensor] inv_scale shape={inv_scale.shape}, stats: min={inv_scale.min().item():.6f}, max={inv_scale.max().item():.6f}, mean={inv_scale.mean().item():.6f}")
    
    return x_quant, inv_scale

  def _parse_iq2s_block_tensor(self, block_idx: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    Parse an I2_S data block into tensors with scale, values and grid indices.
    
    Args:
        block_idx: Index of the block to parse
        
    Returns:
        Tuple of (scales, values, grid_indices)
    """
    print(f"[BitLinear._parse_iq2s_block_tensor] Starting for block_idx={block_idx}")
    
    if self.raw_blocks is None:
      print(f"[BitLinear._parse_iq2s_block_tensor] ERROR: raw_blocks is None")
      raise ValueError("No raw blocks data available. Model must be loaded from GGUF format.")
    
    print(f"[BitLinear._parse_iq2s_block_tensor] raw_blocks shape={self.raw_blocks.shape}, dtype={self.raw_blocks.dtype}")
    
    # Check if iq2s_grid_packed is initialized
    if BitLinear.iq2s_grid_packed is None:
      print(f"[BitLinear._parse_iq2s_block_tensor] WARNING: iq2s_grid_packed is None, dequantization will be incorrect")
    else:
      print(f"[BitLinear._parse_iq2s_block_tensor] iq2s_grid_packed shape={BitLinear.iq2s_grid_packed.shape}, dtype={BitLinear.iq2s_grid_packed.dtype}")
    
    # Placeholder - in a real implementation this would parse the raw block data
    # and return the scales, binary values and grid indices for the block
    # For now, just use the regular weights
    block_size = self.QK_I2_S
    start_idx = block_idx * block_size
    end_idx = min((block_idx + 1) * block_size, self.weight.shape[1])
    
    print(f"[BitLinear._parse_iq2s_block_tensor] Block range: {start_idx} to {end_idx} (size {end_idx-start_idx})")
    
    # Mock implementation - extract a slice of the weight tensor for this block
    block_weights = self.weight[:, start_idx:end_idx]
    block_scale = self.scale.unsqueeze(1)
    
    print(f"[BitLinear._parse_iq2s_block_tensor] block_weights shape={block_weights.shape}, stats: min={block_weights.min().item():.6f}, max={block_weights.max().item():.6f}, mean={block_weights.mean().item():.6f}")
    print(f"[BitLinear._parse_iq2s_block_tensor] block_scale shape={block_scale.shape}, stats: min={block_scale.min().item():.6f}, max={block_scale.max().item():.6f}, mean={block_scale.mean().item():.6f}")
    
    # Check what percentage of weights are non-zero
    non_zero_percent = (block_weights != 0).sum().item() / block_weights.numel() * 100
    print(f"[BitLinear._parse_iq2s_block_tensor] Non-zero weights: {non_zero_percent:.2f}%")
    
    # Debug the distribution of weight values in this block
    if block_weights.numel() > 0:
      unique_vals, counts = np.unique(block_weights.numpy(), return_counts=True)
      print(f"[BitLinear._parse_iq2s_block_tensor] Unique values: {unique_vals[:10] if len(unique_vals) > 10 else unique_vals}")
      print(f"[BitLinear._parse_iq2s_block_tensor] Value counts: {counts[:10] if len(counts) > 10 else counts}")
    
    # For I2_S, we should have mostly -1, 0, 1 values
    if block_weights.numel() > 0:
      neg_one_count = ((block_weights == -1).sum().item() / block_weights.numel() * 100)
      zero_count = ((block_weights == 0).sum().item() / block_weights.numel() * 100)
      pos_one_count = ((block_weights == 1).sum().item() / block_weights.numel() * 100)
      print(f"[BitLinear._parse_iq2s_block_tensor] Value distribution: -1: {neg_one_count:.2f}%, 0: {zero_count:.2f}%, 1: {pos_one_count:.2f}%")
      other_count = 100 - neg_one_count - zero_count - pos_one_count
      print(f"[BitLinear._parse_iq2s_block_tensor] Other values: {other_count:.2f}%")
    
    return block_scale, block_weights, None
    
  def __call__(self, x: Tensor) -> Tensor:
    """
    Execute the BitLinear transformation with I2_S quantized weights.
    
    For BitNet's 1.58-bit weights, this is actually simple:
    1. The weight values are already unpacked to -1, 0, 1 when loaded
    2. We just need to scale them by the scale factor
    3. Perform a standard matrix multiplication
    
    Args:
        x: Input tensor of shape [..., in_features]
        
    Returns:
        Output tensor of shape [..., out_features]
    """
    # Scale the binary weights (-1, 0, 1) by the scale factor
    # The scale should be broadcasted across each output row (feature)
    scaled_weight = self.weight * self.scale.reshape(-1, 1)
    
    # Standard matrix multiplication with the scaled binary weights
    return x.linear(scaled_weight, self.bias)

  @staticmethod
  def unpack_i2_weights(data_bytes, shape):
    """
    Unpacks 2-bit quantized weights from raw bytes into a tensor of shape `shape`.
    Based on the C++ implementation in the ggml-bitnet.cpp code.
    
    The quantization format works as follows:
    - Values are represented using 2 bits: 00 -> -1, 01 -> 0, 10 -> 1 (11 is unused)
    - Data is organized in blocks of 128 (QK_I2_S) elements
    - Each block is 32 bytes, with each byte containing 4 2-bit values
    - The bits are organized in specific positions: bits 6-7, 4-5, 2-3, 0-1
    - A scale factor (float32) is stored after the data
    """
    import struct
    import numpy as np
    from tinygrad.helpers import prod
    
    n = prod(shape)  # Total number of weights
    
    # Constants from the C++ code
    QK_I2_S = BitLinear.QK_I2_S  # 128 elements per block
    bytes_per_block = BitLinear.I2S_BLOCK_SIZE_BYTES  # 32 bytes per block
    nb = n // QK_I2_S  # Number of full blocks
    
    # Calculate the size of the packed data (excluding the scale)
    data_size = (n * 2 + 7) // 8  # Each element is 2 bits
    
    # Extract the scale factor (stored as float32 at the end)
    scale = struct.unpack('f', data_bytes[data_size:data_size+4])[0]
    
    # Prepare the output array
    weights = np.zeros(n, dtype=np.float32)
    
    # Process each block
    for block_idx in range(nb):
        # Each block contains 4 groups, each group has 32 elements
        for group_idx in range(4):
            # Process each of the 32 elements in this group
            for pos in range(32):
                # Find the byte index based on the C++ implementation
                byte_idx = block_idx * bytes_per_block + pos
                byte = data_bytes[byte_idx]
                
                # Extract the 2-bit value using the mask and shift for this group
                mask = BitLinear.kmask_iq2xs[group_idx]
                shift = BitLinear.kshift_iq2xs[group_idx]
                two_bit_val = (byte & mask) >> shift
                
                # Map the 2-bit value to -1, 0, 1
                if two_bit_val == 0:
                    val = -1.0
                elif two_bit_val == 1:
                    val = 0.0
                elif two_bit_val == 2:
                    val = 1.0
                else:  # 3 (11) - unused in the format
                    val = 0.0
                
                # Calculate the index in the weights array
                idx = block_idx * QK_I2_S + group_idx * 32 + pos
                if idx < n:  # Safety check to prevent out-of-bounds
                    weights[idx] = val
    
    # Create tensors for weights and scale
    weight_tensor = Tensor(weights.reshape(shape))
    scale_tensor = Tensor([scale], dtype=dtypes.float32)
    
    return weight_tensor, scale_tensor

class GroupNorm:
  """
  Applies Group Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/group-normalization
  - Paper: https://arxiv.org/abs/1803.08494v3

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.GroupNorm(2, 12)
  t = Tensor.rand(2, 12, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, num_groups:int, num_channels:int, eps=1e-5, affine=True):
    self.num_groups, self.num_channels, self.eps = num_groups, num_channels, eps
    self.weight: Tensor|None = Tensor.ones(num_channels) if affine else None
    self.bias: Tensor|None = Tensor.zeros(num_channels) if affine else None

  def __call__(self, x:Tensor) -> Tensor:
    # reshape for layernorm to work as group norm
    # subtract mean and divide stddev
    x = x.reshape(x.shape[0], self.num_groups, -1).layernorm(eps=self.eps).reshape(x.shape)

    if self.weight is None or self.bias is None: return x
    # elementwise_affine on channels
    return x * self.weight.reshape(1, -1, *[1] * (x.ndim-2)) + self.bias.reshape(1, -1, *[1] * (x.ndim-2))

class InstanceNorm:
  """
  Applies Instance Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/instance-normalization
  - Paper: https://arxiv.org/abs/1607.08022v3

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.InstanceNorm(3)
  t = Tensor.rand(2, 3, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, num_features:int, eps=1e-5, affine=True):
    self.num_features, self.eps = num_features, eps
    self.weight: Tensor|None = Tensor.ones(num_features) if affine else None
    self.bias: Tensor|None = Tensor.zeros(num_features) if affine else None

  def __call__(self, x:Tensor) -> Tensor:
    x = x.reshape(x.shape[0], self.num_features, -1).layernorm(eps=self.eps).reshape(x.shape)
    if self.weight is None or self.bias is None: return x
    return x * self.weight.reshape(1, -1, *[1] * (x.ndim-2)) + self.bias.reshape(1, -1, *[1] * (x.ndim-2))

class LayerNorm:
  """
  Applies Layer Normalization over a mini-batch of inputs.

  - Described: https://paperswithcode.com/method/layer-normalization
  - Paper: https://arxiv.org/abs/1607.06450v1

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.LayerNorm(3)
  t = Tensor.rand(2, 5, 3) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __init__(self, normalized_shape:int|tuple[int, ...], eps=1e-5, elementwise_affine=True):
    self.normalized_shape: tuple[int, ...] = make_tuple(normalized_shape, 1)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight: Tensor|None = Tensor.ones(*self.normalized_shape) if elementwise_affine else None
    self.bias: Tensor|None = Tensor.zeros(*self.normalized_shape) if elementwise_affine else None

  def __call__(self, x:Tensor) -> Tensor:
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    x = x.layernorm(eps=self.eps, axis=self.axis)
    if not self.elementwise_affine: return x
    return x * self.weight + self.bias

class LayerNorm2d(LayerNorm):
  """
  Applies Layer Normalization over a mini-batch of 2D inputs.

  See: `LayerNorm`

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.LayerNorm2d(3)
  t = Tensor.rand(2, 3, 4, 4) * 2 + 1
  print(t.mean().item(), t.std().item())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  t = norm(t)
  print(t.mean().item(), t.std().item())
  ```
  """
  def __call__(self, x: Tensor) -> Tensor: return super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class RMSNorm:
  """
  Applies Root Mean Square Normalization to input.

  - Described: https://paperswithcode.com/method/rmsnorm
  - Paper: https://arxiv.org/abs/1910.07467

  ```python exec="true" source="above" session="tensor" result="python"
  norm = nn.RMSNorm(4)
  t = Tensor.arange(12, dtype=dtypes.float).reshape(3, 4)
  print(t.numpy())
  ```
  ```python exec="true" source="above" session="tensor" result="python"
  print(norm(t).numpy())
  ```
  """
  def __init__(self, dim:int, eps=1e-6): self.eps, self.weight = eps, Tensor.ones(dim)

  def _norm(self, x:Tensor) -> Tensor: return x * (x.square().mean(-1, keepdim=True) + self.eps).rsqrt()

  def __call__(self, x:Tensor) -> Tensor: return self._norm(x.float()).cast(x.dtype) * self.weight

class Embedding:
  """
  A simple lookup table that stores embeddings of a fixed dictionary and size.

  See: https://pytorch.org/docs/stable/generated/torch.nn.Embedding

  ```python exec="true" source="above" session="tensor" result="python"
  emb = nn.Embedding(10, 3)
  print(emb(Tensor([1, 2, 3, 1])).numpy())
  ```
  """
  def __init__(self, vocab_size:int, embed_size:int):
    self.vocab_sz, self.embed_sz, self.weight = vocab_size, embed_size, Tensor.glorot_uniform(vocab_size, embed_size)

  def __call__(self, idx:Tensor) -> Tensor:
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).unsqueeze(-1)
    big_shp = idx.shape+(self.vocab_sz, self.embed_sz)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1)).expand(big_shp), self.weight.expand(big_shp)
    return (arange == idx).mul(vals).sum(-2, dtype=vals.dtype)

class LSTMCell:
  """
  A long short-term memory (LSTM) cell.

  Args:
    input_size: The number of expected features in the input `x`
    hidden_size: The number of features in the hidden state `h`
    bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`
  """
  def __init__(self, input_size:int, hidden_size:int, bias:bool=True):
    stdv = 1.0 / math.sqrt(hidden_size)
    self.weight_ih = Tensor.uniform(hidden_size*4, input_size, low=-stdv, high=stdv)
    self.weight_hh = Tensor.uniform(hidden_size*4, hidden_size, low=-stdv, high=stdv)
    self.bias_ih: Tensor|None = Tensor.zeros(hidden_size*4) if bias else None
    self.bias_hh: Tensor|None = Tensor.zeros(hidden_size*4) if bias else None

  def __call__(self, x:Tensor, hc:tuple[Tensor, Tensor]|None=None) -> tuple[Tensor, Tensor]:
    if hc is None: hc = (Tensor.zeros(x.size(0), self.weight_hh.size(1), dtype=x.dtype, device=x.device),)*2
    gates = x.linear(self.weight_ih.T, self.bias_ih) + hc[0].linear(self.weight_hh.T, self.bias_hh)
    i, f, g, o = gates.chunk(4, dim=1)
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    new_c = f * hc[1] + i * g
    new_h = o * new_c.tanh()
    return (new_h.contiguous(), new_c.contiguous())
